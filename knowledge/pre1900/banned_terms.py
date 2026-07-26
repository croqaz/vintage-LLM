"""Standalone check for post-1900 anachronisms in a piece of text.

This module is intentionally self-contained and dependency-free so you can copy
it (together with the data/ folder) into any project and reuse it.

It flags a text as "anachronistic" (i.e. it could NOT have been written before
1900) when the text contains either:

  1. a BANNED term/phrase  -- from data/banned.txt (a curated blocklist of
     post-1900 concepts, brands, people, tech, etc.), minus anything you have
     listed in data/allowed.txt (your escape hatch for false positives), or
  2. an explicit post-1900 YEAR (1900-2099) -- a pre-1900 author cannot refer
     to a year that is still in their future. Pre-1900 years (e.g. 1865) are
     fine and never flagged.

Library use:
    from banned_terms import contains_anachronism, find_anachronisms
    if contains_anachronism(text):      # -> True / False
        ...
    find_anachronisms(text)             # -> sorted list of what matched

Command-line use:
    python banned_terms.py "We watched a movie in 2016."   # prints matches
    echo "some text" | python banned_terms.py              # reads STDIN
    python banned_terms.py --quiet file.txt && echo CLEAN   # exit code only
        exit code 0 = clean (no anachronism), 1 = anachronism found
"""

import os
import re
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# A pre-1900 author cannot reference a year >= 1900. Matched on raw text because
# years are digits. Bounded to 1900-2099 so we don't catch arbitrary numbers
# like "the year 1066" (pre-1900, fine) or unrelated 4-digit quantities far off.
# Match 1901-2099, but exclude:
#   - 1900 (pre-1900 year, allowed)
#   - years followed by B.C. / BCE (e.g. "2080 B. C.", "2000 B.C.")
#   - years that are the endpoint of a BC range (e.g. "from 2700 B.C. to 2080")
_YEAR_RE = re.compile(r'\b(?!1900\b)(?:19\d\d|20\d\d)\b(?!\s*B\.?\s*C\.?)')
_BC_RANGE_RE = re.compile(r'\bB\.?\s*C\.?\s*,?\s*to\s+(\d{4})\b', re.I)

_REGEX = None


def _load_list(name):
    """Return the lowercased non-comment lines of data/<name> as a set."""
    path = os.path.join(DATA, name)
    out = set()
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            line = line.strip().lower()
            if line and not line.startswith('#'):
                out.add(line)
    return out


def _build_regex():
    """Compile a whole-word/phrase regex of (banned - allowed) terms."""
    banned = _load_list('banned.txt')
    allowed = _load_list('allowed.txt')  # terms here are never flagged
    terms = banned - allowed
    if not terms:
        return None
    # longest first so multi-word phrases win over their substrings.
    # A trailing (?:'s|es|s)? lets a term match its plural / possessive without
    # having to list every form: "website" also catches "websites", "nuclear
    # bomb" catches "nuclear bombs", "google" catches "google's".
    pats = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    # (?<!\w)...(?!\w) instead of \b...\b so terms ending in punctuation match
    # too: \b needs a word char at the edge, so it never matched "c#" or "c++".
    return re.compile(r'(?<!\w)(?:' + '|'.join(pats) + r")(?:'s|es|s|ed)?(?!\w)", re.I)


def _regex():
    global _REGEX
    if _REGEX is None:
        _REGEX = _build_regex()
    return _REGEX


# ---- fast path -----------------------------------------------------------
# The single big regex is slow on long documents (it scans megabytes with a
# 700+ alternation and lookarounds). Split it: pure single-word terms become
# O(1) set lookups against the already-tokenized words; only multi-word / special
# terms (spaces, '+', '#', '<', code fences, ...) stay in a smaller regex.
_ALPHA = re.compile(r'[a-z]+')
_ALPHA_PHRASE = re.compile(r'[a-z]+( [a-z]+)+')
_FAST = None


def _build_fast():
    """single: pure-word terms (token lookup); by_n: {n: set of n-word phrases}
    (token n-gram lookup); other_re: regex for the rest (c++, <html>, ```py, ...)."""
    banned = _load_list('banned.txt')
    allowed = _load_list('allowed.txt')
    single, by_n, other = set(), {}, []
    for t in banned - allowed:
        if _ALPHA.fullmatch(t):
            single.add(t)
        elif _ALPHA_PHRASE.fullmatch(t):
            by_n.setdefault(t.count(' ') + 1, set()).add(t)
        else:
            other.append(t)
    pats = sorted((re.escape(t) for t in other), key=len, reverse=True)
    other_re = re.compile(r'(?<!\w)(?:' + '|'.join(pats) + r")(?:'s|es|s|ed)?(?!\w)", re.I) if pats else None
    return single, by_n, other_re


def _fast():
    global _FAST
    if _FAST is None:
        _FAST = _build_fast()
    return _FAST


def _forms(w):
    """Yield a token and its singular/base forms (mirrors the '?:'s|es|s|ed' suffix)."""
    yield w
    if "'" in w:
        yield w.split("'", 1)[0]
        return
    if len(w) > 2:
        if w.endswith(('es', 'ed')):
            yield w[:-2]
        if w.endswith('s'):
            yield w[:-1]


def find_anachronisms_fast(text, tokens, check_years=True) -> list[str]:
    """Fast equivalent of find_anachronisms() when the tokens are already known.

    Same keep/drop decision as find_anachronisms() (may report a superset of hit
    labels, never fewer). Single words + phrases are matched against the tokens;
    only special terms (punctuation/code/HTML) and years touch the raw text.
    """
    single, by_n, other_re = _fast()
    hits = set()
    if check_years:
        hits.update(m.group(0) for m in _YEAR_RE.finditer(text))
        # Remove years that are the endpoint of a BC range (e.g. "from 2700 B.C. to 2080")
        bc_years = set(m.group(1) for m in _BC_RANGE_RE.finditer(text))
        hits.difference_update(bc_years)
    if other_re:
        hits.update(m.group(0).lower() for m in other_re.finditer(text))
    # single words
    for tok in set(tokens):
        for f in _forms(tok):
            if f in single:
                hits.add(f)
                break
    # multi-word phrases via token n-grams (plural allowed on the last word)
    ntok = len(tokens)
    for n, phrases in by_n.items():
        for i in range(ntok - n + 1):
            head = ' '.join(tokens[i : i + n - 1])
            prefix = head + ' ' if head else ''
            for last in _forms(tokens[i + n - 1]):
                p = prefix + last
                if p in phrases:
                    hits.add(p)
                    break
    return sorted(hits)


def find_anachronisms(text, check_years=True) -> list[str]:
    """Return a sorted list of every banned term / post-1900 year found in text.

    check_years=False restricts the check to the banned.txt term list only.
    """
    hits = set()
    rx = _regex()
    if rx:
        hits.update(m.group(0).lower() for m in rx.finditer(text))
    if check_years:
        hits.update(m.group(0) for m in _YEAR_RE.finditer(text))
        bc_years = set(m.group(1) for m in _BC_RANGE_RE.finditer(text))
        hits.difference_update(bc_years)
    return sorted(hits)


def contains_anachronism(text, check_years=True) -> bool:
    """True if the text contains ANY banned term or post-1900 year."""
    return bool(find_anachronisms(text, check_years=check_years))


def _main(argv):
    args = [a for a in argv if not a.startswith('-')]
    quiet = '--quiet' in argv or '-q' in argv
    no_years = '--no-years' in argv
    if args:
        # treat the argument as a file path if it exists, else as literal text
        text = open(args[0], encoding='utf-8', errors='ignore').read() if os.path.exists(args[0]) else ' '.join(args)
    else:
        text = sys.stdin.read()
    hits = find_anachronisms(text, check_years=not no_years)
    if not quiet:
        print('ANACHRONISTIC' if hits else 'CLEAN', '|', 'hits:', hits)
    return 1 if hits else 0


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))
