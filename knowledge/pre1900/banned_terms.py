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
_YEAR_RE = re.compile(r'\b(?:19\d\d|20\d\d)\b')

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
    # longest first so multi-word phrases win over their substrings
    pats = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    return re.compile(r'\b(' + '|'.join(pats) + r')\b', re.I)


def _regex():
    global _REGEX
    if _REGEX is None:
        _REGEX = _build_regex()
    return _REGEX


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
    return sorted(hits)


def contains_anachronism(text, check_years=True) -> bool:
    """True if the text contains ANY banned term or post-1900 year."""
    rx = _regex()
    if rx and rx.search(text):
        return True
    if check_years and _YEAR_RE.search(text):
        return True
    return False


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
