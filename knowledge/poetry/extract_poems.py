#!/usr/bin/env python3
"""Extract poems from Poems1.txt and Poems2.md into a single poems.jsonl.

Poems1.txt format (semi-markdown):
    # TITLE by AUTHOR
    <blank line>
    <poem body>
    <blank line>
    (next poem)

    - The author is always in FULL CAPS, including things like commas (e.g.
      "HENRY HOWARD, EARL OF SURREY").
    - The title is the rest of the header line, with the leading "# " removed.
    - Some titles themselves contain the word "by" (e.g. "Down By the Salley
      Gardens"). To handle this we split the header on the *last* occurrence
      of " by " (lowercase), which is what separates the title from the
      FULL-CAPS author.

Poems2.md format (full Markdown):
    ## TITLE
    **Author:** Author Name
    <blank line>
    <poem body>
    <blank line>
    ----------   <-- separator before next poem (except before the very first
                       poem, where the file starts with the separator)

    - The poem title is the text after "## ".
    - The author is the text after "**Author:**" on the very next line.
    - Each poem is separated from the next by a line of ten or more dashes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
POEMS1 = HERE / 'Poems1.txt'
POEMS2 = HERE / 'Poems2.md'
OUTPUT = HERE / 'poems.jsonl'


# ---------------------------------------------------------------------------
# Poems1.txt
# ---------------------------------------------------------------------------

# Header line of a poem in Poems1.txt: "# TITLE by AUTHOR"
POEM1_HEADER_RE = re.compile(r'^#\s+(.*?)\s+by\s+(.+)$')


# Words that should remain lowercased in Title Case even when they are
# the first word of the name (we'll handle "first word" separately so this
# is just for non-leading positions).
SMALL_WORDS = {
    'a',
    'an',
    'and',
    'as',
    'at',
    'but',
    'by',
    'for',
    'in',
    'of',
    'on',
    'or',
    'the',
    'to',
    'vs',
    'via',
    'de',
    'du',
    'von',
    'van',
    'der',
    'den',
    'ten',
    'ter',
}

# Roman numerals used as regnal/royal numbers. Anything in this set is
# preserved in its original uppercase form so that "HENRY VIII, KING OF
# ENGLAND" -> "Henry VIII, King of England".
ROMAN_NUMERALS = {
    'I',
    'II',
    'III',
    'IV',
    'V',
    'VI',
    'VII',
    'VIII',
    'IX',
    'X',
    'XI',
    'XII',
    'XIII',
    'XIV',
    'XV',
    'XVI',
    'XVII',
    'XVIII',
    'XIX',
    'XX',
    'XXI',
    'XXII',
    'XXIII',
    'XXIV',
    'XXV',
}


def _title_case_author(name: str) -> str:
    """Convert a FULL-CAPS author name to Title Case.

    Preserves all-uppercase short tokens like "E.", "E.E.", "II", "III"
    (so "E. E. CUMMINGS" -> "E. E. Cummings") and small words like "of"
    are lowercased when they appear mid-name (so "HENRY HOWARD, EARL OF
    SURREY" -> "Henry Howard, Earl of Surrey"). Small words appearing as
    the first word of the name are still capitalized (so "A Poet" stays
    "A Poet").
    """
    if not name:
        return name

    # Split on whitespace, preserving the original separators so we can
    # rejoin faithfully.
    tokens = re.split(r'(\s+)', name)
    out: list[str] = []
    word_index = 0  # counts only non-whitespace tokens
    for tok in tokens:
        if tok.isspace() or not tok:
            out.append(tok)
            continue
        # Letters-only view for analysis.
        stripped = re.sub(r'[^A-Za-z]', '', tok)
        stripped_lower = stripped.lower()
        # 1. If this token (case-insensitive) is a SMALL_WORD, lowercase it
        #    unless it's the very first word of the name.
        if stripped_lower in SMALL_WORDS:
            if word_index == 0:
                # First word: still capitalize the first letter.
                lower = tok.lower()
                out.append(lower[:1].upper() + lower[1:])
            else:
                # Mid-name: fully lowercase.
                out.append(tok.lower())
            word_index += 1
            continue
        # 2. Otherwise, preserve tokens that look like initials/acronyms
        #    (contain a period, e.g. "E.", "E.E.", "U.S.") or that are
        #    Roman numerals (e.g. "VIII", "IV"). Everything else gets
        #    plain title-cased, so "EN" -> "En", "SIR" -> "Sir", "KING"
        #    -> "King", even though they are short and uppercase.
        if '.' in tok or stripped in ROMAN_NUMERALS:
            out.append(tok)
            word_index += 1
            continue
        # 3. Otherwise: lowercase, then capitalize the first letter.
        lower = tok.lower()
        out.append(lower[:1].upper() + lower[1:])
        word_index += 1
    return ''.join(out)


def _strip_quotes(s: str) -> str:
    """Strip matching outer quote pairs (single or double) from a string.

    Only strips when the entire string is enclosed in a matching pair, so
    titles like ``Amoretti LXII: "The weary yeare..."`` (where the quote
    is internal and unbalanced) are left untouched.
    """
    s = s.strip()
    # Strip at most one matching double-quote pair from each end.
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    # Then at most one matching single-quote pair from each end.
    if len(s) >= 2 and s.startswith("'") and s.endswith("'"):
        s = s[1:-1]
    return s


def parse_poems1(text: str) -> list[dict]:
    """Parse Poems1.txt into a list of {title, author, text} dicts."""
    poems: list[dict] = []
    current_title: str | None = None
    current_author: str | None = None
    current_body: list[str] = []

    def flush() -> None:
        if current_title is None:
            return
        # Trim trailing blank lines but preserve internal blank lines which
        # are part of the poem's verse structure.
        body = '\n'.join(current_body).rstrip('\n')
        poems.append(
            {
                'title': _strip_quotes(current_title),
                'author': _title_case_author(current_author or ''),
                'text': body,
            }
        )

    for raw_line in text.splitlines():
        line = raw_line.rstrip('\n')
        # Match the header line "# TITLE by AUTHOR".
        # We split on the LAST " by " to handle titles that themselves
        # contain the word "by" (e.g. "Down By the Salley Gardens").
        if line.startswith('# ') and ' by ' in line:
            # Flush the previous poem before starting a new one.
            flush()
            header = line[2:]  # drop leading "# "
            # Split on the LAST occurrence of " by " (lowercase) so that any
            # "by" appearing inside the title stays in the title.
            idx = header.rfind(' by ')
            title = header[:idx].strip()
            author = header[idx + len(' by ') :].strip()
            current_title = title
            current_author = author
            current_body = []
        else:
            # Body line of the current poem. We keep blank lines because they
            # are part of the poem's structure; we only strip trailing blanks
            # at flush time.
            current_body.append(line)

    flush()
    return poems


# ---------------------------------------------------------------------------
# Poems2.md
# ---------------------------------------------------------------------------

POEM2_TITLE_RE = re.compile(r'^##\s+(.*)$')
POEM2_AUTHOR_RE = re.compile(r'^\*\*Author:\*\*\s*(.*)$')
POEM2_SEPARATOR_RE = re.compile(r'^-{10,}\s*$')


def parse_poems2(text: str) -> list[dict]:
    """Parse Poems2.md into a list of {title, author, text} dicts."""
    poems: list[dict] = []
    lines = text.splitlines()

    i = 0
    n = len(lines)

    # Skip the leading "----------" separator that may appear at the very
    # top of the file before the first poem.
    while i < n and POEM2_SEPARATOR_RE.match(lines[i]):
        i += 1

    current_title: str | None = None
    current_author: str | None = None
    current_body: list[str] = []
    expecting_author = False

    def flush() -> None:
        nonlocal current_title, current_author, current_body, expecting_author
        if current_title is None:
            return
        body = '\n'.join(current_body).rstrip('\n')
        poems.append(
            {
                'title': _strip_quotes(current_title),
                'author': current_author or '',
                'text': body,
            }
        )
        current_title = None
        current_author = None
        current_body = []
        expecting_author = False

    while i < n:
        line = lines[i]
        if POEM2_SEPARATOR_RE.match(line):
            # Separator ends the current poem; the next non-separator line
            # will be the next poem's title.
            flush()
            while i < n and POEM2_SEPARATOR_RE.match(lines[i]):
                i += 1
            continue

        m_title = POEM2_TITLE_RE.match(line)
        if m_title:
            # If we hit a "## ..." while we are still in the body of the
            # previous poem (shouldn't normally happen because of the
            # separator), flush the previous one first.
            if current_title is not None and current_body:
                flush()
            current_title = m_title.group(1).strip()
            expecting_author = True
            i += 1
            continue

        if expecting_author:
            m_author = POEM2_AUTHOR_RE.match(line)
            if m_author:
                current_author = m_author.group(1).strip()
                expecting_author = False
                i += 1
                continue
            # If we don't find an author line right after the title, that's
            # unexpected - keep looking but stop expecting the author.
            expecting_author = False

        # Otherwise, this line belongs to the body of the current poem.
        if current_title is not None:
            current_body.append(line)
        i += 1

    flush()
    return poems


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    poems1_text = POEMS1.read_text(encoding='utf-8')
    poems2_text = POEMS2.read_text(encoding='utf-8')

    poems1 = parse_poems1(poems1_text)
    poems2 = parse_poems2(poems2_text)

    print(f'Poems1.txt: {len(poems1)} poems')
    print(f'Poems2.md:  {len(poems2)} poems')
    print(f'Total:      {len(poems1) + len(poems2)} poems')

    with OUTPUT.open('w', encoding='utf-8') as f:
        for p in poems1 + poems2:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f'Wrote {OUTPUT}')


if __name__ == '__main__':
    main()
