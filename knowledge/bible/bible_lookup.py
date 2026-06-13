#!/usr/bin/env python3
"""
Bible reference lookup tool.
Reads the King James Bible (1611) text file and returns verses for given references.

Usage:
    python bible_lookup.py "1 Corinthians 13:13"
    python bible_lookup.py "1 John 4:19" "Galatians 5:22-23" "Matthew 18:21-22"
"""

import re
import sys
from pathlib import Path

# Path to the Bible text file
BIBLE_PATH = Path(__file__).parent / 'King James Bible 1611.txt'

# ---------------------------------------------------------------------------
# Book name mapping: friendly name -> full title in the KJV text file
# ---------------------------------------------------------------------------

BOOK_MAP = {
    # ---- Old Testament ----
    'genesis': 'The First Book of Moses: Called Genesis',
    'exodus': 'The Second Book of Moses: Called Exodus',
    'leviticus': 'The Third Book of Moses: Called Leviticus',
    'numbers': 'The Fourth Book of Moses: Called Numbers',
    'deuteronomy': 'The Fifth Book of Moses: Called Deuteronomy',
    'joshua': 'The Book of Joshua',
    'judges': 'The Book of Judges',
    'ruth': 'The Book of Ruth',
    '1 samuel': 'The First Book of Samuel',
    'i samuel': 'The First Book of Samuel',
    '2 samuel': 'The Second Book of Samuel',
    'ii samuel': 'The Second Book of Samuel',
    '1 kings': 'The First Book of the Kings',
    'i kings': 'The First Book of the Kings',
    '2 kings': 'The Second Book of the Kings',
    'ii kings': 'The Second Book of the Kings',
    '1 chronicles': 'The First Book of the Chronicles',
    'i chronicles': 'The First Book of the Chronicles',
    '2 chronicles': 'The Second Book of the Chronicles',
    'ii chronicles': 'The Second Book of the Chronicles',
    'ezra': 'Ezra',
    'nehemiah': 'The Book of Nehemiah',
    'esther': 'The Book of Esther',
    'job': 'The Book of Job',
    'psalms': 'The Book of Psalms',
    'psalm': 'The Book of Psalms',
    'proverbs': 'The Proverbs',
    'ecclesiastes': 'Ecclesiastes',
    'song of solomon': 'The Song of Solomon',
    'song of songs': 'The Song of Solomon',
    'isaiah': 'The Book of the Prophet Isaiah',
    'jeremiah': 'The Book of the Prophet Jeremiah',
    'lamentations': 'The Lamentations of Jeremiah',
    'ezekiel': 'The Book of the Prophet Ezekiel',
    'daniel': 'The Book of Daniel',
    'hosea': 'Hosea',
    'joel': 'Joel',
    'amos': 'Amos',
    'obadiah': 'Obadiah',
    'jonah': 'Jonah',
    'micah': 'Micah',
    'nahum': 'Nahum',
    'habakkuk': 'Habakkuk',
    'zephaniah': 'Zephaniah',
    'haggai': 'Haggai',
    'zechariah': 'Zechariah',
    'malachi': 'Malachi',
    # ---- New Testament ----
    'matthew': 'The Gospel According to Saint Matthew',
    'mark': 'The Gospel According to Saint Mark',
    'luke': 'The Gospel According to Saint Luke',
    'john': 'The Gospel According to Saint John',
    'acts': 'The Acts of the Apostles',
    'romans': 'The Epistle of Paul the Apostle to the Romans',
    '1 corinthians': 'The First Epistle of Paul the Apostle to the Corinthians',
    'i corinthians': 'The First Epistle of Paul the Apostle to the Corinthians',
    '2 corinthians': 'The Second Epistle of Paul the Apostle to the Corinthians',
    'ii corinthians': 'The Second Epistle of Paul the Apostle to the Corinthians',
    'galatians': 'The Epistle of Paul the Apostle to the Galatians',
    'ephesians': 'The Epistle of Paul the Apostle to the Ephesians',
    'philippians': 'The Epistle of Paul the Apostle to the Philippians',
    'colossians': 'The Epistle of Paul the Apostle to the Colossians',
    '1 thessalonians': 'The First Epistle of Paul the Apostle to the Thessalonians',
    'i thessalonians': 'The First Epistle of Paul the Apostle to the Thessalonians',
    '2 thessalonians': 'The Second Epistle of Paul the Apostle to the Thessalonians',
    'ii thessalonians': 'The Second Epistle of Paul the Apostle to the Thessalonians',
    '1 timothy': 'The First Epistle of Paul the Apostle to Timothy',
    'i timothy': 'The First Epistle of Paul the Apostle to Timothy',
    '2 timothy': 'The Second Epistle of Paul the Apostle to Timothy',
    'ii timothy': 'The Second Epistle of Paul the Apostle to Timothy',
    'titus': 'The Epistle of Paul the Apostle to Titus',
    'philemon': 'The Epistle of Paul the Apostle to Philemon',
    'hebrews': 'The Epistle of Paul the Apostle to the Hebrews',
    'james': 'The General Epistle of James',
    '1 peter': 'The First Epistle General of Peter',
    'i peter': 'The First Epistle General of Peter',
    '2 peter': 'The Second General Epistle of Peter',
    'ii peter': 'The Second General Epistle of Peter',
    '1 john': 'The First Epistle General of John',
    'i john': 'The First Epistle General of John',
    '2 john': 'The Second Epistle General of John',
    'ii john': 'The Second Epistle General of John',
    '3 john': 'The Third Epistle General of John',
    'iii john': 'The Third Epistle General of John',
    'jude': 'The General Epistle of Jude',
    'revelation': 'The Revelation of Saint John the Divine',
}

# Build a reverse mapping: full title -> friendly name (for display)
TITLE_TO_FRIENDLY = {v: k for k, v in BOOK_MAP.items()}


def resolve_book(name: str) -> str:
    """Convert a user-supplied book name into the full KJV title."""
    key = name.strip().lower()
    if key in BOOK_MAP:
        return BOOK_MAP[key]
    raise ValueError(f"Unknown book: '{name}'")


# ---------------------------------------------------------------------------
# Verse extraction engine
# ---------------------------------------------------------------------------

# Matches chapter:verse markers, e.g. "1:1", "13:13", "2:1"
VERSE_MARKER_RE = re.compile(r'\b(\d+):(\d+)\b')


def find_book_start(full_text: str, book_title: str) -> int:
    """Return the character offset where *book_title* appears as a standalone
    book header in the Bible body (not in the TOC or inside other books)."""
    # Books in the body appear as standalone header lines surrounded by blank
    # lines. We use rfind with this pattern to avoid matching personal names
    # (e.g. "Amos", "Job", "Zechariah") that appear inside other books.
    pos = full_text.rfind(f'\n\n{book_title}\n\n')
    if pos == -1:
        raise ValueError(f'Book title not found in file: {book_title}')
    return pos + 2  # +2 to skip the "\n\n" prefix, pointing to the title itself


def find_next_book_start(full_text: str, current_book_start: int) -> int:
    """Return the start offset of the next book after current_book_start,
    or len(full_text) if this is the last book."""
    # Books in the body appear as standalone lines: they are preceded by
    # "\n\n" (a blank line before the title) and followed by "\n\n\n"
    # (a blank line after). We require this pattern to avoid matching
    # personal names (e.g. "Job", "Zechariah") inside other books' text.
    best = len(full_text)
    # Skip past the current book's title line
    search_start = full_text.find('\n', current_book_start) + 1
    for title in BOOK_MAP.values():
        # Look for the pattern: blank line + title + blank line
        pattern = '\n\n' + title + '\n\n'
        pos = full_text.find(pattern, search_start)
        if pos != -1 and pos < best:
            best = pos + 2  # +2 to point to the title itself (past "\n\n")
    return best


def extract_verses(
    full_text: str,
    book_title: str,
    chapter: int,
    start_verse: int,
    end_verse: int,
) -> list[str]:
    """Extract verses [start_verse .. end_verse] from the given chapter of book.

    Returns a list of strings, one per verse, each cleaned of the verse marker.
    """
    book_start = find_book_start(full_text, book_title)
    book_end = find_next_book_start(full_text, book_start)
    book_text = full_text[book_start:book_end]

    # Split into (verse_marker, text_between) segments using regex
    # We find all markers with their positions
    segments: list[tuple[int, int, str]] = []  # (chapter, verse, text)
    last_pos = 0

    for m in VERSE_MARKER_RE.finditer(book_text):
        ch = int(m.group(1))
        vs = int(m.group(2))
        text_start = m.end()
        segments.append((ch, vs, m.start(), text_start))

    results = []
    for i, (ch, vs, marker_start, text_start) in enumerate(segments):
        if ch != chapter:
            continue
        if vs < start_verse or vs > end_verse:
            continue

        # Find where this verse's text ends = start of next verse marker in
        # the same or next chapter, or end of book_text
        text_end = len(book_text)
        for j in range(i + 1, len(segments)):
            _, _, next_marker_start, _ = segments[j]
            text_end = next_marker_start
            break

        raw = book_text[text_start:text_end]
        # Clean: remove extra whitespace, join lines into a single paragraph
        clean = ' '.join(raw.split())
        # Remove any trailing verse marker that may have been caught:
        # (e.g. if the next verse started inline like "text 2:1 next verse")
        # The text_end already excludes the next marker, so we are safe.

        results.append(clean)

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lookup(full_text: str, reference: str) -> str:
    """Parse a reference like '1 Corinthians 13:13' or 'Galatians 5:22-23'
    and return the verse text(s), formatted with reference headers.

    Returns a multi-line string.
    """
    # Split into book name and verse spec
    # e.g. "1 Corinthians 13:13" -> book="1 Corinthians", spec="13:13"
    # e.g. "Song of Solomon 1:5" -> book="Song of Solomon", spec="1:5"
    # e.g. "1 John 4:19-21" -> book="1 John", spec="4:19-21"

    full_ref = reference.strip()
    # Use regex to separate book from verse spec
    m2 = re.match(r'^(.+?)\s+(\d+:\d+(?:-\d+)?)$', full_ref)
    if not m2:
        return f"[ERROR] Could not parse reference: '{reference}'"

    book_name = m2.group(1).strip()
    verse_spec = m2.group(2).strip()

    # Parse verse_spec:  "chapter:verse" or "chapter:start-end"
    parts = verse_spec.split(':')
    if len(parts) != 2:
        return f"[ERROR] Bad verse spec: '{verse_spec}'"

    chapter = int(parts[0])
    verse_part = parts[1]
    if '-' in verse_part:
        vs_start, vs_end = verse_part.split('-', 1)
        start_verse = int(vs_start)
        end_verse = int(vs_end)
    else:
        start_verse = int(verse_part)
        end_verse = start_verse

    if start_verse > end_verse:
        return f'[ERROR] Invalid verse range: {start_verse}-{end_verse}'

    # Resolve book
    full_title = resolve_book(book_name)

    # Extract verses
    verses = extract_verses(full_text, full_title, chapter, start_verse, end_verse)

    if not verses:
        return f'[NOT FOUND] {reference}'

    # Build output – return only the verse text(s)
    return '\n'.join(verses)


def lookup_all(references: list[str]) -> str:
    """Look up multiple references and return combined output."""
    text = BIBLE_PATH.read_text(encoding='utf-8-sig')
    results = []
    for ref in references:
        results.append(lookup(text, ref))
    return '\n\n'.join(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print('Examples:')
        print('  python bible_lookup.py "1 Corinthians 13:13"')
        print('  python bible_lookup.py "1 John 4:19" "Galatians 5:22-23"')
        print('  python bible_lookup.py "Matthew 18:21-22"')
        sys.exit(0)

    references = sys.argv[1:]
    output = lookup_all(references)
    print(output)


if __name__ == '__main__':
    main()
