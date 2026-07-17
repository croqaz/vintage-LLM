#!/usr/bin/env python3
"""Turn a `completions.txt`-style dump into clean JSON lines: {"text": ...}.

Input format
------------
Records are separated by a line containing exactly `-----` (5 dashes). Each
record is a chunk of (often OCR'd) text. We normalize whitespace, then apply a
few cheap quality filters and emit the survivors as JSONL.

Filters
-------
- text length (in characters)
- number of distinct characters (guards against degenerate / repeated junk)
- number of words
- short-text punctuation rule: very short texts must end with a sentence-ending
  punctuation character, otherwise they are dropped. Longer texts are exempt
  (they are often legitimately truncated mid-sentence).
"""

import argparse
import json
import re
import sys

SEP_RE = re.compile(r'^-{5,}$')
# Collapse runs of blank lines to at most one, and trim trailing spaces.
TRAILING_WS_RE = re.compile(r'[ \t]+$', re.MULTILINE)
MULTI_BLANK_RE = re.compile(r'\n{3,}')
END_PUNCT = tuple('.!?"\'”’)')


def clean_text(raw: str) -> str:
    """Normalize whitespace without destroying paragraph structure."""
    text = raw.replace('\r\n', '\n').replace('\r', '\n')
    text = TRAILING_WS_RE.sub('', text)
    text = MULTI_BLANK_RE.sub('\n\n', text)
    return text.strip()


def iter_records(fh):
    """Yield raw records split on lines of 5+ dashes."""
    buf = []
    for line in fh:
        if SEP_RE.match(line.rstrip('\n')):
            yield ''.join(buf)
            buf = []
        else:
            buf.append(line)
    if buf:
        yield ''.join(buf)


def keep(text: str, args) -> bool:
    if len(text) < args.min_chars:
        return False
    if args.max_chars and len(text) > args.max_chars:
        return False
    if len(set(text)) < args.min_unique_chars:
        return False
    n_words = len(text.split())
    if n_words < args.min_words:
        return False
    # Short texts must look like complete sentences.
    if len(text) < args.short_chars and not text.rstrip().endswith(END_PUNCT):
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', nargs='?', default='completions.txt', help='input file (default: completions.txt)')
    ap.add_argument('-o', '--output', default='-', help="output JSONL file, or '-' for stdout (default)")
    ap.add_argument('--min-chars', type=int, default=32, help='drop text shorter than this many characters')
    ap.add_argument('--max-chars', type=int, default=0, help='drop text longer than this (0 = no limit)')
    ap.add_argument('--min-unique-chars', type=int, default=8, help='drop text with fewer than this many distinct characters')
    ap.add_argument('--min-words', type=int, default=5, help='drop text with fewer than this many words')
    ap.add_argument('--short-chars', type=int, default=200, help='texts shorter than this must end with punctuation')
    args = ap.parse_args()

    out = sys.stdout if args.output == '-' else open(args.output, 'w', encoding='utf-8')
    total = kept = 0
    try:
        with open(args.input, encoding='utf-8', errors='replace') as fh:
            for raw in iter_records(fh):
                total += 1
                text = clean_text(raw)
                if not text or not keep(text, args):
                    continue
                out.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                kept += 1
    finally:
        if out is not sys.stdout:
            out.close()

    print(f'records: {total}  kept: {kept}  dropped: {total - kept} ({kept / total:.1%} kept)' if total else 'no records', file=sys.stderr)


if __name__ == '__main__':
    main()
