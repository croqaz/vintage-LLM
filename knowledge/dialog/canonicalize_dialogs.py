#!/usr/bin/env python3
"""
Post-process a merged book's dialogues into a single, speaker-consistent file.

Three steps, on the per-chunk output of ``merge_dialogs.py``:

  1. CANONICAL SPEAKERS - unify the many surface/alias forms a character is
     labelled with into one canonical name, using ``speaker_map.json``:
       * global  : proper-name aliases + title variants merged to the real
                   person (e.g. M. Madeleine / M. Leblanc -> Jean Valjean;
                   Jondrette -> Thénardier; prioress -> Mother Innocente),
                   matched case-insensitively.
       * per_chunk: scene-local generic labels that mean different people in
                   different chunks (e.g. "the stranger" = Jean Valjean in
                   chunk 0053 but = Thénardier in chunk 0174).
       * surface : any name not in the map is case-folded to its most common
                   casing across the book ("the man"/"The Man" -> one form),
                   which is safe (no claim two people are the same).

  2. COALESCE - consecutive entries by the SAME canonical speaker are joined
     into one entry (a quote split by a speech tag becomes whole again).
     "Unknown" turns are never coalesced (they are not the same speaker).

  3. JOIN - emit one book-level JSONL with all chunks in order.

Usage:
    python3 canonicalize_dialogs.py pg135-clean
    python3 canonicalize_dialogs.py pg135-clean --map speaker_map.json -i dialogs-merged
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

_WS = re.compile(r'\s+')


def normalize_name(s: str) -> str:
    """Light surface cleanup of a speaker label (NOT a semantic mapping)."""
    s = str(s).strip().strip('"“”\'').strip()
    s = _WS.sub(' ', s)
    s = s.rstrip(':,').strip()
    return s


def is_unknown(name: str) -> bool:
    return name.strip().lower() in ('', 'unknown', 'unknown speaker', '?')


def join_turns(prev: str, nxt: str, sep: str) -> str:
    """Join two consecutive same-speaker turns, cleanly.

    When the previous turn ends in a comma but the next begins with a capital
    letter, the two were separate sentences/paragraphs (not a speech-tag split
    like "Sire," ... "you ..."), so the trailing comma becomes a full stop.
    A next-turn starting lowercase is left as a comma (genuine continuation).
    """
    if not prev:
        return nxt
    p, n = prev.rstrip(), nxt.lstrip()
    if not n:
        return p
    if p.endswith(',') and n[:1].isupper():
        p = p[:-1] + '.'
    return (p + sep + n).strip()


def chunk_num(path: Path) -> str:
    m = re.search(r'_chunk_(\d+)', path.stem)
    return m.group(1) if m else ''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('book', help='book id, e.g. pg135-clean')
    ap.add_argument('-i', '--input', default='dialogs-merged', help='merged output root (default: dialogs-merged)')
    ap.add_argument('--map', default='speaker_map.json', help='speaker map JSON (default: speaker_map.json)')
    ap.add_argument('--join-with', default=' ', help='string used to join coalesced turns (default: single space)')
    ap.add_argument(
        '--gap-max',
        type=int,
        default=500,
        help='max source-text gap (normalized chars, end-to-start) to still coalesce two same-speaker turns (default: 500)',
    )
    args = ap.parse_args()

    book_dir = Path(args.input) / args.book
    if not book_dir.is_dir():
        raise SystemExit(f'error: not found: {book_dir}')

    spec = json.loads(Path(args.map).read_text(encoding='utf-8'))
    glob = {k.lower(): v for k, v in spec.get('global', {}).items()}
    per_chunk = {ch: {k.lower(): v for k, v in m.items()} for ch, m in spec.get('per_chunk', {}).items()}

    # gather entries in book order, remembering each one's source chunk.
    # Input rows are {"speaker","text","pos"} (pos = book-global normalized offset).
    chunk_files = sorted((p for p in book_dir.glob(f'{args.book}_chunk_*.jsonl')), key=lambda p: chunk_num(p))
    entries = []  # (chunk, raw_speaker, text, pos)
    for f in chunk_files:
        ch = chunk_num(f)
        for line in f.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            entries.append((ch, normalize_name(obj['speaker']), obj['text'], obj['pos']))

    # surface case-folding table: most common original casing per lowercase form
    casing = defaultdict(Counter)
    for _, sp, _, _ in entries:
        casing[sp.lower()][sp] += 1
    case_canon = {low: cnt.most_common(1)[0][0] for low, cnt in casing.items()}

    def canon(ch: str, sp: str) -> str:
        if is_unknown(sp):
            return 'Unknown'
        low = sp.lower()
        if ch in per_chunk and low in per_chunk[ch]:
            return per_chunk[ch][low]
        if low in glob:
            return glob[low]
        return case_canon.get(low, sp)

    # apply canonicalisation
    canon_entries = [(ch, canon(ch, sp), text, pos) for ch, sp, text, pos in entries]

    # coalesce consecutive same-speaker turns, but ONLY when they are close in the
    # source text (gap end-to-start <= GAP_MAX). Distant same-speaker turns (a scene
    # or chapter apart) stay separate. "Unknown" turns are never coalesced.
    book = []  # list of {speaker, text, pos, _end}
    coalesced = 0
    for ch, sp, text, pos in canon_entries:
        prev = book[-1] if book else None
        gap = pos - prev['_end'] if prev is not None else None
        if prev is not None and prev['speaker'] == sp and not is_unknown(sp) and gap <= args.gap_max:
            prev['text'] = join_turns(prev['text'], text, args.join_with)
            prev['_end'] = pos + len(text)  # end of the just-appended turn
            coalesced += 1
        else:
            book.append({'speaker': sp, 'text': text, 'pos': pos, '_end': pos + len(text)})

    # write the single book JSONL
    out_path = book_dir / f'{args.book}_book.jsonl'
    with out_path.open('w', encoding='utf-8') as fh:
        for e in book:
            fh.write(json.dumps({'speaker': e['speaker'], 'text': e['text'], 'pos': e['pos']}, ensure_ascii=False) + '\n')

    # report
    before = Counter(sp for _, sp, _, _ in entries)
    after = Counter(e['speaker'] for e in book)
    # names that were neither mapped nor are obvious proper names (generic, flag)
    generic = re.compile(r'^(the|a|an)\b', re.I)
    unresolved_generic = sorted(((after[n], n) for n in after if generic.match(n)), reverse=True)

    lines = []
    lines.append(f'CANONICALISE REPORT - {args.book}')
    lines.append('=' * 60)
    lines.append(f'entries in (merged):     {len(entries)}')
    lines.append(f'entries out (book):      {len(book)}')
    lines.append(f'turns coalesced away:    {coalesced}')
    lines.append(f'distinct speakers before:{len(before)}')
    lines.append(f'distinct speakers after: {len(after)}')
    lines.append(f'"Unknown" entries:       {after.get("Unknown", 0)}')
    lines.append('')
    lines.append('top 25 speakers after canonicalisation:')
    for name, n in after.most_common(25):
        lines.append(f'  {n:5}  {name}')
    lines.append('')
    lines.append(f'remaining generic labels (review): {len(unresolved_generic)}')
    for n, name in unresolved_generic[:40]:
        lines.append(f'  {n:5}  {name}')
    report = '\n'.join(lines) + '\n'
    (book_dir / f'{args.book}_canonicalize_report.txt').write_text(report, encoding='utf-8')
    print(report)
    print(f'book file -> {out_path}')


if __name__ == '__main__':
    main()
