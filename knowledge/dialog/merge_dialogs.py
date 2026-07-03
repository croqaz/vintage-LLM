#!/usr/bin/env python3
"""
Merge two LLM dialogue extractions (dialogs-1, dialogs-2) into one complete,
correctly-ordered, well-attributed set, anchored to the original book text.

The original ``gutenberg_chunks/<book>/*_chunk_NNNN.txt`` is the source of truth:
every extracted line is located (anchored) by character offset inside the chunk,
and the merge is performed purely by anchor position. This guarantees:

  * ORDER  - emitted lines follow the order they appear in the book, not the
             order either model happened to output them (dialogs-2 is sometimes
             out of order).
  * NO HALLUCINATIONS - a line that cannot be located in the source text is
             dropped from the output and logged to *_unlocated.jsonl.
  * COMPLETENESS - the union of both models is kept (a line one model missed but
             the other found, and which IS in the text, survives).
  * DEDUPE - a line both models found appears once.

Speaker policy (per the agreed plan): a real name beats "Unknown"; when both name
a speaker, dialogs-1 wins (it is the better attributor here). Cross-chunk speaker
canonicalisation ("The Man" -> "Jean Valjean") is a deliberately separate later
pass and is NOT done here.

Usage:
    python3 merge_dialogs.py pg135
    python3 merge_dialogs.py pg135 --d1 dialogs-1 --d2 dialogs-2 -o dialogs-merged
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

# --- normalisation ----------------------------------------------------------

_WS = re.compile(r'\s+')
_BRACKET = re.compile(r'\[[^\]]{0,60}\]')  # editorial inserts: [_grandeur_], [Matters ...]
# length-bounded so an unbalanced '[' can't greedily delete a huge span up to a
# far-away ']' (e.g. pg2759 ch0010: an oath's '[' matched an editorial ']' ~8800
# chars later, erasing 27 real dialogue lines from the normalised source).
_QUOTE_MAP = str.maketrans(
    {
        '“': '"',
        '”': '"',
        '„': '"',
        '‟': '"',
        '″': '"',
        '‘': "'",
        '’': "'",
        '‚': "'",
        '‛': "'",
        '′': "'",
        '‐': '-',
        '‑': '-',
        '‒': '-',
        '–': '-',
        '—': '-',
        '―': '-',
    }
)
_OPEN, _CLOSE = '“"‘', '”"’'


def norm(s: str) -> str:
    """Normalised form used ONLY for locating text inside the source chunk."""
    s = s.translate(_QUOTE_MAP)
    s = _BRACKET.sub(' ', s)
    s = s.replace('_', '')  # italic delimiters: strip, don't space-pad
    s = _WS.sub(' ', s).strip()
    return s


def norm_line(s: str) -> str:
    """Normalised form of a dialog line, with surrounding quotes stripped."""
    s = norm(s)
    while len(s) >= 2 and s[0] in _OPEN and s[-1] in _CLOSE:
        s = s[1:-1].strip()
    return s.strip(' "\'')


def clean_text(s: str) -> str:
    """Faithful emitted text: strip markup/brackets, keep original quote glyphs."""
    s = _BRACKET.sub('', s)
    s = s.replace('_', '')
    s = _WS.sub(' ', s).strip()
    while len(s) >= 2 and s[0] in _OPEN and s[-1] in _CLOSE:
        s = s[1:-1].strip()
    return s


def is_unknown(speaker: str) -> bool:
    return str(speaker).strip().lower() in ('', 'unknown', 'unknown speaker', '?')


# --- loading ----------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Return [{'speaker':..,'text':..}, ...] from a dialogs JSONL file."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip().rstrip(',')
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not obj:
            continue
        if {'speaker', 'text'} <= obj.keys():
            speaker, text = obj.get('speaker'), obj.get('text')
        elif len(obj) == 1:
            ((speaker, text),) = obj.items()
        else:
            continue
        speaker = str(speaker).strip() if speaker is not None else ''
        text = str(text).strip() if text is not None else ''
        if text:
            out.append({'speaker': speaker, 'text': text})
    return out


# --- locating ---------------------------------------------------------------


def fuzzy_locate(tn_norm: str, tl_norm: str, probe: str, threshold: float = 0.6):
    """Approximate anchor via longest common block. Returns (anchor, end) or None."""
    if not probe:
        return None
    sm = difflib.SequenceMatcher(None, tn_norm, probe, autojunk=False)
    m = sm.find_longest_match(0, len(tn_norm), 0, len(probe))
    if m.size >= threshold * len(probe):
        anchor = max(0, m.a - m.b)
        return anchor, anchor + len(probe)
    return None


def locate_entries(entries: list[dict], tn: str, source: str) -> list[dict]:
    """Anchor each entry to offsets in normalised chunk text ``tn``.

    A per-source monotonic cursor disambiguates repeated short quotes
    ("Yes."/"No."): the k-th such line maps to the k-th occurrence. An entry the
    model emitted out of order still anchors (search falls back to offset 0) but
    does not drag the cursor backwards.
    """
    tl = tn.lower()
    cursor = 0
    out = []
    for e in entries:
        probe = norm_line(e['text'])
        rec = {
            'speaker': e['speaker'],
            'text': e['text'],
            'source': source,
            'probe': probe,
            'anchor': None,
            'end': None,
            'status': 'unlocated',
        }
        if not probe:
            rec['status'] = 'empty'
            out.append(rec)
            continue

        idx = tn.find(probe, cursor)
        advance = idx != -1
        if idx == -1:
            idx = tn.find(probe, 0)  # out-of-order line
        if idx == -1:  # case-insensitive
            pl = probe.lower()
            idx = tl.find(pl, cursor)
            advance = idx != -1
            if idx == -1:
                idx = tl.find(pl, 0)
        if idx == -1 and len(probe) >= 24:  # two-part rescue:
            # one model merged a quote the book splits across a speech tag
            # ("Bah, Madame," he said, "let her play!") and the OTHER model
            # missed it entirely. Locate head + tail across a short narration gap
            # so the line is not lost.
            head, tail = probe[:12], probe[-12:]
            h = tn.find(head, cursor)
            if h == -1:
                h = tn.find(head, 0)
            if h != -1:
                region = h + len(head)
                t = tn.find(tail, region)
                if t != -1 and t <= region + 120:
                    idx, end = h, t + len(tail)
                    rec.update(anchor=idx, end=end, status='located_split')
                    if idx >= cursor:
                        cursor = end
                    out.append(rec)
                    continue

        if idx == -1 and len(probe) >= 28:  # prefix-shingle rescue:
            # a long line whose TAIL diverges (OCR slip, cipher, paraphrased
            # ending) but whose distinctive opening is verbatim in the source.
            pre = probe[:28]
            j = tn.find(pre, cursor)
            if j == -1:
                j = tn.find(pre, 0)
            if j == -1:
                j = tn.lower().find(pre.lower(), 0)
            if j != -1:
                idx, end = j, j + len(probe)
                rec.update(anchor=idx, end=end, status='located_prefix')
                if idx >= cursor:
                    cursor = end
                out.append(rec)
                continue

        if idx == -1:  # fuzzy (rare)
            fz = fuzzy_locate(tn, tn.lower(), probe)
            if fz is not None:
                idx, end = fz
                rec.update(anchor=idx, end=end, status='located_fuzzy')
                out.append(rec)
                continue

        if idx == -1:
            out.append(rec)  # genuinely unlocated
            continue

        end = idx + len(probe)
        rec.update(anchor=idx, end=end, status='located')
        if advance or idx >= cursor:
            cursor = end
        out.append(rec)
    return out


# --- merging ----------------------------------------------------------------


def overlaps(a: dict, b: dict) -> bool:
    return a['anchor'] < b['end'] and b['anchor'] < a['end']


def pick_speaker(seg: dict, group: list[dict]) -> tuple[str, bool]:
    """Choose a speaker for ``seg``: real name beats Unknown, dialogs-1 wins ties.

    Returns (speaker, conflict) where conflict is True if d1 and d2 both gave
    different real names for an overlapping span.
    """
    over = [r for r in group if overlaps(seg, r)]
    d1 = [r['speaker'] for r in over if r['source'] == 'd1' and not is_unknown(r['speaker'])]
    d2 = [r['speaker'] for r in over if r['source'] == 'd2' and not is_unknown(r['speaker'])]
    conflict = bool(d1) and bool(d2) and set(map(str.lower, d1)) != set(map(str.lower, d2))
    if d1:
        return d1[0], conflict
    if d2:
        return d2[0], conflict
    return 'Unknown', False


def best_text_rec(seg: dict, group: list[dict], tn: str) -> dict:
    """Among same-span candidates, pick the text closest to the SOURCE book.

    Fixes cases where one model has a transcription slip (e.g. "tell me who you
    are" vs the book's "tell you who you are"): the book-faithful version wins,
    regardless of which model it came from. Segmentation still follows ``seg``.
    """
    cands = {seg['probe']: seg}
    for r in group:
        if r is seg or not overlaps(seg, r):
            continue
        if 0.7 * len(seg['probe']) <= len(r['probe']) <= 1.4 * len(seg['probe']):
            cands.setdefault(r['probe'], r)
    if len(cands) == 1:
        return seg
    window = tn[seg['anchor'] : seg['end'] + 8].lower()
    return max(cands.values(), key=lambda c: difflib.SequenceMatcher(None, window, c['probe'].lower(), autojunk=False).ratio())


def merge_chunk(d1_recs: list[dict], d2_recs: list[dict], stats: dict, tn: str) -> list[dict]:
    located = [r for r in d1_recs + d2_recs if r['anchor'] is not None]
    located.sort(key=lambda r: (r['anchor'], r['end'], 0 if r['source'] == 'd1' else 1))

    # group records whose spans overlap or are immediately adjacent
    groups = []
    cur, cur_end = [], -1
    for r in located:
        if cur and r['anchor'] <= cur_end + 1:
            cur.append(r)
            cur_end = max(cur_end, r['end'])
        else:
            if cur:
                groups.append(cur)
            cur, cur_end = [r], r['end']
    if cur:
        groups.append(cur)

    emitted = []
    for g in groups:
        d1 = [r for r in g if r['source'] == 'd1']
        d2 = [r for r in g if r['source'] == 'd2']
        # finer segmentation wins (matches the "split on speech tag" rule);
        # dialogs-1 wins ties.
        base = d1 if len(d1) >= len(d2) else d2
        if len(d1) > 1 or len(d2) > 1:
            stats['split_groups'] += 1
        seen = set()
        for seg in base:
            key = seg['probe']
            if key in seen:
                continue
            seen.add(key)
            speaker, conflict = pick_speaker(seg, g)
            if conflict:
                stats['speaker_conflicts'] += 1
            text = clean_text(best_text_rec(seg, g, tn)['text'])
            if not text:
                continue
            emitted.append({'speaker': speaker, 'text': text, 'anchor': seg['anchor']})
    emitted.sort(key=lambda r: r['anchor'])
    return emitted


# --- driver -----------------------------------------------------------------


def chunk_id(p: Path) -> int:
    m = re.search(r'_chunk_(\d+)', p.stem)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('book', help='book id, e.g. pg135')
    ap.add_argument('--chunks', default='gutenberg_chunks', help='source chunk root (default: gutenberg_chunks)')
    ap.add_argument('--d1', default='dialogs-1', help='first extraction root (default: dialogs-1)')
    ap.add_argument('--d2', default='dialogs-2', help='second extraction root (default: dialogs-2)')
    ap.add_argument('-o', '--output', default='dialogs-merged', help='output root (default: dialogs-merged)')
    args = ap.parse_args()

    book = args.book
    src_dir = Path(args.chunks) / book
    d1_dir = Path(args.d1) / book
    d2_dir = Path(args.d2) / book
    out_dir = Path(args.output) / book
    if not src_dir.is_dir():
        sys.exit(f'error: source chunk dir not found: {src_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)

    src_chunks = sorted(src_dir.glob('*_chunk_*.txt'), key=chunk_id)
    if not src_chunks:
        sys.exit(f'error: no chunks in {src_dir}')

    stats = dict(
        chunks=0,
        d1_total=0,
        d2_total=0,
        d1_located=0,
        d2_located=0,
        d1_fuzzy=0,
        d2_fuzzy=0,
        d1_unlocated=0,
        d2_unlocated=0,
        merged=0,
        unknown_in_d1=0,
        unknown_in_d2=0,
        unknown_out=0,
        split_groups=0,
        speaker_conflicts=0,
    )
    combined = []
    unlocated = []
    book_offset = 0  # cumulative normalized length of chunks processed so far

    for src in src_chunks:
        stem = src.stem
        tn = norm(src.read_text(encoding='utf-8', errors='replace'))

        d1_raw = load_jsonl(d1_dir / f'{stem}.jsonl')
        d2_raw = load_jsonl(d2_dir / f'{stem}.jsonl')
        stats['d1_total'] += len(d1_raw)
        stats['d2_total'] += len(d2_raw)
        stats['unknown_in_d1'] += sum(is_unknown(e['speaker']) for e in d1_raw)
        stats['unknown_in_d2'] += sum(is_unknown(e['speaker']) for e in d2_raw)

        d1_recs = locate_entries(d1_raw, tn, 'd1')
        d2_recs = locate_entries(d2_raw, tn, 'd2')

        for recs, tag in ((d1_recs, 'd1'), (d2_recs, 'd2')):
            for r in recs:
                if r['status'] == 'located':
                    stats[f'{tag}_located'] += 1
                elif r['status'] in ('located_fuzzy', 'located_prefix', 'located_split'):
                    stats[f'{tag}_located'] += 1
                    stats[f'{tag}_fuzzy'] += 1
                elif r['status'] == 'unlocated':
                    stats[f'{tag}_unlocated'] += 1
                    unlocated.append({'chunk': stem, 'source': tag, 'speaker': r['speaker'], 'text': r['text']})

        emitted = merge_chunk(d1_recs, d2_recs, stats, tn)
        stats['merged'] += len(emitted)
        stats['unknown_out'] += sum(is_unknown(e['speaker']) for e in emitted)
        stats['chunks'] += 1

        # sanity: order must be non-decreasing by anchor
        anchors = [e['anchor'] for e in emitted]
        assert anchors == sorted(anchors), f'order violation in {stem}'

        # book-global position (normalized coords): prior chunks' length + anchor
        for e in emitted:
            e['pos'] = book_offset + e['anchor']

        out_path = out_dir / f'{stem}.jsonl'
        with out_path.open('w', encoding='utf-8') as f:
            for e in emitted:
                f.write(json.dumps({'speaker': e['speaker'], 'text': e['text'], 'pos': e['pos']}, ensure_ascii=False) + '\n')
        combined.extend(emitted)
        book_offset += len(tn)

    # combined book file
    with (out_dir / f'{book}_dialogs.jsonl').open('w', encoding='utf-8') as f:
        for e in combined:
            f.write(json.dumps({'speaker': e['speaker'], 'text': e['text'], 'pos': e['pos']}, ensure_ascii=False) + '\n')

    # unlocated log
    with (out_dir / f'{book}_unlocated.jsonl').open('w', encoding='utf-8') as f:
        for u in unlocated:
            f.write(json.dumps(u, ensure_ascii=False) + '\n')

    # report
    def pct(n, d):
        return f'{100 * n / d:.1f}%' if d else 'n/a'

    report = f"""MERGE REPORT - {book}
{'=' * 60}
chunks processed:        {stats['chunks']}

dialogs-1 lines in:      {stats['d1_total']}
  located:               {stats['d1_located']}  ({pct(stats['d1_located'], stats['d1_total'])})  (fuzzy: {stats['d1_fuzzy']})
  unlocated (dropped):   {stats['d1_unlocated']}
  "Unknown" speakers in: {stats['unknown_in_d1']}  ({pct(stats['unknown_in_d1'], stats['d1_total'])})

dialogs-2 lines in:      {stats['d2_total']}
  located:               {stats['d2_located']}  ({pct(stats['d2_located'], stats['d2_total'])})  (fuzzy: {stats['d2_fuzzy']})
  unlocated (dropped):   {stats['d2_unlocated']}
  "Unknown" speakers in: {stats['unknown_in_d2']}  ({pct(stats['unknown_in_d2'], stats['d2_total'])})

MERGED lines out:        {stats['merged']}
  "Unknown" speakers out:{stats['unknown_out']}  ({pct(stats['unknown_out'], stats['merged'])})
  split/merge groups:    {stats['split_groups']}
  speaker conflicts:     {stats['speaker_conflicts']}  (d1 kept, d2 discarded)
  unlocated logged:      {len(unlocated)}  -> {book}_unlocated.jsonl
{'=' * 60}
"""
    (out_dir / f'{book}_merge_report.txt').write_text(report, encoding='utf-8')
    print(report)
    print(f'output -> {out_dir}/')


if __name__ == '__main__':
    main()
