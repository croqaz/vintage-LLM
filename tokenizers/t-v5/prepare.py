"""prepare.py — the raw-to-clean pipeline for the v7 tokenizer.

Cleans every supported file of an input directory into an output directory
of plain-text training shards. Supported inputs, discovered by extension:

  *.txt        flat text        -> flat-clean   -> final-pass -> <name>.txt
  *.jsonl      {"text": ...}    -> record-clean -> final-pass -> <name>.jsonl.txt
  *.jsonl.xz   compressed jsonl -> record-clean -> final-pass -> <name>.jsonl.txt

  flat-clean    (per 8M-char newline-aligned chunk): odd spaces -> ' ',
                narrow <=1900-English charset allowlist, char runs 4+ -> 3,
                ASCII rulers -> 3x their dominant character, repeated
                punctuation units -> once.
  record-clean  (per JSONL record): NFC, CRLF -> LF, odd spaces, C0/format
                controls removed, soft hyphens removed, wide charset
                allowlist (extended Latin + Greek + period typography),
                run/ruler/unit collapse, standalone page-number /
                roman-numeral / ruler lines dropped, 3+ blank lines -> 2.
                Records are written with a trailing newline; empty records
                are skipped.
  final-pass    CRLF -> LF, odd spaces and tabs -> ' ', narrow charset
                (final gate, including the deliberate OCR_KEEP symbols),
                and NUMBER EXPLOSION: every 3+-digit run becomes 2-digit
                groups (1884 -> "18 84", 100 -> "10 0"), so the tokenizer
                trainer can never learn a token with 3+ consecutive digits.

Every regex operates locally (never across a newline-aligned chunk or record
boundary), so the pipeline is deterministic and streaming; outputs that
already exist are skipped, making interrupted runs resumable (--force
rebuilds).

Usage:
    python3 prepare.py [SRC_DIR] [OUT_DIR] [--force]
"""

import json
import lzma
import os
import re
import unicodedata
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'DATA'
OUT = HERE.parent / 'DATA-CLEAN'
CHUNK_CHARS = 1 << 23

# Common OCR-era symbols kept deliberately - the intersection of
# the out-of-domain character sets of two published period tokenizers
# (typeWriter and talkie). Frequent enough in scanned <=1900 sources
# that a handful of vocabulary tokens pays for itself.
OCR_KEEP = '§©«»½•™■'

# --- flat-clean (for flat .txt sources) --------------------------------
FLAT_ODD_SPACE_RE = re.compile('[  -  　]')
FLAT_OCR_JUNK_RE = re.compile('[^\t\n\r\x20-\x7eͰ-Ͽἀ-῿àáâäæçèéêëìíîïñòóôöœùúûüÿßÀÁÂÄÆÇÈÉÊËÌÍÎÏÑÒÓÔÖŒÙÚÛÜ“”‘’‚„—–…°£†‡ſ' + OCR_KEEP + ']')
RUN_RE = re.compile(r'(.)\1{3,}')
RULER_RE = re.compile(r'[-=_~+*#.]{4,}')
UNIT_RE = re.compile(r'([^\w\s]{2,8}?)\1+')


def _squash_ruler(match):
    return Counter(match.group()).most_common(1)[0][0] * 3


def flat_clean(text):
    text = FLAT_ODD_SPACE_RE.sub(' ', text)
    text = FLAT_OCR_JUNK_RE.sub('', text)
    text = RUN_RE.sub(r'\1\1\1', text)
    text = RULER_RE.sub(_squash_ruler, text)
    text = UNIT_RE.sub(r'\1', text)
    return text


# --- record-clean (for the JSONL sources) -------------------------------------
RECORD_ODD_SPACE_RE = re.compile('[  -  　]')
RECORD_CONTROL_RE = re.compile('[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f​-‏⁠﻿]')
RECORD_OCR_JUNK_RE = re.compile('[^\t\n\r\x20-\x7eͰ-Ͽἀ-῿À-ɏ̀-ͯḀ-ỿ“”‘’‚„—–…°£†‡ſ' + OCR_KEEP + ']')
PAGE_NUMBER_RE = re.compile(r'^\s*\d{1,5}\s*$')
ROMAN_PAGE_RE = re.compile(r'^\s*[IVXLCDM]{2,12}\s*$', re.IGNORECASE)
PUNCTUATION_LINE_RE = re.compile(r'^[!-/:-@\[-`{-~]{3,}$')
BLANK_LINES_RE = re.compile(r'\n{3,}')


def _drop_structural_lines(text):
    kept = []
    for line in text.split('\n'):
        body = line.strip()
        if PAGE_NUMBER_RE.fullmatch(line):
            continue
        if ROMAN_PAGE_RE.fullmatch(line):
            continue
        if PUNCTUATION_LINE_RE.fullmatch(body):
            continue
        kept.append(line)
    return BLANK_LINES_RE.sub('\n\n', '\n'.join(kept))


def record_clean(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\r\n?', '\n', text)
    text = RECORD_ODD_SPACE_RE.sub(' ', text)
    text = RECORD_CONTROL_RE.sub('', text)
    text = text.replace('­', '')
    text = RECORD_OCR_JUNK_RE.sub('', text)
    text = RUN_RE.sub(r'\1\1\1', text)
    text = RULER_RE.sub(_squash_ruler, text)
    text = UNIT_RE.sub(r'\1', text)
    text = _drop_structural_lines(text)
    return text


# --- final-pass (charset gate + number explosion, all sources) ----------------
FINAL_ODD_SPACE_RE = re.compile('[  -  　\t]')
FINAL_CRLF_RE = re.compile(r'\r\n?')
FINAL_OCR_JUNK_RE = re.compile('[^\\n\\x20-\\x7eͰ-Ͽἀ-῿àáâäæçèéêëìíîïñòóôöœùúûüÿßÀÁÂÄÆÇÈÉÊËÌÍÎÏÑÒÓÔÖŒÙÚÛÜ“”‘’‚„—–…°£†‡ſ' + OCR_KEEP + ']')
LONG_NUMBER_RE = re.compile(r'\d{3,}')


def _explode(match):
    s = match.group()
    return ' '.join(s[i : i + 2] for i in range(0, len(s), 2))


def final_pass(text):
    text = FINAL_CRLF_RE.sub('\n', text)
    text = FINAL_ODD_SPACE_RE.sub(' ', text)
    text = FINAL_OCR_JUNK_RE.sub('', text)
    text = LONG_NUMBER_RE.sub(_explode, text)
    return text


# --- compositions per source group ------------------------------------------------
def clean_dataset_chunk(chunk):
    return final_pass(flat_clean(chunk))


def clean_jsonl_record(text):
    cleaned = record_clean(text)
    if not cleaned:
        return ''
    if not cleaned.endswith('\n'):  # one record per line block
        cleaned += '\n'
    return final_pass(cleaned)


def _read_chunks(path):
    with open(path, 'rt', encoding='utf-8') as f:
        while chunk := f.read(CHUNK_CHARS):
            yield chunk + f.readline()


def iter_jsonl_records(path):
    opener = lzma.open if path.suffix == '.xz' else open
    with opener(path, 'rt', encoding='utf-8') as source:
        for line in source:
            record = json.loads(line)
            text = record.get('text') if isinstance(record, dict) else None
            if not isinstance(text, str):
                raise ValueError(f'{path}: expected an object with string text')
            yield text


def write_stream(dst, cleaner, items):
    tmp = str(dst) + '.tmp'
    with Pool() as pool, open(tmp, 'w', encoding='utf-8') as out:
        for cleaned in pool.imap(cleaner, items, chunksize=8):
            out.write(cleaned)
    os.replace(tmp, dst)
    print(f'done: {dst.name} ({os.path.getsize(dst) / 1e9:.2f} GB)', flush=True)


def discover_sources(src_dir):
    """Every supported file in src_dir: .txt (flat text), .jsonl, .jsonl.xz."""
    flat, records = [], []
    for p in sorted(src_dir.iterdir()):
        if p.name.endswith('.txt'):
            flat.append(p)
        elif p.name.endswith('.jsonl') or p.name.endswith('.jsonl.xz'):
            records.append(p)
    return flat, records


def output_path(out_dir, source):
    """Traceable, collision-free naming: foo.txt -> foo.txt,
    foo.jsonl -> foo.jsonl.txt, foo.jsonl.xz -> foo.jsonl.txt."""
    name = source.name
    if name.endswith('.jsonl.xz'):
        name = name[: -len('.xz')] + '.txt'
    elif name.endswith('.jsonl'):
        name += '.txt'
    return out_dir / name


def run(src_dir, out_dir, force=False):
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    assert src_dir.is_dir(), f'not a directory: {src_dir}'
    assert src_dir.resolve() != out_dir.resolve(), 'source and output directories must differ'
    flat, records = discover_sources(src_dir)
    assert flat or records, f'no supported files (.txt/.jsonl/.jsonl.xz) in {src_dir}'
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for source, cleaner, items, label in [(p, clean_dataset_chunk, _read_chunks(p), 'flat-clean') for p in flat] + [
        (p, clean_jsonl_record, iter_jsonl_records(p), 'record-clean') for p in records
    ]:
        dst = output_path(out_dir, source)
        if dst.exists() and not force:
            print(f'skipping {source.name}: {dst.name} exists (use --force to rebuild)', flush=True)
            continue
        print(f'cleaning {source.name} ({label} -> final-pass)', flush=True)
        write_stream(dst, cleaner, items)
        done += 1
    print(f'{done} file(s) prepared into {out_dir} ({len(flat) + len(records) - done} skipped)', flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('src', nargs='?', default=str(DATA), help='input directory of raw .txt / .jsonl / .jsonl.xz files')
    parser.add_argument('out', nargs='?', default=str(OUT), help='output directory for the cleaned .txt files')
    parser.add_argument('--force', action='store_true', help='rebuild outputs that already exist')
    args = parser.parse_args()
    run(args.src, args.out, force=args.force)


if __name__ == '__main__':
    main()
