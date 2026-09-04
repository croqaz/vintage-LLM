#!/usr/bin/env python3
"""
Split TEXT/ JSONL dataset files into train and validation sets.

Reads files where each line is either:
  - A JSON object: {"text": "..."}
  - Plain text:    the line itself is the text

Ends each text with the tokenizer's EOS tokens and writes two output
files: <name>-train.<ext> and <name>-valid.<ext>, stored alongside the input.

If total character count across all texts in a file is < 1000, only a
-train file is written (no -valid).
"""

import argparse
import json
import re
import sys
import tomllib
from array import array
from pathlib import Path
from random import Random

from transformers import AutoTokenizer

MIN_CHARS_FOR_SPLIT = 1000


def resolve_tokenizer(cfg: dict, config_path) -> str:
    """
    Resolve data.tokenizer the same way base_train.load_config() does: relative
    paths are anchored to the CONFIG FILE's directory, not the shell's cwd, so
    the value means the same thing from anywhere.  Non-path values (HuggingFace
    hub ids) are passed through untouched.
    """
    from pathlib import Path

    value = cfg['data']['tokenizer']
    base = Path(config_path).resolve().parent
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    if value.startswith(('.', '~', '/')) or (base / path).exists():
        return str((base / path).resolve())
    return value  # hub id


def detect_format(line: str) -> str:
    """Return 'jsonl' if the line is a JSON object with a 'text' key, else 'text'."""
    try:
        obj = json.loads(line)
        if isinstance(obj, dict) and 'text' in obj:
            return 'jsonl'
    except (json.JSONDecodeError, ValueError):
        pass
    return 'text'


def sniff_format(path: Path) -> str:
    """Detect 'jsonl' vs 'text' by reading ONLY the first non-empty line."""
    with open(path, encoding='utf-8') as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                return detect_format(line)
    return 'text'


def read_whole_text(path: Path) -> list[str]:
    """Read a plain-text file as ONE document. Only used for the non-JSONL path."""
    with open(path, encoding='utf-8') as fh:
        stripped = fh.read().strip()
    return [stripped] if stripped else []


def scan_jsonl(path: Path) -> tuple[array, array, array]:
    """
    Pass 1 of the streaming JSONL split: index the file without holding it.

    Returns three parallel arrays, one entry per non-empty line:
      offsets  — byte offset of the line in the file
      lengths  — character length of that line's "text" field
      scores   — _boundary_score() of that text

    Cost is ~26 bytes/line of RAM (vs. the whole corpus), so a 38 GB / 10 M-line
    shard indexes in well under a gigabyte.
    """
    offsets, lengths, scores = array('q'), array('q'), array('b')
    size = path.stat().st_size
    next_report = 1 << 30
    with open(path, 'rb') as fh:
        offset = 0
        for raw in fh:
            start = offset
            offset += len(raw)
            if not raw.strip():
                continue
            text = json.loads(raw)['text']
            offsets.append(start)
            lengths.append(len(text))
            scores.append(_boundary_score(text))
            if start >= next_report:
                print(f'        indexing {start / size:5.1%}  ({len(offsets):,} docs)', flush=True)
                next_report = start + (1 << 30)
    return offsets, lengths, scores


def read_jsonl_text(fh, offset: int) -> str:
    """
    Pass 2 helper: pull one document back out of the file by byte offset.

    *fh* must be opened in BINARY mode. scan_jsonl() records raw byte offsets,
    and seeking a text-mode handle to an arbitrary byte position is not
    supported by TextIOWrapper -- it only promises to accept opaque cookies
    from tell(). It happens to work for UTF-8 at a line boundary, but the
    contract does not hold, so read bytes and let json.loads do the decoding.
    """
    fh.seek(offset)
    return json.loads(fh.readline())['text']


# Patterns for boundary detection
_PARAGRAPH_RE = re.compile(r'(\n\n|\r\n\r\n)\s*$')  # end-of-entry check
_SENTENCE_RE = re.compile(r'[.!?\u2026]["\')\]]*\s*$')  # end-of-entry check


def _boundary_score(text: str) -> int:
    """Score how clean a split would be *after* this entry. Higher is better."""
    if _PARAGRAPH_RE.search(text):
        return 3  # paragraph boundary
    if _SENTENCE_RE.search(text):
        return 2  # sentence boundary
    stripped = text.rstrip()
    if stripped and not stripped[-1].isalpha():
        return 1  # word boundary (ends on punctuation / digit / symbol)
    return 0  # ends mid-word


def find_char_split(text: str, train_ratio: float, tolerance: float = 0.05) -> int:
    """
    Find the best character index to split a single text string.
    Returns i: text[:i] → train, text[i:] → valid.
    Priority: paragraph boundary > sentence boundary > word boundary > exact target.
    Searches within ±tolerance of the target position.
    """
    n = len(text)
    target = int(n * train_ratio)
    window = int(n * tolerance)
    lo = max(1, target - window)
    hi = min(n - 1, target + window)

    def best_in_range(positions: list[int]) -> int | None:
        in_range = [p for p in positions if lo <= p <= hi]
        return min(in_range, key=lambda p: abs(p - target)) if in_range else None

    # Paragraph: position right after \n\n
    pos = best_in_range([m.end() for m in re.finditer(r'\n\n+', text)])
    if pos is not None:
        return pos

    # Sentence: position right after sentence-ending punctuation + whitespace
    pos = best_in_range([m.end() for m in re.finditer(r'[.!?\u2026]["\')\]]*\s+', text)])
    if pos is not None:
        return pos

    # Word: position right after any whitespace run
    pos = best_in_range([m.end() for m in re.finditer(r'\s+', text)])
    if pos is not None:
        return pos

    return target  # last resort: hard cut


def find_split_index(lengths, scores, train_ratio: float = 0.9, tolerance: float = 0.05) -> int:
    """
    Return index i so that entries [:i] go to train and [i:] to valid.

    Takes per-entry CHARACTER LENGTHS and pre-computed _boundary_score() values
    rather than the texts themselves, so the caller never has to hold the corpus
    in memory. Looks for the cleanest boundary (paragraph > sentence > word)
    within tolerance of the target character position; falls back to the nearest
    entry boundary when nothing lands in that window.
    """
    total = sum(lengths)
    target = total * train_ratio
    tol = total * tolerance

    # cumulative[i] = total chars after including entry i
    cumulative = array('q')
    acc = 0
    for n in lengths:
        acc += n
        cumulative.append(acc)

    # Consider all entry-end positions inside the tolerance window.
    # Exclude the very last entry so validation always has at least one entry.
    candidates = [i for i in range(len(cumulative) - 1) if target - tol <= cumulative[i] <= target + tol]

    if not candidates:
        # Nothing in the window — pick the nearest entry boundary.
        candidates = [min(range(len(lengths) - 1), key=lambda i: abs(cumulative[i] - target))]

    # Highest score wins; ties broken by proximity to the target position.
    best = max(candidates, key=lambda i: (scores[i], -abs(cumulative[i] - target)))
    # entries[:best+1] → train, entries[best+1:] → valid
    return best + 1


def _assert_no_loss(label: str, original: int, *parts: int) -> None:
    """Abort loudly if the sum of *parts* doesn't equal *original*."""
    total = sum(parts)
    if total != original:
        raise RuntimeError(
            f'Integrity check FAILED for {label}: original {original} chars, output {total} chars (delta {total - original:+d})'
        )
    print(f'[ok]    {label}  — {original:,} lengths verified (no loss)')


def write_texts(path: Path, texts: list[str], fmt: str) -> None:
    with open(path, 'w', encoding='utf-8') as fh:
        for text in texts:
            if fmt == 'jsonl':
                fh.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            else:
                fh.write(text.strip() + '\n')


def split_jsonl_streaming(
    input_path: Path,
    train_path: Path,
    valid_path: Path,
    wrap,
    valid_ratio: float,
    split_tolerance: float,
    shuffle: bool,
    seed: int,
) -> None:
    """
    Split a JSONL file without ever holding it in memory.

    Two passes over the file:
      1. scan_jsonl() builds a compact index (byte offset, char length, boundary
         score) — a few tens of bytes per document instead of the document.
      2. Documents are re-read one at a time by byte offset and streamed straight
         out to the -train / -valid files.

    This is what makes 38 GB shards splittable on a 62 GB machine. The result is
    identical to the old in-memory path, including the shuffled output ordering.
    """
    print(f'        pass 1/2: indexing {input_path.name} …', flush=True)
    offsets, lengths, scores = scan_jsonl(input_path)
    n_docs = len(offsets)
    if not n_docs:
        print(f'[skip]  {input_path.name}  — empty file')
        return
    if n_docs < 2:
        print(f'[train] {input_path.name}  — single document, writing as train only')
        with open(input_path, 'rb') as src, open(train_path, 'w', encoding='utf-8') as out:
            out.write(json.dumps({'text': wrap(read_jsonl_text(src, offsets[0]))}, ensure_ascii=False) + '\n')
        return

    order = list(range(n_docs))
    if shuffle:
        Random(seed).shuffle(order)

    # find_split_index works on the SHUFFLED sequence, exactly as before.
    ordered_lengths = array('q', (lengths[i] for i in order))
    ordered_scores = array('b', (scores[i] for i in order))
    split_at = find_split_index(ordered_lengths, ordered_scores, 1.0 - valid_ratio, tolerance=split_tolerance)

    raw_total = sum(ordered_lengths)
    raw_train = sum(ordered_lengths[:split_at])
    raw_valid = sum(ordered_lengths[split_at:])
    _assert_no_loss(input_path.name, raw_total, raw_train, raw_valid)

    print(f'        pass 2/2: writing {split_at:,} train / {n_docs - split_at:,} valid …', flush=True)
    written = 0
    with (
        open(input_path, 'rb') as src,
        open(train_path, 'w', encoding='utf-8', buffering=1 << 22) as train_fh,
        open(valid_path, 'w', encoding='utf-8', buffering=1 << 22) as valid_fh,
    ):
        for rank, idx in enumerate(order):
            out = train_fh if rank < split_at else valid_fh
            text = wrap(read_jsonl_text(src, offsets[idx]))
            out.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            written += 1
            if written % 1_000_000 == 0:
                print(f'        wrote {written:,}/{n_docs:,} docs', flush=True)

    actual_ratio = raw_valid / raw_total
    print(
        f'[split] {input_path.name}  — {split_at:,} train / {n_docs - split_at:,} valid '
        f'(valid {actual_ratio:.1%} of chars, jsonl, streamed)'
    )


def process_file(
    input_path: Path,
    tokenizer,
    delete_original: bool = False,
    valid_ratio: float = 0.1,
    split_tolerance: float = 0.05,
    shuffle: bool = False,
    seed: int = 42,
) -> None:
    stem = input_path.stem
    suffix = input_path.suffix
    parent = input_path.parent

    if stem.endswith('-train') or stem.endswith('-valid'):
        base = stem[:-6]  # strip '-train' or '-valid' (both 6 chars)
        sibling = parent / f'{base}{"-valid" if stem.endswith("-train") else "-train"}{suffix}'
        if sibling.exists():
            print(f'[skip]  {input_path.name}  — already split (sibling {sibling.name} exists)')
        else:
            print(f'[skip]  {input_path.name}  — already has a -train/-valid suffix, skipping to avoid double-split')
        return

    fmt = sniff_format(input_path)

    eos = tokenizer.eos_token or ''
    if not eos:
        print('Warning: tokenizer has no eos_token', file=sys.stderr)

    train_path = parent / f'{stem}-train{suffix}'
    valid_path = parent / f'{stem}-valid{suffix}'

    def wrap(t: str) -> str:
        if not t.rstrip().endswith(eos):
            t = f'{t}\n{eos}'
        return t

    if fmt == 'jsonl':
        split_jsonl_streaming(input_path, train_path, valid_path, wrap, valid_ratio, split_tolerance, shuffle, seed)
        if delete_original:
            input_path.unlink()
        return

    texts = read_whole_text(input_path)
    if not texts:
        print(f'[skip]  {input_path.name}  — empty file')
        return

    # Single document — wrap the whole file once, carve validation from the centre.
    text = wrap('\n'.join(t.strip() for t in texts).strip())

    if len(text) < MIN_CHARS_FOR_SPLIT:
        print(f'[train] {input_path.name}  — {len(text)} chars < {MIN_CHARS_FOR_SPLIT}, writing as train only')
        write_texts(train_path, [text], fmt)
        if delete_original:
            input_path.unlink()
        return

    # Validation is carved from the middle so both halves of the training text
    # stay contextually clean.  Two boundaries are found independently:
    #   split1 — end of train-part-1  (~45 % for valid_ratio=0.10)
    #   split2 — start of train-part-2 (~55 %)
    # train = text[:split1] + text[split2:]
    # valid = text[split1:split2]
    mid_start = (1.0 - valid_ratio) / 2  # e.g. 0.45
    mid_end = mid_start + valid_ratio  # e.g. 0.55

    split1 = find_char_split(text, mid_start, tolerance=split_tolerance)
    split2 = find_char_split(text, mid_end, tolerance=split_tolerance)

    if split2 <= split1:  # degenerate edge-case guard
        split2 = split1 + 1

    train_text = text[:split1] + text[split2:]
    valid_text = text[split1:split2]
    _assert_no_loss(input_path.name, len(text), len(train_text), len(valid_text))
    write_texts(train_path, [train_text], fmt)
    write_texts(valid_path, [valid_text], fmt)
    actual_ratio = len(valid_text) / len(text)
    print(f'[split] {input_path.name}  — valid {actual_ratio:.1%} of chars from centre (text)')

    if delete_original:
        input_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Split text/JSONL files into -train and -valid sets.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('files', nargs='+', help='Input file(s) to split')
    parser.add_argument(
        '--config',
        default='config.toml',
        help='Path to config.toml (reads data.tokenizer)',
    )
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.05,
        metavar='RATIO',
        help='Fraction of entries reserved for validation',
    )
    parser.add_argument(
        '--split-tolerance',
        type=float,
        default=0.05,
        metavar='FRAC',
        help='Tolerance window (±fraction of total chars) for finding a clean split boundary',
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        default=False,
        help='Shuffle entries before splitting (disabled by default)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed used when --shuffle is active',
    )
    parser.add_argument(
        '--delete-original',
        action='store_true',
        default=False,
        help='Delete each input file after it has been split',
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f'Error: config file not found: {config_path}', file=sys.stderr)
        sys.exit(1)

    with open(config_path, 'rb') as fh:
        config = tomllib.load(fh)

    tokenizer_path = resolve_tokenizer(config, config_path)
    print(f'Loading tokenizer from: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    for file_arg in args.files:
        path = Path(file_arg)
        if not path.exists():
            print(f'Warning: file not found: {path}', file=sys.stderr)
            continue
        if not path.is_file():
            print(f'Warning: not a file: {path}', file=sys.stderr)
            continue
        process_file(
            path,
            tokenizer,
            args.delete_original,
            args.split_ratio,
            args.split_tolerance,
            args.shuffle,
            args.seed,
        )


if __name__ == '__main__':
    main()
