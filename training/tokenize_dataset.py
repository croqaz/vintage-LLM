#!/usr/bin/env python3
"""
Tokenize text files into sharded binary (.bin) files for LLM pre-training.

Reads plain text (.txt), JSON Lines (.jsonl/.ndjson), or Parquet (.parquet)
files and produces uint16 binary shards (e.g. train_0000.bin, train_0001.bin)
compatible with BinaryTokenDataset in base_train.py.

Input files are expected to already contain EOS ending (from
split_dataset.py), so the tokenizer is called with add_special_tokens=False.

Tokenizes with gigatoken by default (~27x less CPU than HF, bit-identical
output — see REPORT-gigatoken-feasibility.md); pass --engine hf to use the
HF tokenizer instead. On first run the gigatoken engine writes a
tokenizer.gigatoken.json next to tokenizer.json (byte-alphabet completion);
only this script reads that file — training and inference are unaffected.

Usage:
    python training/tokenize_dataset.py train-text/*.txt --output training/train.bin
    python training/tokenize_dataset.py valid-text/*.txt --output training/valid.bin
    python training/tokenize_dataset.py data/**/*.jsonl --output training/train.bin --no-shuffle
"""

import argparse
import glob
import math
import sys
import time
import tomllib
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from random import Random

import numpy as np
import orjson
import pyarrow.parquet as pq
from tokenizers.pre_tokenizers import ByteLevel
from transformers import AutoTokenizer

# 1 GiB shard cap
MAX_SHARD_BYTES = 1 * 1024 * 1024 * 1024


# ============================================================================
# Format readers
# ============================================================================


def wrap(t: str, eos: str) -> str:
    if not t.rstrip().endswith(eos):
        t = f'{t}\n{eos}'
    return t


def read_txt(path: Path, eos: str) -> Iterator[str]:
    """Yield the entire file as a single document."""
    content = path.read_text(encoding='utf-8').strip()
    if content:
        yield wrap(content, eos)


def read_jsonl(path: Path, eos: str) -> Iterator[str]:
    """Yield documents from a JSON Lines file, extracting the 'text' field
    from each line. Reads lazily, one line at a time."""
    with open(path, encoding='utf-8') as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = orjson.loads(line)
            except orjson.JSONDecodeError:
                print(f'  Warning: {path.name}:{line_num} — invalid JSON, skipping', file=sys.stderr)
                continue
            if not isinstance(obj, dict) or 'text' not in obj:
                print(f'  Warning: {path.name}:{line_num} — no "text" key, skipping', file=sys.stderr)
                continue
            text = obj['text']
            if text:
                yield wrap(text, eos)


def read_parquet(path: Path, eos: str) -> Iterator[str]:
    """Yield documents from a Parquet file, extracting the 'text' column via
    PyArrow. Iterates row-group batches lazily."""
    pf = pq.ParquetFile(path)
    schema_names = [f.name for f in pf.schema_arrow]
    if 'text' not in schema_names:
        print(f'  Error: {path.name} has no "text" column (columns: {schema_names})', file=sys.stderr)
        return

    for batch in pf.iter_batches(columns=['text']):
        for val in batch.column('text'):
            text = val.as_py()
            if text:
                yield wrap(text, eos)


READERS = {
    '.md': read_txt,
    '.txt': read_txt,
    '.text': read_txt,
    '.jsonl': read_jsonl,
    '.ndjson': read_jsonl,
    '.parquet': read_parquet,
}


# ============================================================================
# Sharded binary writer
# ============================================================================


class ShardWriter:
    """Streams uint16 token arrays into 1-GiB-capped binary shards."""

    def __init__(self, base_path: Path):
        self.dir = base_path.parent
        self.stem = base_path.stem
        self.suffix = base_path.suffix or '.bin'
        self.dir.mkdir(parents=True, exist_ok=True)

        self.shard_index = 0
        self.shard_bytes = 0
        self._fh = None
        self._open_shard()

    @property
    def shard_path(self) -> Path:
        return self.dir / f'{self.stem}_{self.shard_index:04d}{self.suffix}'

    def write(self, tokens: np.ndarray) -> None:
        """Write a uint16 array (one document) to the current shard.
        Rolls to a new shard if the current one would exceed the cap,
        but never splits a document across shards.
        """
        nbytes = tokens.nbytes
        if self.shard_bytes > 0 and self.shard_bytes + nbytes > MAX_SHARD_BYTES:
            self._close_shard()
            self.shard_index += 1
            self._open_shard()

        # Print some stats from time to time
        shard_100mb = 100 * 1024 * 1024
        if self.shard_bytes // shard_100mb != (self.shard_bytes + nbytes) // shard_100mb:
            print(f'... wrote {fmt_bytes((self.shard_bytes + nbytes) // shard_100mb * shard_100mb)} to {self.shard_path.name}')

        tokens.tofile(self._fh)
        self.shard_bytes += nbytes

    def _open_shard(self) -> None:
        path = self.shard_path
        print(f'Creating shard: {path.name}')
        self._fh = open(path, 'wb')
        self.shard_bytes = 0

    def _close_shard(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def close(self) -> int:
        """Close the active shard and return the number of shards written."""
        self._close_shard()
        return self.shard_index + 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ============================================================================
# Helpers
# ============================================================================


def batched(iterable: Iterator[str], n: int) -> Iterator[list[str]]:
    """Yield successive lists of up to n items from an iterator."""
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


def build_gigatoken_tokenizer(tokenizer_path: str):
    """Return a gigatoken.Tokenizer equivalent to the HF tokenizer at
    `tokenizer_path`.
    gigatoken's byte-level BPE backend requires all 256 single-byte tokens in
    the vocab; ours lacks 39 of them (HF maps those bytes to unk). Generate a
    sibling tokenizer.gigatoken.json that assigns the missing byte chars fresh
    ids above the base vocab — callers must remap ids >= base vocab back to
    unk_id so the output is bit-identical to HF. Only this script reads the
    generated file; training/inference keep loading tokenizer.json.
    """
    import gigatoken as gt

    src = Path(tokenizer_path) / 'tokenizer.json'
    if not src.is_file():
        sys.exit(f'Error: {src} not found — gigatoken needs a local tokenizer.json (use --engine hf for hub ids)')

    data = orjson.loads(src.read_text(encoding='utf-8'))
    vocab = data['model']['vocab']
    missing = sorted(c for c in ByteLevel.alphabet() if c not in vocab)
    if not missing:
        return gt.Tokenizer(src)

    dst = src.with_name('tokenizer.gigatoken.json')
    if not dst.is_file() or dst.stat().st_mtime < src.stat().st_mtime:
        next_id = max(vocab.values()) + 1
        for char in missing:
            vocab[char] = next_id
            next_id += 1
        dst.write_text(orjson.dumps(data).decode('utf-8'), encoding='utf-8')
        print(f'Wrote {dst.name} (+{len(missing)} byte-alphabet entries for gigatoken)')
    return gt.Tokenizer(dst)


def resolve_inputs(patterns: list[str]) -> list[Path]:
    """Expand globs, deduplicate, keep only existing files.
    If an input is a directory, recursively collect all known extensions.
    """
    seen: set[Path] = set()
    result: list[Path] = []

    def _add(p: Path) -> None:
        p = p.resolve()
        if p not in seen and p.is_file():
            seen.add(p)
            result.append(p)

    for pattern in patterns:
        plain = Path(pattern)
        if plain.is_dir():
            for ext in READERS:
                for match in sorted(plain.rglob(f'*{ext}')):
                    _add(match)
        elif '*' in pattern or '?' in pattern:
            for m in sorted(glob.glob(pattern, recursive=True)):
                _add(Path(m))
        else:
            _add(plain)

    return sorted(result)


def fmt_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if abs(n) < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} TiB'


def format_number_with_unit(number: int, unit: str = '') -> str:
    """Format a large integer into a human-readable string with suffixes.

    Examples:
        >>> format_number_with_unit(83_210_646_761, 'tokens')
        '83.2B tokens'
        >>> format_number_with_unit(1_500, 'items')
        '1.5K items'
        >>> format_number_with_unit(500, 'points')
        '500 points'
        >>> format_number_with_unit(1_000_000_000_000, 'bytes')
        '1.0T bytes'
    """
    if number < 1_000:
        return f'{number:,}{f" {unit}" if unit else ""}'
    suffixes = ['', 'K', 'M', 'B', 'T']
    magnitude = min(int(math.log10(number) // 3), len(suffixes) - 1)
    divisor = 10 ** (magnitude * 3)
    value = number / divisor
    if value == int(value):
        formatted = f'{int(value)}'
    else:
        formatted = f'{value:.1f}'
    return f'{formatted}{suffixes[magnitude]}{f" {unit}" if unit else ""}'


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Tokenize text/JSONL/Parquet files into sharded .bin files for training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'inputs',
        nargs='+',
        help='Input files or glob patterns (e.g. train-text/*.txt)',
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output base path (e.g. training/train.bin). Shards are named train_0000.bin, …',
    )
    parser.add_argument(
        '--config',
        default=str(Path(__file__).resolve().parent / 'config.toml'),
        help='Path to config.toml (reads data.tokenizer)',
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        default=False,
        help='Do not shuffle input files before processing',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for file shuffling',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of documents to tokenize per batch',
    )
    parser.add_argument(
        '--engine',
        choices=('gigatoken', 'hf'),
        default='gigatoken',
        help='Tokenization engine (gigatoken is ~40x faster, bit-identical output)',
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load config & tokenizer
    # ------------------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        print(f'Error: config file not found: {config_path}', file=sys.stderr)
        sys.exit(1)

    with open(config_path, 'rb') as fh:
        cfg = tomllib.load(fh)

    tokenizer_path = cfg['data']['tokenizer']
    print(f'Loading tokenizer from: {tokenizer_path}...\n')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    vocab_size = len(tokenizer)
    if vocab_size > 65535:
        print(
            f'Error: vocab size {vocab_size} exceeds uint16 max (65535). Binary format cannot represent these token IDs.',
            file=sys.stderr,
        )
        sys.exit(1)
    print(f'Vocab size: {vocab_size}')

    # gigatoken emits ids >= vocab_size for the byte-alphabet entries added in
    # tokenizer.gigatoken.json; those are remapped to unk_id below, matching
    # what HF does with the corresponding bytes.
    gtok = build_gigatoken_tokenizer(tokenizer_path) if args.engine == 'gigatoken' else None
    unk_id = tokenizer.unk_token_id

    eos = tokenizer.eos_token or ''
    if not eos:
        print('Warning: tokenizer has no eos_token', file=sys.stderr)

    # ------------------------------------------------------------------
    # Resolve and shuffle input files
    # ------------------------------------------------------------------
    files = resolve_inputs(args.inputs)
    if not files:
        print('Error: no input files found', file=sys.stderr)
        sys.exit(1)

    if not args.no_shuffle:
        Random(args.seed).shuffle(files)

    print(f'\nInput files found: {len(files)}\n')

    # ------------------------------------------------------------------
    # Process files
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    total_start = time.perf_counter()
    total_input_bytes = 0
    total_docs = 0
    total_tokens = 0
    files_ok = 0

    with ShardWriter(output_path) as writer:
        for file_idx, fpath in enumerate(files, 1):
            ext = fpath.suffix.lower()
            reader = READERS.get(ext)
            if reader is None:
                print(f'[{file_idx}/{len(files)}] {fpath.name}  — unknown extension {ext}, skipping')
                continue

            t0 = time.perf_counter()
            input_bytes = fpath.stat().st_size

            # Stream documents through tokenization in batches so we never
            # hold an entire file's worth of text/tokens in memory at once.
            file_tokens = 0
            file_docs = 0
            try:
                for batch in batched(reader(fpath, eos), args.batch_size):
                    if gtok is not None:
                        encoded = gtok.encode_batch_list(batch)
                    else:
                        encoded = tokenizer(batch, add_special_tokens=False)['input_ids']
                    file_docs += len(batch)

                    for ids in encoded:
                        arr = np.asarray(ids, dtype=np.uint32)
                        if len(arr) > 0 and arr.max() >= vocab_size:
                            # gigatoken byte-alphabet ids -> unk, like HF does
                            if arr.max() >= 65536 or unk_id is None:
                                print(
                                    f'Error: token ID {arr.max()} out of range in {fpath.name} — vocab too large for uint16 format',
                                    file=sys.stderr,
                                )
                                sys.exit(1)
                            arr = arr.copy()
                            arr[arr >= vocab_size] = unk_id
                        writer.write(arr.astype(np.uint16))
                        file_tokens += len(arr)
            except Exception as exc:
                print(
                    f'[{file_idx}/{len(files)}] {fpath.name}  — read error: {exc}, skipping',
                    file=sys.stderr,
                )
                continue

            if file_docs == 0:
                print(f'[{file_idx}/{len(files)}] {fpath.name}  — empty, skipping')
                continue

            elapsed = time.perf_counter() - t0
            tok_per_sec = file_tokens / elapsed if elapsed > 0 else 0

            print(
                f'[{file_idx}/{len(files)}] {fpath.name}  '
                f'docs={file_docs:,}  '
                f'in={fmt_bytes(input_bytes)}  '
                f'tokens={file_tokens:,}  '
                f'{tok_per_sec:,.0f} tok/s  '
                f'in {elapsed:.1f}s'
            )

            total_input_bytes += input_bytes
            total_docs += file_docs
            total_tokens += file_tokens
            files_ok += 1

        num_shards = writer.close()

    total_elapsed = time.perf_counter() - total_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('TOKENIZATION SUMMARY')
    print('=' * 70)
    print(f'  Files processed:  {files_ok}/{len(files)}')
    print(f'  Documents:        {total_docs:,}')
    print(f'  Input size:       {fmt_bytes(total_input_bytes)}')
    print(f'  Output tokens:    {total_tokens:,} ({format_number_with_unit(total_tokens, "tok")})')
    print(f'  Output size:      {fmt_bytes(total_tokens * 2)} (uint16)')
    print(f'  Shards written:   {num_shards}')
    for i in range(num_shards):
        shard = output_path.parent / f'{output_path.stem}_{i:04d}{output_path.suffix or ".bin"}'
        if shard.exists():
            print(f'    {shard.name}: {fmt_bytes(shard.stat().st_size)}')
    print(f'  Wall time:        {total_elapsed:.1f}s')
    if total_elapsed > 0:
        print(f'  Throughput:       {int(total_tokens / total_elapsed):,} tok/s')
    print('=' * 70)


if __name__ == '__main__':
    main()
