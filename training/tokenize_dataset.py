#!/usr/bin/env python3
"""
Tokenize text files into sharded binary (.bin) files for LLM pre-training.

Reads plain text (.txt), JSON Lines (.jsonl/.ndjson), or Parquet (.parquet)
files and produces uint16 binary shards (e.g. train_0000.bin, train_0001.bin)
compatible with BinaryTokenDataset in base_train.py.

Input files are expected to already contain EOS ending (from
split_dataset.py), so the tokenizer is called with add_special_tokens=False.

Usage:
    python training/tokenize_dataset.py train-text/*.txt --output training/train.bin
    python training/tokenize_dataset.py valid-text/*.txt --output training/valid.bin
    python training/tokenize_dataset.py data/**/*.jsonl --output training/train.bin --no-shuffle
"""

import argparse
import glob
import json
import math
import sys
import time
import tomllib
from pathlib import Path
from random import Random

import numpy as np
from transformers import AutoTokenizer

# 1 GiB shard cap
MAX_SHARD_BYTES = 1 * 1024 * 1024 * 1024


# ============================================================================
# Format readers
# ============================================================================


def read_txt(path: Path) -> list[str]:
    """Read entire file as a single document."""
    content = path.read_text(encoding='utf-8').strip()
    return [content] if content else []


def read_jsonl(path: Path) -> list[str]:
    """Read JSON Lines file, extracting the 'text' field from each line."""
    docs = []
    with open(path, encoding='utf-8') as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f'  Warning: {path.name}:{line_num} — invalid JSON, skipping', file=sys.stderr)
                continue
            if not isinstance(obj, dict) or 'text' not in obj:
                print(f'  Warning: {path.name}:{line_num} — no "text" key, skipping', file=sys.stderr)
                continue
            text = obj['text']
            if text:
                docs.append(text)
    return docs


def read_parquet(path: Path) -> list[str]:
    """Read Parquet file, extracting the 'text' column via PyArrow."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(f'  Error: pyarrow not installed — cannot read {path.name}', file=sys.stderr)
        print('  Install with: pip install pyarrow', file=sys.stderr)
        return []

    pf = pq.ParquetFile(path)
    schema_names = [f.name for f in pf.schema_arrow]
    if 'text' not in schema_names:
        print(f'  Error: {path.name} has no "text" column (columns: {schema_names})', file=sys.stderr)
        return []

    docs = []
    for batch in pf.iter_batches(columns=['text']):
        for val in batch.column('text'):
            text = val.as_py()
            if text:
                docs.append(text)
    return docs


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
        tokens.tofile(self._fh)
        self.shard_bytes += nbytes

    def _open_shard(self) -> None:
        path = self.dir / f'{self.stem}_{self.shard_index:04d}{self.suffix}'
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
        default='training/config.toml',
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

            try:
                docs = reader(fpath)
            except Exception as exc:
                print(
                    f'[{file_idx}/{len(files)}] {fpath.name}  — read error: {exc}, skipping',
                    file=sys.stderr,
                )
                continue

            if not docs:
                print(f'[{file_idx}/{len(files)}] {fpath.name}  — empty, skipping')
                continue

            # Tokenize in batches
            file_tokens = 0
            for batch_start in range(0, len(docs), args.batch_size):
                batch = docs[batch_start : batch_start + args.batch_size]
                encoded = tokenizer(batch, add_special_tokens=False)['input_ids']

                for ids in encoded:
                    arr = np.array(ids, dtype=np.uint16)
                    # uint16 overflow guard
                    if len(arr) > 0 and arr.max() >= 65536:
                        print(
                            f'Error: token ID >= 65536 in {fpath.name} — vocab too large for uint16 format',
                            file=sys.stderr,
                        )
                        sys.exit(1)
                    writer.write(arr)
                    file_tokens += len(arr)

            elapsed = time.perf_counter() - t0
            tok_per_sec = file_tokens / elapsed if elapsed > 0 else 0

            print(
                f'[{file_idx}/{len(files)}] {fpath.name}  '
                f'docs={len(docs):,}  '
                f'in={fmt_bytes(input_bytes)}  '
                f'tokens={file_tokens:,}  '
                f'{tok_per_sec:,.0f} tok/s  '
                f'in {elapsed:.1f}s'
            )

            total_input_bytes += input_bytes
            total_docs += len(docs)
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
