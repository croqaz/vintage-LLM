"""
Ingest JSONL → compute statistics & hashes → deduplicate → sharded Parquet.

Pipeline per input file:
  1. Load JSONL          → HF Dataset         (rows_loaded)
  2. Compute stats+hash  → HF Dataset         (rows_after_compute)
                                               rows_dropped_length skipped silently
  3. Deduplicate via     → filtered batches   (rows_new_unique kept, rows_dupes skipped,
     persistent LMDB                           rows_short_bypass kept without LMDB check)
  4. Write sharded Parquet                    (rows_written = rows_new_unique + rows_short_bypass)

Row accounting invariants (verified after every file):
  rows_loaded        == rows_dropped_length + rows_after_compute
  rows_after_compute == rows_short_bypass + rows_new_unique + rows_dupes
  rows_written       == rows_short_bypass + rows_new_unique

Final verification: reads back Parquet metadata (num_rows) for every shard written
in this run, sums them, and asserts the total equals grand_total_written.

LMDB layout (persistent across runs):
  key   — xxh64 hex digest, UTF-8  (16 bytes per key)
  value — text[:preview_len], UTF-8  (for human inspection only)

Output columns (PA_SCHEMA):
  text, len, len_uniq, words, sentences, xxh64, minhash

Notes:
  - Texts shorter than --min-dedup-len chars bypass LMDB dedup; they are always
    kept and never stored in LMDB, so short titles like "Poems" are preserved.
  - Each JSONL file is processed and flushed independently; memory stays bounded.
  - Shard naming: {input_stem}-{shard:05d}.parquet (no "of-N" suffix because the
    final shard count is unknown before dedup completes).

Examples:
  python ingest_dedup.py \\
      --input-glob "raw/*.jsonl" \\
      --out-dir parquet_out/ \\
      --lmdb-dir lmdb_store/

  python ingest_dedup.py \\
      --input-glob "raw/**/*.jsonl" \\
      --out-dir parquet_out/ \\
      --lmdb-dir lmdb_store/ \\
      --num-proc 16 \\
      --dupes-file dupes.jsonl \\
      --skip-existing
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import lmdb
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash
from datasets import Features, Sequence, Value, load_dataset
from datasketch import MinHash

# ──────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────
DEFAULT_TEXT_KEY = 'text'
DEFAULT_NUM_PERM = 128
DEFAULT_NGRAM_SIZE = 5
DEFAULT_COMPUTE_BATCH_SIZE = 1_000
DEFAULT_DEDUP_BATCH_SIZE = 10_000
DEFAULT_ROWS_PER_SHARD = 500_000
DEFAULT_LMDB_MAP_SIZE = 16 * 1024**3  # 16 GiB virtual address space
DEFAULT_PREVIEW_LEN = 250
DEFAULT_MIN_DEDUP_LEN = 100  # texts shorter than this bypass LMDB dedup

# ──────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────
OUT_FEATURES = Features(
    {
        'text': Value('large_string'),
        'len': Value('uint32'),
        'len_uniq': Value('uint32'),
        'words': Value('uint32'),
        'sentences': Value('uint32'),
        'xxh64': Value('string'),
        'minhash': Sequence(Value('uint64')),
    }
)

PA_SCHEMA = pa.schema(
    [
        ('text', pa.large_string()),
        ('len', pa.uint32()),
        ('len_uniq', pa.uint32()),
        ('words', pa.uint32()),
        ('sentences', pa.uint32()),
        ('xxh64', pa.string()),
        ('minhash', pa.list_(pa.uint64())),
    ]
)

OUTPUT_COLUMNS = set(OUT_FEATURES)


SENTENCE_BOUNDARY_REGEX = re.compile(r'((?:[.!?]["\']?)\s+(?=[A-Z"\'])|(?:[\n\r]{2,}\s*(?=[a-zA-Z"\'])))')


def split_into_sentences(text: str) -> list[str]:
    parts = SENTENCE_BOUNDARY_REGEX.split(text)
    return [parts[i] + (parts[i + 1] if i + 1 < len(parts) else '') for i in range(0, len(parts), 2)]


# ──────────────────────────────────────────────────────────────────
# Compute batch
# Pickle-safe: no closures, all config via fn_kwargs.
# Required for multiprocessing on Linux (fork) and macOS (spawn).
# ──────────────────────────────────────────────────────────────────
def compute_batch(
    batch: dict,
    text_key: str = DEFAULT_TEXT_KEY,
    num_perm: int = DEFAULT_NUM_PERM,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
) -> dict:
    """
    HF .map(batched=True) worker — computes per-document statistics and hashes.
    Rows with len(text) <= 3 or > 10_000_000 are silently dropped (sanity guard).
    """
    out: dict[str, list] = {c: [] for c in OUTPUT_COLUMNS}

    for text in batch[text_key]:
        if text is None:
            text = ''
        if len(text) <= 3 or len(text) > 10_000_000:
            continue  # counted as rows_dropped_length in the caller

        tokens = text.split()

        out['text'].append(text)
        out['len'].append(len(text))
        out['len_uniq'].append(len(set(text)))
        out['words'].append(len(tokens))

        if text.strip():
            segs = split_into_sentences(text)
            out['sentences'].append(max(1, len(segs)))
        else:
            out['sentences'].append(0)

        out['xxh64'].append(xxhash.xxh64_hexdigest(text.encode('utf-8')))

        mh = MinHash(num_perm=num_perm)
        n = ngram_size
        if len(tokens) >= n:
            for i in range(len(tokens) - n + 1):
                mh.update(' '.join(tokens[i : i + n]).encode('utf-8'))
        else:
            mh.update(text.encode('utf-8'))

        out['minhash'].append(mh.hashvalues.tolist())

    return out


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def strip_extensions(path: str) -> str:
    """Derive an output prefix from an input filename."""
    name = Path(path).name
    for suffix in ('.jsonl.gz', '.jsonl.zst', '.jsonl'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


# ──────────────────────────────────────────────────────────────────
# Row-accounting dataclass
# ──────────────────────────────────────────────────────────────────
@dataclass
class FileStats:
    path: str
    rows_loaded: int = 0
    rows_dropped_length: int = 0  # len ≤ 3 or > 10 M; dropped by compute_batch
    rows_after_compute: int = 0
    rows_short_bypass: int = 0  # len < min_dedup_len; kept, never stored in LMDB
    rows_dupes: int = 0  # LMDB hit or intra-batch dupe; skipped
    rows_new_unique: int = 0  # new entry stored in LMDB; written to Parquet
    rows_written: int = 0  # short_bypass + new_unique
    shards: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """
        Raise AssertionError if any accounting invariant is violated.
        This is a historic dataset — every row must be accounted for.
        """
        ok1 = self.rows_loaded == self.rows_dropped_length + self.rows_after_compute
        ok2 = self.rows_after_compute == (self.rows_short_bypass + self.rows_new_unique + self.rows_dupes)
        ok3 = self.rows_written == self.rows_short_bypass + self.rows_new_unique

        if not (ok1 and ok2 and ok3):
            raise AssertionError(
                f'\n❌  Row accounting mismatch for: {self.path}\n'
                f'    loaded={self.rows_loaded:,}  '
                f'dropped_len={self.rows_dropped_length:,}  '
                f'after_compute={self.rows_after_compute:,}\n'
                f'    short_bypass={self.rows_short_bypass:,}  '
                f'dupes={self.rows_dupes:,}  '
                f'new_unique={self.rows_new_unique:,}  '
                f'written={self.rows_written:,}\n'
                f'    Invariant 1 — loaded = dropped + after_compute: {ok1}\n'
                f'    Invariant 2 — after_compute = short_bypass + new_unique + dupes: {ok2}\n'
                f'    Invariant 3 — written = short_bypass + new_unique: {ok3}'
            )

    def print_summary(self, min_dedup_len: int) -> None:
        print(
            f'  Loaded:               {self.rows_loaded:>14,}\n'
            f'  Dropped (length):     {self.rows_dropped_length:>14,}  (len ≤ 3 or > 10 M)\n'
            f'  After compute:        {self.rows_after_compute:>14,}\n'
            f'  Short bypass (<{min_dedup_len}c):  {self.rows_short_bypass:>14,}  (always kept)\n'
            f'  Duplicates skipped:   {self.rows_dupes:>14,}\n'
            f'  New unique:           {self.rows_new_unique:>14,}\n'
            f'  Written:              {self.rows_written:>14,}\n'
            f'  Shards written:       {len(self.shards):>14,}'
        )


# ──────────────────────────────────────────────────────────────────
# Shard writer
# ──────────────────────────────────────────────────────────────────
def flush_shard(
    buffer: list[pa.Table],
    out_dir: str,
    stem: str,
    shard_idx: int,
) -> tuple[int, str]:
    """
    Concatenate buffered PyArrow tables and write one Parquet shard (zstd).
    Returns (num_rows_written, absolute_path). Returns (0, '') if buffer is empty.
    """
    if not buffer:
        return 0, ''
    table = pa.concat_tables(buffer).cast(PA_SCHEMA)
    path = os.path.join(out_dir, f'{stem}-{shard_idx:05d}.parquet')
    pq.write_table(table, path, compression='zstd')
    n = len(table)
    print(f'  ✓ {path}  ({n:,} rows)')
    return n, path


# ──────────────────────────────────────────────────────────────────
# Per-file pipeline
# ──────────────────────────────────────────────────────────────────
def process_file(
    input_path: str,
    out_dir: str,
    env: lmdb.Environment,
    text_key: str,
    num_perm: int,
    ngram_size: int,
    cache_dir: str | None,
    num_proc: int,
    compute_batch_size: int,
    dedup_batch_size: int,
    rows_per_shard: int,
    preview_len: int,
    min_dedup_len: int,
    dupes_fh,
) -> FileStats:
    stats = FileStats(path=input_path)
    stem = strip_extensions(input_path)

    # ── 1. Load ────────────────────────────────────────────────────
    ds = load_dataset(
        'json',
        split='train',
        data_files=input_path,
        cache_dir=cache_dir,
    )
    if text_key not in ds.column_names:
        raise ValueError(f"Field '{text_key}' not found in {input_path}. Available columns: {ds.column_names}")
    stats.rows_loaded = len(ds)
    print(f'  Loaded:        {stats.rows_loaded:,} rows')

    # ── 2. Compute (multiprocessing, CPU-heavy) ────────────────────
    # LMDB dedup happens after this returns, in the main process only.
    # This keeps LMDB single-writer while still parallelising hashing.
    ds = ds.map(
        compute_batch,
        batched=True,
        batch_size=compute_batch_size,
        writer_batch_size=compute_batch_size,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        features=OUT_FEATURES,
        fn_kwargs=dict(text_key=text_key, num_perm=num_perm, ngram_size=ngram_size),
        desc='  Computing stats & hashes',
    )
    stats.rows_after_compute = len(ds)
    stats.rows_dropped_length = stats.rows_loaded - stats.rows_after_compute
    print(f'  After compute: {stats.rows_after_compute:,} rows  ({stats.rows_dropped_length:,} dropped by length guard)')

    # ── 3. Dedup + buffer (single-threaded; LMDB is single-writer) ─
    shard_idx = 0
    buffer: list[pa.Table] = []
    buffer_rows = 0

    # .with_format('arrow').iter() yields pyarrow.Table objects per batch.
    for batch in ds.with_format('arrow').iter(batch_size=dedup_batch_size):
        xxh64_col = batch.column('xxh64')
        text_col = batch.column('text')
        keys = [v.as_py().encode() for v in xxh64_col]

        keep_indices: list[int] = []
        new_kv: dict[bytes, bytes] = {}
        seen_in_batch: set[bytes] = set()

        # Single read transaction covers the whole batch for efficiency.
        with env.begin(write=False) as txn:
            for i, key in enumerate(keys):
                text = text_col[i].as_py() or ''

                if len(text) < min_dedup_len:
                    # Short texts: always keep, never stored in LMDB.
                    # Identical short titles (e.g. "Poems") must not be deduplicated.
                    keep_indices.append(i)
                    stats.rows_short_bypass += 1
                    continue

                if txn.get(key) is not None or key in seen_in_batch:
                    # Duplicate: already in LMDB or seen earlier in this batch.
                    stats.rows_dupes += 1
                    if dupes_fh is not None:
                        dupes_fh.write(
                            json.dumps(
                                {'xxh64': key.decode(), 'text': text[:preview_len]},
                                ensure_ascii=False,
                            )
                            + '\n'
                        )
                else:
                    keep_indices.append(i)
                    seen_in_batch.add(key)
                    new_kv[key] = text[:preview_len].encode('utf-8')
                    stats.rows_new_unique += 1

        # Write all new keys in a single transaction.
        if new_kv:
            with env.begin(write=True) as txn:
                for k, v in new_kv.items():
                    txn.put(k, v)

        if keep_indices:
            filtered = batch.take(keep_indices)
            buffer.append(filtered)
            buffer_rows += len(filtered)

            if buffer_rows >= rows_per_shard:
                n, path = flush_shard(buffer, out_dir, stem, shard_idx)
                stats.rows_written += n
                stats.shards.append(path)
                shard_idx += 1
                buffer = []
                buffer_rows = 0

    # Flush remaining rows for this file.
    if buffer:
        n, path = flush_shard(buffer, out_dir, stem, shard_idx)
        stats.rows_written += n
        stats.shards.append(path)

    # ── 4. Validate accounting ──────────────────────────────────────
    stats.validate()
    return stats


# ──────────────────────────────────────────────────────────────────
# Final verification
# ──────────────────────────────────────────────────────────────────
def verify_parquet_rows(shard_paths: list[str], expected: int) -> int:
    """
    Read Parquet metadata (num_rows) for each shard written in this run.
    Asserts the sum equals expected. Raises AssertionError on mismatch.
    """
    total = 0
    for path in sorted(set(shard_paths)):
        meta = pq.read_metadata(path)
        total += meta.num_rows

    if total != expected:
        raise AssertionError(
            f'\n❌  VERIFICATION FAILED\n'
            f'    Parquet metadata total : {total:,}\n'
            f'    Expected (in-memory)   : {expected:,}\n'
            f'    Delta                  : {total - expected:+,}'
        )
    return total


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Ingest JSONL → compute stats+hashes → deduplicate via LMDB → sharded Parquet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--input-glob',
        required=True,
        help="Glob pattern for JSONL input files (e.g. 'raw/*.jsonl')",
    )
    p.add_argument(
        '--out-dir',
        required=True,
        help='Output directory for deduplicated Parquet shards',
    )
    p.add_argument(
        '--lmdb-dir',
        required=True,
        help='LMDB database directory (created if absent; persistent across runs)',
    )
    p.add_argument('--text-key', default=DEFAULT_TEXT_KEY, help='JSON field containing document text')
    p.add_argument('--num-perm', type=int, default=DEFAULT_NUM_PERM, help='MinHash permutations')
    p.add_argument('--ngram-size', type=int, default=DEFAULT_NGRAM_SIZE, help='Word n-gram size for MinHash')
    p.add_argument(
        '--rows-per-shard',
        type=int,
        default=DEFAULT_ROWS_PER_SHARD,
        help='Maximum rows per output Parquet shard (may be slightly exceeded by up to --dedup-batch-size)',
    )
    p.add_argument(
        '--compute-batch-size',
        type=int,
        default=DEFAULT_COMPUTE_BATCH_SIZE,
        help='Batch size for the HF datasets.map() compute phase',
    )
    p.add_argument(
        '--dedup-batch-size',
        type=int,
        default=DEFAULT_DEDUP_BATCH_SIZE,
        help='Batch size for the LMDB dedup iteration',
    )
    p.add_argument(
        '--num-proc',
        type=int,
        default=os.cpu_count() or 4,
        help='Parallel workers for the compute phase (dedup is always single-threaded)',
    )
    p.add_argument('--cache-dir', default=None, help='HF datasets Arrow cache directory (fast SSD recommended)')
    p.add_argument(
        '--lmdb-map-size',
        type=int,
        default=DEFAULT_LMDB_MAP_SIZE,
        help='LMDB virtual address space in bytes (ceiling, not a physical reservation)',
    )
    p.add_argument(
        '--preview-len',
        type=int,
        default=DEFAULT_PREVIEW_LEN,
        help='Max text characters stored as the LMDB value (for human inspection only)',
    )
    p.add_argument(
        '--min-dedup-len',
        type=int,
        default=DEFAULT_MIN_DEDUP_LEN,
        help='Texts shorter than this many characters bypass LMDB dedup and are always kept',
    )
    p.add_argument(
        '--dupes-file',
        default=None,
        help='Optional JSONL file to append duplicate entries to: {"xxh64": ..., "text": ...}',
    )
    p.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip JSONL files whose output shards already exist in --out-dir',
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    files = sorted(glob.glob(args.input_glob, recursive=True))
    if not files:
        raise SystemExit(f'No files matched: {args.input_glob}')
    print(f'Found {len(files)} input file(s)')

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.lmdb_dir, exist_ok=True)

    env = lmdb.open(
        args.lmdb_dir,
        map_size=args.lmdb_map_size,
        subdir=True,
        max_spare_txns=4,
    )
    dupes_fh = open(args.dupes_file, 'a', encoding='utf-8') if args.dupes_file else None

    # Grand totals accumulated across all processed files.
    grand_loaded = 0
    grand_dropped_length = 0
    grand_short_bypass = 0
    grand_dupes = 0
    grand_new_unique = 0
    grand_written = 0
    grand_shards: list[str] = []
    skipped_files: list[str] = []

    for input_path in files:
        print(f'\n{"=" * 60}')
        print(f'Processing: {input_path}')
        print(f'{"=" * 60}')

        if args.skip_existing:
            stem = strip_extensions(input_path)
            existing = glob.glob(os.path.join(args.out_dir, f'{stem}-*.parquet'))
            if existing:
                print(f'⏭  Skipping — {len(existing)} shard(s) already exist')
                skipped_files.append(input_path)
                continue

        stats = process_file(
            input_path=input_path,
            out_dir=args.out_dir,
            env=env,
            text_key=args.text_key,
            num_perm=args.num_perm,
            ngram_size=args.ngram_size,
            cache_dir=args.cache_dir,
            num_proc=args.num_proc,
            compute_batch_size=args.compute_batch_size,
            dedup_batch_size=args.dedup_batch_size,
            rows_per_shard=args.rows_per_shard,
            preview_len=args.preview_len,
            min_dedup_len=args.min_dedup_len,
            dupes_fh=dupes_fh,
        )

        stats.print_summary(args.min_dedup_len)

        grand_loaded += stats.rows_loaded
        grand_dropped_length += stats.rows_dropped_length
        grand_short_bypass += stats.rows_short_bypass
        grand_dupes += stats.rows_dupes
        grand_new_unique += stats.rows_new_unique
        grand_written += stats.rows_written
        grand_shards.extend(stats.shards)

    if dupes_fh is not None:
        dupes_fh.close()

    # ── Final verification ─────────────────────────────────────────
    print(f'\n{"=" * 60}')
    print('FINAL VERIFICATION')
    print(f'{"=" * 60}')

    if skipped_files:
        print(
            f'  ⚠  {len(skipped_files)} file(s) were skipped (--skip-existing).\n'
            f'     Parquet verification covers only shards written in this run.'
        )

    if grand_shards:
        verified = verify_parquet_rows(grand_shards, grand_written)
        verification_line = f'  Parquet verified rows:       {verified:>14,}  ✓'
    else:
        verification_line = '  Parquet verified rows:              (no shards written this run)'

    lmdb_entries = env.stat()['entries']
    env.close()

    rows_after_compute = grand_loaded - grand_dropped_length
    print(
        f'\n  Grand total loaded:          {grand_loaded:>14,}\n'
        f'  Dropped (length guard):      {grand_dropped_length:>14,}  (len ≤ 3 or > 10 M)\n'
        f'  After compute:               {rows_after_compute:>14,}\n'
        f'  Short bypass (<{args.min_dedup_len}c):        {grand_short_bypass:>14,}  (always kept)\n'
        f'  Duplicates skipped:          {grand_dupes:>14,}\n'
        f'  New unique (→ LMDB):         {grand_new_unique:>14,}\n'
        f'  Written (total):             {grand_written:>14,}\n'
        f'  {"─" * 50}\n'
        f'{verification_line}\n'
        f'  LMDB entries (cumulative):   {lmdb_entries:>14,}\n'
        f'  Output shards (this run):    {len(grand_shards):>14,}'
    )

    if args.dupes_file:
        print(f'  Dupes log:                   {args.dupes_file}')

    cache_loc = args.cache_dir or '~/.cache/huggingface/datasets'
    print('\n💡  HF datasets Arrow cache may be large. Clean up with:')
    print(f'    rm -rf {cache_loc}')


if __name__ == '__main__':
    main()
