"""
Merge multiple Lance datasets (same schema) into a single fresh output dataset.

For each row:
  - Re-compute the composite ID (xxh64 + blake2b on whitespace-normalized UTF-8 text).
  - Re-compute `length` (character count of the original text).
  - All other columns pass through unchanged.

Duplicate IDs are resolved via in-memory dedup + Lance append:
  - First-seen row wins for text / source / unique_chars / words / sentences.
  - minhash / lsh_bands / embed1 are coalesced: if a later row brings a non-null
    hash that the first-seen row lacked, the output is updated via a single
    merge_insert pass at the end.

Rows where len(text) <= 10 or > 32 000 are dropped (same filter as index_lance.py).
All-zero minhash / embed1 vectors are treated as NULL.

Resumable: use --resume to continue from where a previous run left off.
The script rebuilds the seen-IDs set from the existing output and re-scans
all inputs, skipping already-indexed rows instantly.

Examples
────────
  # Merge two datasets
  python merge_lance.py \\
      --inputs /data/ds1.lance /data/ds2.lance \\
      --output-path /data/merged.lance

  # Resume after interruption
  python merge_lance.py \\
      --inputs /data/ds1.lance /data/ds2.lance \\
      --output-path /data/merged.lance --resume

  # Overwrite existing output
  python merge_lance.py \\
      --inputs /data/ds1.lance \\
      --output-path /data/merged.lance --overwrite
"""

import argparse
import hashlib
import os

import lance
import pyarrow as pa
import xxhash
from tqdm.auto import tqdm

# ──────────────────────────────────────────────────────────────────
# Constants (must match index_lance.py)
# ──────────────────────────────────────────────────────────────────
MAX_CHARS = 32_000
DEFAULT_READ_BATCH_SIZE = 4096
DEFAULT_WRITE_BATCH_SIZE = 8192

# Hash-fill bitmask flags
HAS_MINHASH = 1
HAS_LSH = 2
HAS_EMBED = 4
HASH_FULL = HAS_MINHASH | HAS_LSH | HAS_EMBED

# All expected source columns (minus id/length which we recompute).
EXPECTED_SOURCE_COLS = {'text', 'source', 'unique_chars', 'words', 'sentences', 'minhash', 'lsh_bands', 'embed1'}

# Source columns to read (we recompute id + length).
READ_COLS = ['text', 'source', 'unique_chars', 'words', 'sentences', 'minhash', 'lsh_bands', 'embed1']


# ──────────────────────────────────────────────────────────────────
# ID computation — mirrors index_lance.py prefilter_batch exactly.
# ──────────────────────────────────────────────────────────────────
def compute_id(text: str) -> str:
    """
    Whitespace-normalize text to bytes, then produce a 48-char hex ID:
    xxh3_64 (16 hex) + blake2b/16 (32 hex).
    """
    enc_text = b' '.join(text.encode('utf-8').split())
    xxh64_hex = xxhash.xxh3_64_hexdigest(enc_text)
    blake2b_hex = hashlib.blake2b(enc_text, digest_size=16).hexdigest()
    return f'{xxh64_hex}{blake2b_hex}'


# ──────────────────────────────────────────────────────────────────
# Schema builder
# ──────────────────────────────────────────────────────────────────
def make_output_schema(num_perm: int, embedding_dim: int) -> pa.Schema:
    """Build the Arrow schema for the merged output dataset."""
    return pa.schema(
        [
            pa.field('id', pa.string(), nullable=False),
            pa.field('text', pa.large_string(), nullable=False),
            pa.field('source', pa.string(), nullable=True),
            pa.field('length', pa.uint32(), nullable=False),
            pa.field('unique_chars', pa.uint32(), nullable=False),
            pa.field('words', pa.uint32(), nullable=False),
            pa.field('sentences', pa.uint32(), nullable=False),
            pa.field('minhash', pa.list_(pa.uint64(), num_perm), nullable=True),
            pa.field('lsh_bands', pa.list_(pa.string()), nullable=True),
            pa.field('embed1', pa.list_(pa.float16(), embedding_dim), nullable=True),
        ]
    )


# ──────────────────────────────────────────────────────────────────
# Null / zero helpers
# ──────────────────────────────────────────────────────────────────
def _is_null_or_none(val) -> bool:
    """Check if a value is None, empty, or an all-zero numeric list."""
    if val is None:
        return True
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return True
        # All-zero minhash / embed1 vectors are useless — treat as null.
        if isinstance(val[0], (int, float)) and not any(val):
            return True
    return False


def _normalize(val):
    """Return None if val is null-like (None, empty, all-zero), else val as-is."""
    return None if _is_null_or_none(val) else val


def _hash_bitmask_from_row(minhash, lsh_bands, embed1) -> int:
    """Compute a 3-bit bitmask indicating which hash columns are non-null."""
    mask = 0
    if minhash is not None:
        mask |= HAS_MINHASH
    if lsh_bands is not None:
        mask |= HAS_LSH
    if embed1 is not None:
        mask |= HAS_EMBED
    return mask


# ──────────────────────────────────────────────────────────────────
# Schema detection & validation
# ──────────────────────────────────────────────────────────────────
def detect_schema_params(ds: lance.LanceDataset) -> tuple[int, int]:
    """Detect num_perm and embedding_dim from an existing dataset's schema."""
    schema = ds.schema
    num_perm = schema.field('minhash').type.list_size
    embedding_dim = schema.field('embed1').type.list_size
    return num_perm, embedding_dim


def validate_input_columns(ds: lance.LanceDataset, path: str) -> None:
    """Check that the input dataset has all the columns we need."""
    col_names = set(ds.schema.names)
    missing = EXPECTED_SOURCE_COLS - col_names
    if missing:
        raise SystemExit(f'Input {path} is missing columns: {sorted(missing)}. Available: {sorted(col_names)}')


# ──────────────────────────────────────────────────────────────────
# Resume: rebuild seen_ids from existing output
# ──────────────────────────────────────────────────────────────────
def rebuild_seen_ids(output_path: str, batch_size: int = 65536) -> dict[str, int]:
    """
    Scan the output dataset and return {id: hash_bitmask} for every row.
    Used on --resume to skip already-indexed rows without re-writing them.
    """
    ds = lance.dataset(output_path)
    total = ds.count_rows()
    if total == 0:
        return {}

    seen: dict[str, int] = {}
    pbar = tqdm(total=total, desc='  Rebuilding seen IDs', unit='row', dynamic_ncols=True)

    for batch in ds.to_batches(
        columns=['id', 'minhash', 'lsh_bands', 'embed1'],
        batch_size=batch_size,
    ):
        ids = batch.column('id').to_pylist()
        mh_valid = batch.column('minhash').is_valid().to_pylist()
        lsh_valid = batch.column('lsh_bands').is_valid().to_pylist()
        em_valid = batch.column('embed1').is_valid().to_pylist()

        for i, rid in enumerate(ids):
            mask = 0
            if mh_valid[i]:
                mask |= HAS_MINHASH
            if lsh_valid[i]:
                mask |= HAS_LSH
            if em_valid[i]:
                mask |= HAS_EMBED
            seen[rid] = mask

        pbar.update(batch.num_rows)

    pbar.close()
    return seen


# ──────────────────────────────────────────────────────────────────
# Build a PyArrow table from a list of row dicts
# ──────────────────────────────────────────────────────────────────
def _build_fixed_list_column(
    values: list,
    list_size: int,
    value_type: pa.DataType,
) -> pa.FixedSizeListArray:
    """
    Build a nullable FixedSizeListArray from a list of (list | None).
    None entries become null in the output array.
    """
    n = len(values)
    if n == 0:
        return pa.array([], type=pa.list_(value_type, list_size))

    all_null = all(v is None for v in values)
    if all_null:
        return pa.nulls(n, type=pa.list_(value_type, list_size))

    # Let pyarrow handle None → null natively; no manual buffer tricks.
    return pa.array(values, type=pa.list_(value_type, list_size))


def rows_to_table(
    rows: list[dict],
    schema: pa.Schema,
    num_perm: int,
    embedding_dim: int,
) -> pa.Table:
    """Convert a list of row dicts into a pa.Table matching the output schema."""
    n = len(rows)
    if n == 0:
        return pa.table(
            {f.name: pa.array([], type=f.type) for f in schema},
            schema=schema,
        )

    return pa.table(
        {
            'id': pa.array([r['id'] for r in rows], type=pa.string()),
            'text': pa.array([r['text'] for r in rows], type=pa.large_string()),
            'source': pa.array([r.get('source') for r in rows], type=pa.string()),
            'length': pa.array([r['length'] for r in rows], type=pa.uint32()),
            'unique_chars': pa.array([r['unique_chars'] for r in rows], type=pa.uint32()),
            'words': pa.array([r['words'] for r in rows], type=pa.uint32()),
            'sentences': pa.array([r['sentences'] for r in rows], type=pa.uint32()),
            'minhash': _build_fixed_list_column(
                [r.get('minhash') for r in rows],
                num_perm,
                pa.uint64(),
            ),
            'lsh_bands': pa.array(
                [r.get('lsh_bands') for r in rows],
                type=pa.list_(pa.string()),
            ),
            'embed1': _build_fixed_list_column(
                [r.get('embed1') for r in rows],
                embedding_dim,
                pa.float16(),
            ),
        },
        schema=schema,
    )


# ──────────────────────────────────────────────────────────────────
# Per-input processing (append-based, fast)
# ──────────────────────────────────────────────────────────────────
def process_input(
    input_path: str,
    output_path: str,
    schema: pa.Schema,
    num_perm: int,
    embedding_dim: int,
    read_batch_size: int,
    write_batch_size: int,
    seen_ids: dict[str, int],
) -> tuple[dict[str, int], list[dict]]:
    """
    Stream rows from one input Lance dataset, recompute id + length,
    filter, dedup against seen_ids, and append new rows in bulk.

    Returns (stats_dict, list_of_update_rows).
    Update rows are rows whose id already exists in the output but
    that carry hash columns the existing row lacks — they are collected
    and flushed via a single merge_insert at the very end.
    """
    stats = {
        'rows_loaded': 0,
        'rows_dropped': 0,
        'rows_skipped': 0,
        'rows_inserted': 0,
    }

    input_ds = lance.dataset(input_path)
    total_rows = input_ds.count_rows()
    stats['rows_loaded'] = total_rows

    pbar = tqdm(
        total=total_rows,
        desc=f'  {os.path.basename(input_path)}',
        unit='row',
        dynamic_ncols=True,
    )

    append_buffer: list[dict] = []
    update_buffer: list[dict] = []

    for batch in input_ds.to_batches(columns=READ_COLS, batch_size=read_batch_size):
        pydict = batch.to_pydict()
        batch_n = batch.num_rows

        for i in range(batch_n):
            text = pydict['text'][i]
            if not text:
                text = ''

            # Length filter
            if len(text) <= 10 or len(text) > MAX_CHARS:
                stats['rows_dropped'] += 1
                pbar.update(1)
                continue

            # Normalize hash values (all-zero → None)
            minhash = _normalize(pydict['minhash'][i])
            lsh_bands = _normalize(pydict['lsh_bands'][i])
            embed1 = _normalize(pydict['embed1'][i])

            rid = compute_id(text)
            new_mask = _hash_bitmask_from_row(minhash, lsh_bands, embed1)

            if rid not in seen_ids:
                # New row → append buffer
                seen_ids[rid] = new_mask
                append_buffer.append(
                    {
                        'id': rid,
                        'text': text,
                        'source': pydict['source'][i],
                        'length': len(text),
                        'unique_chars': pydict['unique_chars'][i],
                        'words': pydict['words'][i],
                        'sentences': pydict['sentences'][i],
                        'minhash': minhash,
                        'lsh_bands': lsh_bands,
                        'embed1': embed1,
                    }
                )
            else:
                # Already seen — does this row fill a hash gap?
                existing_mask = seen_ids[rid]
                fills_gap = (new_mask & ~existing_mask) != 0

                if fills_gap:
                    seen_ids[rid] = existing_mask | new_mask
                    update_buffer.append(
                        {
                            'id': rid,
                            'text': text,
                            'source': pydict['source'][i],
                            'length': len(text),
                            'unique_chars': pydict['unique_chars'][i],
                            'words': pydict['words'][i],
                            'sentences': pydict['sentences'][i],
                            'minhash': minhash,
                            'lsh_bands': lsh_bands,
                            'embed1': embed1,
                        }
                    )
                else:
                    stats['rows_skipped'] += 1

            pbar.update(1)

            # Flush append buffer when full
            if len(append_buffer) >= write_batch_size:
                tbl = rows_to_table(append_buffer, schema, num_perm, embedding_dim)
                lance.write_dataset(tbl, output_path, mode='append')
                stats['rows_inserted'] += len(append_buffer)
                append_buffer = []

    # Flush remaining appends
    if append_buffer:
        tbl = rows_to_table(append_buffer, schema, num_perm, embedding_dim)
        lance.write_dataset(tbl, output_path, mode='append')
        stats['rows_inserted'] += len(append_buffer)

    pbar.close()
    return stats, update_buffer


# ──────────────────────────────────────────────────────────────────
# Flush update rows via merge_insert (single pass at the end)
# ──────────────────────────────────────────────────────────────────
def flush_updates(
    update_rows: list[dict],
    output_path: str,
    schema: pa.Schema,
    num_perm: int,
    embedding_dim: int,
    batch_size: int = 4096,
) -> int:
    """
    Upsert rows that fill hash gaps in existing output records.
    Uses merge_insert so the target row gets the new hash columns.
    Returns total number of rows updated.
    """
    if not update_rows:
        return 0

    total_updated = 0

    for start in range(0, len(update_rows), batch_size):
        chunk = update_rows[start : start + batch_size]
        tbl = rows_to_table(chunk, schema, num_perm, embedding_dim)

        ds = lance.dataset(output_path)
        result = ds.merge_insert('id').when_matched_update_all().when_not_matched_insert_all().execute(tbl)
        total_updated += result.get('num_updated_rows', 0)

    return total_updated


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Merge multiple Lance datasets into one, deduplicating by recomputed ID.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--inputs', nargs='+', required=True, help='Paths to input Lance datasets')
    p.add_argument('--output-path', required=True, help='Path for the merged output Lance dataset')
    p.add_argument(
        '--read-batch-size', type=int, default=DEFAULT_READ_BATCH_SIZE, help=f'Rows per read batch (default: {DEFAULT_READ_BATCH_SIZE})'
    )
    p.add_argument(
        '--write-batch-size',
        type=int,
        default=DEFAULT_WRITE_BATCH_SIZE,
        help=f'Rows per append batch (default: {DEFAULT_WRITE_BATCH_SIZE})',
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--overwrite', action='store_true', help='Delete existing output dataset before starting')
    mode.add_argument('--resume', action='store_true', help='Resume from existing output (rebuild seen IDs, skip indexed rows)')

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_args()

    # Validate inputs exist
    for path in args.inputs:
        if not os.path.isdir(path):
            raise SystemExit(f'Input dataset not found: {path}')

    print('Inputs:\n  ' + '\n  '.join(args.inputs))

    # ── Handle output path ────────────────────────────────────────
    seen_ids: dict[str, int] = {}

    if os.path.isdir(args.output_path):
        if args.resume:
            print(f'Resuming from existing output: {args.output_path}')
            seen_ids = rebuild_seen_ids(args.output_path)
            print(f'  Rebuilt {len(seen_ids):,} seen IDs')
        else:
            raise SystemExit(f'Output already exists: {args.output_path}\nUse --overwrite to replace it, or --resume to continue.')

    # Detect schema params from the first input
    first_ds = lance.dataset(args.inputs[0])
    validate_input_columns(first_ds, args.inputs[0])
    num_perm, embedding_dim = detect_schema_params(first_ds)
    print(f'Detected: num_perm={num_perm}, embedding_dim={embedding_dim}')

    # Build output schema
    schema = make_output_schema(num_perm, embedding_dim)
    print(f'\nOutput schema:\n{schema}')

    # Create empty output dataset if needed
    if not os.path.isdir(args.output_path):
        empty = pa.table(
            {f.name: pa.array([], type=f.type) for f in schema},
            schema=schema,
        )
        lance.write_dataset(empty, args.output_path)
        print(f'\nCreated output: {args.output_path}')
    else:
        print(f'\nOutput: {args.output_path}')

    # ── Process each input ────────────────────────────────────────
    grand = {
        'rows_loaded': 0,
        'rows_dropped': 0,
        'rows_skipped': 0,
        'rows_inserted': 0,
    }
    all_updates: list[dict] = []

    for input_path in args.inputs:
        print(f'\n{"=" * 60}')
        print(f'Input: {input_path}')
        print(f'{"=" * 60}')

        input_ds = lance.dataset(input_path)
        validate_input_columns(input_ds, input_path)

        file_stats, updates = process_input(
            input_path=input_path,
            output_path=args.output_path,
            schema=schema,
            num_perm=num_perm,
            embedding_dim=embedding_dim,
            read_batch_size=args.read_batch_size,
            write_batch_size=args.write_batch_size,
            seen_ids=seen_ids,
        )
        all_updates.extend(updates)

        print(
            f'  Loaded:     {file_stats["rows_loaded"]:>12,}\n'
            f'  Dropped:    {file_stats["rows_dropped"]:>12,}  (len ≤ 10 or > 32K)\n'
            f'  Skipped:    {file_stats["rows_skipped"]:>12,}  (duplicate, no new hashes)\n'
            f'  Inserted:   {file_stats["rows_inserted"]:>12,}\n'
            f'  Pending:    {len(updates):>12,}  (hash-gap updates queued)'
        )

        for k in grand:
            grand[k] += file_stats[k]

    # ── Flush hash-gap updates ────────────────────────────────────
    total_updated = 0
    if all_updates:
        print(f'\n{"─" * 60}')
        print(f'Flushing {len(all_updates):,} hash-gap updates via merge_insert…')
        total_updated = flush_updates(
            all_updates,
            args.output_path,
            schema,
            num_perm,
            embedding_dim,
        )
        print(f'  Updated: {total_updated:,}')

    # ── Final summary ─────────────────────────────────────────────
    output_ds = lance.dataset(args.output_path)
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    print(
        f'  Grand loaded:     {grand["rows_loaded"]:>12,}\n'
        f'  Grand dropped:    {grand["rows_dropped"]:>12,}\n'
        f'  Grand skipped:    {grand["rows_skipped"]:>12,}\n'
        f'  Grand inserted:   {grand["rows_inserted"]:>12,}\n'
        f'  Grand updated:    {total_updated:>12,}  (hash-gap fills)'
    )
    print(f'  Seen IDs:         {len(seen_ids):>12,}')
    print(f'  Output path:      {args.output_path}')
    print(f'  Total rows:       {output_ds.count_rows():>12,}')
    print(f'  Fragments:        {len(output_ds.get_fragments()):>12,}')
    print(f'\nOutput schema:\n{output_ds.schema}')


if __name__ == '__main__':
    main()
