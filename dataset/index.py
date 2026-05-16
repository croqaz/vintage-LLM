import argparse
import hashlib
import os
import shutil
import string
import tempfile
import timeit
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Literal

import lance
import pyarrow as pa
import xxhash
from datasets import Features, Sequence, Value, load_dataset
from datasketch import MinHash
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from .config import (
    DEFAULT_LSH_BANDS,
    DEFAULT_NGRAM_SIZE,
    DEFAULT_NUM_PERM,
    DEFAULT_TEXT_KEY,
    DEVICE,
    MAX_CHARS,
)
from .fields import char_entropy, compression_ratio, embed_texts, minhash_lsh_band_hashes, quality_score, split_into_sentences


# ─────────────────────────────────────────────────────────
# Lance schema + open/create
# ─────────────────────────────────────────────────────────
def make_arrow_schema(
    embedding_dim_1: int,
    embedding_dim_2: int,
    num_perm: int = DEFAULT_NUM_PERM,
) -> pa.Schema:
    """Build the Arrow schema for the Lance dataset."""
    fields = [
        pa.field('id', pa.string(), nullable=False),
        pa.field('text', pa.large_string(), nullable=False),
        pa.field('source', pa.string(), nullable=True),
        pa.field('length', pa.uint32(), nullable=False),
        pa.field('unique_chars', pa.uint32(), nullable=False),
        pa.field('words', pa.uint32(), nullable=False),
        pa.field('sentences', pa.uint32(), nullable=False),
        pa.field('quality_score', pa.float16(), nullable=False),
        pa.field('compression_ratio', pa.float16(), nullable=False),
        pa.field('char_entropy', pa.float16(), nullable=False),
        pa.field('minhash', pa.list_(pa.uint64(), num_perm), nullable=True),
        pa.field('lsh_bands', pa.list_(pa.string()), nullable=True),
        pa.field('embed1', pa.list_(pa.float16(), embedding_dim_1), nullable=True),
        pa.field('embed2', pa.list_(pa.float16(), embedding_dim_2), nullable=True),
    ]
    return pa.schema(fields)


# ─────────────────────────────────────────────────────────
# HF dataset features for compute phase
# ─────────────────────────────────────────────────────────
def compute_features(minhash_enabled: bool = True, minhash_perm: int = DEFAULT_NUM_PERM) -> Features:
    """
    Build HF dataset Features for the compute phase.
    The embeddings are computed in the final write loop.
    """
    feats: dict = {
        'text': Value('large_string'),
        'id': Value('string'),
        'length': Value('uint32'),
        'unique_chars': Value('uint32'),
        'words': Value('uint32'),
        'sentences': Value('uint32'),
        'quality_score': Value('float32'),
        'compression_ratio': Value('float32'),
        'char_entropy': Value('float32'),
    }
    if minhash_enabled:
        feats['minhash'] = Sequence(Value('uint64'), length=minhash_perm)
        feats['lsh_bands'] = Sequence(Value('string'))
    return Features(feats)


PREFILTER_FEATURES = Features(
    {
        'id': Value('string'),
        'text': Value('large_string'),
        'length': Value('uint32'),
        'unique_chars': Value('uint32'),
    }
)


# ─────────────────────────────────────────────────────────
# Row-accounting dataclass
# ─────────────────────────────────────────────────────────
@dataclass
class FileStats:
    path: str
    rows_loaded: int = 0
    rows_dropped: int = 0  # Due to length filter
    rows_duplicate: int = 0  # Already in DB
    rows_indexed: int = 0  # Successfully indexed

    def print_summary(self) -> None:
        print(
            f'  Loaded:     {self.rows_loaded:>12,}\n'
            f'  Dropped:    {self.rows_dropped:>12,}  (len ≤ 10 or > 32K)\n'
            f'  Duplicates: {self.rows_duplicate:>12,}  (already indexed)\n'
            f'  Indexed:    {self.rows_indexed:>12,}'
        )


def open_or_create(
    db_path: str,
    embedding_dim_1: int,
    embedding_dim_2: int,
    num_perm: int = DEFAULT_NUM_PERM,
) -> lance.LanceDataset:
    """Open existing Lance dataset or create a new empty one."""
    if os.path.isdir(db_path):
        return lance.dataset(db_path)

    schema = make_arrow_schema(embedding_dim_1, embedding_dim_2, num_perm)
    empty = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    ds = lance.write_dataset(empty, db_path)
    try:
        ds.create_scalar_index('id', index_type='BTREE', replace=True)
    except Exception as exc:
        print(f'  Warning: Failed to create initial index on "id": {exc}')
    return lance.dataset(db_path)


def ensure_indexes(ds: lance.LanceDataset, replace=False) -> None:
    """Create Vector indexes if they don't already exist."""
    # Ensure scalar indexes exist before ingestion
    scalar_indexes: list[tuple[str, Literal['BTREE', 'LABEL_LIST']]] = [
        ('id', 'BTREE'),
    ]
    for col, idx_type in scalar_indexes:
        try:
            ds.create_scalar_index(col, index_type=idx_type, replace=replace)
            print(f'  Created {idx_type} index on "{col}"')
        except Exception as exc:
            print(f'  Skipped {idx_type} index on "{col}": {exc}')

# ─────────────────────────────────────────────────────────
# Subcommand: index
# ─────────────────────────────────────────────────────────
def cmd_index(args: argparse.Namespace) -> None:
    input_files = sorted(glob(args.input_glob, recursive=True))
    if not input_files:
        raise SystemExit(f'No files matched: {args.input_glob}')
    print(f'Found {len(input_files)} input file(s)')

    # Resolve embedding dimensions and optionally load models
    model_1: SentenceTransformer | None = None
    model_2: SentenceTransformer | None = None
    embed_dim_1: int = -1
    embed_dim_2: int = -1

    # Open or create dataset
    ds = open_or_create(args.db_path, embed_dim_1, embed_dim_2, args.num_perm)

    # ── Model 1 ───────────────────────────────────────────
    if 1 in args.calc_embeds:
        print(f'Loading embedding model 1: {args.embed_model_1}')
        model_1 = SentenceTransformer(args.embed_model_1, device=DEVICE).half()
        embed_dim_1 = model_1.get_embedding_dimension()
        print(f'Embedding dimension 1: {embed_dim_1}')
    else:
        print(f'Loading embedding model 1 (dim only): {args.embed_model_1}')
        _model = SentenceTransformer(args.embed_model_1, device='cpu')
        embed_dim_1 = _model.get_embedding_dimension()
        del _model
        print(f'Embedding dimension 1: {embed_dim_1}  (disabled)')

    # ── Model 2 ───────────────────────────────────────────
    if 2 in args.calc_embeds:
        print(f'Loading embedding model 2: {args.embed_model_2}')
        model_2 = SentenceTransformer(args.embed_model_2, device=DEVICE).half()
        embed_dim_2 = model_2.get_embedding_dimension()
        print(f'Embedding dimension 2: {embed_dim_2}')
    else:
        print(f'Loading embedding model 2 (dim only): {args.embed_model_2}')
        _model = SentenceTransformer(args.embed_model_2, device='cpu')
        embed_dim_2 = _model.get_embedding_dimension()
        del _model
        print(f'Embedding dimension 2: {embed_dim_2}  (disabled)')

    if not args.calc_minhash:
        print('MinHash & LSH bands: disabled')

    print(f'Dataset path: {args.db_path}')

    grand_loaded = 0
    grand_dropped = 0
    grand_duplicate = 0
    grand_indexed = 0

    for input_path in input_files:
        print(f'\n{"=" * 60}')
        print(f'Processing: {input_path}')
        print(f'{"=" * 60}')

        fstats = process_file(
            input_path=input_path,
            model_1=model_1,
            model_2=model_2,
            embed_dim_1=embed_dim_1,
            embed_dim_2=embed_dim_2,
            args=args,
        )
        fstats.print_summary()

        grand_loaded += fstats.rows_loaded
        grand_dropped += fstats.rows_dropped
        grand_duplicate += fstats.rows_duplicate
        grand_indexed += fstats.rows_indexed

    ds = lance.dataset(args.db_path)

    if args.optimize:
        # NOOP: Lance optimize is currently broken!
        print('\nCompacting dataset…')
        ds.optimize.compact_files(target_rows_per_fragment=2_097_152)  # 2M rows
        # ds.optimize.optimize_indices()
        # ds = lance.dataset(args.db_path)
        print('  Done.')

        print('\nChecking indexes…')
        ensure_indexes(ds)
        print('  Done.')

    # ── Final summary ──────────────────────────────────────────────
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    print(
        f'  Grand loaded:     {grand_loaded:>12,}\n'
        f'  Grand dropped:    {grand_dropped:>12,}  (len ≤ 10 or > 32K)\n'
        f'  Grand duplicates: {grand_duplicate:>12,}  (already indexed)\n'
        f'  Grand indexed:    {grand_indexed:>12,}'
    )
    print(f'  Dataset:          {args.db_path}')
    print(f'  Total rows:       {ds.count_rows():>12,}')
    print(f'  Fragments:        {len(ds.get_fragments()):>12,}')
    if args.source:
        print(f'  Source label:     {args.source}')
    if 1 in args.calc_embeds:
        print(f'  Embed model 1:    {args.embed_model_1}')
    if 2 in args.calc_embeds:
        print(f'  Embed model 2:    {args.embed_model_2}')


# ─────────────────────────────────────────────────────────
# Per-file ingest pipeline
# ─────────────────────────────────────────────────────────
def process_file(
    input_path: str,
    model_1: SentenceTransformer | None,
    model_2: SentenceTransformer | None,
    embed_dim_1: int,
    embed_dim_2: int,
    args: argparse.Namespace,
) -> FileStats:
    stats = FileStats(path=input_path)

    # ── 1. Load ──────────────────────────────────────────
    ext = Path(input_path).suffix.lower()
    ext = 'parquet' if ext == '.parquet' else 'json'

    _tmp_cache: str | None = None
    if args.no_cache:
        _tmp_cache = tempfile.mkdtemp(prefix='index_lance_')
        effective_cache = _tmp_cache
    else:
        effective_cache = args.cache_dir

    try:
        hf_ds = load_dataset(ext, split='train', data_files=input_path, cache_dir=effective_cache)
        if args.text_key not in hf_ds.column_names:
            raise ValueError(f"Field '{args.text_key}' not found in {input_path}. Available columns: {hf_ds.column_names}")
        stats.rows_loaded = len(hf_ds)
        print(f'  Loaded:   {stats.rows_loaded:,} rows')

        # ── 2. Pre-filter: length guard + composite ID (cheap, CPU multiproc) ─
        hf_ds = hf_ds.map(
            prefilter_batch,
            batched=True,
            batch_size=args.compute_batch_size,
            writer_batch_size=args.compute_batch_size,
            num_proc=args.num_proc,
            remove_columns=hf_ds.column_names,
            features=PREFILTER_FEATURES,
            fn_kwargs={'text_key': args.text_key},
            desc='  Pre-filter (quick QA pass)',
        )
        stats.rows_dropped = stats.rows_loaded - len(hf_ds)
        print(f'  Pre-filtered: {len(hf_ds):,} rows' + (f' ({stats.rows_dropped:,} dropped by QA)' if stats.rows_dropped else ''))

        # ── 3. Deduplicate: within-file + Lance DB (single bulk read) ──────────
        t_start = timeit.default_timer()
        ds_lance = lance.dataset(args.db_path)
        prefiltered_count = len(hf_ds)
        # Load all existing IDs from Lance once (much faster than per-batch queries)
        if ds_lance.count_rows() > 0:
            existing_ids: set[str] = set(ds_lance.to_table(columns=['id']).column('id').to_pylist())
        else:
            existing_ids = set()
        del ds_lance

        # Deduplicate: remove within-file dupes and already-indexed IDs
        seen_ids: set[str] = set()
        keep_ids: set[str] = set()
        for bid in hf_ds['id']:
            if bid not in seen_ids and bid not in existing_ids:
                keep_ids.add(bid)
            seen_ids.add(bid)
        t_end = timeit.default_timer()
        print(f'  Calculated: {len(keep_ids):,} keep rows in {t_end - t_start:.2f} seconds')
        del existing_ids, seen_ids

        stats.rows_duplicate = prefiltered_count - len(keep_ids)
        print(
            f'  Unique new rows: {len(keep_ids):,}' + (f'  ({stats.rows_duplicate:,} duplicates skipped)' if stats.rows_duplicate else '')
        )

        if not keep_ids:
            print('  Nothing new to index.')
            return stats

        # ── 4. Filter dataset to new rows only ────────────────────────────────
        hf_ds = hf_ds.filter(
            lambda batch: [bid in keep_ids for bid in batch['id']],
            batched=True,
            batch_size=args.compute_batch_size,
            desc='  Filtering duplicates',
        )

        # ── 5. Compute stats + hashes + MinHash (CPU multiproc) ───────────────
        features = compute_features(args.calc_minhash, args.num_perm)
        hf_ds = hf_ds.map(
            compute_batch,
            batched=True,
            batch_size=args.compute_batch_size,
            writer_batch_size=args.compute_batch_size,
            num_proc=args.num_proc,
            remove_columns=hf_ds.column_names,
            features=features,
            fn_kwargs={
                'text_key': args.text_key,
                'minhash_enabled': args.calc_minhash,
                'num_perm': args.num_perm,
                'ngram_size': args.ngram_size,
                'lsh_bands': args.lsh_bands,
            },
            desc='  Computing stats' + (' & hashes' if args.calc_minhash else ''),
        )
        print(f'  After map: {len(hf_ds):,} rows')

        arrow_schema = make_arrow_schema(embed_dim_1, embed_dim_2, args.num_perm)
        desc = '  Index + Embeds' if args.calc_embeds else '  Writing data'
        pbar = tqdm(total=len(hf_ds), desc=desc, unit='doc', dynamic_ncols=True)

        # ── 6. Embed → build Arrow table → append ─────────────────────────────
        for batch in hf_ds.iter(batch_size=args.write_batch_size):
            texts = batch['text']
            doc_ids = batch['id']
            n = len(texts)

            columns = {
                'id': pa.array(doc_ids, type=pa.string()),
                'text': pa.array(texts, type=pa.large_string()),
                'source': pa.array([args.source] * n if args.source else [None] * n, type=pa.string()),
                'length': pa.array([batch['length'][i] for i in range(n)], type=pa.uint64()),
                'unique_chars': pa.array([batch['unique_chars'][i] for i in range(n)], type=pa.uint32()),
                'words': pa.array([batch['words'][i] for i in range(n)], type=pa.uint32()),
                'sentences': pa.array([batch['sentences'][i] for i in range(n)], type=pa.uint32()),
                'quality_score': pa.array([batch['quality_score'][i] for i in range(n)], type=pa.float16()),
                'compression_ratio': pa.array([batch['compression_ratio'][i] for i in range(n)], type=pa.float16()),
                'char_entropy': pa.array([batch['char_entropy'][i] for i in range(n)], type=pa.float16()),
            }

            if not args.calc_minhash:
                columns['minhash'] = pa.array([None] * n, type=pa.list_(pa.uint64(), args.num_perm))
                columns['lsh_bands'] = pa.array([None] * n, type=pa.list_(pa.string()))
            else:
                columns['minhash'] = pa.FixedSizeListArray.from_arrays(
                    pa.array([v for i in range(n) for v in batch['minhash'][i]], type=pa.uint64()),
                    list_size=args.num_perm,
                )
                columns['lsh_bands'] = pa.array([batch['lsh_bands'][i] for i in range(n)], type=pa.list_(pa.string()))

            if 1 not in args.calc_embeds:
                columns['embed1'] = pa.array([None] * n, type=pa.list_(pa.float16(), embed_dim_1))
            else:
                embed_f16 = embed_texts(model_1, texts, batch_size=args.embed_batch_size)
                columns['embed1'] = pa.FixedSizeListArray.from_arrays(
                    pa.array([v for e in embed_f16 for v in e], type=pa.float16()),
                    list_size=embed_dim_1,
                )

            if 2 not in args.calc_embeds:
                columns['embed2'] = pa.array([None] * n, type=pa.list_(pa.float16(), embed_dim_2))
            else:
                embed_f16 = embed_texts(model_2, texts, batch_size=args.embed_batch_size)
                columns['embed2'] = pa.FixedSizeListArray.from_arrays(
                    pa.array([v for e in embed_f16 for v in e], type=pa.float16()),
                    list_size=embed_dim_2,
                )

            tbl = pa.table(columns, schema=arrow_schema)

            lance.write_dataset(tbl, args.db_path, mode='append')
            stats.rows_indexed += n

            pbar.update(n)

        pbar.close()
    finally:
        if _tmp_cache is not None:
            shutil.rmtree(_tmp_cache, ignore_errors=True)

    return stats


# ──────────────────────────────────────────────────────────────────
# Pre-filter batch (cheap pass: length guard + composite ID only)
# Pickle-safe: no closures, all config via fn_kwargs.
# ──────────────────────────────────────────────────────────────────
def prefilter_batch(
    batch: dict[str, list[str]],
    text_key: str = DEFAULT_TEXT_KEY,
) -> dict[str, list]:
    """
    HF .map(batched=True) worker for a cheap pre-filter pass.
    Drops texts that are too short or too long, then calc the
    composite xxh64+blake2b ID.
    Returns {id, text, length, unique_chars} for surviving rows.
    """
    out: dict[str, list] = {'id': [], 'text': [], 'length': [], 'unique_chars': []}
    for text in batch[text_key]:
        # Filter by length of text
        length = len(text) if text else 0
        if length <= 10 or length > MAX_CHARS:
            continue
        # Filter by unique character count
        # (cheap proxy for "binary" or "garbage" text)
        unique_chars = len(set(text))
        if unique_chars <= 4:
            continue

        enc_text = b' '.join(text.encode('utf-8').split())
        xxh64_hex = xxhash.xxh3_64_hexdigest(enc_text)
        blake2b_hex = hashlib.blake2b(enc_text, digest_size=16).hexdigest()
        # ID length = 16 + 32 = 48 characters
        out['id'].append(f'{xxh64_hex}{blake2b_hex}')
        out['text'].append(text)
        out['length'].append(length)
        out['unique_chars'].append(unique_chars)
    return out


# ──────────────────────────────────────────────────────────────────
# Compute batch
# Pickle-safe: no closures, all config via fn_kwargs.
# ──────────────────────────────────────────────────────────────────
def compute_batch(
    batch: dict[str, list[str]],
    text_key: str = DEFAULT_TEXT_KEY,
    num_perm: int = DEFAULT_NUM_PERM,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
    lsh_bands: int = DEFAULT_LSH_BANDS,
    minhash_enabled: bool = False,
) -> dict[str, list]:
    """
    HF .map(batched=True) worker — computes per-document stats, hashes,
    and composite ID.  Text is passed through for storage + embedding.
    Rows too small or too large are silently dropped (sanity guard).
    """
    out: dict[str, list] = {
        'text': [],
        'id': [],
        'length': [],
        'unique_chars': [],
        'words': [],
        'sentences': [],
        'quality_score': [],
        'compression_ratio': [],
        'char_entropy': [],
    }
    if minhash_enabled:
        out['minhash'] = []
        out['lsh_bands'] = []

    for i, text in enumerate(batch[text_key]):
        enc_text = b' '.join(text.encode('utf-8').split())

        out['text'].append(text)
        out['id'].append(batch['id'][i])
        out['length'].append(batch['length'][i])
        out['unique_chars'].append(batch['unique_chars'][i])

        tokens = [t for t in text.split() if t not in string.punctuation]
        out['words'].append(len(tokens))

        sentences = split_into_sentences(text)
        out['sentences'].append(len(sentences))

        out['quality_score'].append(quality_score(text))
        out['compression_ratio'].append(compression_ratio(text))
        out['char_entropy'].append(char_entropy(text))

        if minhash_enabled:
            mh = MinHash(num_perm=num_perm)
            n = ngram_size
            if len(tokens) >= n:
                for i in range(len(tokens) - n + 1):
                    mh.update(' '.join(tokens[i : i + n]).encode('utf-8'))
            else:
                mh.update(enc_text)
            mh_vals = mh.hashvalues.tolist()
            out['minhash'].append(mh_vals)

            band_hashes = minhash_lsh_band_hashes(mh_vals, lsh_bands)
            out['lsh_bands'].append([f'{band_id}_{bh}' for band_id, bh in enumerate(band_hashes)])

    return out


# ─────────────────────────────────────────────────────────
# Subcommand: del
# ─────────────────────────────────────────────────────────
def cmd_del(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.db_path):
        raise SystemExit(f'Dataset not found: {args.db_path}')

    ds = lance.dataset(args.db_path)
    total_before = ds.count_rows()

    # Count rows matching the filter
    matched = ds.count_rows(filter=args.query)
    remaining = total_before - matched

    print(f'Filter:     {args.query}')
    print(f'Total rows: {total_before:,}')
    print(f'To delete:  {matched:,}')
    print(f'Remaining:  {remaining:,}')

    if matched == 0:
        print('\nNo rows match the filter. Nothing to delete.')
        return

    answer = input(f'\nDelete {matched:,} row(s)? [y/N] ').strip().lower()
    if answer not in ('y', 'yes'):
        print('Aborted.')
        return

    ds.delete(args.query)

    # Reopen to verify
    ds = lance.dataset(args.db_path)
    total_after = ds.count_rows()
    deleted = total_before - total_after

    print(f'\n{"─" * 40}')
    print(f'Before:    {total_before:,}')
    print(f'Deleted:   {deleted:,}')
    print(f'Remaining: {total_after:,}')
    print(f'{"─" * 40}')
