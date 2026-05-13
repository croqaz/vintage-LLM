import argparse
import hashlib
import os
import shutil
import string
import tempfile
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
def make_arrow_schema(embedding_dim: int, num_perm: int = DEFAULT_NUM_PERM) -> pa.Schema:
    """Build the Arrow schema for the Lance dataset."""
    return pa.schema(
        [
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
            pa.field('embed1', pa.list_(pa.float16(), embedding_dim), nullable=True),
        ]
    )


# ─────────────────────────────────────────────────────────
# HF dataset features for compute phase
# ─────────────────────────────────────────────────────────
def compute_features(minhash_enabled: bool = True, minhash_perm: int = DEFAULT_NUM_PERM) -> Features:
    """Build HF dataset Features for the compute phase."""
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


def open_or_create(db_path: str, embedding_dim: int, num_perm: int = DEFAULT_NUM_PERM) -> lance.LanceDataset:
    """Open existing Lance dataset or create a new empty one."""
    if os.path.isdir(db_path):
        return lance.dataset(db_path)

    schema = make_arrow_schema(embedding_dim, num_perm)
    empty = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    ds = lance.write_dataset(empty, db_path)
    try:
        ds.create_scalar_index('id', index_type='BTREE', replace=False)
    except Exception as exc:
        print(f'  Warning: Failed to create initial index on "id": {exc}')
    return ds


def ensure_indexes(ds: lance.LanceDataset, replace=False) -> None:
    """Create Vector indexes if they don't already exist."""
    # Ensure scalar indexes exist before ingestion
    scalar_indexes: list[tuple[str, Literal['BTREE', 'LABEL_LIST']]] = [
        ('id', 'BTREE'),
        ('length', 'BTREE'),
        ('lsh_bands', 'LABEL_LIST'),
    ]
    for col, idx_type in scalar_indexes:
        try:
            ds.create_scalar_index(col, index_type=idx_type, replace=replace)
            print(f'  Created {idx_type} index on "{col}"')
        except Exception as exc:
            print(f'  Skipped {idx_type} index on "{col}": {exc}')

    # # Detect embedding size from schema
    # dim = ds.schema.field('embed1').type.list_size
    # num_sub = max(1, dim // 16)
    # # Ensure SIMD alignment: (dim / num_sub) % 8 == 0
    # while num_sub > 1 and (dim // num_sub) % 8 != 0:
    #     num_sub -= 1
    # # Ensure vector index for sentence-transformer embeddings exists
    # try:
    #     ds.create_index(
    #         'embed1',
    #         index_type='IVF_PQ',
    #         metric='cosine',
    #         num_partitions=256,
    #         num_sub_vectors=num_sub,
    #         replace=replace,
    #     )
    #     # Created IVF_PQ index on "embed1" (sub_vectors=48)
    #     print(f'  Created IVF_PQ index on "embed1" (sub_vectors={num_sub})')
    # except Exception as exc:
    #     print(f'  Skipped IVF_PQ index on "embed1": {exc}')


# ─────────────────────────────────────────────────────────
# Subcommand: index
# ─────────────────────────────────────────────────────────
def cmd_index(args: argparse.Namespace) -> None:
    input_files = sorted(glob(args.input_glob, recursive=True))
    if not input_files:
        raise SystemExit(f'No files matched: {args.input_glob}')
    print(f'Found {len(input_files)} input file(s)')

    # Resolve embedding dimension and optionally load the model
    model: SentenceTransformer | None = None
    embedding_dim: int = -1
    if not args.calc_embeds:
        if os.path.isdir(args.db_path):
            embedding_dim = lance.dataset(args.db_path).schema.field('embed1').type.list_size
        else:
            print(f'Loading embedding model (dim only): {args.embedding_model}')
            _model = SentenceTransformer(args.embedding_model, device='cpu')
            embedding_dim = _model.get_embedding_dimension()
            del _model
        print(f'Embedding dimension: {embedding_dim}  (embedding disabled)')
    else:
        print(f'Loading embedding model: {args.embedding_model}')
        model = SentenceTransformer(args.embedding_model, device=DEVICE).half()
        embedding_dim = model.get_embedding_dimension()
        print(f'Embedding dimension: {embedding_dim}')

    if not args.calc_minhash:
        print('MinHash & LSH bands: disabled')

    # Open or create dataset
    open_or_create(args.db_path, embedding_dim, args.num_perm)
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
            model=model,
            embedding_dim=embedding_dim,
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
        # ds.optimize.compact_files(target_rows_per_fragment=2_097_152)  # 2M rows
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
    if args.calc_embeds:
        print(f'  Embedding model:  {args.embedding_model}')


# ─────────────────────────────────────────────────────────
# Per-file ingest pipeline
# ─────────────────────────────────────────────────────────
def process_file(
    input_path: str,
    model: SentenceTransformer | None,
    embedding_dim: int,
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
            desc='  Pre-filtering (length + IDs)',
        )
        stats.rows_dropped = stats.rows_loaded - len(hf_ds)
        print(f'  Pre-filtered: {len(hf_ds):,} rows' + (f'  ({stats.rows_dropped:,} dropped by length)' if stats.rows_dropped else ''))

        # ── 3. Deduplicate: within-file + Lance DB (batch ID lookup) ──────────
        ds_lance = lance.dataset(args.db_path)
        db_has_rows = ds_lance.count_rows() > 0
        seen_ids: set[str] = set()
        keep_ids: set[str] = set()
        prefiltered_count = len(hf_ds)

        for dedup_batch in hf_ds.iter(batch_size=args.compute_batch_size):
            batch_ids: list[str] = dedup_batch['id']
            novel_ids = [bid for bid in batch_ids if bid not in seen_ids]
            seen_ids.update(batch_ids)
            if db_has_rows and novel_ids:
                quoted = ', '.join(f"'{bid}'" for bid in novel_ids)
                existing = set(ds_lance.to_table(filter=f'id IN ({quoted})', columns=['id']).column('id').to_pylist())
                keep_ids.update(bid for bid in novel_ids if bid not in existing)
            else:
                keep_ids.update(novel_ids)

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

        arrow_schema = make_arrow_schema(embedding_dim, args.num_perm)
        pbar = tqdm(total=len(hf_ds), desc='  Writing index', unit='doc', dynamic_ncols=True)

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

            if not args.calc_embeds:
                columns['embed1'] = pa.array([None] * n, type=pa.list_(pa.float16(), embedding_dim))
            else:
                embed_f16 = embed_texts(model, texts, batch_size=args.embed_batch_size)
                columns['embed1'] = pa.FixedSizeListArray.from_arrays(
                    pa.array([v for e in embed_f16 for v in e], type=pa.float16()),
                    list_size=embedding_dim,
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
    HF .map(batched=True) worker — cheap pre-filter pass.
    Drops texts that are too short or too long, then calc only the
    composite xxh64+blake2b ID (no MinHash / LSH / stats).
    Returns {id, text} for surviving rows only.
    """
    out: dict[str, list] = {'id': [], 'text': []}
    for text in batch[text_key]:
        if not text:
            text = ''
        if len(text) <= 10 or len(text) > MAX_CHARS:
            continue
        enc_text = b' '.join(text.encode('utf-8').split())
        xxh64_hex = xxhash.xxh3_64_hexdigest(enc_text)
        blake2b_hex = hashlib.blake2b(enc_text, digest_size=16).hexdigest()
        # ID length = 16 + 32 = 48 characters
        out['id'].append(f'{xxh64_hex}{blake2b_hex}')
        out['text'].append(text)
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
        out['length'].append(len(text))
        out['unique_chars'].append(len(set(text)))

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
