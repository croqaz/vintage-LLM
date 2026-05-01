"""
Index and query text documents using Lance (Open Lakehouse Format).

Subcommands
───────────
  index          Ingest JSONL/Parquet files → Lance dataset (stats, hashes, MinHash, embedding).
  stats          Print dataset schema, row count, indexes, fragment info.
  get            Retrieve a single record by composite ID or raw text.
  find-similar   Find duplicate or similar texts:
                   exact     — identical text (hash lookup)
                   similar   — near-duplicates via MinHash LSH + Jaccard verification
                   semantic  — semantically similar texts via IVF_PQ embedding search

Embeddings are computed with sentence-transformers.
Before embedding, each text is trimmed: keep the first 32 000 and the last 32 000
characters only (passthrough if shorter than 64 000).

Examples
────────
  # Index files
  python index_lance.py index --input-glob "raw/*.jsonl" --source "British Library"

  # Re-index (existing docs are skipped by default)
  python index_lance.py index --input-glob "raw/*.jsonl"

  # Dataset stats
  python index_lance.py stats

  # Look up a record
  python index_lance.py get --text "The quick brown fox"

  # Near-duplicate search
  python index_lance.py find-similar --mode similar --text "some text" --threshold 0.4

  # Semantic search
  python index_lance.py find-similar --mode semantic --text "some text" --limit 10
"""

import argparse
import glob as globmod
import hashlib
import os
import re
import shutil
import string
import struct
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lance
import numpy as np
import pyarrow as pa
import torch
import xxhash
from datasets import Features, Sequence, Value, load_dataset
from datasketch import MinHash
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

warnings.filterwarnings('ignore', message='lance is not fork-safe', category=UserWarning)

# ──────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────
DEFAULT_TEXT_KEY = 'text'
DEFAULT_DB_PATH = './lance-data/text_index.lance'
DEFAULT_EMBEDDING_MODEL = 'nomic-ai/nomic-embed-text-v1.5'
DEFAULT_NUM_PERM = 128
DEFAULT_NGRAM_SIZE = 5
DEFAULT_LSH_BANDS = 16
DEFAULT_COMPUTE_BATCH_SIZE = 512
DEFAULT_WRITE_BATCH_SIZE = 256
DEFAULT_EMBED_BATCH_SIZE = 32
MAX_CHARS = 32_000
SEMANTIC_PREFIX = 'clustering:'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ──────────────────────────────────────────────────────────────────
# HF dataset features for compute phase (text kept for storage + embedding)
# ──────────────────────────────────────────────────────────────────
COMPUTE_FEATURES = Features(
    {
        'text': Value('large_string'),
        'id': Value('string'),
        'length': Value('uint32'),
        'unique_chars': Value('uint32'),
        'words': Value('uint32'),
        'sentences': Value('uint32'),
        'minhash': Sequence(Value('uint64')),
        'lsh_bands': Sequence(Value('string')),
    }
)

PREFILTER_FEATURES = Features(
    {
        'id': Value('string'),
        'text': Value('large_string'),
    }
)

SENTENCE_BOUNDARY_REGEX = re.compile(r'((?:[.!?]["\']?)\s+(?=[A-Z"\'])|(?:[\n\r]{2,}\s*(?=[a-zA-Z"\'])))')


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using a regex."""
    parts = SENTENCE_BOUNDARY_REGEX.split(text)
    return [parts[i] + (parts[i + 1] if i + 1 < len(parts) else '') for i in range(0, len(parts), 2)]


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
    Drops texts that are too short or too long, then computes only the
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
) -> dict[str, list]:
    """
    HF .map(batched=True) worker — computes per-document stats, hashes,
    and composite ID.  Text is passed through for storage + embedding.
    Rows too small or too large are silently dropped (sanity guard).
    """
    out: dict[str, list] = {c: [] for c in COMPUTE_FEATURES}

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


# ──────────────────────────────────────────────────────────────────
# Hashing / MinHash helpers
# ──────────────────────────────────────────────────────────────────
def compute_minhash(text: str, num_perm: int = DEFAULT_NUM_PERM, ngram_size: int = DEFAULT_NGRAM_SIZE) -> MinHash:
    """Compute a datasketch MinHash for a single text using word n-grams."""
    tokens = [t for t in text.split() if t not in string.punctuation]
    mh = MinHash(num_perm=num_perm)
    if len(tokens) >= ngram_size:
        for i in range(len(tokens) - ngram_size + 1):
            mh.update(' '.join(tokens[i : i + ngram_size]).encode('utf-8'))
    else:
        mh.update(text.encode('utf-8'))
    return mh


def minhash_lsh_band_hashes(sig: list[int], num_bands: int) -> list[str]:
    """
    Split a MinHash signature into LSH bands and return one xxh64 hex digest per band.

    sig:       list of num_perm uint64 values (MinHash.hashvalues)
    num_bands: number of bands (num_perm must be divisible by num_bands)

    Returns num_bands hex strings.  Documents sharing any band digest are
    near-duplicate candidates.
    """
    rows = len(sig) // num_bands
    return [xxhash.xxh64_hexdigest(struct.pack(f'<{rows}Q', *sig[b * rows : (b + 1) * rows])) for b in range(num_bands)]


# ──────────────────────────────────────────────────────────────────
# Embedding helpers
# ──────────────────────────────────────────────────────────────────
def trim_for_embedding(text: str) -> str:
    """Keep the first MAX_CHARS and the last MAX_CHARS characters."""
    if len(text) <= MAX_CHARS * 2:
        return text
    return text[:MAX_CHARS] + text[-MAX_CHARS:]


def _adaptive_encode_batch_size(texts: list[str], base_batch_size: int) -> int:
    """
    Scale batch_size down for batches with long texts.
    GPU memory is proportional to batch_size × max_seq_len (due to padding).
    We scale inversely with the longest text, using ~4 chars/token as a heuristic.
    """
    if not texts:
        return base_batch_size
    max_chars = max(len(t) for t in texts)
    approx_tokens = min(max_chars // 4, 8192)
    if approx_tokens == 0:
        return base_batch_size
    bs = (base_batch_size * 512) // approx_tokens
    return max(1, min(bs, base_batch_size))


def embed_texts(model, texts: list[str], batch_size: int = DEFAULT_EMBED_BATCH_SIZE) -> list[list[float]]:
    """
    Embed texts with sentence-transformers, with text trimming and OOM retry.
    Sorts texts by length to minimise padding waste, then restores original order.
    On CUDA OOM the batch size is halved and retried down to 1.
    """
    trimmed = [SEMANTIC_PREFIX + trim_for_embedding(t) for t in texts]

    # Sort by length so sub-batches group similar-length texts (less padding waste)
    order = sorted(range(len(trimmed)), key=lambda i: len(trimmed[i]))
    sorted_texts = [trimmed[i] for i in order]

    enc_bs = _adaptive_encode_batch_size(sorted_texts, batch_size)
    current_bs = enc_bs
    while True:
        try:
            raw_sorted = model.encode(sorted_texts, batch_size=current_bs, convert_to_tensor=True, normalize_embeddings=False)
            break
        except torch.cuda.OutOfMemoryError:
            if current_bs <= 1:
                raise
            new_bs = max(1, current_bs // 2)
            print(f'  [OOM] encode batch_size {current_bs} → {new_bs}, retrying…')
            torch.cuda.empty_cache()
            current_bs = new_bs

    # Restore original order
    emb_list = [[0.0]] * len(trimmed)
    for sorted_i, orig_i in enumerate(order):
        emb = raw_sorted[sorted_i]
        # emb.dtype=torch.float16 ; emb.shape=torch.Size([embed_dim])
        emb_list[orig_i] = (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy().tolist()
    return emb_list


# ──────────────────────────────────────────────────────────────────
# Lance schema + open/create
# ──────────────────────────────────────────────────────────────────
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
            pa.field('minhash', pa.list_(pa.uint64(), num_perm), nullable=False),
            pa.field('lsh_bands', pa.list_(pa.string()), nullable=False),
            pa.field('embed1', pa.list_(pa.float16(), embedding_dim), nullable=False),
        ]
    )


def open_or_create(db_path: str, embedding_dim: int, num_perm: int = DEFAULT_NUM_PERM) -> lance.LanceDataset:
    """Open existing Lance dataset or create a new empty one."""
    if os.path.isdir(db_path):
        return lance.dataset(db_path)

    schema = make_arrow_schema(embedding_dim, num_perm)
    empty = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    ds = lance.write_dataset(empty, db_path)

    print('\nCreating indexes…')
    # Ensure scalar indexes exist before ingestion
    scalar_indexes: list[tuple[str, Literal['BTREE', 'LABEL_LIST']]] = [
        ('id', 'BTREE'),
        ('length', 'BTREE'),
        ('lsh_bands', 'LABEL_LIST'),
    ]
    for col, idx_type in scalar_indexes:
        try:
            ds.create_scalar_index(col, index_type=idx_type, replace=False)
            print(f'  Created {idx_type} index on "{col}"')
        except Exception as exc:
            print(f'  Skipped {idx_type} index on "{col}": {exc}')

    # Detect embedding size from schema
    dim = schema.field('embed1').type.list_size
    num_sub = max(1, dim // 16)
    # Ensure SIMD alignment: (dim / num_sub) % 8 == 0
    while num_sub > 1 and (dim // num_sub) % 8 != 0:
        num_sub -= 1
    # Ensure vector index for sentence-transformer embeddings exists
    try:
        ds.create_index(
            'embed1',
            index_type='IVF_PQ',
            metric='cosine',
            num_partitions=256,
            num_sub_vectors=num_sub,
            replace=False,
        )
        # Created IVF_PQ index on "embed1" (sub_vectors=48)
        print(f'  Created IVF_PQ index on "embed1" (sub_vectors={num_sub})')
    except Exception as exc:
        print(f'  Skipped IVF_PQ index on "embed1": {exc}')
    print('  Done.')

    return ds


def ensure_indexes(ds: lance.LanceDataset) -> None:
    """Create Vector indexes if they don't already exist."""

    # Detect embedding size from schema
    dim = ds.schema.field('embed1').type.list_size
    num_sub = max(1, dim // 16)
    # Ensure SIMD alignment: (dim / num_sub) % 8 == 0
    while num_sub > 1 and (dim // num_sub) % 8 != 0:
        num_sub -= 1
    # Ensure vector index for sentence-transformer embeddings exists
    try:
        ds.create_index(
            'embed1',
            index_type='IVF_PQ',
            metric='cosine',
            num_partitions=256,
            num_sub_vectors=num_sub,
            replace=False,
        )
        # Created IVF_PQ index on "embed1" (sub_vectors=48)
        print(f'  Created IVF_PQ index on "embed1" (sub_vectors={num_sub})')
    except Exception as exc:
        print(f'  Skipped IVF_PQ index on "embed1": {exc}')


# ──────────────────────────────────────────────────────────────────
# Row-accounting dataclass
# ──────────────────────────────────────────────────────────────────
@dataclass
class FileStats:
    path: str
    rows_loaded: int = 0
    rows_dropped: int = 0  # Due to length filter
    rows_duplicate: int = 0  # Already in DB
    rows_indexed: int = 0

    def print_summary(self) -> None:
        print(
            f'  Loaded:      {self.rows_loaded:>12,}\n'
            f'  Dropped:     {self.rows_dropped:>12,}  (len ≤ 10 or > 32K)\n'
            f'  Duplicates:  {self.rows_duplicate:>12,}  (already indexed)\n'
            f'  Indexed:     {self.rows_indexed:>12,}'
        )


# ──────────────────────────────────────────────────────────────────
# Infer dataset format
# ──────────────────────────────────────────────────────────────────
def _infer_dataset_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == '.parquet':
        return 'parquet'
    return 'json'


# ──────────────────────────────────────────────────────────────────
# Per-file ingest pipeline
# ──────────────────────────────────────────────────────────────────
def process_file(
    input_path: str,
    model: SentenceTransformer,
    args: argparse.Namespace,
) -> FileStats:
    stats = FileStats(path=input_path)

    # ── 1. Load ────────────────────────────────────────────────────
    fmt = _infer_dataset_format(input_path)
    _tmp_cache: str | None = None
    if args.no_cache:
        _tmp_cache = tempfile.mkdtemp(prefix='index_lance_')
        effective_cache = _tmp_cache
    else:
        effective_cache = args.cache_dir

    embedding_dim = model.get_embedding_dimension()
    arrow_schema = make_arrow_schema(embedding_dim, args.num_perm)

    try:
        hf_ds = load_dataset(fmt, split='train', data_files=input_path, cache_dir=effective_cache)
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
        hf_ds = hf_ds.map(
            compute_batch,
            batched=True,
            batch_size=args.compute_batch_size,
            writer_batch_size=args.compute_batch_size,
            num_proc=args.num_proc,
            remove_columns=hf_ds.column_names,
            features=COMPUTE_FEATURES,
            fn_kwargs={'text_key': args.text_key, 'num_perm': args.num_perm, 'ngram_size': args.ngram_size, 'lsh_bands': args.lsh_bands},
            desc='  Computing stats & hashes',
        )
        print(f'  After map: {len(hf_ds):,} rows')

        # ── 6. Embed → build Arrow table → append ─────────────────────────────
        pbar = tqdm(total=len(hf_ds), desc='  Writing index', unit='doc', dynamic_ncols=True)
        for batch in hf_ds.iter(batch_size=args.write_batch_size):
            texts = batch['text']
            doc_ids = batch['id']
            n = len(texts)

            # Run embedding in a separate loop to isolate OOM risk and ensure it doesn't
            # affect the filter and compute batch
            embed_f16 = embed_texts(model, texts, batch_size=args.embed_batch_size)

            tbl = pa.table(
                {
                    'id': pa.array(doc_ids, type=pa.string()),
                    'text': pa.array(texts, type=pa.large_string()),
                    'source': pa.array([args.source] * n if args.source else [None] * n, type=pa.string()),
                    'length': pa.array([batch['length'][i] for i in range(n)], type=pa.uint64()),
                    'unique_chars': pa.array([batch['unique_chars'][i] for i in range(n)], type=pa.uint32()),
                    'words': pa.array([batch['words'][i] for i in range(n)], type=pa.uint32()),
                    'sentences': pa.array([batch['sentences'][i] for i in range(n)], type=pa.uint32()),
                    'minhash': pa.FixedSizeListArray.from_arrays(
                        pa.array([v for i in range(n) for v in batch['minhash'][i]], type=pa.uint64()),
                        list_size=args.num_perm,
                    ),
                    'lsh_bands': pa.array([batch['lsh_bands'][i] for i in range(n)], type=pa.list_(pa.string())),
                    'embed1': pa.FixedSizeListArray.from_arrays(
                        pa.array([v for e in embed_f16 for v in e], type=pa.float16()),
                        list_size=embedding_dim,
                    ),
                },
                schema=arrow_schema,
            )

            lance.write_dataset(tbl, args.db_path, mode='append')
            stats.rows_indexed += n

            pbar.update(n)

        pbar.close()
    finally:
        if _tmp_cache is not None:
            shutil.rmtree(_tmp_cache, ignore_errors=True)

    return stats


# ══════════════════════════════════════════════════════════════════
# Subcommand: index
# ══════════════════════════════════════════════════════════════════
def cmd_index(args: argparse.Namespace) -> None:
    files = sorted(globmod.glob(args.input_glob, recursive=True))
    if not files:
        raise SystemExit(f'No files matched: {args.input_glob}')
    print(f'Found {len(files)} input file(s)')

    # Load embedding model in F16 for smaller index size and faster embedding on GPU
    print(f'Loading embedding model: {args.embedding_model}')
    model = SentenceTransformer(args.embedding_model, device=DEVICE).half()
    embedding_dim = model.get_embedding_dimension()
    print(f'Embedding dimension: {embedding_dim}')

    # Open or create dataset
    open_or_create(args.db_path, embedding_dim, args.num_perm)
    print(f'Dataset path: {args.db_path}')

    grand_loaded = 0
    grand_dropped = 0
    grand_duplicate = 0
    grand_indexed = 0

    for input_path in files:
        print(f'\n{"=" * 60}')
        print(f'Processing: {input_path}')
        print(f'{"=" * 60}')

        fstats = process_file(
            input_path=input_path,
            model=model,
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
        # ds.optimize.compact_files(target_rows_per_fragment=8_388_608)  # 8M rows
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
    print(f'  Embedding model:  {args.embedding_model}')


# ══════════════════════════════════════════════════════════════════
# Subcommand: stats
# ══════════════════════════════════════════════════════════════════
def cmd_stats(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.db_path):
        raise SystemExit(f'Dataset not found: {args.db_path}')
    ds = lance.dataset(args.db_path)
    print(f'Path: {args.db_path}')
    print(f'\nSchema:\n{ds.schema}')
    print(f'\nTotal rows:  {ds.count_rows():,}')
    print(f'Fragments:   {len(ds.get_fragments()):,}')
    print(f'Version:     {ds.version}')
    indices = ds.list_indices()
    if indices:
        print(f'\nIndexes ({len(indices)}):')
        for idx in indices:
            print(f'  {idx}')
    else:
        print('\nNo indexes.')


# ══════════════════════════════════════════════════════════════════
# Pretty-print a row dict
# ══════════════════════════════════════════════════════════════════
def _print_doc_row(row: dict, indent: int = 2) -> None:
    pad = ' ' * indent
    for name, val in row.items():
        if name == 'minhash' and isinstance(val, (list, tuple, np.ndarray)) and len(val) > 8:
            display = f'[{val[0]}, {val[1]}, {val[2]}, {val[3]}, ... {val[-2]}, {val[-1]}]  ({len(val)} values)'
        elif name == 'lsh_bands' and isinstance(val, (list, tuple)) and len(val) > 4:
            display = f'[{val[0]}, {val[1]}, ... {val[-1]}]  ({len(val)} tags)'
        elif name == 'embed1' and isinstance(val, (list, tuple, np.ndarray)) and len(val) > 6:
            v = list(val)
            display = f'[{v[0]:.5f}, {v[1]:.5f}, {v[2]:.5f}, ... {v[-1]:.5f}]  ({len(v)} floats)'
        elif name == 'text' and isinstance(val, str) and len(val) > 200:
            display = repr(val[:100] + '[...]' + val[-100:])
        else:
            display = repr(val) if isinstance(val, str) else str(val)
        print(f'{pad}{name}: {display}')


# ══════════════════════════════════════════════════════════════════
# ID resolution helper
# ══════════════════════════════════════════════════════════════════
def _resolve_id(args: argparse.Namespace) -> str | None:
    """Return the composite xxh64-blake2b id from --id or by hashing --text."""
    if args.id:
        if len(args.id) != 48 or not re.match(r'^[0-9a-f]{16}[0-9a-f]{32}$', args.id):
            raise SystemExit(f'--id must be in xxh64-blake2b format, got: {args.id!r}')
        return args.id
    if getattr(args, 'text', None):
        enc_text = b' '.join(args.text.encode('utf-8').split())
        xxh64_hex = xxhash.xxh3_64_hexdigest(enc_text)
        blake2b_hex = hashlib.blake2b(enc_text, digest_size=16).hexdigest()
        # ID length = 16 + 32 = 48 characters
        return f'{xxh64_hex}{blake2b_hex}'
    return None


# ══════════════════════════════════════════════════════════════════
# Subcommand: get
# ══════════════════════════════════════════════════════════════════
def cmd_get(args: argparse.Namespace) -> None:
    doc_id = _resolve_id(args)
    if doc_id is None:
        raise SystemExit('Provide --id xxh64-blake2b or --text "some text"')

    if not os.path.isdir(args.db_path):
        raise SystemExit(f'Dataset not found: {args.db_path}')
    ds = lance.dataset(args.db_path)

    print(f'Looking up: {doc_id}')
    tbl = ds.to_table(filter=f"id = '{doc_id}'", limit=1)
    if tbl.num_rows == 0:
        print('Not found.')
        return

    row = tbl.to_pylist()[0]
    print(f'{"─" * 60}')
    _print_doc_row(row)
    print(f'{"─" * 60}')


# ══════════════════════════════════════════════════════════════════
# Subcommand: find-similar
# ══════════════════════════════════════════════════════════════════
def cmd_find_similar(args: argparse.Namespace) -> None:
    mode = args.mode
    if mode == 'exact':
        _find_exact(args)
    elif mode == 'similar':
        _find_similar(args)
    elif mode == 'semantic':
        _find_semantic(args)
    else:
        raise SystemExit(f'Unknown mode: {mode!r}')


# ── exact ─────────────────────────────────────────────────────────
def _find_exact(args: argparse.Namespace) -> None:
    doc_id = _resolve_id(args)
    if doc_id is None:
        raise SystemExit('Provide --id or --text for exact mode!')

    ds = lance.dataset(args.db_path)
    print(f'Exact lookup: {doc_id}')
    tbl = ds.to_table(filter=f"id = '{doc_id}'", limit=1)
    if tbl.num_rows > 0:
        print('Found (exact match):')
        print(f'{"─" * 60}')
        _print_doc_row(tbl.to_pylist()[0])
        print(f'{"─" * 60}')
    else:
        print('Not found.')


# ── similar ───────────────────────────────────────────────────────
def _find_similar(args: argparse.Namespace) -> None:
    if not args.text and not args.id:
        raise SystemExit('Provide --text or --id for similar mode!')

    ds = lance.dataset(args.db_path)

    # Build query MinHash
    if args.text:
        query_mh = compute_minhash(args.text, args.num_perm, args.ngram_size)
    else:
        tbl = ds.to_table(filter=f"id = '{args.id}'", columns=['minhash'], limit=1)
        if tbl.num_rows == 0:
            raise SystemExit(f'Record not found: {args.id}')
        mh_vals = tbl.to_pylist()[0]['minhash']
        query_mh = MinHash(num_perm=args.num_perm)
        query_mh.hashvalues = np.array(mh_vals, dtype=np.uint64)

    print(f'MinHash near-duplicate search  (threshold≥{args.threshold}, limit={args.limit})')
    band_hashes = minhash_lsh_band_hashes(query_mh.hashvalues.tolist(), args.lsh_bands)
    band_tags = [f'{band_id}_{bh}' for band_id, bh in enumerate(band_hashes)]

    # Build filter using array_has_any for LABEL_LIST index
    quoted_tags = ', '.join(f"'{t}'" for t in band_tags)
    filt = f'array_has_any(lsh_bands, [{quoted_tags}])'
    if args.text_like:
        filt += f" AND text LIKE '{args.text_like}'"

    candidates = ds.to_table(
        filter=filt,
        columns=['id', 'minhash', 'text', 'length', 'source'],
        limit=args.limit * 16,
    )

    print(f'  LSH candidates: {candidates.num_rows:,}')

    matches: list[tuple[float, str, str, int, str]] = []
    for row in candidates.to_pylist():
        mh_vals = row.get('minhash')
        if not mh_vals:
            continue
        cand_mh = MinHash(num_perm=args.num_perm)
        cand_mh.hashvalues = np.array(mh_vals, dtype=np.uint64)
        jaccard = query_mh.jaccard(cand_mh)
        if jaccard >= args.threshold:
            text = row.get('text', '')
            if len(text) > 100:
                snippet = text[:50] + '[...]' + text[-50:]
            else:
                snippet = text
            length = row.get('length', 0)
            source = row.get('source', '') or ''
            matches.append((jaccard, row['id'], snippet, length, source))

    matches.sort(key=lambda x: -x[0])
    matches = matches[: args.limit]
    if not matches:
        print('  No matches above threshold.')
        return

    print(f'  Matches: {len(matches)} (showing up to {args.limit})\n')
    print(f'  {"Jaccard":>8}  {"Length":>8}  {"Source":<16}  ID / Snippet')
    print(f'  {"─" * 8}  {"─" * 8}  {"─" * 16}  {"─" * 40}')
    for jaccard, doc_id, snippet, length, source in matches:
        src = (source or '')[:16]
        print(f'  {jaccard:>8.4f}  {length:>8,}  {src:<16}  [{doc_id}]  {repr(snippet)}')


# ── semantic ──────────────────────────────────────────────────────
def _find_semantic(args: argparse.Namespace) -> None:
    if not args.text and not args.id:
        raise SystemExit('Provide --text or --id for semantic mode!')

    ds = lance.dataset(args.db_path)

    print(f'Semantic search  (limit={args.limit})')
    print(f'Loading embedding model: {args.embedding_model}')
    model = SentenceTransformer(args.embedding_model, device=DEVICE).half()

    if args.text:
        query_vec = embed_texts(model, [args.text])[0]
    else:
        tbl = ds.to_table(filter=f"id = '{args.id}'", columns=['embed1'], limit=1)
        if tbl.num_rows == 0:
            raise SystemExit(f'Record not found: {args.id}')
        query_vec = tbl.to_pylist()[0]['embed1']

    print(f'Query embedded ({len(query_vec)}-dim). Searching…')

    nearest_params = {
        'column': 'embed1',
        'q': np.array(query_vec, dtype=np.float32),
        'k': args.limit * 4,
        'metric': 'cosine',
        'nprobes': 32,
        'refine_factor': 10,
    }

    scan_kwargs: dict = dict(
        columns=['id', 'text', 'length', 'source', '_distance'],
        nearest=nearest_params,
        limit=args.limit * 4,
    )
    if args.text_like:
        scan_kwargs['filter'] = f"text LIKE '{args.text_like}'"

    results = ds.to_table(**scan_kwargs)

    if results.num_rows == 0:
        print('  No results.')
        return

    matches: list[tuple[float, str, str, int, str]] = []
    for row in results.to_pylist():
        # Lance returns _distance for nearest queries;
        # for cosine distance: similarity = 1 - distance
        score = 1.0 - (row.get('_distance', 1.0))
        if score < args.threshold:
            continue
        text = row.get('text', '')
        if len(text) > 100:
            snippet = text[:50] + '[...]' + text[-50:]
        else:
            snippet = text
        length = row.get('length', 0)
        source = row.get('source', '') or ''
        matches.append((score, row['id'], snippet, length, source))

    matches.sort(key=lambda x: -x[0])
    matches = matches[: args.limit]
    if not matches:
        print('  No matches above threshold.')
        return

    print(f'  Matches: {len(matches)} (showing up to {args.limit})\n')
    print(f'\n  {"Score":>8}  {"Length":>8}  {"Source":<16}  ID / Snippet')
    print(f'  {"─" * 8}  {"─" * 8}  {"─" * 16}  {"─" * 40}')
    for score, doc_id, snippet, length, source in matches:
        src = (source or '')[:16]
        length = length or 0
        score = score or 0.0
        print(f'  {score:>8.4f}  {length:>8,}  {src:<16}  [{doc_id}]  {repr(snippet)}')


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    root = argparse.ArgumentParser(
        description='Index and query text documents using Lance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Global args ────────────────────────────────────────────────
    root.add_argument('--db-path', default=DEFAULT_DB_PATH, help='Path to the Lance dataset directory')
    root.add_argument('--num-perm', type=int, default=DEFAULT_NUM_PERM, help='MinHash permutations')
    root.add_argument('--ngram-size', type=int, default=DEFAULT_NGRAM_SIZE, help='Word n-gram size for MinHash')
    root.add_argument('--lsh-bands', type=int, default=DEFAULT_LSH_BANDS, help='MinHash LSH band count')

    sub = root.add_subparsers(dest='subcommand', required=True)

    # ── index ──────────────────────────────────────────────────────
    p_index = sub.add_parser('index', help='Ingest JSONL/Parquet files into the Lance dataset')
    p_index.add_argument('--input-glob', required=True, help="Glob pattern for input files (e.g. 'raw/*.jsonl')")
    p_index.add_argument('--source', default=None, help='Provenance label added to every record')
    p_index.add_argument('--text-key', default=DEFAULT_TEXT_KEY, help='JSON field containing document text')
    p_index.add_argument('--embedding-model', default=DEFAULT_EMBEDDING_MODEL, help='SentenceTransformer model name')
    p_index.add_argument('--num-proc', type=int, default=os.cpu_count() or 4, help='Parallel workers for compute phase')
    p_index.add_argument('--compute-batch-size', type=int, default=DEFAULT_COMPUTE_BATCH_SIZE, help='HF datasets.map() batch size')
    p_index.add_argument('--write-batch-size', type=int, default=DEFAULT_WRITE_BATCH_SIZE, help='Iteration batch size for write phase')
    p_index.add_argument('--embed-batch-size', type=int, default=DEFAULT_EMBED_BATCH_SIZE, help='Batch size for model.encode()')
    p_index.add_argument('--overwrite', action='store_true', help='Overwrite existing dataset (if not set, identical records are skipped)')
    p_index.add_argument('--optimize', action='store_true', help='Compact files and optimize indexes after ingestion')
    cache_group = p_index.add_mutually_exclusive_group()
    cache_group.add_argument('--cache-dir', default=None, help='HF datasets cache directory')
    cache_group.add_argument('--no-cache', action='store_true', help='Use a temporary cache dir per file')

    # ── stats ──────────────────────────────────────────────────────
    p_stats = sub.add_parser('stats', help='Print dataset schema and statistics')

    # ── get ────────────────────────────────────────────────────────
    p_get = sub.add_parser('get', help='Retrieve a single record by composite ID or raw text')
    id_group = p_get.add_mutually_exclusive_group(required=True)
    id_group.add_argument('--id', help='Composite ID in xxh64-blake2b format')
    id_group.add_argument('--text', help='Raw text to hash and look up')

    # ── find-similar ───────────────────────────────────────────────
    p_find = sub.add_parser('find-similar', help='Find identical, near-duplicate, or semantically similar texts')
    p_find.add_argument(
        '--mode',
        choices=['exact', 'similar', 'semantic'],
        required=True,
        help='exact: hash lookup. similar: MinHash LSH + Jaccard. semantic: embedding ANN.',
    )
    input_group = p_find.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', help='Input text')
    input_group.add_argument('--id', help='Composite ID — fetch stored data from the index')
    p_find.add_argument('--text-like', default=None, help="Filter text with LIKE pattern (e.g. 'Word%%')")
    p_find.add_argument('--threshold', type=float, default=0.25, help='Minimum similarity threshold')
    p_find.add_argument('--limit', type=int, default=20, help='Maximum number of results')
    p_find.add_argument('--embedding-model', default=DEFAULT_EMBEDDING_MODEL, help='SentenceTransformer model name (semantic mode)')

    return root.parse_args()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_args()
    if args.num_perm % args.lsh_bands != 0:
        raise SystemExit(f'--lsh-bands {args.lsh_bands} must evenly divide --num-perm {args.num_perm}')
    dispatch = {
        'index': cmd_index,
        'stats': cmd_stats,
        'get': cmd_get,
        'find-similar': cmd_find_similar,
    }
    dispatch[args.subcommand](args)


if __name__ == '__main__':
    main()
