"""
Index and query text documents using Zvec (local embedded vector database).

Subcommands
───────────
  index          Ingest JSONL/Parquet files → Zvec collection (stats, hashes, MinHash, embedding).
  stats          Print collection schema and statistics.
  get            Retrieve a single record by composite ID or raw text.
  find-similar   Find duplicate or similar texts:
                   exact     — identical text (hash lookup)
                   similar   — near-duplicates via MinHash LSH + Jaccard verification
                   semantic  — semantically similar texts via HNSW embedding search

Embeddings are computed with sentence-transformers.
Before embedding, each text is trimmed: keep the first 32 000 and the last 32 000
characters only (passthrough if shorter than 64 000).

Examples
────────
  # Index files
  python index_zvec.py index --input-glob "raw/*.jsonl" --source "British Library"

  # Re-index (existing docs are skipped by default)
  python index_zvec.py index --input-glob "raw/*.jsonl"

  # Collection stats
  python index_zvec.py stats

  # Look up a record
  python index_zvec.py get --text "The quick brown fox"

  # Near-duplicate search
  python index_zvec.py find-similar --mode similar --text "some text" --threshold 0.4

  # Semantic search
  python index_zvec.py find-similar --mode semantic --text "some text" --limit 10
"""

import argparse
import glob
import hashlib
import os
import re
import shutil
import string
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xxhash
import zvec
from datasets import Features, Sequence, Value, load_dataset
from datasketch import MinHash
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ──────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────
DEFAULT_TEXT_KEY = 'text'
DEFAULT_DB_PATH = './zvec-data/text_index'
DEFAULT_EMBEDDING_MODEL = 'nomic-ai/nomic-embed-text-v1.5'
DEFAULT_NUM_PERM = 128
DEFAULT_NGRAM_SIZE = 5
DEFAULT_LSH_BANDS = 16
DEFAULT_COMPUTE_BATCH_SIZE = 1_000
DEFAULT_WRITE_BATCH_SIZE = 512
DEFAULT_EMBED_BATCH_SIZE = 32
TRIM_CHARS = 32_000  # keep first N and last N chars before embedding
SEMANTIC_PREFIX = 'clustering:'

# ──────────────────────────────────────────────────────────────────
# HF dataset features for compute phase (text kept temporarily for embedding)
# ──────────────────────────────────────────────────────────────────
COMPUTE_FEATURES = Features(
    {
        'text': Value('large_string'),
        'xxh64': Value('string'),
        'blake2b': Value('string'),
        'length': Value('uint32'),
        'unique_chars': Value('uint32'),
        'words': Value('uint32'),
        'sentences': Value('uint32'),
        'snippet': Value('string'),
        'minhash': Sequence(Value('uint64')),
    }
)

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
    HF .map(batched=True) worker — computes per-document stats, hashes, and MinHash.
    Text is passed through so the embedding pass can access it after .map() completes.
    Rows with len(text) <= 10 or > 10_000_000 are silently dropped (sanity guard).
    """
    out: dict[str, list] = {c: [] for c in COMPUTE_FEATURES}

    for text in batch[text_key]:
        if text is None:
            text = ''
        if len(text) <= 10 or len(text) > 10_000_000:
            continue

        tokens = [t for t in text.split() if t not in string.punctuation]
        enc_text = b' '.join(text.encode('utf-8').split())

        out['text'].append(text)
        out['xxh64'].append(xxhash.xxh64_hexdigest(enc_text))
        out['blake2b'].append(hashlib.blake2b(enc_text, digest_size=16).hexdigest())
        out['length'].append(len(text))
        out['unique_chars'].append(len(set(text)))
        out['words'].append(len(tokens))

        segs = split_into_sentences(text) if enc_text else []
        out['sentences'].append(max(1, len(segs)))

        if len(text) <= 100:
            out['snippet'].append(text)
        else:
            out['snippet'].append(text[:50] + '[...]' + text[-50:])

        mh = MinHash(num_perm=num_perm)
        n = ngram_size
        if len(tokens) >= n:
            for i in range(len(tokens) - n + 1):
                mh.update(' '.join(tokens[i : i + n]).encode('utf-8'))
        else:
            mh.update(enc_text)
        out['minhash'].append(mh.hashvalues.tolist())

    return out


# ──────────────────────────────────────────────────────────────────
# Hashing / MinHash helpers
# ──────────────────────────────────────────────────────────────────
def compute_text_hashes(text: str) -> tuple[str, str]:
    """Return (xxh64_hex, blake2b_hex) for a text string."""
    encoded = text.encode('utf-8')
    return xxhash.xxh64_hexdigest(encoded), hashlib.blake2b(encoded, digest_size=16).hexdigest()


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

    Returns num_bands hex strings. Documents sharing any band digest are near-duplicate
    candidates (probability of sharing rises steeply around the configured Jaccard threshold).
    """
    rows = len(sig) // num_bands
    return [xxhash.xxh64_hexdigest(struct.pack(f'<{rows}Q', *sig[b * rows : (b + 1) * rows])) for b in range(num_bands)]


# ──────────────────────────────────────────────────────────────────
# Embedding helpers
# ──────────────────────────────────────────────────────────────────
def trim_for_embedding(text: str) -> str:
    """Keep the first TRIM_CHARS and the last TRIM_CHARS characters."""
    if len(text) <= TRIM_CHARS * 2:
        return text
    return text[:TRIM_CHARS] + text[-TRIM_CHARS:]


def get_embedding_dim(model) -> int:
    """Return the embedding dimension from the model."""
    return model.get_embedding_dimension()


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
# Zvec schema + open/create
# ──────────────────────────────────────────────────────────────────
def make_schema(embedding_dim: int) -> zvec.CollectionSchema:
    """Build the Zvec collection schema."""
    return zvec.CollectionSchema(
        name='text_index',
        fields=[
            zvec.FieldSchema(name='source', data_type=zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema(
                name='length',
                data_type=zvec.DataType.UINT64,
                index_param=zvec.InvertIndexParam(enable_range_optimization=True),
            ),
            zvec.FieldSchema(name='unique_chars', data_type=zvec.DataType.UINT32),
            zvec.FieldSchema(name='words', data_type=zvec.DataType.UINT32),
            zvec.FieldSchema(name='sentences', data_type=zvec.DataType.UINT32),
            zvec.FieldSchema(name='snippet', data_type=zvec.DataType.STRING),
            zvec.FieldSchema(name='minhash', data_type=zvec.DataType.ARRAY_UINT64),
            zvec.FieldSchema(
                name='lsh_bands',
                data_type=zvec.DataType.ARRAY_STRING,
                index_param=zvec.InvertIndexParam(),
            ),
        ],
        vectors=[
            zvec.VectorSchema(
                name='embedding',
                data_type=zvec.DataType.VECTOR_FP16,
                dimension=embedding_dim,
                index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
            ),
        ],
    )


def open_or_create(db_path: str, embedding_dim: int) -> zvec.model.collection.Collection:
    """Open existing collection or create a new one."""
    if os.path.isdir(db_path):
        return zvec.open(db_path)
    schema = make_schema(embedding_dim)
    os.makedirs(Path(db_path).parent, exist_ok=True)
    return zvec.create_and_open(path=db_path, schema=schema)


# ──────────────────────────────────────────────────────────────────
# Row-accounting dataclass
# ──────────────────────────────────────────────────────────────────
@dataclass
class FileStats:
    path: str
    rows_loaded: int = 0
    rows_dropped: int = 0
    rows_skipped: int = 0  # already in DB
    rows_dup_insert: int = 0  # insert() failed (duplicate ID)
    rows_indexed: int = 0

    def print_summary(self) -> None:
        print(
            f'  Loaded:      {self.rows_loaded:>12,}\n'
            f'  Dropped:     {self.rows_dropped:>12,}  (len ≤ 10 or > 10 M)\n'
            f'  Skipped:     {self.rows_skipped:>12,}  (already in DB)\n'
            f'  Dup insert:  {self.rows_dup_insert:>12,}  (insert exception)\n'
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
    collection: zvec.model.collection.Collection,
    source: str | None,
    model,
    text_key: str,
    num_perm: int,
    ngram_size: int,
    lsh_bands: int,
    num_proc: int,
    compute_batch_size: int,
    write_batch_size: int,
    embed_batch_size: int,
    skip_existing: bool,
    cache_dir: str | None = None,
    no_cache: bool = False,
) -> FileStats:
    stats = FileStats(path=input_path)

    # ── 1. Load ────────────────────────────────────────────────────
    fmt = _infer_dataset_format(input_path)
    _tmp_cache: str | None = None
    if no_cache:
        _tmp_cache = tempfile.mkdtemp(prefix='index_zvec_')
        effective_cache = _tmp_cache
    else:
        effective_cache = cache_dir

    try:
        ds = load_dataset(fmt, split='train', data_files=input_path, cache_dir=effective_cache)
        if text_key not in ds.column_names:
            raise ValueError(f"Field '{text_key}' not found in {input_path}. Available columns: {ds.column_names}")
        stats.rows_loaded = len(ds)
        print(f'  Loaded:   {stats.rows_loaded:,} rows')

        # ── 2. Compute stats + hashes + MinHash (CPU multiproc) ───────
        ds = ds.map(
            compute_batch,
            batched=True,
            batch_size=compute_batch_size,
            writer_batch_size=compute_batch_size,
            num_proc=num_proc,
            remove_columns=ds.column_names,
            features=COMPUTE_FEATURES,
            fn_kwargs={'text_key': text_key, 'num_perm': num_perm, 'ngram_size': ngram_size},
            desc='  Computing stats & hashes',
        )
        stats.rows_dropped = stats.rows_loaded - len(ds)
        print(f'  After compute: {len(ds):,} rows' + (f'  ({stats.rows_dropped:,} dropped)' if stats.rows_dropped else ''))

        # ── 3. Batch: skip-existing → embed → insert ──────────────────
        pbar = tqdm(total=len(ds), desc='  Writing index', unit='doc', dynamic_ncols=True)
        for batch in ds.iter(batch_size=write_batch_size):
            texts = batch['text']
            n = len(texts)
            skipped = 0

            # Build doc IDs
            doc_ids = [f'{batch["xxh64"][i]}_{batch["blake2b"][i]}' for i in range(n)]

            # Compute LSH band tags
            batch_lsh_bands = []
            for i in range(n):
                band_hashes = minhash_lsh_band_hashes(batch['minhash'][i], lsh_bands)
                tags = [f'{band_id}_{bh}' for band_id, bh in enumerate(band_hashes)]
                batch_lsh_bands.append(tags)

            # Skip-existing check via fetch
            if skip_existing:
                existing = collection.fetch(ids=doc_ids)
                keep_indices = [i for i in range(n) if doc_ids[i] not in existing]
                skipped = n - len(keep_indices)
                stats.rows_skipped += skipped
                if not keep_indices:
                    pbar.update(n)
                    continue
                # Filter batch to only new docs
                texts = [texts[i] for i in keep_indices]
                doc_ids = [doc_ids[i] for i in keep_indices]
                batch_lsh_bands = [batch_lsh_bands[i] for i in keep_indices]
                # Re-index batch fields
                filtered_batch = {}
                for key in batch:
                    filtered_batch[key] = [batch[key][i] for i in keep_indices]
                batch = filtered_batch
                n = len(texts)

            # Embed
            embeddings = embed_texts(model, texts, batch_size=embed_batch_size)

            # Build zvec.Doc list
            docs = []
            for i in range(n):
                fields = {
                    'length': batch['length'][i],
                    'unique_chars': batch['unique_chars'][i],
                    'words': batch['words'][i],
                    'sentences': batch['sentences'][i],
                    'snippet': batch['snippet'][i],
                    'minhash': batch['minhash'][i],
                    'lsh_bands': batch_lsh_bands[i],
                }
                if source is not None:
                    fields['source'] = source
                docs.append(
                    zvec.Doc(
                        id=doc_ids[i],
                        vectors={'embedding': embeddings[i]},
                        fields=fields,
                    )
                )

            results = collection.insert(docs)
            if isinstance(results, list):
                for j, st in enumerate(results):
                    if st.ok():
                        stats.rows_indexed += 1
                    else:
                        stats.rows_dup_insert += 1
                        print(f'    Insert failed [{doc_ids[j][:24]}…]: {st.message()}')
            else:
                if results.ok():
                    stats.rows_indexed += 1
                else:
                    stats.rows_dup_insert += 1
                    print(f'    Insert failed: {results.message()}')

            pbar.update(n + skipped)

        pbar.close()
    finally:
        if _tmp_cache is not None:
            shutil.rmtree(_tmp_cache, ignore_errors=True)

    return stats


# ══════════════════════════════════════════════════════════════════
# Subcommand: index
# ══════════════════════════════════════════════════════════════════
def cmd_index(args: argparse.Namespace) -> None:
    files = sorted(glob.glob(args.input_glob, recursive=True))
    if not files:
        raise SystemExit(f'No files matched: {args.input_glob}')
    print(f'Found {len(files)} input file(s)')

    # Load embedding model
    print(f'Loading embedding model: {args.embedding_model}')
    model = SentenceTransformer(args.embedding_model, device=DEVICE).half()
    embedding_dim = get_embedding_dim(model)
    print(f'Embedding dimension: {embedding_dim}')

    # Open or create collection
    collection = open_or_create(args.db_path, embedding_dim)
    print(f'Collection path: {collection.path}')

    grand_loaded = 0
    grand_dropped = 0
    grand_skipped = 0
    grand_dup = 0
    grand_indexed = 0

    for input_path in files:
        print(f'\n{"=" * 60}')
        print(f'Processing: {input_path}')
        print(f'{"=" * 60}')

        fstats = process_file(
            input_path=input_path,
            collection=collection,
            source=args.source,
            model=model,
            text_key=args.text_key,
            num_perm=args.num_perm,
            ngram_size=args.ngram_size,
            lsh_bands=args.lsh_bands,
            num_proc=args.num_proc,
            compute_batch_size=args.compute_batch_size,
            write_batch_size=args.write_batch_size,
            embed_batch_size=args.embed_batch_size,
            skip_existing=not args.no_skip_existing,
            cache_dir=args.cache_dir,
            no_cache=args.no_cache,
        )
        fstats.print_summary()

        grand_loaded += fstats.rows_loaded
        grand_dropped += fstats.rows_dropped
        grand_skipped += fstats.rows_skipped
        grand_dup += fstats.rows_dup_insert
        grand_indexed += fstats.rows_indexed

    if args.optimize:
        print('\nOptimizing collection…')
        collection.optimize()
        print('  Done.')

    # ── Final summary ──────────────────────────────────────────────
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    print(
        f'  Grand loaded:     {grand_loaded:>12,}\n'
        f'  Grand dropped:    {grand_dropped:>12,}  (len ≤ 10 or > 10 M)\n'
        f'  Grand skipped:    {grand_skipped:>12,}  (already in DB)\n'
        f'  Grand dup insert: {grand_dup:>12,}\n'
        f'  Grand indexed:    {grand_indexed:>12,}'
    )
    print(f'  Collection:       {args.db_path}')
    print(f'  Stats:            {collection.stats}')
    if args.source:
        print(f'  Source label:     {args.source}')
    print(f'  Embedding model:  {args.embedding_model}')


# ══════════════════════════════════════════════════════════════════
# Subcommand: stats
# ══════════════════════════════════════════════════════════════════
def cmd_stats(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.db_path):
        raise SystemExit(f'Collection not found: {args.db_path}')
    collection = zvec.open(args.db_path)
    print(f'Path: {collection.path}')
    print(f'\nSchema:\n{collection.schema}')
    print('\nFields:')
    for f in collection.schema.fields:
        print(f'  {f.name:20s}  {f.data_type}  nullable={f.nullable}')
    print('\nVectors:')
    for v in collection.schema.vectors:
        print(f'  {v.name:20s}  {v.data_type}  dim={v.dimension}')
    print(f'\nStats:\n{collection.stats}')


# ══════════════════════════════════════════════════════════════════
# Pretty-print a Doc
# ══════════════════════════════════════════════════════════════════
def _print_doc(doc: zvec.Doc, indent: int = 2) -> None:
    pad = ' ' * indent
    print(f'{pad}id: {doc.id}')
    if doc.score is not None:
        print(f'{pad}score: {doc.score:.5f}')
    for name in doc.field_names():
        val = doc.field(name)
        if name == 'minhash' and isinstance(val, (list, tuple)) and len(val) > 8:
            display = f'[{val[0]}, {val[1]}, {val[2]}, {val[3]}, ... {val[-2]}, {val[-1]}]  ({len(val)} values)'
        elif name == 'lsh_bands' and isinstance(val, (list, tuple)) and len(val) > 4:
            display = f'[{val[0]}, {val[1]}, ... {val[-1]}]  ({len(val)} tags)'
        else:
            display = str(val)
        print(f'{pad}{name}: {display}')
    for name in doc.vector_names():
        vec = doc.vector(name)
        if isinstance(vec, (list, tuple)) and len(vec) > 6:
            display = f'[{vec[0]:.5f}, {vec[1]:.5f}, {vec[2]:.5f}, ... {vec[-1]:.5f}]  ({len(vec)} floats)'
        else:
            display = str(vec)
        print(f'{pad}{name}: {display}')


# ══════════════════════════════════════════════════════════════════
# Subcommand: get
# ══════════════════════════════════════════════════════════════════
def cmd_get(args: argparse.Namespace) -> None:
    doc_id = _resolve_id(args)
    if doc_id is None:
        raise SystemExit('Provide --id xxh64_blake2b or --text "some text"')

    if not os.path.isdir(args.db_path):
        raise SystemExit(f'Collection not found: {args.db_path}')
    collection = zvec.open(args.db_path)

    print(f'Looking up: {doc_id}')
    result = collection.fetch(ids=doc_id)
    if doc_id not in result:
        print('Not found.')
        return

    print(f'{"─" * 60}')
    _print_doc(result[doc_id])
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

    collection = zvec.open(args.db_path)
    print(f'Exact lookup: {doc_id}')
    result = collection.fetch(ids=doc_id)
    if doc_id in result:
        print('Found (exact match):')
        print(f'{"─" * 60}')
        _print_doc(result[doc_id])
        print(f'{"─" * 60}')
    else:
        print('Not found.')


# ── similar ───────────────────────────────────────────────────────
def _find_similar(args: argparse.Namespace) -> None:
    if not args.text and not args.id:
        raise SystemExit('Provide --text or --id for similar mode!')

    collection = zvec.open(args.db_path)

    # Build query MinHash + LSH band tags
    if args.text:
        query_mh = compute_minhash(args.text, args.num_perm, args.ngram_size)
    else:
        # Fetch stored minhash from the collection
        result = collection.fetch(ids=args.id)
        if args.id not in result:
            raise SystemExit(f'Record not found: {args.id}')
        mh_vals = result[args.id].field('minhash')
        if mh_vals is None:
            raise SystemExit(f'Record has no minhash field: {args.id}')
        query_mh = MinHash(num_perm=args.num_perm)
        query_mh.hashvalues = np.array(mh_vals, dtype=np.uint64)

    print(f'MinHash near-duplicate search  (threshold≥{args.threshold}, limit={args.limit})')
    band_hashes = minhash_lsh_band_hashes(query_mh.hashvalues.tolist(), args.lsh_bands)
    band_tags = [f'{band_id}_{bh}' for band_id, bh in enumerate(band_hashes)]

    # Build filter: lsh_bands CONTAIN_ANY(tag1, tag2, ...)
    quoted_tags = ', '.join(f"'{t}'" for t in band_tags)

    # Use a filter-only query to get LSH candidates
    # topk must be large enough to capture all candidates (we post-filter by Jaccard)
    filt = f'lsh_bands CONTAIN_ANY({quoted_tags})'
    if args.text_like:
        filt += f" AND snippet LIKE '{args.text_like}'"
    candidates = collection.query(
        filter=filt,
        topk=args.limit * 16,  # fetch more candidates than needed
        output_fields=['minhash', 'snippet', 'length', 'source'],
    )

    print(f'  LSH candidates: {len(candidates):,}')

    matches: list[tuple[float, str, str, int, str]] = []
    for doc in candidates:
        mh_vals = doc.field('minhash')
        if not mh_vals:
            continue
        cand_mh = MinHash(num_perm=args.num_perm)
        cand_mh.hashvalues = np.array(mh_vals, dtype=np.uint64)
        jaccard = query_mh.jaccard(cand_mh)
        if jaccard >= args.threshold:
            snippet = doc.field('snippet') or ''
            length = doc.field('length') or 0
            source = doc.field('source') or ''
            matches.append((jaccard, doc.id, snippet, length, source))

    matches.sort(key=lambda x: -x[0])
    matches = matches[: args.limit]
    if not matches:
        print('  No matches above threshold.')
        return

    print(f'  Matches: {len(matches)} (showing up to {args.limit})\n')
    print(f'  {"Jaccard":>8}  {"Length":>8}  {"Source":<20}  ID / Snippet')
    print(f'  {"─" * 8}  {"─" * 8}  {"─" * 20}  {"─" * 40}')
    for jaccard, doc_id, snippet, length, source in matches:
        src = (source or '')[:20]
        print(f'  {jaccard:>8.4f}  {length:>8,}  {src:<20}  [{doc_id}]  {repr(snippet)}')


# ── semantic ──────────────────────────────────────────────────────
def _find_semantic(args: argparse.Namespace) -> None:
    if not args.text and not args.id:
        raise SystemExit('Provide --text or --id for semantic mode!')

    collection = zvec.open(args.db_path)

    print(f'Semantic search  (limit={args.limit})')
    print(f'Loading embedding model: {args.embedding_model}')
    model = SentenceTransformer(args.embedding_model, device=DEVICE).half()
    if args.text:
        query_vec = embed_texts(model, [args.text])[0]
    else:
        # Fetch stored embedding from the collection
        result = collection.fetch(ids=args.id)
        if args.id not in result:
            raise SystemExit(f'Record not found: {args.id}')
        embedding = result[args.id].vector('embedding')
        if embedding is None:
            raise SystemExit(f'Record has no embedding vector: {args.id}')
        query_vec = embedding
    print(f'Query embedded ({len(query_vec)}-dim). Searching…')

    query_kwargs = dict(
        vectors=zvec.VectorQuery(
            field_name='embedding',
            vector=query_vec,
        ),
        topk=args.limit,
        output_fields=['snippet', 'length', 'source'],
    )
    if args.text_like:
        query_kwargs['filter'] = f"snippet LIKE '{args.text_like}'"

    results = collection.query(**query_kwargs)

    if not results:
        print('  No results.')
        return

    matches: list[tuple[float, str, str, int, str]] = []
    for doc in results:
        # zvec COSINE metric returns cosine distance (0 = identical);
        # convert to cosine similarity (1 = identical) for display/threshold.
        score = 1.0 - (doc.score if doc.score is not None else 1.0)
        if score < args.threshold:
            continue
        snippet = doc.field('snippet') or ''
        length = doc.field('length') or 0
        source = doc.field('source') or ''
        matches.append((score, doc.id, snippet, length, source))

    matches.sort(key=lambda x: -x[0])
    matches = matches[: args.limit]
    if not matches:
        print('  No matches above threshold.')
        return

    print(f'  Matches: {len(matches)} (showing up to {args.limit})\n')
    print(f'\n  {"Score":>8}  {"Length":>8}  {"Source":<20}  ID / Snippet')
    print(f'  {"─" * 8}  {"─" * 8}  {"─" * 20}  {"─" * 40}')
    for score, doc_id, snippet, length, source in matches:
        src = (source or '')[:20]
        length = length or 0
        score = score or 0.0
        print(f'  {score:>8.4f}  {length:>8,}  {src:<20}  [{doc_id}]  {repr(snippet)}')


# ══════════════════════════════════════════════════════════════════
# ID resolution helper
# ══════════════════════════════════════════════════════════════════
def _resolve_id(args: argparse.Namespace) -> str | None:
    """Return the composite xxh64_blake2b id from --id or by hashing --text."""
    if args.id:
        if '_' not in args.id:
            raise SystemExit(f'--id must be in xxh64_blake2b format, got: {args.id!r}')
        return args.id
    if getattr(args, 'text', None):
        xxh, bl = compute_text_hashes(args.text)
        return f'{xxh}_{bl}'
    return None


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    root = argparse.ArgumentParser(
        description='Index and query text documents using Zvec',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Global args ────────────────────────────────────────────────
    root.add_argument('--db-path', default=DEFAULT_DB_PATH, help='Path to the Zvec collection directory')
    root.add_argument('--num-perm', type=int, default=DEFAULT_NUM_PERM, help='MinHash permutations')
    root.add_argument('--ngram-size', type=int, default=DEFAULT_NGRAM_SIZE, help='Word n-gram size for MinHash')
    root.add_argument('--lsh-bands', type=int, default=DEFAULT_LSH_BANDS, help='MinHash LSH band count')

    sub = root.add_subparsers(dest='subcommand', required=True)

    # ── index ──────────────────────────────────────────────────────
    p_index = sub.add_parser('index', help='Ingest JSONL/Parquet files into the Zvec collection')
    p_index.add_argument('--input-glob', required=True, help="Glob pattern for input files (e.g. 'raw/*.jsonl')")
    p_index.add_argument('--source', default=None, help='Provenance label added to every record')
    p_index.add_argument('--text-key', default=DEFAULT_TEXT_KEY, help='JSON field containing document text')
    p_index.add_argument('--embedding-model', default=DEFAULT_EMBEDDING_MODEL, help='SentenceTransformer model name')
    p_index.add_argument('--num-proc', type=int, default=os.cpu_count() or 4, help='Parallel workers for compute phase')
    p_index.add_argument('--compute-batch-size', type=int, default=DEFAULT_COMPUTE_BATCH_SIZE, help='HF datasets.map() batch size')
    p_index.add_argument('--write-batch-size', type=int, default=DEFAULT_WRITE_BATCH_SIZE, help='Iteration batch size for write phase')
    p_index.add_argument('--embed-batch-size', type=int, default=DEFAULT_EMBED_BATCH_SIZE, help='Batch size for model.encode()')
    p_index.add_argument('--no-skip-existing', action='store_true', help='Do not check for existing docs (re-insert all)')
    p_index.add_argument('--optimize', action='store_true', help='Call collection.optimize() after ingestion')
    cache_group = p_index.add_mutually_exclusive_group()
    cache_group.add_argument('--cache-dir', default=None, help='HF datasets cache directory')
    cache_group.add_argument('--no-cache', action='store_true', help='Use a temporary cache dir per file')

    # ── stats ──────────────────────────────────────────────────────
    sub.add_parser('stats', help='Print collection schema and statistics')

    # ── get ────────────────────────────────────────────────────────
    p_get = sub.add_parser('get', help='Retrieve a single record by composite ID or raw text')
    id_group = p_get.add_mutually_exclusive_group(required=True)
    id_group.add_argument('--id', help='Composite ID in xxh64:blake2b format')
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
    p_find.add_argument('--text-like', default=None, help="Filter snippets with LIKE pattern (e.g. 'Word%%' or '%%.log')")
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
