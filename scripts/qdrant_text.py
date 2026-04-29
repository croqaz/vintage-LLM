"""
Qdrant-only text indexing and querying.

Combines indexing (from JSONL / Parquet files) and querying (stats, get,
find-similar) into a single script backed exclusively by Qdrant.

Dense embeddings are computed with fastembed (default model:
nomic-ai/nomic-embed-text-v1.5, 768-dim).  MinHash signatures live in the
point payload (base64-packed uint64 array) alongside an ``lsh_bands`` keyword
index for fast near-duplicate candidate retrieval; Jaccard similarity is
verified in Python after the LSH candidate filter.

Data model (per Qdrant point)
─────────────────────────────
  Vector    dense embedding (float32, COSINE distance, on-disk HNSW)
  Payload   xxh64, blake2b, length, unique_chars, words, sentences, snippet,
            source, source_file, minhash (base64-packed uint64),
            lsh_bands (list[str]: "<band_id>_<hex>")

Payload indices (created on first collection init):
  xxh64, blake2b, source, lsh_bands  →  KEYWORD
  length                             →  INTEGER

Subcommands
───────────
  index          Ingest JSONL / Parquet files into Qdrant.
  stats          Print aggregate statistics for all indexed records.
  get            Retrieve a single record by composite ID or raw text.
  find-similar   Find duplicate or similar texts:
                   exact     — identical text (hash lookup)
                   minhash   — near-duplicates via MinHash LSH + Jaccard
                   semantic  — ANN embedding search via Qdrant HNSW

Usage examples
──────────────
  # Index
  python qdrant_text.py index --input-glob "raw/*.jsonl" --source "British Library"
  python qdrant_text.py index --input-glob "raw/*.parquet" --skip-existing

  # Stats
  python qdrant_text.py stats
  python qdrant_text.py stats --collection my_collection

  # Get
  python qdrant_text.py get --text "The quick brown fox"
  python qdrant_text.py get --id "a1b2c3d4e5f6a7b8:abcdef..."

  # Find similar
  python qdrant_text.py find-similar --mode exact --text "some text"
  python qdrant_text.py find-similar --mode minhash --text "some text" --threshold 0.4
  python qdrant_text.py find-similar --mode semantic --text "some text" --limit 10
"""

import argparse
import base64
import glob
import hashlib
import math
import os
import re
import shutil
import string
import struct
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xxhash
from datasketch import MinHash
from tqdm.auto import tqdm

# ──────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────
DEFAULT_TEXT_KEY = 'text'
DEFAULT_NUM_PERM = 128
DEFAULT_NGRAM_SIZE = 5
DEFAULT_COMPUTE_BATCH_SIZE = 1_000
DEFAULT_WRITE_BATCH_SIZE = 512
DEFAULT_ENCODE_BATCH_SIZE = 64
DEFAULT_LSH_BANDS = 16  # 16 bands × 8 rows = 128 perms; Jaccard threshold ≈ 0.53
DEFAULT_COLLECTION = 'text_index'
DEFAULT_EMBEDDING_MODEL = 'nomic-ai/nomic-embed-text-v1.5'
DEFAULT_QDRANT_URL = 'http://127.0.0.1:6333'

# ──────────────────────────────────────────────────────────────────
# HF dataset features for the compute phase (text kept temporarily)
# ──────────────────────────────────────────────────────────────────
from datasets import Features, Sequence, Value

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


# ──────────────────────────────────────────────────────────────────
# Inline helpers (duplicated from index_texts.py / query_index.py)
# ──────────────────────────────────────────────────────────────────
def split_into_sentences(text: str) -> list[str]:
    parts = SENTENCE_BOUNDARY_REGEX.split(text)
    return [parts[i] + (parts[i + 1] if i + 1 < len(parts) else '') for i in range(0, len(parts), 2)]


def compute_text_hashes(text: str) -> tuple[str, str]:
    """Return (xxh64_hex, blake2b_hex) for a text string."""
    encoded = text.encode('utf-8')
    return xxhash.xxh64_hexdigest(encoded), hashlib.blake2b(encoded, digest_size=32).hexdigest()


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
    """Split a MinHash signature into LSH bands and return one xxh64 hex digest per band."""
    rows = len(sig) // num_bands
    return [xxhash.xxh64_hexdigest(struct.pack(f'<{rows}Q', *sig[b * rows : (b + 1) * rows])) for b in range(num_bands)]


def compute_batch(
    batch: dict,
    text_key: str = DEFAULT_TEXT_KEY,
    num_perm: int = DEFAULT_NUM_PERM,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
) -> dict:
    """HF .map(batched=True) worker — computes per-document stats, hashes, and MinHash."""
    out: dict[str, list] = {c: [] for c in COMPUTE_FEATURES}

    for text in batch[text_key]:
        if text is None:
            text = ''
        if len(text) <= 10 or len(text) > 10_000_000:
            continue

        tokens = [t for t in text.split() if t not in string.punctuation]

        out['text'].append(text)
        out['xxh64'].append(xxhash.xxh64_hexdigest(text.encode('utf-8')))
        out['blake2b'].append(hashlib.blake2b(text.encode('utf-8'), digest_size=32).hexdigest())
        out['length'].append(len(text))
        out['unique_chars'].append(len(set(text)))
        out['words'].append(len(tokens))

        segs = split_into_sentences(text) if text.strip() else []
        out['sentences'].append(max(1, len(segs)) if text.strip() else 0)

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
            mh.update(text.encode('utf-8'))
        out['minhash'].append(mh.hashvalues.tolist())

    return out


def _unpack_minhash(data: bytes, num_perm: int = DEFAULT_NUM_PERM) -> MinHash:
    """Reconstruct a datasketch MinHash from little-endian packed uint64 bytes."""
    vals = struct.unpack(f'<{num_perm}Q', data)
    mh = MinHash(num_perm=num_perm)
    mh.hashvalues = np.array(vals, dtype=np.uint64)
    return mh


def _b64_to_minhash(b64str: str, num_perm: int = DEFAULT_NUM_PERM) -> MinHash:
    """Reconstruct a datasketch MinHash from a base64-encoded packed-bytes string."""
    return _unpack_minhash(base64.b64decode(b64str), num_perm)


# ──────────────────────────────────────────────────────────────────
# Running statistics accumulator (Welford online algorithm)
# ──────────────────────────────────────────────────────────────────
class _RunningStats:
    """Streaming mean, variance (population), min, max — no values stored."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._M2 = 0.0
        self.min = math.inf
        self.max = -math.inf

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._M2 += delta * delta2
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x

    @property
    def stddev(self) -> float:
        return math.sqrt(self._M2 / self.n) if self.n > 1 else 0.0

    def summary(self, label: str, width: int = 12) -> str:
        if self.n == 0:
            return f'  {label}: no data'
        return (
            f'  {label}: '
            f'min={self.min:>{width},.0f}  '
            f'max={self.max:>{width},.0f}  '
            f'mean={self.mean:>{width},.1f}  '
            f'stddev={self.stddev:>{width},.1f}'
        )


# ──────────────────────────────────────────────────────────────────
# Pretty-print a record dict
# ──────────────────────────────────────────────────────────────────
def _print_record(rec: dict, indent: int = 2) -> None:
    pad = ' ' * indent
    for k, v in rec.items():
        if k == 'minhash':
            if isinstance(v, list) and len(v) > 8:
                display = f'[{v[0]}, {v[1]}, {v[2]}, {v[3]}, ... {v[-2]}, {v[-1]}]  ({len(v)} values)'
            else:
                display = str(v)
        elif k == 'embedding':
            if isinstance(v, list) and len(v) > 6:
                display = f'[{v[0]:.5f}, {v[1]:.5f}, {v[2]:.5f}, ... {v[-1]:.5f}]  ({len(v)} floats)'
            else:
                display = str(v)
        else:
            display = str(v)
        print(f'{pad}{k}: {display}')


# ──────────────────────────────────────────────────────────────────
# Row-accounting dataclass
# ──────────────────────────────────────────────────────────────────
@dataclass
class FileStats:
    path: str
    rows_loaded: int = 0
    rows_dropped: int = 0
    rows_indexed: int = 0

    def validate(self) -> None:
        expected = self.rows_loaded - self.rows_dropped
        if self.rows_indexed != expected:
            raise AssertionError(
                f'\n  Row accounting error for: {self.path}\n'
                f'  loaded={self.rows_loaded:,}  dropped={self.rows_dropped:,}'
                f'  → expected indexed={expected:,}  actual indexed={self.rows_indexed:,}'
            )

    def print_summary(self) -> None:
        print(
            f'  Loaded:   {self.rows_loaded:>14,}\n'
            f'  Dropped:  {self.rows_dropped:>14,}  (len ≤ 10 or > 10 M)\n'
            f'  Indexed:  {self.rows_indexed:>14,}'
        )


# ──────────────────────────────────────────────────────────────────
# Qdrant helpers
# ──────────────────────────────────────────────────────────────────
def _qdrant_connect(url: str):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url)


def _ensure_collection(client, collection: str, embedding_dim: int) -> None:
    """Create the Qdrant collection and payload indices if they do not exist."""
    from qdrant_client.models import Distance, HnswConfigDiff, PayloadSchemaType, VectorParams

    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
                on_disk=True,
            ),
            hnsw_config=HnswConfigDiff(on_disk=True),
        )
        for field, schema in [
            ('xxh64', PayloadSchemaType.KEYWORD),
            ('blake2b', PayloadSchemaType.KEYWORD),
            ('length', PayloadSchemaType.INTEGER),
            ('source', PayloadSchemaType.KEYWORD),
            ('lsh_bands', PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=schema,
            )
        print(f'  Created collection {collection!r} (dim={embedding_dim})')
    else:
        info = client.get_collection(collection)
        # Validate that the existing collection's vector dimension matches the model
        existing_dim = info.config.params.vectors.size  # type: ignore[union-attr]
        if existing_dim != embedding_dim:
            raise SystemExit(
                f'Collection {collection!r} already exists with vector dim={existing_dim}, '
                f'but the embedding model produces dim={embedding_dim}.\n'
                f'Delete the collection first or use --collection <new_name>.\n'
                f'  To delete:  curl -X DELETE {client._client.rest_uri}/collections/{collection}'
            )
        print(f'  Using existing collection {collection!r} (points={info.points_count or 0:,}, dim={existing_dim})')


def _resolve_id(args: argparse.Namespace) -> str | None:
    """Return the composite xxh64:blake2b id from --id or by hashing --text."""
    if args.id:
        if ':' not in args.id:
            raise SystemExit(f'--id must be in xxh64:blake2b format, got: {args.id!r}')
        return args.id
    if getattr(args, 'text', None):
        xxh, bl = compute_text_hashes(args.text)
        return f'{xxh}:{bl}'
    return None


def _qdrant_get_record(client, collection: str, doc_id: str, num_perm: int) -> dict | None:
    """Look up a single record by composite xxh64:blake2b id."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    parts = doc_id.split(':', 1)
    if len(parts) != 2:
        print(f'  Invalid id format. Expected xxh64:blake2b, got: {doc_id!r}', file=sys.stderr)
        return None
    xxh64_val, blake2b_val = parts
    results, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(key='xxh64', match=MatchValue(value=xxh64_val)),
                FieldCondition(key='blake2b', match=MatchValue(value=blake2b_val)),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not results:
        return None
    pt = results[0]
    rec = _decode_payload(pt.payload or {}, num_perm)
    rec['_id'] = doc_id
    rec['_point_id'] = pt.id
    return rec


def _decode_payload(payload: dict, num_perm: int) -> dict:
    """Decode a Qdrant point payload (base64 minhash → list[int])."""
    rec = dict(payload)
    if 'minhash' in rec and isinstance(rec['minhash'], str):
        try:
            mh = _b64_to_minhash(rec['minhash'], num_perm)
            rec['minhash'] = mh.hashvalues.tolist()
        except Exception:
            pass
    return rec


# ──────────────────────────────────────────────────────────────────
# fastembed helpers
# ──────────────────────────────────────────────────────────────────
def _load_embedding_model(model_name: str):
    """Load a fastembed TextEmbedding model and return (model, embedding_dim)."""
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name)
    # Probe embedding dimension with a single-doc encode
    probe = list(model.embed(['hello']))
    dim = len(probe[0])
    return model, dim


# ══════════════════════════════════════════════════════════════════
# Embedding helpers
# ══════════════════════════════════════════════════════════════════

# ONNX attention memory ∝ batch_size × num_heads × seq_len² × sizeof(float32).
# A single 8192-token text: 1 × 12 × 8192² × 4 ≈ 3.2 GB  — fine.
# 64 of them:              64 × 12 × 8192² × 4 ≈ 206 GB   — OOM-killed.
#
# Instead of heuristic batch-size formulas (which the OS kills before Python
# can catch the error), we split texts by character length:
#   • Short texts (≤ LONG_TEXT_THRESHOLD) → batched at full batch_size.
#     At ≤ 4 000 chars ≈ 1 000 tokens: 64 × 12 × 1000² × 4 ≈ 3 GB.
#   • Long texts (> LONG_TEXT_THRESHOLD) → embedded one at a time.
#     Even at 8192 tokens, a single text needs only ~3.2 GB.
#
# Short texts are sorted by length before embedding so fastembed sub-batches
# group similarly-sized texts, minimising padding waste.

LONG_TEXT_THRESHOLD = 4_000  # chars; ~1 000 tokens at ~4 chars/token
EMBED_TRUNCATE_THRESHOLD = 64_000  # chars; texts longer than this are truncated for embedding


def _truncate_for_embedding(text: str) -> str:
    """Keep first 32K + last 32K chars of very long texts for embedding."""
    if len(text) <= EMBED_TRUNCATE_THRESHOLD:
        return text
    return text[:32_000] + text[-32_000:]


def _embed_texts(model, texts: list[str], batch_size: int) -> list:
    """Embed texts with length-aware batching to prevent OOM."""
    # Truncate very long texts for embedding only (payload/hashes use full text)
    embed_texts = [_truncate_for_embedding(t) for t in texts]

    short_idx = [i for i in range(len(embed_texts)) if len(embed_texts[i]) <= LONG_TEXT_THRESHOLD]
    long_idx = [i for i in range(len(embed_texts)) if len(embed_texts[i]) > LONG_TEXT_THRESHOLD]

    embeddings = [None] * len(texts)

    # Batch-embed short texts (sorted for minimal padding waste)
    if short_idx:
        short_idx.sort(key=lambda i: len(embed_texts[i]))
        short_texts = [embed_texts[i] for i in short_idx]
        for j, emb in enumerate(model.embed(short_texts, batch_size=batch_size)):
            embeddings[short_idx[j]] = emb

    # Embed long texts one at a time — safe even at max context length
    if long_idx:
        if len(long_idx) > 5:
            print(f'  {len(long_idx)} long texts (>{LONG_TEXT_THRESHOLD:,} chars) — embedding individually')
        long_texts = [embed_texts[i] for i in long_idx]
        for j, emb in enumerate(model.embed(long_texts, batch_size=1)):
            embeddings[long_idx[j]] = emb

    return embeddings


# ══════════════════════════════════════════════════════════════════
# Subcommand: index
# ══════════════════════════════════════════════════════════════════
def _infer_dataset_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == '.parquet':
        return 'parquet'
    return 'json'


def cmd_index(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

    files = sorted(glob.glob(args.input_glob, recursive=True))
    if not files:
        raise SystemExit(f'No files matched: {args.input_glob}')
    print(f'Found {len(files)} input file(s)')

    # Load fastembed model
    print(f'Loading embedding model: {args.embedding_model}')
    model, embedding_dim = _load_embedding_model(args.embedding_model)
    print(f'  Embedding dimension: {embedding_dim}')

    # Connect to Qdrant
    client = _qdrant_connect(args.qdrant_url)
    _ensure_collection(client, args.collection, embedding_dim)

    # Resume: seed sequential IDs from current points_count
    info = client.get_collection(args.collection)
    next_id = info.points_count or 0

    grand_loaded = 0
    grand_dropped = 0
    grand_indexed = 0
    grand_skipped = 0

    for input_path in files:
        print(f'\n{"=" * 60}')
        print(f'Processing: {input_path}')
        print(f'{"=" * 60}')

        stats = FileStats(path=input_path)

        fmt = _infer_dataset_format(input_path)
        _tmp_cache: str | None = None
        if args.no_cache:
            _tmp_cache = tempfile.mkdtemp(prefix='qdrant_text_')
            effective_cache = _tmp_cache
        else:
            effective_cache = args.cache_dir

        try:
            ds = load_dataset(fmt, split='train', data_files=input_path, cache_dir=effective_cache)
            if args.text_key not in ds.column_names:
                raise ValueError(f"Field '{args.text_key}' not found in {input_path}. Available columns: {ds.column_names}")
            stats.rows_loaded = len(ds)
            print(f'  Loaded:   {stats.rows_loaded:,} rows')

            # Compute phase (CPU, multiprocessing)
            ds = ds.map(
                compute_batch,
                batched=True,
                batch_size=args.compute_batch_size,
                writer_batch_size=args.compute_batch_size,
                num_proc=args.num_proc,
                remove_columns=ds.column_names,
                features=COMPUTE_FEATURES,
                fn_kwargs=dict(text_key=args.text_key, num_perm=args.num_perm, ngram_size=args.ngram_size),
                desc='  Computing stats & hashes',
            )
            stats.rows_dropped = stats.rows_loaded - len(ds)
            if stats.rows_dropped:
                print(f'  After compute: {len(ds):,} rows  ({stats.rows_dropped:,} dropped)')

            # Embed + write in batches
            pbar = tqdm(total=len(ds), desc='  Writing index', unit='doc', dynamic_ncols=True)
            for batch in ds.iter(batch_size=args.write_batch_size):
                texts = batch['text']

                # Embed (short texts batched, long texts individually)
                embeddings = _embed_texts(model, texts, args.encode_batch_size)

                # Optional skip-existing check
                if args.skip_existing:
                    keep = []
                    for i in range(len(texts)):
                        existing, _ = client.scroll(
                            collection_name=args.collection,
                            scroll_filter=Filter(
                                must=[
                                    FieldCondition(key='xxh64', match=MatchValue(value=batch['xxh64'][i])),
                                    FieldCondition(key='blake2b', match=MatchValue(value=batch['blake2b'][i])),
                                ]
                            ),
                            limit=1,
                            with_payload=False,
                            with_vectors=False,
                        )
                        if existing:
                            grand_skipped += 1
                        else:
                            keep.append(i)
                    if not keep:
                        pbar.update(len(texts))
                        continue
                else:
                    keep = list(range(len(texts)))

                # Build points
                points = []
                for i in keep:
                    minhash_vals = batch['minhash'][i]
                    lsh_tags = [
                        f'{band_id}_{band_hex}' for band_id, band_hex in enumerate(minhash_lsh_band_hashes(minhash_vals, args.lsh_bands))
                    ]
                    minhash_b64 = base64.b64encode(struct.pack(f'<{len(minhash_vals)}Q', *minhash_vals)).decode('ascii')

                    payload = {
                        'xxh64': batch['xxh64'][i],
                        'blake2b': batch['blake2b'][i],
                        'length': batch['length'][i],
                        'unique_chars': batch['unique_chars'][i],
                        'words': batch['words'][i],
                        'sentences': batch['sentences'][i],
                        'snippet': batch['snippet'][i],
                        'minhash': minhash_b64,
                        'lsh_bands': lsh_tags,
                    }
                    if args.source is not None:
                        payload['source'] = args.source
                    payload['source_file'] = Path(input_path).name

                    points.append(PointStruct(id=next_id, vector=embeddings[i].tolist(), payload=payload))
                    next_id += 1

                client.upsert(collection_name=args.collection, points=points)
                stats.rows_indexed += len(points)
                pbar.update(len(texts))

            pbar.close()
            stats.validate()
            stats.print_summary()

        finally:
            if _tmp_cache is not None:
                shutil.rmtree(_tmp_cache, ignore_errors=True)

        grand_loaded += stats.rows_loaded
        grand_dropped += stats.rows_dropped
        grand_indexed += stats.rows_indexed

    client.close()

    # Final summary
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    if grand_skipped:
        print(f'  Already exists: {grand_skipped:>14,}  (--skip-existing)')
    print(
        f'  Grand loaded:   {grand_loaded:>14,}\n'
        f'  Grand dropped:  {grand_dropped:>14,}  (len ≤ 10 or > 10 M)\n'
        f'  Grand indexed:  {grand_indexed:>14,}\n'
        f'  Qdrant URL:     {args.qdrant_url}\n'
        f'  Collection:     {args.collection}\n'
        f'  Embedding model: {args.embedding_model}'
    )
    if args.source:
        print(f'  Source label:    {args.source}')

    cache_loc = args.cache_dir or '~/.cache/huggingface/datasets'
    print('\n💡  HF datasets Arrow cache may be large. Clean up with:')
    print(f'    rm -rf {cache_loc}')


# ══════════════════════════════════════════════════════════════════
# Subcommand: stats
# ══════════════════════════════════════════════════════════════════
def cmd_stats(args: argparse.Namespace) -> None:
    client = _qdrant_connect(args.qdrant_url)
    info = client.get_collection(args.collection)
    print(f'Collection: {args.collection}')
    print(f'  Status:          {info.status}')
    print(f'  Points total:    {info.points_count or 0:>12,}')
    print(f'  Vectors on disk: {info.config.params.vectors.on_disk}')  # type: ignore[union-attr]

    total = 0
    has_source_file = 0
    sources: dict[str, int] = {}
    s_length = _RunningStats()
    s_words = _RunningStats()
    s_sentences = _RunningStats()

    offset = None
    print('Scrolling Qdrant (payload only)…')
    while True:
        results, next_offset = client.scroll(
            collection_name=args.collection,
            offset=offset,
            limit=1_000,
            with_payload=['length', 'words', 'sentences', 'source', 'source_file'],
            with_vectors=False,
        )
        if not results:
            break
        for pt in results:
            p = pt.payload or {}
            total += 1
            if p.get('length') is not None:
                s_length.update(float(p['length']))
            if p.get('words') is not None:
                s_words.update(float(p['words']))
            if p.get('sentences') is not None:
                s_sentences.update(float(p['sentences']))
            src = p.get('source')
            if src:
                sources[src] = sources.get(src, 0) + 1
            if p.get('source_file'):
                has_source_file += 1
        offset = next_offset
        if offset is None:
            break

    print(f'\n{"─" * 60}')
    print(f'  Total records:               {total:>12,}')
    print(f'  Records with source_file:    {has_source_file:>12,}')
    print()
    print(s_length.summary('length    '))
    print(s_words.summary('words     '))
    print(s_sentences.summary('sentences '))
    if sources:
        print(f'\n  Sources ({len(sources)} unique):')
        for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
            print(f'    {src!r}: {cnt:,}')
    print(f'{"─" * 60}')
    client.close()


# ══════════════════════════════════════════════════════════════════
# Subcommand: get
# ══════════════════════════════════════════════════════════════════
def cmd_get(args: argparse.Namespace) -> None:
    doc_id = _resolve_id(args)
    if doc_id is None:
        raise SystemExit('Provide --id xxh64:blake2b or --text "some text"')

    print(f'Looking up: {doc_id}')
    client = _qdrant_connect(args.qdrant_url)
    rec = _qdrant_get_record(client, args.collection, doc_id, args.num_perm)

    if args.with_vector and rec:
        # Re-fetch with vector
        point_id = rec['_point_id']
        pts = client.retrieve(collection_name=args.collection, ids=[point_id], with_payload=False, with_vectors=True)
        if pts:
            rec['embedding'] = pts[0].vector

    client.close()

    if rec is None:
        print('Not found.')
        return

    print(f'{"─" * 60}')
    _print_record(rec)
    print(f'{"─" * 60}')


# ══════════════════════════════════════════════════════════════════
# Subcommand: find-similar
# ══════════════════════════════════════════════════════════════════
def cmd_find_similar(args: argparse.Namespace) -> None:
    mode = args.mode
    if mode == 'exact':
        _find_exact(args)
    elif mode == 'minhash':
        _find_minhash(args)
    elif mode == 'semantic':
        _find_semantic(args)
    else:
        raise SystemExit(f'Unknown mode: {mode!r}')


# ── exact ─────────────────────────────────────────────────────────
def _find_exact(args: argparse.Namespace) -> None:
    doc_id = _resolve_id(args)
    if doc_id is None:
        raise SystemExit('Provide --id or --text for exact mode')
    print(f'Exact lookup: {doc_id}')

    client = _qdrant_connect(args.qdrant_url)
    rec = _qdrant_get_record(client, args.collection, doc_id, args.num_perm)
    client.close()

    if rec:
        print('Found (exact match):')
        print(f'{"─" * 60}')
        _print_record(rec)
        print(f'{"─" * 60}')
    else:
        print('Not found.')


# ── minhash ───────────────────────────────────────────────────────
def _get_query_minhash(args: argparse.Namespace) -> tuple[MinHash, list[str]]:
    """Return (query_minhash, lsh_band_hashes) from --text or --id."""
    if args.text:
        mh = compute_minhash(args.text, args.num_perm, args.ngram_size)
        bands = minhash_lsh_band_hashes(mh.hashvalues.tolist(), args.lsh_bands)
        return mh, bands

    if args.id:
        client = _qdrant_connect(args.qdrant_url)
        rec = _qdrant_get_record(client, args.collection, args.id, args.num_perm)
        client.close()
        if rec is None:
            raise SystemExit(f'Record not found: {args.id}')
        mh_vals = rec.get('minhash')
        if mh_vals is None:
            raise SystemExit(f'Record has no minhash field: {args.id}')
        mh = MinHash(num_perm=args.num_perm)
        mh.hashvalues = np.array(mh_vals, dtype=np.uint64)
        bands = minhash_lsh_band_hashes(mh.hashvalues.tolist(), args.lsh_bands)
        return mh, bands

    raise SystemExit('Provide --text or --id for minhash mode')


def _find_minhash(args: argparse.Namespace) -> None:
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    query_mh, band_hashes = _get_query_minhash(args)
    query_id = args.id or '<query text>'
    print(f'MinHash near-duplicate search  (threshold≥{args.threshold}, limit={args.limit})')
    print(f'Query: {query_id}')

    client = _qdrant_connect(args.qdrant_url)
    band_tags = [f'{band_id}_{bh}' for band_id, bh in enumerate(band_hashes)]
    scroll_filter = Filter(should=[FieldCondition(key='lsh_bands', match=MatchAny(any=band_tags))])

    candidates = []
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=args.collection,
            scroll_filter=scroll_filter,
            offset=offset,
            limit=1_000,
            with_payload=['minhash', 'snippet', 'length', 'source', 'xxh64', 'blake2b'],
            with_vectors=False,
        )
        candidates.extend(results)
        offset = next_offset
        if offset is None:
            break

    client.close()
    print(f'  LSH candidates: {len(candidates):,}')

    matches: list[tuple[float, str, str, int, str]] = []
    for pt in candidates:
        p = pt.payload or {}
        mh_field = p.get('minhash')
        if not mh_field:
            continue
        cand_mh = _b64_to_minhash(mh_field, args.num_perm)
        jaccard = query_mh.jaccard(cand_mh)
        if jaccard >= args.threshold:
            doc_id = f'{p.get("xxh64", "")}:{p.get("blake2b", "")}'
            snippet = p.get('snippet', '')
            length = int(p.get('length', 0))
            source = p.get('source', '')
            matches.append((jaccard, doc_id, snippet, length, source))

    _print_minhash_results(matches, args.limit)


def _print_minhash_results(matches: list[tuple[float, str, str, int, str]], limit: int) -> None:
    matches.sort(key=lambda x: -x[0])
    matches = matches[:limit]
    if not matches:
        print('  No matches above threshold.')
        return
    print(f'  Matches: {len(matches)} (showing up to {limit})\n')
    print(f'  {"Jaccard":>8}  {"Length":>8}  {"Source":<20}  ID / Snippet')
    print(f'  {"─" * 8}  {"─" * 8}  {"─" * 20}  {"─" * 40}')
    for jaccard, doc_id, snippet, length, source in matches:
        short_id = doc_id[:20] + '…' if len(doc_id) > 20 else doc_id
        short_snippet = textwrap.shorten(snippet, width=50, placeholder='…')
        src = (source or '')[:20]
        print(f'  {jaccard:>8.4f}  {length:>8,}  {src:<20}  [{short_id}]  {short_snippet}')


# ── semantic ──────────────────────────────────────────────────────
def _find_semantic(args: argparse.Namespace) -> None:
    if not args.text:
        raise SystemExit('--text is required for semantic mode')

    print(f'Semantic search  (limit={args.limit})')
    print(f'Loading embedding model: {args.embedding_model}')

    model, _dim = _load_embedding_model(args.embedding_model)
    query_vec = list(model.embed([args.text]))[0].tolist()
    print(f'Query embedded ({len(query_vec)}-dim). Searching…')

    client = _qdrant_connect(args.qdrant_url)
    results = client.query_points(
        collection_name=args.collection,
        query=query_vec,
        limit=args.limit,
        with_payload=['xxh64', 'blake2b', 'snippet', 'length', 'source'],
        with_vectors=False,
    ).points
    client.close()

    if not results:
        print('  No results.')
        return

    print(f'\n  {"Score":>8}  {"Length":>8}  {"Source":<20}  ID / Snippet')
    print(f'  {"─" * 8}  {"─" * 8}  {"─" * 20}  {"─" * 40}')
    for pt in results:
        p = pt.payload or {}
        doc_id = f'{p.get("xxh64", "")}:{p.get("blake2b", "")}'
        short_id = doc_id[:20] + '…' if len(doc_id) > 20 else doc_id
        snippet = textwrap.shorten(p.get('snippet', ''), width=50, placeholder='…')
        src = (p.get('source') or '')[:20]
        length = int(p.get('length', 0))
        print(f'  {pt.score:>8.4f}  {length:>8,}  {src:<20}  [{short_id}]  {snippet}')


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    root = argparse.ArgumentParser(
        description='Qdrant-only text indexing and querying',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Global flags
    root.add_argument('--qdrant-url', default=DEFAULT_QDRANT_URL, help='Qdrant server URL')
    root.add_argument('--collection', default=DEFAULT_COLLECTION, help='Qdrant collection name')
    root.add_argument('--num-perm', type=int, default=DEFAULT_NUM_PERM, help='MinHash permutation count')
    root.add_argument('--ngram-size', type=int, default=DEFAULT_NGRAM_SIZE, help='Word n-gram size for MinHash')
    root.add_argument('--lsh-bands', type=int, default=DEFAULT_LSH_BANDS, help='LSH band count (num_perm must be divisible)')

    sub = root.add_subparsers(dest='subcommand', required=True)

    # ── index ──────────────────────────────────────────────────────
    p_index = sub.add_parser('index', help='Ingest JSONL / Parquet files into Qdrant')
    p_index.add_argument('--input-glob', required=True, help="Glob pattern for input files (e.g. 'raw/*.jsonl')")
    p_index.add_argument('--source', default=None, help='Provenance label added to every record')
    p_index.add_argument('--text-key', default=DEFAULT_TEXT_KEY, help='JSON field containing document text')
    p_index.add_argument('--embedding-model', default=DEFAULT_EMBEDDING_MODEL, help='fastembed model name or path')
    p_index.add_argument('--num-proc', type=int, default=os.cpu_count() or 4, help='Parallel workers for compute phase')
    p_index.add_argument('--compute-batch-size', type=int, default=DEFAULT_COMPUTE_BATCH_SIZE, help='Batch size for HF datasets.map()')
    p_index.add_argument('--write-batch-size', type=int, default=DEFAULT_WRITE_BATCH_SIZE, help='Batch size for the embed + write phase')
    p_index.add_argument('--encode-batch-size', type=int, default=DEFAULT_ENCODE_BATCH_SIZE, help='Batch size for fastembed model.embed()')
    p_index.add_argument('--skip-existing', action='store_true', help='Skip records whose xxh64+blake2b already exist')
    cache_group = p_index.add_mutually_exclusive_group()
    cache_group.add_argument('--cache-dir', default=None, help='HF datasets cache directory')
    cache_group.add_argument('--no-cache', action='store_true', help='Use a temp cache dir per file and delete after')

    # ── stats ──────────────────────────────────────────────────────
    sub.add_parser('stats', help='Print aggregate statistics for all indexed records')

    # ── get ────────────────────────────────────────────────────────
    p_get = sub.add_parser('get', help='Retrieve a single record by composite ID or raw text')
    id_group = p_get.add_mutually_exclusive_group(required=True)
    id_group.add_argument('--id', help='Composite ID in xxh64:blake2b format')
    id_group.add_argument('--text', help='Raw text to hash and look up')
    p_get.add_argument('--with-vector', action='store_true', help='Also retrieve and display the embedding vector')

    # ── find-similar ───────────────────────────────────────────────
    p_find = sub.add_parser('find-similar', help='Find identical, near-duplicate, or semantically similar texts')
    p_find.add_argument(
        '--mode',
        choices=['exact', 'minhash', 'semantic'],
        required=True,
        help='exact: hash lookup. minhash: LSH + Jaccard. semantic: embedding ANN.',
    )
    input_group = p_find.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', help='Input text')
    input_group.add_argument('--id', help='Composite ID — fetch stored MinHash/embedding')
    p_find.add_argument('--threshold', type=float, default=0.5, help='Minimum Jaccard similarity (minhash mode)')
    p_find.add_argument('--limit', type=int, default=20, help='Maximum results to print')
    p_find.add_argument('--embedding-model', default=DEFAULT_EMBEDDING_MODEL, help='fastembed model (semantic mode)')

    return root.parse_args()


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
