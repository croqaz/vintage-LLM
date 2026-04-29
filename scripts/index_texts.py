"""
Index datasets → metadata-only index.

Pipeline per input file:
  1. Load dataset         → HF Dataset
  2. Compute stats+hash   → HF Dataset   (rows with len ≤ 10 or > 10 M dropped)
  3. Embedding pass       → model.encode() on text batches (optional)
  4. Write index records → IndexWriter   (text is never written to the index)

Index schema (per record):
  source_file   str        — basename of the origin JSONL file
  source        str        — user-provided provenance label (--source; omitted if not set)
  xxh64         str        — xxHash-64 hex digest
  blake2b       str        — BLAKE2b-256 hex digest
  length        int        — len(text)
  unique_chars  int        — len(set(text))
  words         int        — word count
  sentences     int        — sentence count
  snippet       str        — text[:50] + "[...]" + text[-50:]  (or full text if len ≤ 100)
  minhash       list[int]  — MinHash signature (num_perm uint64 values)
  embedding     list[float]— sentence embedding (omitted if --no-embedding)

Storage backends (--backend):
  jsonl    Newline-delimited JSON, optionally gzip-compressed (--compress). Default.
  valkey   Valkey / Redis-compatible server (redis-py). Hash per record + MinHash LSH Set index.
  qdrant   Qdrant vector database (qdrant-client). Requires embeddings.

Examples:
  python index_texts.py \\
      --input-glob "raw/*.jsonl" \\
      --output index.jsonl \\
      --source "British Library" \\
      --no-embedding

  python index_texts.py \\
      --input-glob "raw/**/*.jsonl" \\
      --output index.jsonl.gz \\
      --compress \\
      --source "Project Gutenberg" \\
      --embedding-model "ibm-granite/granite-embedding-small-english-r2" \\
      --num-proc 16
"""

import abc
import argparse
import base64
import glob
import gzip
import hashlib
import os
import re
import shutil
import string
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import orjson
import xxhash
from datasets import Features, Sequence, Value, load_dataset
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
DEFAULT_ENCODE_BATCH_SIZE = 32
DEFAULT_LSH_BANDS = 16  # 16 bands × 8 rows = 128 perms; Jaccard threshold ≈ 0.53
DEFAULT_DB_COLLECTION = 'text_index'
DEFAULT_EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-0.6B'

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

        out['text'].append(text)
        out['xxh64'].append(xxhash.xxh64_hexdigest(text.encode('utf-8')))
        out['blake2b'].append(hashlib.blake2b(text.encode('utf-8'), digest_size=32).hexdigest())
        out['length'].append(len(text))
        out['unique_chars'].append(len(set(text)))
        out['words'].append(len(tokens))

        segs = split_into_sentences(text) if text.strip() else []
        out['sentences'].append(max(1, len(segs)) if text.strip() else 0)

        # Single snippet field: full text if short, otherwise start[...]end
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


# ──────────────────────────────────────────────────────────────────
# Storage abstraction
# ──────────────────────────────────────────────────────────────────
class IndexWriter(abc.ABC):
    """
    Abstract index writer. Implement this interface to add new storage backends.

    Usage::

        with make_writer(backend, output, compress) as writer:
            writer.write_batch(records)
    """

    @abc.abstractmethod
    def open(self) -> None:
        """Open / initialise the storage resource."""

    @abc.abstractmethod
    def write_batch(self, records: list[dict]) -> None:
        """Write a batch of index record dicts."""

    @abc.abstractmethod
    def close(self) -> None:
        """Flush and close the storage resource."""

    @abc.abstractmethod
    def count(self) -> int:
        """Return the total number of records written so far."""

    def __enter__(self) -> 'IndexWriter':
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()


class JsonlIndexWriter(IndexWriter):
    """
    Writes index records as newline-delimited JSON using orjson.
    Supports optional gzip compression (--compress).

    Notes:
        - Opens in append mode so multiple script invocations accumulate into the same file.
        - minhash is stored as a JSON array of integers.
        - embedding is stored as a JSON array of floats (float32, 6 decimal places).
    """

    def __init__(self, path: str, compress: bool = False) -> None:
        self._path = path
        self._compress = compress
        self._fh = None
        self._count: int = 0

    def open(self) -> None:
        if self._compress:
            self._fh = gzip.open(self._path, 'ab')
        else:
            self._fh = open(self._path, 'ab')

    def write_batch(self, records: list[dict]) -> None:
        for rec in records:
            self._fh.write(orjson.dumps(rec) + b'\n')
        self._count += len(records)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def count(self) -> int:
        return self._count


# ──────────────────────────────────────────────────────────────────
# Public shared helpers (imported by query_index.py)
# ──────────────────────────────────────────────────────────────────
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
    """
    Split a MinHash signature into LSH bands and return one xxh64 hex digest per band.

    sig:       list of num_perm uint64 values (MinHash.hashvalues)
    num_bands: number of bands (num_perm must be divisible by num_bands)

    Returns num_bands hex strings. Documents sharing any band digest are near-duplicate
    candidates (probability of sharing rises steeply around the configured Jaccard threshold).
    """
    rows = len(sig) // num_bands
    return [xxhash.xxh64_hexdigest(struct.pack(f'<{rows}Q', *sig[b * rows : (b + 1) * rows])) for b in range(num_bands)]


# Keep the private alias so internal callers are not broken during any missed replacement.
_minhash_lsh_band_hashes = minhash_lsh_band_hashes


# ──────────────────────────────────────────────────────────────────
# Valkey / Redis-compatible backend
# ──────────────────────────────────────────────────────────────────
class ValKeyIndexWriter(IndexWriter):
    """
    Writes index records to a Valkey / Redis-compatible server via redis-py.
    Works unchanged with Valkey, Kvrocks (disk-backed), or plain Redis.

    Data model
    ──────────
    Primary record (Hash):
        Key     doc:{xxh64}:{blake2b}
        Fields  source (omitted if absent), length, unique_chars, words, sentences,
                snippet, minhash (num_perm × uint64, little-endian packed bytes),
                embedding (N × float32, little-endian packed bytes; omitted if absent)

    MinHash LSH band index (Sets):
        Key     mhlsh:{band_id}:{band_hex}
        Members {xxh64}:{blake2b}
        SUNION across matching bands → near-duplicate candidate set.

    Exact-dedup check:   EXISTS doc:{xxh64}:{blake2b}   — O(1)
    Near-dedup query:    compute band hashes for query doc → SUNION matching band Sets
                         → HGET minhash for each candidate → compute Jaccard similarity.
    """

    def __init__(
        self,
        url: str = 'redis://localhost:6379/0',
        lsh_bands: int = DEFAULT_LSH_BANDS,
        store_embedding: bool = True,
        skip_existing: bool = False,
        embedding_dim: int = 0,
        index_name: str = DEFAULT_DB_COLLECTION,
        semantic_dedup_threshold: float | None = None,
    ) -> None:
        self._url = url
        self._lsh_bands = lsh_bands
        self._store_embedding = store_embedding
        self._skip_existing = skip_existing
        self._embedding_dim = embedding_dim
        self._index_name = index_name
        self._semantic_dedup_threshold = semantic_dedup_threshold
        self._client = None
        self._count: int = 0
        self._skipped: int = 0

    def open(self) -> None:
        import redis as _redis

        self._client = _redis.from_url(self._url, decode_responses=False)
        self._client.ping()
        self._index_exists: bool = False
        if self._store_embedding and self._embedding_dim > 0:
            self._index_exists = self._check_index_exists()
            if self._index_exists:
                print(f'  FT HNSW index {self._index_name!r} found (incremental mode — per-write HNSW updates active)')
            else:
                print(
                    f'  FT HNSW index {self._index_name!r} not found.\n'
                    f'  Writing without HNSW updates (fast bulk load).\n'
                    f'  Index will be created in close() after all data is written.'
                )
                if self._semantic_dedup_threshold is not None:
                    print(
                        '  Warning: --semantic-dedup-threshold has no effect on first run\n'
                        '  (index does not exist yet). It will apply on subsequent incremental runs.'
                    )

    def _check_index_exists(self) -> bool:
        """Return True if the FT index already exists in Valkey."""
        try:
            self._client.ft(self._index_name).info()
            return True
        except Exception:
            return False

    def _ensure_search_index(self) -> None:
        """Create the valkey-search HNSW vector index if it does not already exist."""
        from redis.commands.search.field import VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        try:
            self._client.ft(self._index_name).create_index(
                [
                    VectorField(
                        'embedding',
                        'HNSW',
                        {
                            'TYPE': 'FLOAT32',
                            'DIM': self._embedding_dim,
                            'DISTANCE_METRIC': 'COSINE',
                        },
                    )
                ],
                definition=IndexDefinition(prefix=['doc:'], index_type=IndexType.HASH),
            )
            print(
                f'  Created FT HNSW index {self._index_name!r} (dim={self._embedding_dim}).\n'
                f'  Background scan started — valkey-search will index all doc:* keys.\n'
                f'  Query FT.INFO {self._index_name} to monitor progress before running semantic search.'
            )
        except Exception as exc:
            msg = str(exc).lower()
            if 'already exists' in msg or 'index already exists' in msg:
                pass
            else:
                print(f'  Warning: could not create FT search index: {exc}')

    @property
    def skipped(self) -> int:
        """Number of records skipped because they already existed (--skip-existing)."""
        return self._skipped

    def write_batch(self, records: list[dict]) -> None:
        pipe = self._client.pipeline(transaction=False)

        if self._skip_existing:
            # Pipeline EXISTS checks first, then filter
            for rec in records:
                pipe.exists(f'doc:{rec["xxh64"]}:{rec["blake2b"]}')
            exists_flags = pipe.execute()
            records = [r for r, exists in zip(records, exists_flags) if not exists]
            self._skipped += sum(exists_flags)
            pipe = self._client.pipeline(transaction=False)

        if self._semantic_dedup_threshold is not None and self._store_embedding and self._index_exists:
            from redis.commands.search.query import Query as _FTQuery

            deduped: list[dict] = []
            for rec in records:
                emb = rec.get('embedding')
                if emb is None:
                    deduped.append(rec)
                    continue
                query_bytes = struct.pack(f'<{len(emb)}f', *emb)
                q = _FTQuery('*=>[KNN 1 @embedding $vec AS __dist]').return_fields('__dist').dialect(2)
                try:
                    res = self._client.ft(self._index_name).search(q, query_params={'vec': query_bytes})
                    if res.docs and float(res.docs[0].__dist) < (1.0 - self._semantic_dedup_threshold):
                        self._skipped += 1
                        continue
                except Exception:
                    pass  # index empty or not yet available — write the record
                deduped.append(rec)
            records = deduped

        for rec in records:
            doc_key = f'doc:{rec["xxh64"]}:{rec["blake2b"]}'
            minhash_vals = rec['minhash']
            fields: dict = {
                'length': rec['length'],
                'unique_chars': rec['unique_chars'],
                'words': rec['words'],
                'sentences': rec['sentences'],
                'snippet': rec['snippet'],
                'minhash': struct.pack(f'<{len(minhash_vals)}Q', *minhash_vals),
            }
            if 'source' in rec:
                fields['source'] = rec['source']
            if 'source_file' in rec:
                fields['source_file'] = rec['source_file']
            if self._store_embedding and 'embedding' in rec:
                emb = rec['embedding']
                fields['embedding'] = struct.pack(f'<{len(emb)}f', *emb)

            pipe.hset(doc_key, mapping=fields)

            suffix = f'{rec["xxh64"]}:{rec["blake2b"]}'
            for band_id, band_hex in enumerate(minhash_lsh_band_hashes(minhash_vals, self._lsh_bands)):
                pipe.sadd(f'mhlsh:{band_id}:{band_hex}', suffix)

        pipe.execute()
        self._count += len(records)

    def close(self) -> None:
        if self._client is not None:
            if self._store_embedding and self._embedding_dim > 0 and not self._index_exists:
                self._ensure_search_index()
            self._client.close()
            self._client = None

    def count(self) -> int:
        return self._count


# ──────────────────────────────────────────────────────────────────
# Qdrant backend
# ──────────────────────────────────────────────────────────────────
class QdrantIndexWriter(IndexWriter):
    """
    Writes index records to a Qdrant vector database via qdrant-client.

    Data model
    ──────────
    Each record is stored as a Qdrant point:
        Vector   embedding_dim-float32 embedding — REQUIRED (do not use --no-embedding).
        Payload  xxh64, blake2b, length, unique_chars, words, sentences, snippet,
                 minhash (list[int]), lsh_bands (list[str]: "<band_id>_<hex>"),
                 source (omitted if absent)

    Payload indices (created on first open() when the collection is new):
        xxh64, blake2b  keyword — exact-dedup / point lookup
        source          keyword — provenance filter
        lsh_bands       keyword — near-dedup candidate retrieval
        length          integer — range filter

    Exact-dedup:   scroll with filter matching both xxh64 and blake2b keyword fields.
    Near-dedup:    filter on lsh_bands sharing any band tag → Python Jaccard on minhash payload.
    Semantic search: query_points(collection, query_vector, limit=k, query_filter=...)

    Notes:
        - Raises ValueError in write_batch() if records lack 'embedding'.
        - Collection created with on_disk=True vectors on first open(); reused otherwise.
        - On resume, _next_id continues from the collection's current points_count.
        - Upserts are idempotent — re-indexing with the same sequential IDs is safe.
    """

    def __init__(
        self,
        url: str = 'http://localhost:6333',
        collection: str = DEFAULT_DB_COLLECTION,
        embedding_dim: int = 384,
        lsh_bands: int = DEFAULT_LSH_BANDS,
        skip_existing: bool = False,
    ) -> None:
        self._url = url
        self._collection = collection
        self._embedding_dim = embedding_dim
        self._lsh_bands = lsh_bands
        self._skip_existing = skip_existing
        self._client = None
        self._next_id: int = 0
        self._count: int = 0
        self._skipped: int = 0

    def open(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, HnswConfigDiff, PayloadSchemaType, VectorParams

        self._client = QdrantClient(url=self._url)
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
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
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=schema,
                )
        else:
            info = self._client.get_collection(self._collection)
            self._next_id = info.points_count or 0

    @property
    def skipped(self) -> int:
        """Number of records skipped because they already existed (--skip-existing)."""
        return self._skipped

    def write_batch(self, records: list[dict]) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

        if records and 'embedding' not in records[0]:
            raise ValueError('QdrantIndexWriter requires embeddings. Do not use --no-embedding with --backend qdrant.')

        if self._skip_existing:
            filtered = []
            for rec in records:
                result, _ = self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key='xxh64', match=MatchValue(value=rec['xxh64'])),
                            FieldCondition(key='blake2b', match=MatchValue(value=rec['blake2b'])),
                        ]
                    ),
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                )
                if result:
                    self._skipped += 1
                else:
                    filtered.append(rec)
            records = filtered

        points = []
        for rec in records:
            minhash_vals = rec['minhash']
            lsh_tags = [f'{band_id}_{band_hex}' for band_id, band_hex in enumerate(minhash_lsh_band_hashes(minhash_vals, self._lsh_bands))]
            # Store minhash as base64-encoded little-endian packed bytes (~171 B) instead of
            # a JSON int array (~560 B). query_index.py decodes with base64 + struct.unpack.
            minhash_b64 = base64.b64encode(struct.pack(f'<{len(minhash_vals)}Q', *minhash_vals)).decode('ascii')
            payload = {
                'xxh64': rec['xxh64'],
                'blake2b': rec['blake2b'],
                'length': rec['length'],
                'unique_chars': rec['unique_chars'],
                'words': rec['words'],
                'sentences': rec['sentences'],
                'snippet': rec['snippet'],
                'minhash': minhash_b64,
                'lsh_bands': lsh_tags,
            }
            if 'source' in rec:
                payload['source'] = rec['source']
            if 'source_file' in rec:
                payload['source_file'] = rec['source_file']
            points.append(PointStruct(id=self._next_id, vector=rec['embedding'], payload=payload))
            self._next_id += 1

        self._client.upsert(collection_name=self._collection, points=points)
        self._count += len(records)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def count(self) -> int:
        return self._count


def make_writer(
    backend: str,
    output: str,
    compress: bool = False,
    *,
    lsh_bands: int = DEFAULT_LSH_BANDS,
    valkey_url: str = 'redis://localhost:6379/0',
    valkey_store_embedding: bool = True,
    valkey_embedding_dim: int = 0,
    valkey_index_name: str = DEFAULT_DB_COLLECTION,
    valkey_semantic_dedup_threshold: float | None = None,
    qdrant_url: str = 'http://localhost:6333',
    qdrant_collection: str = DEFAULT_DB_COLLECTION,
    embedding_dim: int = 384,
    skip_existing: bool = False,
) -> IndexWriter:
    """Factory: return an IndexWriter for the requested backend."""
    if backend == 'jsonl':
        return JsonlIndexWriter(output, compress=compress)
    if backend == 'valkey':
        return ValKeyIndexWriter(
            url=valkey_url,
            lsh_bands=lsh_bands,
            store_embedding=valkey_store_embedding,
            skip_existing=skip_existing,
            embedding_dim=valkey_embedding_dim,
            index_name=valkey_index_name,
            semantic_dedup_threshold=valkey_semantic_dedup_threshold,
        )
    if backend == 'qdrant':
        return QdrantIndexWriter(
            url=qdrant_url, collection=qdrant_collection, embedding_dim=embedding_dim, lsh_bands=lsh_bands, skip_existing=skip_existing
        )
    raise ValueError(f"Unknown backend: {backend!r}. Available: 'jsonl', 'valkey', 'qdrant'")


# ──────────────────────────────────────────────────────────────────
# Row-accounting dataclass
# ──────────────────────────────────────────────────────────────────
@dataclass
class FileStats:
    path: str
    rows_loaded: int = 0
    rows_dropped: int = 0  # len ≤ 10 or > 10 M
    rows_indexed: int = 0  # written to the index

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
# Embedding helpers
# ──────────────────────────────────────────────────────────────────
def _adaptive_encode_batch_size(texts: list[str], base_batch_size: int) -> int:
    """
    Compute a safe batch_size for model.encode() based on the longest text.

    GPU memory is proportional to batch_size × max_seq_len (due to padding all
    texts in a sub-batch to the longest). We scale batch_size down inversely
    with the longest text, using ~4 chars/token as a heuristic and capping at
    the model's 8192-token maximum.
    """
    if not texts:
        return base_batch_size
    max_chars = max(len(t) for t in texts)
    approx_tokens = min(max_chars // 4, 8192)
    if approx_tokens == 0:
        return base_batch_size
    # Keep batch_size × approx_tokens ≤ base_batch_size × 512
    bs = (base_batch_size * 512) // approx_tokens
    return max(1, min(bs, base_batch_size))


def safe_encode(model, texts: list[str], batch_size: int, normalize_embeddings: bool = True):
    """
    Encode texts with automatic OOM recovery.

    On torch.cuda.OutOfMemoryError the GPU cache is cleared, batch_size is
    halved, and encoding is retried. This repeats down to batch_size=1.
    If OOM persists at batch_size=1 the exception is re-raised.
    """
    import torch

    current_bs = batch_size
    while True:
        try:
            return model.encode(texts, batch_size=current_bs, normalize_embeddings=normalize_embeddings)
        except torch.cuda.OutOfMemoryError:
            if current_bs <= 1:
                raise
            new_bs = max(1, current_bs // 2)
            print(f'  [OOM] encode batch_size {current_bs} → {new_bs}, retrying…')
            torch.cuda.empty_cache()
            current_bs = new_bs


# ──────────────────────────────────────────────────────────────────
# Per-file pipeline
# ──────────────────────────────────────────────────────────────────
def _infer_dataset_format(path: str) -> str:
    """Return the datasets format string based on the file extension."""
    ext = Path(path).suffix.lower()
    if ext == '.parquet':
        return 'parquet'
    return 'json'  # covers .jsonl, .json, .ndjson, etc.


def process_file(
    input_path: str,
    writer: IndexWriter,
    source: str | None,
    model,  # SentenceTransformer | None
    text_key: str,
    num_perm: int,
    ngram_size: int,
    num_proc: int,
    compute_batch_size: int,
    write_batch_size: int,
    encode_batch_size: int = DEFAULT_ENCODE_BATCH_SIZE,
    cache_dir: str | None = None,
    no_cache: bool = False,
    source_file: str | None = None,
) -> FileStats:
    stats = FileStats(path=input_path)

    # ── 1. Load ────────────────────────────────────────────────────
    fmt = _infer_dataset_format(input_path)
    _tmp_cache: str | None = None
    if no_cache:
        _tmp_cache = tempfile.mkdtemp(prefix='index_texts_')
        effective_cache = _tmp_cache
    else:
        effective_cache = cache_dir  # None → HF default (~/.cache/huggingface/datasets)
    try:
        ds = load_dataset(fmt, split='train', data_files=input_path, cache_dir=effective_cache)
        if text_key not in ds.column_names:
            raise ValueError(f"Field '{text_key}' not found in {input_path}. Available columns: {ds.column_names}")
        stats.rows_loaded = len(ds)
        print(f'  Loaded:   {stats.rows_loaded:,} rows')

        # ── 2. Compute (multiprocessing, CPU-heavy) ────────────────────
        # Text is kept in the dataset here so the embedding pass can read it.
        # It is never written to the index.
        ds = ds.map(
            compute_batch,
            batched=True,
            batch_size=compute_batch_size,
            writer_batch_size=compute_batch_size,
            num_proc=num_proc,
            remove_columns=ds.column_names,
            features=COMPUTE_FEATURES,
            fn_kwargs=dict(text_key=text_key, num_perm=num_perm, ngram_size=ngram_size),
            desc='  Computing stats & hashes',
        )
        stats.rows_dropped = stats.rows_loaded - len(ds)
        print(f'  After compute: {len(ds):,} rows' + (f'  ({stats.rows_dropped:,} dropped)' if stats.rows_dropped else ''))

        # ── 3. Embedding + write (batched; GPU or CPU) ─────────────────
        # Iterate the dataset in batches. Optionally encode text for embeddings,
        # then write index records — without the text field.

        pbar = tqdm(total=len(ds), desc='  Writing index', unit='doc', dynamic_ncols=True)
        for batch in ds.iter(batch_size=write_batch_size):
            texts = batch['text']

            if model is not None:
                # Sort by character length so encode()'s internal sub-batches
                # group similar-length texts together, minimising padding waste
                # (GPU memory ∝ batch_size × max_seq_len_in_sub_batch).
                order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
                sorted_texts = [texts[i] for i in order]

                # Scale batch_size down for batches containing long texts, then
                # fall back further via OOM-retry if the GPU still runs out.
                enc_bs = _adaptive_encode_batch_size(sorted_texts, encode_batch_size)
                raw_sorted = safe_encode(model, sorted_texts, batch_size=enc_bs)

                # Restore original order so record assembly below stays aligned.
                emb_list = [None] * len(texts)
                for sorted_i, orig_i in enumerate(order):
                    emb_list[orig_i] = raw_sorted[sorted_i]
                embeddings = [[round(float(x), 5) for x in vec] for vec in emb_list]
            else:
                embeddings = None

            records: list[dict] = []
            for i in range(len(texts)):
                rec: dict = {}
                if source is not None:
                    rec['source'] = source
                if source_file is not None:
                    rec['source_file'] = source_file
                rec.update(
                    {
                        'xxh64': batch['xxh64'][i],
                        'blake2b': batch['blake2b'][i],
                        'length': batch['length'][i],
                        'unique_chars': batch['unique_chars'][i],
                        'words': batch['words'][i],
                        'sentences': batch['sentences'][i],
                        'snippet': batch['snippet'][i],
                        'minhash': batch['minhash'][i],
                    }
                )
                if embeddings is not None:
                    rec['embedding'] = embeddings[i]
                records.append(rec)

            writer.write_batch(records)
            stats.rows_indexed += len(records)
            pbar.update(len(records))

        pbar.close()
        stats.validate()
    finally:
        if _tmp_cache is not None:
            shutil.rmtree(_tmp_cache, ignore_errors=True)
    return stats


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Index JSONL documents → metadata-only index (no text stored)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--input-glob',
        required=True,
        help="Glob pattern for input files (e.g. 'raw/*.jsonl' or 'raw/*.parquet')",
    )
    p.add_argument('--output', default=None, help='Output file path (required for jsonl backend)')
    p.add_argument('--backend', default='jsonl', choices=['jsonl', 'valkey', 'qdrant'], help='Storage backend')
    p.add_argument('--compress', action='store_true', help='Gzip-compress the output (JSONL backend)')
    p.add_argument(
        '--source',
        default=None,
        help='Optional provenance label added to every record (e.g. "British Library")',
    )
    p.add_argument('--text-key', default=DEFAULT_TEXT_KEY, help='JSON field containing document text')
    p.add_argument('--num-perm', type=int, default=DEFAULT_NUM_PERM, help='MinHash permutations')
    p.add_argument('--ngram-size', type=int, default=DEFAULT_NGRAM_SIZE, help='Word n-gram size for MinHash')
    p.add_argument('--no-embedding', action='store_true', help='Skip sentence embedding computation')
    p.add_argument(
        '--embedding-model',
        default=DEFAULT_EMBEDDING_MODEL,
        help='SentenceTransformer model name or local path',
    )
    p.add_argument(
        '--num-proc',
        type=int,
        default=os.cpu_count() or 4,
        help='Parallel workers for the compute phase',
    )
    p.add_argument(
        '--compute-batch-size',
        type=int,
        default=DEFAULT_COMPUTE_BATCH_SIZE,
        help='Batch size for the HF datasets.map() compute phase',
    )
    p.add_argument(
        '--write-batch-size',
        type=int,
        default=DEFAULT_WRITE_BATCH_SIZE,
        help='Iteration batch size for the write phase when embeddings are disabled',
    )
    p.add_argument(
        '--encode-batch-size',
        type=int,
        default=DEFAULT_ENCODE_BATCH_SIZE,
        help=(
            'Base batch size for model.encode(). Automatically reduced for batches '
            'containing long texts (GPU memory ∝ batch_size × max_seq_len due to '
            'padding). OOM-retry halves it further if needed.'
        ),
    )
    cache_group = p.add_mutually_exclusive_group()
    cache_group.add_argument(
        '--cache-dir',
        default=None,
        help='HF datasets cache directory (default: ~/.cache/huggingface/datasets)',
    )
    cache_group.add_argument(
        '--no-cache',
        action='store_true',
        help='Use a temporary cache dir per file and delete it immediately after processing',
    )
    # ── Valkey / Qdrant backends ───────────────────────────────────────────
    p.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip records whose composite key (xxh64+blake2b) already exists in the backend (valkey and qdrant)',
    )
    p.add_argument(
        '--no-valkey-embedding',
        action='store_true',
        help=(
            'Do not store embeddings in the Valkey Hash. Disables HNSW index creation '
            'and semantic search on the Valkey backend (saves ~1.5\u202fKB/record; '
            'use Qdrant for semantic search instead).'
        ),
    )
    p.add_argument(
        '--valkey-index',
        default=DEFAULT_DB_COLLECTION,
        help='Name of the valkey-search FT index for HNSW vector search (valkey backend)',
    )
    p.add_argument(
        '--semantic-dedup-threshold',
        type=float,
        default=None,
        metavar='FLOAT',
        help=(
            'Skip new records whose nearest-neighbour cosine similarity in the HNSW index '
            'exceeds this value (0.0–1.0). Uses FT KNN-1; requires valkey-bundle and '
            '--backend valkey. Approximate — HNSW may miss some neighbours.'
        ),
    )
    p.add_argument(
        '--valkey-url',
        default='redis://localhost:6379/0',
        help='Valkey/Redis connection URL (valkey backend)',
    )
    p.add_argument(
        '--lsh-bands',
        type=int,
        default=DEFAULT_LSH_BANDS,
        help=(
            'MinHash LSH band count (valkey and qdrant backends). '
            'num_perm must be divisible by this value. '
            'Fewer bands → higher Jaccard threshold (stricter near-dedup).'
        ),
    )
    p.add_argument(
        '--qdrant-url',
        default='http://localhost:6333',
        help='Qdrant server URL (qdrant backend)',
    )
    p.add_argument(
        '--qdrant-collection',
        default=DEFAULT_DB_COLLECTION,
        help='Qdrant collection name (qdrant backend)',
    )
    p.add_argument(
        '--embedding-dim',
        type=int,
        default=384,
        help=(
            'Embedding vector dimension for Qdrant collection creation. '
            'Automatically set from the loaded model when --no-embedding is not used.'
        ),
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if args.backend == 'jsonl' and args.output is None:
        raise SystemExit('--output is required for the jsonl backend')
    if args.num_perm % args.lsh_bands != 0:
        raise SystemExit(f'--lsh-bands {args.lsh_bands} must evenly divide --num-perm {args.num_perm}')

    files = sorted(glob.glob(args.input_glob, recursive=True))
    if not files:
        raise SystemExit(f'No files matched: {args.input_glob}')
    print(f'Found {len(files)} input file(s)')

    # Load embedding model once before processing any file
    model = None
    embedding_dim: int = args.embedding_dim
    if not args.no_embedding:
        from sentence_transformers import SentenceTransformer

        print(f'Loading embedding model: {args.embedding_model}')
        # For GPUs, it's worth adding model_kwargs={'torch_dtype': 'bfloat16'}
        model = SentenceTransformer(args.embedding_model)
        embedding_dim = model.get_embedding_dimension() or args.embedding_dim
        print(f'Embedding dimension: {embedding_dim}')

    if args.backend == 'jsonl':
        os.makedirs(Path(args.output).parent, exist_ok=True)

    grand_loaded = 0
    grand_dropped = 0
    grand_indexed = 0
    skipped: list[str] = []

    with make_writer(
        args.backend,
        args.output or '',
        args.compress,
        lsh_bands=args.lsh_bands,
        valkey_url=args.valkey_url,
        valkey_store_embedding=not args.no_valkey_embedding,
        valkey_embedding_dim=embedding_dim if not args.no_valkey_embedding else 0,
        valkey_index_name=args.valkey_index,
        valkey_semantic_dedup_threshold=args.semantic_dedup_threshold,
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.qdrant_collection,
        embedding_dim=embedding_dim,
        skip_existing=args.skip_existing,
    ) as writer:
        for input_path in files:
            print(f'\n{"=" * 60}')
            print(f'Processing: {input_path}')
            print(f'{"=" * 60}')

            stats = process_file(
                input_path=input_path,
                writer=writer,
                source=args.source,
                source_file=Path(input_path).name,
                model=model,
                text_key=args.text_key,
                num_perm=args.num_perm,
                ngram_size=args.ngram_size,
                num_proc=args.num_proc,
                compute_batch_size=args.compute_batch_size,
                write_batch_size=args.write_batch_size,
                encode_batch_size=args.encode_batch_size,
                cache_dir=args.cache_dir,
                no_cache=args.no_cache,
            )
            stats.print_summary()

            grand_loaded += stats.rows_loaded
            grand_dropped += stats.rows_dropped
            grand_indexed += stats.rows_indexed

    # ── Final summary ──────────────────────────────────────────────
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    writer_skipped = getattr(writer, 'skipped', 0)
    if skipped:
        print(f'  Skipped files:  {len(skipped):>14,}')
    if writer_skipped:
        print(f'  Already exists: {writer_skipped:>14,}  (--skip-existing)')
    print(
        f'  Grand loaded:   {grand_loaded:>14,}\n'
        f'  Grand dropped:  {grand_dropped:>14,}  (len ≤ 10 or > 10 M)\n'
        f'  Grand indexed:  {grand_indexed:>14,}\n'
        f'  Writer total:   {writer.count():>14,}\n'
        f'  Backend:        {args.backend}'
    )
    if args.backend == 'jsonl':
        print(f'  Output:         {args.output}')
    elif args.backend == 'valkey':
        print(f'  Valkey URL:     {args.valkey_url}')
    elif args.backend == 'qdrant':
        print(f'  Qdrant URL:     {args.qdrant_url}')
        print(f'  Collection:     {args.qdrant_collection}')
    if not args.no_embedding:
        print(f'  Embedding model: {args.embedding_model}')
    if args.source:
        print(f'  Source label:    {args.source}')

    cache_loc = args.cache_dir or '~/.cache/huggingface/datasets'
    print('\n💡  HF datasets Arrow cache may be large. Clean up with:')
    print(f'    rm -rf {cache_loc}')


if __name__ == '__main__':
    main()
