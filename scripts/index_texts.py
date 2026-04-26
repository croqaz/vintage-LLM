"""
Index JSONL documents → metadata-only index.

Pipeline per input file:
  1. Load JSONL          → HF Dataset
  2. Compute stats+hash  → HF Dataset   (rows with len ≤ 10 or > 10 M dropped)
  3. Embedding pass      → model.encode() on text batches (optional)
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
  embedding     list[float]— sentence embedding, float32 rounded to 6 dp (omitted if --no-embedding)

Storage backends (--backend):
  jsonl    Newline-delimited JSON, optionally gzip-compressed (--compress). Default.
           Future: parquet, lmdb, redis — implement IndexWriter ABC.

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
import glob
import gzip
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import orjson
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
DEFAULT_WRITE_BATCH_SIZE = 512
DEFAULT_EMBEDDING_MODEL = 'ibm-granite/granite-embedding-small-english-r2'

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

        tokens = text.split()

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


def make_writer(backend: str, output: str, compress: bool) -> IndexWriter:
    """Factory: return an IndexWriter for the requested backend."""
    if backend == 'jsonl':
        return JsonlIndexWriter(output, compress=compress)
    raise ValueError(f"Unknown backend: {backend!r}. Available: 'jsonl'")


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
# Per-file pipeline
# ──────────────────────────────────────────────────────────────────
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
) -> FileStats:
    stats = FileStats(path=input_path)

    # ── 1. Load ────────────────────────────────────────────────────
    ds = load_dataset('json', split='train', data_files=input_path)
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

    for batch in ds.iter(batch_size=write_batch_size):
        texts = batch['text']

        if model is not None:
            raw_embs = model.encode(
                texts,
                normalize_embeddings=True,
            )
            embeddings = [[round(float(x), 6) for x in vec] for vec in raw_embs]
        else:
            embeddings = None

        records: list[dict] = []
        for i in range(len(texts)):
            rec: dict = {}
            if source is not None:
                rec['source'] = source
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

    stats.validate()
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
        help="Glob pattern for JSONL input files (e.g. 'raw/*.jsonl')",
    )
    p.add_argument('--output', required=True, help='Output index file path')
    p.add_argument('--backend', default='jsonl', choices=['jsonl'], help='Storage backend')
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

    # Load embedding model once before processing any file
    model = None
    if not args.no_embedding:
        from sentence_transformers import SentenceTransformer

        print(f'Loading embedding model: {args.embedding_model}')
        model = SentenceTransformer(args.embedding_model, model_kwargs={'torch_dtype': 'float16'})
        print(f'Embedding dimension: {model.get_sentence_embedding_dimension()}')

    os.makedirs(Path(args.output).parent, exist_ok=True)

    grand_loaded = 0
    grand_dropped = 0
    grand_indexed = 0
    skipped: list[str] = []

    with make_writer(args.backend, args.output, args.compress) as writer:
        for input_path in files:
            print(f'\n{"=" * 60}')
            print(f'Processing: {input_path}')
            print(f'{"=" * 60}')

            stats = process_file(
                input_path=input_path,
                writer=writer,
                source=args.source,
                model=model,
                text_key=args.text_key,
                num_perm=args.num_perm,
                ngram_size=args.ngram_size,
                num_proc=args.num_proc,
                compute_batch_size=args.compute_batch_size,
                write_batch_size=args.write_batch_size,
            )
            stats.print_summary()

            grand_loaded += stats.rows_loaded
            grand_dropped += stats.rows_dropped
            grand_indexed += stats.rows_indexed

    # ── Final summary ──────────────────────────────────────────────
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    if skipped:
        print(f'  Skipped files:  {len(skipped):>14,}')
    print(
        f'  Grand loaded:   {grand_loaded:>14,}\n'
        f'  Grand dropped:  {grand_dropped:>14,}  (len ≤ 10 or > 10 M)\n'
        f'  Grand indexed:  {grand_indexed:>14,}\n'
        f'  Writer total:   {writer.count():>14,}\n'
        f'  Output:         {args.output}'
    )
    if not args.no_embedding:
        print(f'  Embedding model: {args.embedding_model}')
    if args.source:
        print(f'  Source label:    {args.source}')


if __name__ == '__main__':
    main()
