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

Examples
────────
  # Index files
  python -m dataset index --input-glob "raw/*.jsonl" --source "British Library"

  # Re-index (existing docs are skipped by default)
  python -m dataset index --input-glob "raw/*.jsonl"

  # Dataset stats
  python -m dataset stats

  # Look up a record
  python -m dataset get --text "The quick brown fox"

  # Near-duplicate search
  python -m dataset find-similar --mode similar --text "some text" --threshold 0.4

  # Semantic search
  python -m dataset find-similar --mode semantic --text "some text" --limit 10
"""

import argparse
import os
import warnings

from .config import (
    DEFAULT_COMPUTE_BATCH_SIZE,
    DEFAULT_DB_PATH,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LSH_BANDS,
    DEFAULT_NGRAM_SIZE,
    DEFAULT_NUM_PERM,
    DEFAULT_TEXT_KEY,
    DEFAULT_WRITE_BATCH_SIZE,
)
from .index import cmd_index
from .query import cmd_find_similar, cmd_get
from .report import cmd_report
from .stats import cmd_stats

warnings.filterwarnings('ignore', message='lance is not fork-safe', category=UserWarning)


def parse_args(argv: list[str]) -> argparse.Namespace:
    root = argparse.ArgumentParser(
        description='Index and query text documents using Lance (Open Lakehouse Format).',
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
    p_index.add_argument(
        '--calc-minhash',
        type=str,
        choices=['true', 'false', 'yes', 'no'],
        default='false',
        help='Compute MinHash & LSH bands (default: false)',
    )
    p_index.add_argument(
        '--calc-embeds',
        type=str,
        choices=['true', 'false', 'yes', 'no'],
        default='false',
        help='Compute embeddings (default: false)',
    )
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
    _p_stats = sub.add_parser('stats', help='Print dataset schema and statistics')
    # ── report ─────────────────────────────────────────────────
    p_report = sub.add_parser('report', help='Generate a Markdown report with charts from the Lance dataset')
    p_report.add_argument('--out', '-o', default='dataset_report', help='Output directory (default: ./dataset_report)')
    p_report.add_argument('--limit', type=int, default=None, help='Maximum number of documents to process')
    p_report.add_argument('--bins', type=int, default=60, help='Number of histogram bins (default: 60)')
    p_report.add_argument('--skip-vocab', action='store_true', help='Skip vocabulary analysis (faster, no text scan)')

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

    return root.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    if hasattr(args, 'calc_minhash'):
        args.calc_minhash = args.calc_minhash in ['true', 'yes']
    if hasattr(args, 'calc_embeds'):
        args.calc_embeds = args.calc_embeds in ['true', 'yes']
    if args.num_perm % args.lsh_bands != 0:
        raise SystemExit(f'--lsh-bands {args.lsh_bands} must evenly divide --num-perm {args.num_perm}')

    dispatch = {
        'index': cmd_index,
        'stats': cmd_stats,
        'get': cmd_get,
        'find-similar': cmd_find_similar,
        'report': cmd_report,
    }
    try:
        return dispatch[args.subcommand](args)
    except Exception as err:
        raise SystemExit(f'Error: {err}') from err
