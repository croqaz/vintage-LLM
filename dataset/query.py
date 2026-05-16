#
# Semantic and clustering utilities for Lance datasets.
#

import argparse
import hashlib
import os
import re

import lance
import numpy as np
import xxhash
from datasketch import MinHash
from sentence_transformers import SentenceTransformer

from .config import DEVICE
from .fields import compute_minhash, embed_texts, minhash_lsh_band_hashes


# ─────────────────────────────────────────────────────────
# ID resolution helper
# ─────────────────────────────────────────────────────────
def resolve_id(args: argparse.Namespace) -> str | None:
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


# ─────────────────────────────────────────────────────────
# Pretty-print a row dict
# ─────────────────────────────────────────────────────────
def print_doc_row(row: dict, indent: int = 2) -> None:
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


# ─────────────────────────────────────────────────────────
# Subcommand: sql
# ─────────────────────────────────────────────────────────
def cmd_sql(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.db_path):
        raise SystemExit(f'Dataset not found: {args.db_path}')

    ds = lance.dataset(args.db_path)
    columns = [c.strip() for c in args.columns.split(',')]

    scan_kwargs: dict = {
        'columns': columns,
        'filter': args.query,
    }
    if args.limit is not None and args.limit > 0:
        scan_kwargs['limit'] = args.limit

    print(f'Query: {args.query}  columns={columns}  limit={args.limit}\n')
    tbl = ds.to_table(**scan_kwargs)
    rows = tbl.to_pylist()

    if not rows:
        print('No results.')
        return

    for row in rows:
        print_doc_row(row)
        print()

    print(f'{len(rows)} row(s) found.\n')


# ─────────────────────────────────────────────────────────
# Subcommand: get
# ─────────────────────────────────────────────────────────
def cmd_get(args: argparse.Namespace) -> None:
    doc_id = resolve_id(args)
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
    print_doc_row(row)
    print(f'{"─" * 60}')


# ─────────────────────────────────────────────────────────
# Subcommand: find-similar
# ─────────────────────────────────────────────────────────
def cmd_find_similar(args: argparse.Namespace) -> None:
    mode = args.mode
    if mode == 'exact':
        find_exact(args)
    elif mode == 'similar':
        find_similar(args)
    elif mode == 'semantic':
        find_semantic(args)
    else:
        raise SystemExit(f'Unknown mode: {mode!r}')


# ── exact ─────────────────────────────────────────────────────────
def find_exact(args: argparse.Namespace) -> None:
    doc_id = resolve_id(args)
    if doc_id is None:
        raise SystemExit('Provide --id or --text for exact mode!')

    ds = lance.dataset(args.db_path)
    print(f'Exact lookup: {doc_id}')
    tbl = ds.to_table(filter=f"id = '{doc_id}'", limit=1)
    if tbl.num_rows > 0:
        print('Found (exact match):')
        print(f'{"─" * 60}')
        print_doc_row(tbl.to_pylist()[0])
        print(f'{"─" * 60}')
    else:
        print('Not found.')


# ── semantic ──────────────────────────────────────────────────────
def find_semantic(args: argparse.Namespace) -> None:
    if not args.text and not args.id:
        raise SystemExit('Provide --text or --id for semantic mode!')

    ds = lance.dataset(args.db_path)

    print(f'Semantic search  (limit={args.limit})')
    print(f'Loading embedding model: {args.embed_model}')
    model = SentenceTransformer(args.embed_model, device=DEVICE).half()

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

    scan_kwargs: dict = {
        'columns': ['id', 'text', 'length', 'source', '_distance'],
        'nearest': nearest_params,
        'limit': args.limit * 4,
    }
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
        length = row.get('length', 0)
        source = row.get('source', '') or ''
        snippet = text[:50] + '[...]' + text[-50:] if len(text) > 100 else text
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


# ── similar ───────────────────────────────────────────────────────
def find_similar(args: argparse.Namespace) -> None:
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
            length = row.get('length', 0)
            source = row.get('source', '') or ''
            snippet = text[:50] + '[...]' + text[-50:] if len(text) > 100 else text
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
