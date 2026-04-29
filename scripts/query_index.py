"""
Query and inspect the text index stored in a Valkey or Qdrant backend.

Subcommands
───────────
  stats          Print aggregate statistics for all indexed records.
  get            Retrieve and print a single record by composite ID or raw text.
  find-similar   Find duplicate or similar texts using three strategies:
                   exact     — identical text (O(1) hash lookup)
                   minhash   — near-duplicates via MinHash LSH + Jaccard verification
                   semantic  — semantically similar texts via embedding ANN search (Qdrant
                               or Valkey with valkey-search HNSW index; brute-force cosine
                               scan available via --brute-force)

Usage examples
──────────────
  # Backend stats
  python query_index.py --backend valkey stats
  python query_index.py --backend qdrant --qdrant-collection text_index stats

  # Look up a record
  python query_index.py --backend valkey get --text "The quick brown fox"
  python query_index.py --backend qdrant get --id "a1b2c3d4e5f6a7b8:abcdef..."

  # Find exact duplicates
  python query_index.py --backend valkey find-similar --mode exact --text "some text"

  # Find near-duplicates via MinHash (both backends)
  python query_index.py --backend valkey find-similar --mode minhash --text "some text" --threshold 0.4

  # Find semantically similar texts (Qdrant or Valkey with valkey-search)
  python query_index.py --backend valkey find-similar --mode semantic --text "some text" --limit 10
  python query_index.py --backend qdrant find-similar --mode semantic --text "some text" --limit 10
"""

import argparse
import base64
import math
import struct
import sys
import textwrap

import numpy as np
from datasketch import MinHash

# Import shared constants and helpers from index_texts.py (same directory).
# sys.path manipulation is not needed when running from the scripts/ directory
# or when the package is installed.
try:
    from index_texts import (
        DEFAULT_DB_COLLECTION,
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_LSH_BANDS,
        DEFAULT_NGRAM_SIZE,
        DEFAULT_NUM_PERM,
        compute_minhash,
        compute_text_hashes,
        minhash_lsh_band_hashes,
    )
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).parent))
    from index_texts import (
        DEFAULT_DB_COLLECTION,
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_LSH_BANDS,
        DEFAULT_NGRAM_SIZE,
        DEFAULT_NUM_PERM,
        compute_minhash,
        compute_text_hashes,
        minhash_lsh_band_hashes,
    )

# ──────────────────────────────────────────────────────────────────
# Shared decode helpers
# ──────────────────────────────────────────────────────────────────


def _unpack_minhash(data: bytes, num_perm: int = DEFAULT_NUM_PERM) -> MinHash:
    """Reconstruct a datasketch MinHash from little-endian packed uint64 bytes."""
    vals = struct.unpack(f'<{num_perm}Q', data)
    mh = MinHash(num_perm=num_perm)
    mh.hashvalues = np.array(vals, dtype=np.uint64)
    return mh


def _b64_to_minhash(b64str: str, num_perm: int = DEFAULT_NUM_PERM) -> MinHash:
    """Reconstruct a datasketch MinHash from a base64-encoded packed-bytes string (Qdrant payload)."""
    return _unpack_minhash(base64.b64decode(b64str), num_perm)


def _unpack_embedding(data: bytes) -> list[float]:
    """Decode little-endian float32 packed bytes back to a list of floats."""
    n = len(data) // 4
    return list(struct.unpack(f'<{n}f', data))


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


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
# Valkey helpers
# ──────────────────────────────────────────────────────────────────


def _valkey_connect(url: str):
    import redis

    client = redis.from_url(url, decode_responses=False)
    client.ping()
    return client


def _valkey_decode_hash(raw: dict, num_perm: int) -> dict:
    """Convert a raw HGETALL response (bytes→bytes) to a usable Python dict."""
    rec: dict = {}
    str_fields = {b'source', b'source_file', b'snippet'}
    int_fields = {b'length', b'unique_chars', b'words', b'sentences'}
    for k, v in raw.items():
        if k in str_fields:
            rec[k.decode()] = v.decode('utf-8', errors='replace')
        elif k in int_fields:
            rec[k.decode()] = int(v)
        elif k == b'minhash':
            mh = _unpack_minhash(v, num_perm)
            rec['minhash'] = mh.hashvalues.tolist()
        elif k == b'embedding':
            rec['embedding'] = _unpack_embedding(v)
        else:
            try:
                rec[k.decode()] = v.decode('utf-8', errors='replace')
            except Exception:
                rec[k.decode()] = repr(v)
    return rec


def _valkey_get_record(client, doc_id: str, num_perm: int) -> dict | None:
    raw = client.hgetall(f'doc:{doc_id}')
    if not raw:
        return None
    rec = _valkey_decode_hash(raw, num_perm)
    rec['_id'] = doc_id
    return rec


# ──────────────────────────────────────────────────────────────────
# Qdrant helpers
# ──────────────────────────────────────────────────────────────────


def _qdrant_connect(url: str, collection: str):
    from qdrant_client import QdrantClient

    client = QdrantClient(url=url)
    return client


def _qdrant_decode_payload(payload: dict, num_perm: int) -> dict:
    """Decode a Qdrant point payload to a usable dict (base64 minhash → list[int])."""
    rec = dict(payload)
    if 'minhash' in rec and isinstance(rec['minhash'], str):
        try:
            mh = _b64_to_minhash(rec['minhash'], num_perm)
            rec['minhash'] = mh.hashvalues.tolist()
        except Exception:
            pass  # leave as-is if format is unexpected
    return rec


def _qdrant_get_record(client, collection: str, doc_id: str, num_perm: int) -> dict | None:
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
    rec = _qdrant_decode_payload(pt.payload or {}, num_perm)
    rec['_id'] = doc_id
    rec['_point_id'] = pt.id
    return rec


# ══════════════════════════════════════════════════════════════════
# Subcommand: stats
# ══════════════════════════════════════════════════════════════════


def cmd_stats(args: argparse.Namespace) -> None:
    print(f'Backend: {args.backend}')

    if args.backend == 'valkey':
        _stats_valkey(args)
    else:
        _stats_qdrant(args)


def _stats_valkey(args: argparse.Namespace) -> None:
    client = _valkey_connect(args.valkey_url)

    total = 0
    has_embedding = 0
    has_source_file = 0
    sources: dict[str, int] = {}
    s_length = _RunningStats()
    s_words = _RunningStats()
    s_sentences = _RunningStats()

    cursor = 0
    scan_count = 1_000
    print('Scanning Valkey (doc:* keys)…')
    while True:
        cursor, keys = client.scan(cursor=cursor, match='doc:*', count=scan_count)
        if keys:
            # Pipeline HMGET for all metadata fields (skip minhash/embedding bytes)
            pipe = client.pipeline(transaction=False)
            fields = [b'length', b'words', b'sentences', b'source', b'source_file', b'embedding']
            for key in keys:
                pipe.hmget(key, fields)
            results = pipe.execute()

            for vals in results:
                length_b, words_b, sentences_b, source_b, sf_b, emb_b = vals
                total += 1
                if length_b is not None:
                    s_length.update(int(length_b))
                if words_b is not None:
                    s_words.update(int(words_b))
                if sentences_b is not None:
                    s_sentences.update(int(sentences_b))
                if source_b is not None:
                    src = source_b.decode('utf-8', errors='replace')
                    sources[src] = sources.get(src, 0) + 1
                if sf_b is not None:
                    has_source_file += 1
                if emb_b is not None:
                    has_embedding += 1

        if cursor == 0:
            break

    # LSH band key count
    lsh_cursor = 0
    lsh_count = 0
    while True:
        lsh_cursor, lsh_keys = client.scan(cursor=lsh_cursor, match='mhlsh:*', count=scan_count)
        lsh_count += len(lsh_keys)
        if lsh_cursor == 0:
            break

    _print_stats_table(total, has_embedding, has_source_file, sources, s_length, s_words, s_sentences)
    print(f'\n  LSH band keys (mhlsh:*):     {lsh_count:>12,}')

    # FT index info (requires valkey-bundle with valkey-search)
    try:
        ft_info = client.ft(args.valkey_index).info()
        num_indexed = ft_info.get('num_docs', 'n/a')
        print(f'  FT HNSW indexed docs:        {int(num_indexed):>12,}')
    except Exception:
        pass  # valkey-search not available or index not yet created

    client.close()


def _stats_qdrant(args: argparse.Namespace) -> None:
    client = _qdrant_connect(args.qdrant_url, args.qdrant_collection)
    info = client.get_collection(args.qdrant_collection)
    print(f'Collection: {args.qdrant_collection}')
    print(f'  Status:          {info.status}')
    print(f'  Points total:    {info.points_count or 0:>12,}')
    print(f'  Vectors on disk: {info.config.params.vectors.on_disk}')  # type: ignore[union-attr]

    total = 0
    has_embedding = 0  # always present in Qdrant, but track for completeness
    has_source_file = 0
    sources: dict[str, int] = {}
    s_length = _RunningStats()
    s_words = _RunningStats()
    s_sentences = _RunningStats()

    offset = None
    print('Scrolling Qdrant (payload only)…')
    while True:
        results, next_offset = client.scroll(
            collection_name=args.qdrant_collection,
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
            has_embedding += 1  # all Qdrant points have vectors by design
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

    _print_stats_table(total, has_embedding, has_source_file, sources, s_length, s_words, s_sentences)
    client.close()


def _print_stats_table(
    total: int,
    has_embedding: int,
    has_source_file: int,
    sources: dict,
    s_length: _RunningStats,
    s_words: _RunningStats,
    s_sentences: _RunningStats,
) -> None:
    print(f'\n{"─" * 60}')
    print(f'  Total records:               {total:>12,}')
    print(f'  Records with embedding:      {has_embedding:>12,}')
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


# ══════════════════════════════════════════════════════════════════
# Subcommand: get
# ══════════════════════════════════════════════════════════════════


def cmd_get(args: argparse.Namespace) -> None:
    doc_id = _resolve_id(args)
    if doc_id is None:
        raise SystemExit('Provide --id xxh64:blake2b or --text "some text"')

    print(f'Looking up: {doc_id}')

    if args.backend == 'valkey':
        client = _valkey_connect(args.valkey_url)
        rec = _valkey_get_record(client, doc_id, args.num_perm)
        client.close()
    else:
        client = _qdrant_connect(args.qdrant_url, args.qdrant_collection)
        rec = _qdrant_get_record(client, args.qdrant_collection, doc_id, args.num_perm)
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

    if args.backend == 'valkey':
        client = _valkey_connect(args.valkey_url)
        exists = bool(client.exists(f'doc:{doc_id}'))
        if exists:
            rec = _valkey_get_record(client, doc_id, args.num_perm)
        client.close()
        if exists:
            print('Found (exact match):')
            print(f'{"─" * 60}')
            _print_record(rec)
            print(f'{"─" * 60}')
        else:
            print('Not found.')
    else:
        client = _qdrant_connect(args.qdrant_url, args.qdrant_collection)
        rec = _qdrant_get_record(client, args.qdrant_collection, doc_id, args.num_perm)
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
        # Fetch stored minhash from the backend
        if args.backend == 'valkey':
            client = _valkey_connect(args.valkey_url)
            raw_mh = client.hget(f'doc:{args.id}', 'minhash')
            client.close()
            if raw_mh is None:
                raise SystemExit(f'Record not found or has no minhash: {args.id}')
            mh = _unpack_minhash(raw_mh, args.num_perm)
        else:
            client = _qdrant_connect(args.qdrant_url, args.qdrant_collection)
            rec = _qdrant_get_record(client, args.qdrant_collection, args.id, args.num_perm)
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
    query_mh, band_hashes = _get_query_minhash(args)
    query_id = args.id or '<query text>'
    print(f'MinHash near-duplicate search  (threshold≥{args.threshold}, limit={args.limit})')
    print(f'Query: {query_id}')

    if args.backend == 'valkey':
        _find_minhash_valkey(args, query_mh, band_hashes)
    else:
        _find_minhash_qdrant(args, query_mh, band_hashes)


def _find_minhash_valkey(args: argparse.Namespace, query_mh: MinHash, band_hashes: list[str]) -> None:
    client = _valkey_connect(args.valkey_url)

    # SUNION across all matching band Sets to collect candidate doc-id suffixes
    band_keys = [f'mhlsh:{band_id}:{bh}' for band_id, bh in enumerate(band_hashes)]
    candidates: set[str] = set()
    for bk in band_keys:
        members = client.smembers(bk)
        candidates.update(m.decode() for m in members)

    if not candidates:
        print('No candidates found via LSH bands.')
        client.close()
        return

    print(f'  LSH candidates: {len(candidates):,}')

    # Fetch minhash bytes for all candidates via pipeline
    pipe = client.pipeline(transaction=False)
    cand_list = list(candidates)
    for cid in cand_list:
        pipe.hmget(f'doc:{cid}', [b'minhash', b'snippet', b'length', b'source'])
    results = pipe.execute()
    client.close()

    matches: list[tuple[float, str, str, int, str]] = []  # (jaccard, id, snippet, length, source)
    for cid, vals in zip(cand_list, results):
        mh_bytes, snippet_b, length_b, source_b = vals
        if mh_bytes is None:
            continue
        cand_mh = _unpack_minhash(mh_bytes, args.num_perm)
        jaccard = query_mh.jaccard(cand_mh)
        if jaccard >= args.threshold:
            snippet = snippet_b.decode('utf-8', errors='replace') if snippet_b else ''
            length = int(length_b) if length_b else 0
            source = source_b.decode('utf-8', errors='replace') if source_b else ''
            matches.append((jaccard, cid, snippet, length, source))

    _print_minhash_results(matches, args.limit)


def _find_minhash_qdrant(args: argparse.Namespace, query_mh: MinHash, band_hashes: list[str]) -> None:
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    client = _qdrant_connect(args.qdrant_url, args.qdrant_collection)

    # Scroll with should-filter on lsh_bands keyword tags
    band_tags = [f'{band_id}_{bh}' for band_id, bh in enumerate(band_hashes)]
    scroll_filter = Filter(should=[FieldCondition(key='lsh_bands', match=MatchAny(any=band_tags))])

    candidates = []
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=args.qdrant_collection,
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
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.embedding_model)
    query_vec = model.encode(args.text, normalize_embeddings=True).tolist()
    print(f'Query embedded ({len(query_vec)}-dim). Searching…')

    if args.backend == 'qdrant':
        _semantic_qdrant(args, query_vec)
    elif args.brute_force:
        _semantic_valkey_brute(args, query_vec)
    else:
        _semantic_valkey_search(args, query_vec)


def _semantic_valkey_search(args: argparse.Namespace, query_vec: list[float]) -> None:
    """ANN search via valkey-search HNSW index (requires valkey-bundle)."""
    from redis.commands.search.query import Query as FTQuery

    client = _valkey_connect(args.valkey_url)
    query_bytes = struct.pack(f'<{len(query_vec)}f', *query_vec)
    q = (
        FTQuery(f'*=>[KNN {args.limit} @embedding $vec AS __score]')
        .return_fields('xxh64', 'blake2b', 'snippet', 'length', 'source', '__score')
        .sort_by('__score')
        .paging(0, args.limit)
        .dialect(2)
    )
    try:
        results = client.ft(args.valkey_index).search(q, query_params={'vec': query_bytes})
    except Exception as e:
        print(f'\n  FT.SEARCH failed: {e}', file=sys.stderr)
        print(
            '  Ensure you are running valkey-bundle (not plain valkey) and the HNSW\n'
            '  index was created during indexing (re-index without --no-valkey-embedding).\n'
            '  Use --brute-force to fall back to a full cosine scan (very slow).',
            file=sys.stderr,
        )
        client.close()
        return
    client.close()

    docs = results.docs
    if not docs:
        print('  No results.')
        return

    print(f'\n  {"Score":>8}  {"Length":>8}  {"Source":<20}  ID / Snippet')
    print(f'  {"\u2500" * 8}  {"\u2500" * 8}  {"\u2500" * 20}  {"\u2500" * 40}')
    for doc in docs:
        doc_id = f'{getattr(doc, "xxh64", "")}:{getattr(doc, "blake2b", "")}'
        short_id = doc_id[:20] + '\u2026' if len(doc_id) > 20 else doc_id
        snippet = textwrap.shorten(getattr(doc, 'snippet', '') or '', width=50, placeholder='\u2026')
        src = (getattr(doc, 'source', '') or '')[:20]
        length_val = getattr(doc, 'length', '0') or '0'
        length = int(length_val) if str(length_val).lstrip('-').isdigit() else 0
        # valkey-search COSINE distance = 1 - cosine_similarity; convert back to similarity
        dist = float(getattr(doc, '__score', 1.0) or 1.0)
        score = 1.0 - dist
        print(f'  {score:>8.4f}  {length:>8,}  {src:<20}  [{short_id}]  {snippet}')


def _semantic_qdrant(args: argparse.Namespace, query_vec: list[float]) -> None:
    from qdrant_client.models import ScoredPoint

    client = _qdrant_connect(args.qdrant_url, args.qdrant_collection)
    results: list[ScoredPoint] = client.query_points(
        collection_name=args.qdrant_collection,
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


def _semantic_valkey_brute(args: argparse.Namespace, query_vec: list[float]) -> None:
    """Full scan of all doc:* keys with an embedding field. Very slow for large indexes."""
    client = _valkey_connect(args.valkey_url)
    print('  Brute-force cosine scan in progress…')

    results: list[tuple[float, str, str, int, str]] = []
    cursor = 0
    scanned = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, match='doc:*', count=1_000)
        if keys:
            pipe = client.pipeline(transaction=False)
            for key in keys:
                pipe.hmget(key, [b'embedding', b'snippet', b'length', b'source'])
            batch = pipe.execute()
            for key, vals in zip(keys, batch):
                emb_b, snippet_b, length_b, source_b = vals
                if emb_b is None:
                    continue
                emb = _unpack_embedding(emb_b)
                score = _cosine(query_vec, emb)
                snippet = snippet_b.decode('utf-8', errors='replace') if snippet_b else ''
                length = int(length_b) if length_b else 0
                source = source_b.decode('utf-8', errors='replace') if source_b else ''
                doc_id = key.decode().removeprefix('doc:')
                # Keep a bounded top-k heap
                results.append((score, doc_id, snippet, length, source))
            scanned += len(keys)
            if scanned % 50_000 == 0:
                print(f'    …scanned {scanned:,} docs')
        if cursor == 0:
            break

    client.close()
    results.sort(key=lambda x: -x[0])
    top = results[: args.limit]

    if not top:
        print('  No results.')
        return

    print(f'\n  {"Score":>8}  {"Length":>8}  {"Source":<20}  ID / Snippet')
    print(f'  {"─" * 8}  {"─" * 8}  {"─" * 20}  {"─" * 40}')
    for score, doc_id, snippet, length, source in top:
        short_id = doc_id[:20] + '…' if len(doc_id) > 20 else doc_id
        short_snippet = textwrap.shorten(snippet, width=50, placeholder='…')
        src = (source or '')[:20]
        print(f'  {score:>8.4f}  {length:>8,}  {src:<20}  [{short_id}]  {short_snippet}')


# ══════════════════════════════════════════════════════════════════
# ID resolution helper
# ══════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    root = argparse.ArgumentParser(
        description='Query the text index (Valkey or Qdrant backend)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Global / backend args (before the subcommand) ──────────────
    root.add_argument('--backend', choices=['valkey', 'qdrant'], required=True, help='Storage backend to query')
    root.add_argument('--valkey-url', default='redis://localhost:6379/0', help='Valkey/Redis connection URL')
    root.add_argument('--valkey-index', default=DEFAULT_DB_COLLECTION, help='FT (valkey-search) index name for HNSW semantic search')
    root.add_argument('--qdrant-url', default='http://localhost:6333', help='Qdrant server URL')
    root.add_argument('--qdrant-collection', default=DEFAULT_DB_COLLECTION, help='Qdrant collection name')
    root.add_argument('--num-perm', type=int, default=DEFAULT_NUM_PERM, help='MinHash permutation count (must match indexing)')
    root.add_argument('--ngram-size', type=int, default=DEFAULT_NGRAM_SIZE, help='Word n-gram size (must match indexing)')
    root.add_argument('--lsh-bands', type=int, default=DEFAULT_LSH_BANDS, help='LSH band count (must match indexing)')

    sub = root.add_subparsers(dest='subcommand', required=True)

    # ── stats ──────────────────────────────────────────────────────
    sub.add_parser('stats', help='Print aggregate statistics for all indexed records')

    # ── get ────────────────────────────────────────────────────────
    p_get = sub.add_parser('get', help='Retrieve a single record by composite ID or raw text')
    id_group = p_get.add_mutually_exclusive_group(required=True)
    id_group.add_argument('--id', help='Composite ID in xxh64:blake2b format')
    id_group.add_argument('--text', help='Raw text to hash and look up')

    # ── find-similar ───────────────────────────────────────────────
    p_find = sub.add_parser(
        'find-similar',
        help='Find identical, near-duplicate, or semantically similar texts',
    )
    p_find.add_argument(
        '--mode',
        choices=['exact', 'minhash', 'semantic'],
        required=True,
        help=(
            'exact: identical text via hash lookup.  '
            'minhash: near-duplicates via MinHash LSH + Jaccard.  '
            'semantic: embedding ANN (Qdrant) or brute-force cosine (Valkey + --brute-force).'
        ),
    )
    input_group = p_find.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', help='Input text (hashed / embedded / used to compute MinHash)')
    input_group.add_argument('--id', help='Composite ID — fetch stored MinHash/embedding from the index')
    p_find.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Minimum Jaccard similarity to report (minhash mode)',
    )
    p_find.add_argument('--limit', type=int, default=20, help='Maximum number of results to print')
    p_find.add_argument(
        '--embedding-model',
        default=DEFAULT_EMBEDDING_MODEL,
        help='SentenceTransformer model name or path (semantic mode)',
    )
    p_find.add_argument(
        '--brute-force',
        action='store_true',
        help=(
            'Use full cosine scan instead of HNSW for semantic mode on the Valkey backend '
            '(very slow; only useful if valkey-search is not available)'
        ),
    )

    return root.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_perm % args.lsh_bands != 0:
        raise SystemExit(f'--lsh-bands {args.lsh_bands} must evenly divide --num-perm {args.num_perm}')

    dispatch = {
        'stats': cmd_stats,
        'get': cmd_get,
        'find-similar': cmd_find_similar,
    }
    dispatch[args.subcommand](args)


if __name__ == '__main__':
    main()
