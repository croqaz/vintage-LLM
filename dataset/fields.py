import argparse
import hashlib
import re
import string
import struct
import zlib
from collections import Counter
from math import log2

import torch
import xxhash
from datasketch import MinHash

from .config import (
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_NGRAM_SIZE,
    DEFAULT_NUM_PERM,
    MAX_CHARS,
    SEMANTIC_PREFIX,
)

_SENTENCE_BOUNDARY_RE = re.compile(r'((?:[.!?]["\']?)\s+(?=[A-Z"\'])|(?:[\n\r]{2,}\s*(?=[a-zA-Z"\'])))')


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using a regex."""
    parts = _SENTENCE_BOUNDARY_RE.split(text)
    return [parts[i] + (parts[i + 1] if i + 1 < len(parts) else '') for i in range(0, len(parts), 2)]


# ──────────────────────────────────────────────────────────────────
# ID resolution helper
# ──────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────
# Text quality metrics
# ──────────────────────────────────────────────────────────────────

_LETTER_RE = re.compile(r'[a-zα-ωàâäçèéêëîïôöùûüüÿæœß]$', re.I)
_DIGIT_SPACE_RE = re.compile(r'[0-9 \n]$')
_PUNCT_RE = re.compile(r'[.,;!?\'"_\-]$')


def quality_score(text: str) -> float:
    # Custom quality score
    length = len(text)
    if length == 0:
        return 0.0
    score = 0.0
    for c in text:
        if _LETTER_RE.match(c):
            score += 2
        elif _DIGIT_SPACE_RE.match(c):
            score += 1
        elif _PUNCT_RE.match(c):
            score += 0.5
        else:
            score -= 0.5
    # Normalize by length and shift to range [-0.75, +1.25]
    return (score / length) - 0.75


def compression_ratio(text: str) -> float:
    raw = text.encode('utf-8')
    if len(raw) == 0:
        return 0.0
    compressed = zlib.compress(raw)
    # Normalize to [0.5, 1.5] range
    return len(compressed) / len(raw) + 0.5


def char_entropy(text: str) -> float:
    if len(text) == 0:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * log2(p)
    # Normalise by max entropy of English text (~4.4 bits/char)
    return entropy / 4.4


# ─────────────────────────────────────────────────────────
# Hashing / MinHash helpers
# ─────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────


def trim_for_embedding(text: str) -> str:
    """Keep the first MAX_CHARS and the last MAX_CHARS characters."""
    if len(text) <= MAX_CHARS * 2:
        return text
    return text[:MAX_CHARS] + text[-MAX_CHARS:]


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

    # Scale batch size down for batches with long texts.
    # GPU memory is proportional to batch_size × max_seq_len (due to padding).
    # Scale inversely with the longest text, using ~4 chars/token as a heuristic.
    max_chars = max(len(t) for t in sorted_texts)
    approx_tokens = min(max_chars // 4, 8192)
    if approx_tokens == 0:
        enc_bs = batch_size
    else:
        bs = (batch_size * 512) // approx_tokens
        enc_bs = max(1, min(bs, batch_size))

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
