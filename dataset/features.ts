// ──────────────────────────────────────────────────────────────────────────────
// features.ts — shared document feature + id computation (Bun / Deno)
//
// Extracted from import.ts so the same id-hashing and metric logic can be reused
// by the ingest pipeline AND by tools that mutate stored records (e.g. the
// explorer's edit/save). Keeping this in one place guarantees a record edited in
// the GUI gets the exact same id + metrics it would have received at import time.
//
//   key:   <id>  (SHA-512/256 hex of the normalized text)
//   value: DocValue { source, len, uniqChar, tokens, sentences,
//                     quality, compress, entropy, dictHit, alpha, vowel, ascii, text }
// ──────────────────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

export const MIN_LENGTH = 100;
export const DEFAULT_MAX_LENGTH = 10_000_000;
export const MIN_UNIQUE_CHARS = 10;
export const MAX_UNIQUE_CHARS = 132;

// Sentence boundary regex adapted from fields.py:
// Matches .!? followed by whitespace + uppercase, or double newline + lowercase
const SENTENCE_RE = new RegExp('((?:[.!?][\\"\']?)\\s+(?=[A-Z\\"\'])|(?:[\\n\\r]{2,}\\s*(?=[a-zA-Z\\"\'])))', 'g');
const VOWEL_RE = /[aeiouy]/i;
const ALPHA_TOKEN_RE = /^[A-Za-z][A-Za-z'’\-]*$/;

// Characters that almost never appear in clean prose. All are in the BMP, so we
// can match on UTF-16 code units (charCodeAt) without decoding full codepoints.
const NOISE_CODES = new Set(Array.from('•▪■□●○◦·※†‡§¶¤¦¨¬¯´¸×÷=~^|\\{}<>@#$%&*_+', c => c.charCodeAt(0)));

// ──────────────────────────────────────────────────────────────────────────────
// Stored value type (no `id` — the id is the LevelDB key)
// ──────────────────────────────────────────────────────────────────────────────

export interface DocValue {
  source?: string;
  len: number;
  uniqChar: number;
  tokens: number;
  sentences: number;
  quality: number; // Cro's custom quality score
  compress: number; // normalized ZLIB compression ratio
  entropy: number; // normalized Shannon entropy
  dictHit: number; // dict hit rate over alpha tokens
  alpha: number; // share of well-formed alphabetic tokens
  vowel: number; // share of alpha tokens containing a vowel
  ascii: number; // 1 - amplified share of noise chars
  score?: number; // derived global score (higher is better)
  text: string;
  extra?: Record<string, unknown>; // optional extra fields from input JSONL
}

// ──────────────────────────────────────────────────────────────────────────────
// ID generation — SHA-512/256
// ──────────────────────────────────────────────────────────────────────────────

export function generateId(text: string): string {
  const hasher = new Bun.CryptoHasher('sha512-256');
  hasher.update(text);
  return hasher.digest('hex');
}

// ──────────────────────────────────────────────────────────────────────────────
// Vocabulary
// ──────────────────────────────────────────────────────────────────────────────

// Build a lowercase vocabulary from one or more clean reference text files.
// Keeps tokens that appear at least `minCount` times and are pure alphabetic.
export async function buildVocabFromFiles(paths: string[], minCount = 3): Promise<Set<string>> {
  const counts = new Map<string, number>();
  for (const p of paths) {
    const fname = Bun.file(p);
    if (!(await fname.exists())) {
      console.warn(`File ${fname} doesn't exist, skipping!`);
      continue;
    }
    const text = await fname.text();
    for (const t of tokenize(text)) {
      if (!ALPHA_TOKEN_RE.test(t)) continue;
      const lc = t.toLowerCase().replace(/[’]/g, "'");
      counts.set(lc, (counts.get(lc) ?? 0) + 1);
    }
  }
  const vocab = new Set<string>();
  for (const [w, c] of counts) if (c >= minCount) vocab.add(w);
  return vocab;
}

// Load a pre-calculated lowercase vocabulary from a JSON file.
// The file is expected to be a JSON array of words (e.g. produced by buildVocabFromFiles).
export async function loadVocabFromFile(path: string): Promise<Set<string>> {
  const file = Bun.file(path);
  if (!(await file.exists())) {
    throw new Error(`Vocabulary file ${path} doesn't exist!`);
  }
  const words = JSON.parse(await file.text()) as string[];
  return new Set(words);
}

// ──────────────────────────────────────────────────────────────────────────────
// Feature computation
// ──────────────────────────────────────────────────────────────────────────────

function tokenize(text: string): string[] {
  // Split on whitespace; strip leading/trailing punctuation but keep internal apostrophes/hyphens.
  const raw = text.split(/\s+/).filter(Boolean);
  const out: string[] = [];
  for (const t of raw) {
    const stripped = t.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
    if (stripped) out.push(stripped);
  }
  return out;
}

function countSentences(text: string): number {
  const parts = text.split(SENTENCE_RE);
  if (parts.length === 0) return 1;
  // Rejoin split parts: each boundary is followed by the next segment
  const sentences: string[] = [];
  for (let i = 0; i < parts.length; i += 2) {
    const part = parts[i] + (i + 1 < parts.length ? parts[i + 1] : '');
    if (part.trim().length > 0) {
      sentences.push(part);
    }
  }
  return sentences.length > 0 ? sentences.length : 1;
}

function qualityScore(text: string): number {
  const letterRe = /[a-zα-ωàâäçèéêëîïôöùûüüÿæœß]$/i;
  const digitSpaceRe = /[0-9 \n]$/;
  const punctRe = /[.,;!?'"_\-]$/;
  let score = 0.0;
  for (const c of text) {
    if (letterRe.test(c)) score += 2;
    else if (digitSpaceRe.test(c)) score += 1;
    else if (punctRe.test(c)) score += 0.5;
    else score -= 0.5;
  }
  // Normalize by length and shift to range [-0.75, +1.25]
  return score / text.length - 0.75;
}

function compressionRatio(text: string): number {
  const raw = new TextEncoder().encode(text);
  // Use deflate (no header) to match Python's zlib.compress behavior
  const compressed = Bun.deflateSync(raw);
  return compressed.length / raw.length + 0.5;
}

function charEntropy(text: string): number {
  const counts = new Map<string, number>();
  for (const c of text) {
    counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  const total = text.length;
  let entropy = 0.0;
  for (const count of counts.values()) {
    const p = count / total;
    entropy -= p * Math.log2(p);
  }
  // Normalize by max entropy of English text (~4.4 bits/char)
  return entropy / 4.4;
}

// ──────────────────────────────────────────────────────────────────────────────
// Compute the id + stored value for a document
// ──────────────────────────────────────────────────────────────────────────────

export function computeRecord(
  text: string,
  source: string,
  vocab: Set<string>,
  extra?: Record<string, unknown>
): { id: string; value: DocValue } {
  const normalized = text.split(/\W+/).join(' ');
  const id = generateId(normalized);
  const len = normalized.length;
  const uniqChar = new Set(normalized).size;
  const toks = tokenize(text);
  const tokens = toks.length || 1;

  const sentences = countSentences(text);
  const entropy = +(charEntropy(text) * 100).toFixed(2);
  const quality = +(qualityScore(text) * 100).toFixed(2);
  const compress = +(compressionRatio(text) * 100).toFixed(2);

  // Single pass over characters for the noise count
  let noise = 0;
  for (let i = 0; i < text.length; i++) {
    if (NOISE_CODES.has(text.charCodeAt(i))) noise++;
  }

  // Single pass over tokens: alpha-ness, vowel presence, and dict lookup share
  // the same ALPHA_TOKEN_RE test, so we never re-test a token.
  let dictHits = 0;
  let alphaTokens = 0;
  let vowelTokens = 0;
  for (const t of toks) {
    if (!ALPHA_TOKEN_RE.test(t)) continue;
    alphaTokens++;
    const lc = t.toLowerCase().replace(/’/g, "'");
    if (vocab.has(lc) || (lc.endsWith("'s") && vocab.has(lc.slice(0, -2)))) {
      dictHits++;
    }
    if (VOWEL_RE.test(t)) vowelTokens++;
  }

  const alpha = +((alphaTokens / tokens) * 100).toFixed(2);
  const vowel = +((alphaTokens > 0 ? vowelTokens / alphaTokens : 0) * 100).toFixed(2);
  const ascii = +(Math.max(0, 1 - (noise / len) * 5) * 100).toFixed(2); // amplify; 20% noise ⇒ 0
  const dictHit = +((alphaTokens > 0 ? dictHits / alphaTokens : 0) * 100).toFixed(2);

  const value: DocValue = {
    source,
    len,
    uniqChar,
    tokens,
    sentences,
    entropy,
    quality,
    compress,
    dictHit,
    alpha,
    vowel,
    ascii,
    text,
  };

  // Only attach extra if it has at least one key (keeps LevelDB small)
  if (Object.keys(extra || {}).length > 0) {
    value.extra = extra;
  }

  return { id, value };
}

// ──────────────────────────────────────────────────────────────────────────────
// Global score — higher is better
//
// Reward `quality` above its 100 midpoint, and penalize the absolute distance
// from 100 for six normalized signals (except `quality` itself).
// ──────────────────────────────────────────────────────────────────────────────

export function globalScore(v: DocValue): number {
  return (
    (v.quality ?? 0) -
    Math.abs((v.compress ?? 0) - 100) -
    Math.abs((v.entropy ?? 0) - 100) -
    Math.abs((v.dictHit ?? 0) - 100) -
    Math.abs((v.alpha ?? 0) - 100) -
    Math.abs((v.vowel ?? 0) - 100) -
    Math.abs((v.ascii ?? 0) - 100)
  );
}
