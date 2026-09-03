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
//
// Performance note: the per-character classifiers (quality score, noise chars)
// are precomputed into lookup tables that are built FROM the original regexes at
// module load, so classification is identical by construction while the hot loop
// runs on charCodeAt instead of per-char regex tests. All floating-point
// accumulations keep the original operand order so rounded metrics match
// bit-for-bit what the previous implementation produced.
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
const ALPHA_TOKEN_RE = /^[A-Za-z][A-Za-z'’\-]*$/;

// Characters that almost never appear in clean prose. All are in the BMP, so we
// can match on UTF-16 code units (charCodeAt) without decoding full codepoints.
const NOISE_CODES = new Set(Array.from('•▪■□●○◦·※†‡§¶¤¦¨¬¯´¸×÷=~^|\\{}<>@#$%&*_+', c => c.charCodeAt(0)));

// ──────────────────────────────────────────────────────────────────────────────
// Lookup tables (built once from the reference regexes/sets — identical
// classification, table-speed lookups)
// ──────────────────────────────────────────────────────────────────────────────

// Per-code-unit score deltas for qualityScore. Astral codepoints (surrogate
// pairs) never match any of the classes and score -0.5, which the hot loop
// handles explicitly.
const QUALITY_TABLE = new Float64Array(65536);
{
  const letterRe = /[a-zα-ωàâäçèéêëîïôöùûüüÿæœß]$/i;
  const digitSpaceRe = /[0-9 \n]$/;
  const punctRe = /[.,;!?'"_\-]$/;
  for (let c = 0; c < 65536; c++) {
    const s = String.fromCharCode(c);
    QUALITY_TABLE[c] = letterRe.test(s) ? 2 : digitSpaceRe.test(s) ? 1 : punctRe.test(s) ? 0.5 : -0.5;
  }
}

const NOISE_TABLE = new Uint8Array(65536);
for (const c of NOISE_CODES) NOISE_TABLE[c] = 1;

// Shared encoder — TextEncoder construction is not free and computeRecord is hot.
const ENCODER = new TextEncoder();

const isAsciiAlnum = (c: number): boolean => (c >= 48 && c <= 57) || (c >= 65 && c <= 90) || (c >= 97 && c <= 122);
const isAsciiWord = (c: number): boolean => (c >= 48 && c <= 57) || (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || c === 95;
// a e i o u y (both cases) — the manual form of /[aeiouy]/i
const isVowelCode = (c: number): boolean => {
  const l = c | 32;
  return l === 97 || l === 101 || l === 105 || l === 111 || l === 117 || l === 121;
};

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

// End-of-token strip for the rare non-ASCII token (identical to the fast path
// below for ASCII input; kept as the reference behavior).
const STRIP_RE = /^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu;

function tokenize(text: string): string[] {
  // Split on whitespace; strip leading/trailing punctuation but keep internal apostrophes/hyphens.
  const raw = text.split(/\s+/);
  const out: string[] = [];
  for (const t of raw) {
    const len = t.length;
    if (len === 0) continue;
    // ASCII fast path: leading/trailing [^\p{L}\p{N}] on ASCII is exactly
    // "not alphanumeric", so strip with two index scans instead of the regex.
    let ascii = true;
    for (let i = 0; i < len; i++) {
      if (t.charCodeAt(i) > 127) {
        ascii = false;
        break;
      }
    }
    if (ascii) {
      let s = 0;
      let e = len;
      while (s < e && !isAsciiAlnum(t.charCodeAt(s))) s++;
      while (e > s && !isAsciiAlnum(t.charCodeAt(e - 1))) e--;
      if (s < e) out.push(s === 0 && e === len ? t : t.slice(s, e));
    } else {
      const stripped = t.replace(STRIP_RE, '');
      if (stripped) out.push(stripped);
    }
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
  let score = 0.0;
  const n = text.length;
  for (let i = 0; i < n; i++) {
    const c = text.charCodeAt(i);
    // A surrogate pair is one character for scoring purposes and never matches
    // any class, exactly like the original per-codepoint iteration.
    if (c >= 0xd800 && c < 0xdc00 && i + 1 < n) {
      const d = text.charCodeAt(i + 1);
      if (d >= 0xdc00 && d < 0xe000) {
        score -= 0.5;
        i++;
        continue;
      }
    }
    score += QUALITY_TABLE[c];
  }
  // Normalize by length and shift to range [-0.75, +1.25]
  return score / n - 0.75;
}

function compressionRatio(text: string): number {
  const raw = ENCODER.encode(text);
  // Use deflate (no header) to match Python's zlib.compress behavior
  const compressed = Bun.deflateSync(raw);
  return compressed.length / raw.length + 0.5;
}

// Scratch buffer for charEntropy counts; entries touched during a call are
// zeroed again before it returns.
const ENTROPY_COUNTS = new Uint32Array(65536);

function charEntropy(text: string): number {
  // Codepoints in first-occurrence order — the same order the original
  // Map<string, number> iterated in, so the entropy sum is bit-identical.
  const order: number[] = [];
  let astral: Map<number, number> | null = null;
  const n = text.length;
  for (let i = 0; i < n; i++) {
    let c = text.charCodeAt(i);
    if (c >= 0xd800 && c < 0xdc00 && i + 1 < n) {
      const d = text.charCodeAt(i + 1);
      if (d >= 0xdc00 && d < 0xe000) {
        c = 0x10000 + ((c - 0xd800) << 10) + (d - 0xdc00);
        i++;
      }
    }
    if (c < 65536) {
      if (ENTROPY_COUNTS[c] === 0) order.push(c);
      ENTROPY_COUNTS[c]++;
    } else {
      if (astral === null) astral = new Map();
      const prev = astral.get(c);
      if (prev === undefined) {
        order.push(c);
        astral.set(c, 1);
      } else {
        astral.set(c, prev + 1);
      }
    }
  }
  const total = n;
  let entropy = 0.0;
  for (const c of order) {
    const count = c < 65536 ? ENTROPY_COUNTS[c] : astral!.get(c)!;
    if (c < 65536) ENTROPY_COUNTS[c] = 0;
    const p = count / total;
    entropy -= p * Math.log2(p);
  }
  // Normalize by max entropy of English text (~4.4 bits/char)
  return entropy / 4.4;
}

// Replace every run of non-word code units with a single space — equivalent to
// text.split(/\W+/).join(' ') without materializing the parts array. \W here is
// the non-unicode class [^A-Za-z0-9_], so the result only contains those 63
// word characters plus spaces.
const NON_WORD_RUN_RE = /\W+/g;

function normalizeText(text: string): string {
  return text.replace(NON_WORD_RUN_RE, ' ');
}

// Scratch for counting distinct chars of a normalized string (alphabet is
// [A-Za-z0-9_ ], all < 128).
const UNIQ_SEEN = new Uint8Array(128);

function countUniqueNormalized(normalized: string): number {
  UNIQ_SEEN.fill(0);
  let uniq = 0;
  for (let i = 0; i < normalized.length; i++) {
    const c = normalized.charCodeAt(i);
    if (UNIQ_SEEN[c] === 0) {
      UNIQ_SEEN[c] = 1;
      uniq++;
    }
  }
  return uniq;
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
  const normalized = normalizeText(text);
  const id = generateId(normalized);
  const len = normalized.length;
  const uniqChar = countUniqueNormalized(normalized);
  const toks = tokenize(text);
  const tokens = toks.length || 1;

  const sentences = countSentences(text);
  const entropy = +(charEntropy(text) * 100).toFixed(2);
  const quality = +(qualityScore(text) * 100).toFixed(2);
  const compress = +(compressionRatio(text) * 100).toFixed(2);

  // Single pass over characters for the noise count
  let noise = 0;
  for (let i = 0; i < text.length; i++) {
    if (NOISE_TABLE[text.charCodeAt(i)] === 1) noise++;
  }

  // Single pass over tokens: alpha-ness (^[A-Za-z][A-Za-z'’\-]*$), vowel
  // presence, and case/curly-apostrophe flags all come out of one charCodeAt
  // scan, so the dict lookup only pays for toLowerCase/replace when needed.
  let dictHits = 0;
  let alphaTokens = 0;
  let vowelTokens = 0;
  for (const t of toks) {
    const tlen = t.length;
    const first = t.charCodeAt(0);
    if (!((first >= 65 && first <= 90) || (first >= 97 && first <= 122))) {
      continue;
    }
    let isAlpha = true;
    let hasUpper = first <= 90;
    let hasCurly = false;
    let hasVowel = isVowelCode(first);
    for (let i = 1; i < tlen; i++) {
      const c = t.charCodeAt(i);
      if (c >= 97 && c <= 122) {
        if (hasVowel === false && isVowelCode(c)) hasVowel = true;
      } else if (c >= 65 && c <= 90) {
        hasUpper = true;
        if (hasVowel === false && isVowelCode(c)) hasVowel = true;
      } else if (c === 39 || c === 45) {
        // ' or -
      } else if (c === 0x2019) {
        hasCurly = true; // ’
      } else {
        isAlpha = false;
        break;
      }
    }
    if (!isAlpha) continue;
    alphaTokens++;
    let lc = hasUpper ? t.toLowerCase() : t;
    if (hasCurly) lc = lc.replace(/’/g, "'");
    if (vocab.has(lc) || (lc.endsWith("'s") && vocab.has(lc.slice(0, -2)))) {
      dictHits++;
    }
    if (hasVowel) vowelTokens++;
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
