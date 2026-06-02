#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// import.ts — JSONL → LevelDB indexer (Bun / Deno)
//
// Reads .jsonl files, computes per-document features, and stores them in a
// LevelDB database (via classic-level). The key is a SHA-512/256 hash of the
// normalized text (the document ID); the value is a single JSON object holding
// the metadata together with the original text.
//
//   key:   <id>
//   value: { source, length, uniqueChars, words, sentences,
//            entropy, quality, compress, text }
// ──────────────────────────────────────────────────────────────────────────────

import { ClassicLevel } from 'classic-level';
import { createReadStream, readFileSync } from 'node:fs';
import { basename, extname } from 'node:path';

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

const MIN_LENGTH = 100;
const DEFAULT_MAX_LENGTH = 32_000;
const MIN_UNIQUE_CHARS = 10;
const MAX_UNIQUE_CHARS = 255;
const BATCH_SIZE = 512; // LevelDB batch flush size

// Sentence boundary regex adapted from fields.py:
// Matches .!? followed by whitespace + uppercase, or double newline + lowercase
const SENTENCE_RE = new RegExp('((?:[.!?][\\"\']?)\\s+(?=[A-Z\\"\'])|(?:[\\n\\r]{2,}\\s*(?=[a-zA-Z\\"\'])))', 'g');
const VOWEL_RE = /[aeiouy]/i;
const ALPHA_TOKEN_RE = /^[A-Za-z][A-Za-z'’\-]*$/;
// Characters that almost never appear in clean prose. All are in the BMP, so we
// can match on UTF-16 code units (charCodeAt) without decoding full codepoints.
const NOISE_CODES = new Set(Array.from('•▪■□●○◦·※†‡§¶¤¦¨¬¯´¸×÷=~^|\\{}<>@#$%&*_+', c => c.charCodeAt(0)));

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs(): {
  inputs: string[];
  source: string;
  dbPath: string;
  textKey: string;
  maxLength: number;
  vocabPath: string;
} {
  const args = process.argv.slice(2);
  let inputs: string[] = [];
  let source = 'cli';
  let dbPath = './levelDB';
  let textKey = 'text';
  let maxLength = DEFAULT_MAX_LENGTH;
  // Vocabulary for the dictHit signal. Defaults to vocab.json next to this script
  // so the path is correct regardless of the current working directory.
  let vocabPath = `${(import.meta as any).dirname}/vocab.json`;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-i' || arg === '--input') && i + 1 < args.length) {
      inputs.push(args[++i]);
    } else if ((arg === '-s' || arg === '--source') && i + 1 < args.length) {
      source = args[++i];
    } else if ((arg === '-d' || arg === '--db') && i + 1 < args.length) {
      dbPath = args[++i];
    } else if ((arg === '-k' || arg === '--text-key') && i + 1 < args.length) {
      textKey = args[++i];
    } else if ((arg === '-m' || arg === '--max-length' || arg === '--maxLength') && i + 1 < args.length) {
      maxLength = parseInt(args[++i], 10);
      if (isNaN(maxLength) || maxLength <= MIN_LENGTH) {
        console.error(`Error: --max-length must be an integer greater than ${MIN_LENGTH}.`);
        process.exit(1);
      }
    } else if ((arg === '-v' || arg === '--vocab') && i + 1 < args.length) {
      vocabPath = args[++i];
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun run dataset3/import.ts [options] [inputs...]

Options:
  -i, --input <glob>        JSONL file path (repeatable, required)
  -s, --source <label>      Source label for all documents (default: "cli")
  -d, --db <path>           LevelDB directory (default: "./levelDB")
  -k, --text-key <key>      JSON field name for text (default: "text")
  -m, --max-length <n>      Max character length to import (default: ${DEFAULT_MAX_LENGTH})
  -v, --vocab <path>        Wordlist JSON for the dictHit signal (default: vocab.json next to import.ts)
  -h, --help                Show this help`);
      process.exit(0);
    } else if (!arg.startsWith('-')) {
      inputs.push(arg);
    }
  }

  if (inputs.length === 0) {
    console.error('Error: --input is required. Use -h for help.');
    process.exit(1);
  }

  return { inputs, source, dbPath, textKey, maxLength, vocabPath };
}

// ──────────────────────────────────────────────────────────────────────────────
// ID generation — SHA-512/256
// ──────────────────────────────────────────────────────────────────────────────

function generateId(text: string): string {
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
  // const dict = Array.from(vocab); dict.sort();
  // await Bun.write("dict.json", JSON.stringify(dict, null, 2));
  return vocab;
}

// Load a pre-calculated lowercase vocabulary from a JSON file.
// The file is expected to be a JSON array of words (e.g. produced by buildVocabFromFiles).
export async function loadVocabFromFile(path: string): Promise<Set<string>> {
  const file = Bun.file(path);
  if (!(await file.exists())) {
    throw new Error(`Vocab file ${path} doesn't exist!`);
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
// Stored value type (no `id` — the id is the LevelDB key)
// ──────────────────────────────────────────────────────────────────────────────

interface DocValue {
  source: string;
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
  text: string;
}

// ──────────────────────────────────────────────────────────────────────────────
// Compute the id + stored value for a document
// ──────────────────────────────────────────────────────────────────────────────

function computeRecord(text: string, source: string, vocab: Set<string>): { id: string; value: DocValue } {
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

  return {
    id,
    value: {
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
    },
  };
}

// ──────────────────────────────────────────────────────────────────────────────
// Conflict resolution
//
// Called whenever two documents resolve to the same id (within a run or against
// a value already stored in the DB). Returns the value that should be kept in
// the database.
//
// TODO: implement a real strategy (e.g. keep the higher-quality or longer text).
// For now this is a stub that keeps the existing/older value ("first wins").
// ──────────────────────────────────────────────────────────────────────────────

function calcScore(value: DocValue): number {
  return (
    value.quality -
    100 +
    (value.compress > 100 ? 100 - value.compress : value.compress - 100) +
    (value.entropy > 100 ? 100 - value.entropy : value.entropy - 100)
  );
}

function onConflict(oldValue: DocValue, newValue: DocValue): DocValue | null {
  if (oldValue.source !== 'cli' && oldValue.text === newValue.text) {
    return null;
  }
  if (newValue.source !== 'cli' && oldValue.source === 'cli') {
    // Always prefer non-CLI sources over CLI
    oldValue.source = newValue.source;
  }
  return calcScore(newValue) >= calcScore(oldValue) ? newValue : oldValue;
}

// ──────────────────────────────────────────────────────────────────────────────
// Per-file stats
// ──────────────────────────────────────────────────────────────────────────────

interface FileStats {
  path: string;
  rowsLoaded: number;
  rowsDropped: number;
  rowsDuplicate: number;
  rowsIndexed: number;
}

function printSummary(stats: FileStats, maxLength: number): void {
  console.log(
    `  Loaded:     ${String(stats.rowsLoaded).padStart(12)}\n` +
      `  Dropped:    ${String(stats.rowsDropped).padStart(12)}  (len ≤ ${MIN_LENGTH} or > ${maxLength} or uniqueChars out of range)\n` +
      `  Duplicates: ${String(stats.rowsDuplicate).padStart(12)}  (id collision)\n` +
      `  Indexed:    ${String(stats.rowsIndexed).padStart(12)}`
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Pre-filter: length + unique chars
// ──────────────────────────────────────────────────────────────────────────────

function prefilter(text: string, maxLength: number): Record<string, any> {
  const length = text.length;
  if (length <= MIN_LENGTH || length > maxLength) return { ok: false, length };
  const uniqueChars = new Set(text).size;
  if (uniqueChars <= MIN_UNIQUE_CHARS || uniqueChars > MAX_UNIQUE_CHARS) {
    return { ok: false, uniqueChars };
  }
  const toks = text.split(/\s+/).filter(t => t.length > 0);
  const words = toks.length;
  if (words <= 2) return { ok: false, words };
  return { ok: true };
}

// ──────────────────────────────────────────────────────────────────────────────
// File type detection
// ──────────────────────────────────────────────────────────────────────────────

const JSONL_EXTENSIONS = new Set(['.json', '.jsonl', '.ndjson']);
const TEXT_EXTENSIONS = new Set(['.txt', '.md']);

function detectFileType(filePath: string): 'jsonl' | 'text' {
  const ext = extname(filePath).toLowerCase();
  if (JSONL_EXTENSIONS.has(ext)) return 'jsonl';
  if (TEXT_EXTENSIONS.has(ext)) return 'text';
  console.error(`Fatal: unsupported file extension "${ext}" for ${basename(filePath)}.`);
  console.error(`Supported: ${[...JSONL_EXTENSIONS, ...TEXT_EXTENSIONS].join(', ')}`);
  process.exit(1);
}

// ──────────────────────────────────────────────────────────────────────────────
// Process a single .txt / .md file (entire file = one document)
// ──────────────────────────────────────────────────────────────────────────────

async function processTextFile(
  filePath: string,
  source: string,
  db: ClassicLevel<string, DocValue>,
  maxLength: number,
  vocab?: Set<string>
): Promise<FileStats> {
  const stats: FileStats = {
    path: filePath,
    rowsLoaded: 1,
    rowsDropped: 0,
    rowsDuplicate: 0,
    rowsIndexed: 0,
  };

  const text = readFileSync(filePath, 'utf8');

  const filtered = prefilter(text, maxLength);
  if (!filtered.ok) {
    delete filtered.ok;
    console.log(`  Skipping ${basename(filePath)}: failed pre-filter (${JSON.stringify(filtered)})`);
    stats.rowsDropped = 1;
    return stats;
  }

  const { id, value } = computeRecord(text, source, vocab);

  // Check for existing entry in the DB.
  const existing = await db.getMany([id]);
  if (existing[0] !== undefined) {
    stats.rowsDuplicate = 1;
    const resolved = onConflict(existing[0], value);
    if (resolved) {
      await db.put(id, resolved);
      stats.rowsIndexed = 1;
      stats.rowsDuplicate = 0;
    }
  } else {
    await db.put(id, value);
    stats.rowsIndexed = 1;
  }

  return stats;
}

// ──────────────────────────────────────────────────────────────────────────────
// Process a single JSONL file
// ──────────────────────────────────────────────────────────────────────────────

async function processFile(
  filePath: string,
  source: string,
  db: ClassicLevel<string, DocValue>,
  textKey: string,
  maxLength: number,
  vocab?: Set<string>
): Promise<FileStats> {
  const stats: FileStats = {
    path: filePath,
    rowsLoaded: 0,
    rowsDropped: 0,
    rowsDuplicate: 0,
    rowsIndexed: 0,
  };

  // Pending writes keyed by id. Using a Map dedups within-batch collisions and
  // lets us resolve them via onConflict before they ever hit the DB.
  const batch = new Map<string, DocValue>();

  async function flushBatch(): Promise<void> {
    if (batch.size === 0) return;

    const keys = [...batch.keys()];
    // Single round-trip existence check for the whole batch.
    const existing = await db.getMany(keys);

    const ops: { type: 'put'; key: string; value: DocValue }[] = [];
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      const newValue = batch.get(key)!;
      const oldValue = existing[i];

      if (oldValue !== undefined) {
        // Already in the DB → conflict.
        stats.rowsDuplicate++;
        const resolved = onConflict(oldValue, newValue);
        if (resolved) {
          ops.push({ type: 'put', key, value: resolved });
        }
      } else {
        stats.rowsIndexed++;
        ops.push({ type: 'put', key, value: newValue });
      }
    }

    if (ops.length > 0) {
      await db.batch(ops);
    }

    console.log(`  Flushed ${keys.length} keys (${ops.length} writes) to LevelDB...`);
    batch.clear();
  }

  // Add a computed record to the batch, resolving within-batch collisions.
  function addToBatch(id: string, value: DocValue): void {
    const existing = batch.get(id);
    if (existing !== undefined) {
      stats.rowsDuplicate++;
      const resolved = onConflict(existing, value);
      if (resolved) {
        batch.set(id, resolved);
      }
    } else {
      batch.set(id, value);
    }
  }

  // Stream the file line by line
  const fileStream = createReadStream(filePath, { encoding: 'utf8' });
  let lineBuffer = '';

  for await (const chunk of fileStream) {
    lineBuffer += chunk;

    // Process complete lines
    let newlineIdx: number;
    while ((newlineIdx = lineBuffer.indexOf('\n')) !== -1) {
      const line = lineBuffer.slice(0, newlineIdx).trim();
      lineBuffer = lineBuffer.slice(newlineIdx + 1);
      if (line.length <= 10) continue;

      stats.rowsLoaded++;

      if (stats.rowsLoaded % 100_000 === 0) {
        console.log(`  Processed ${stats.rowsLoaded} lines...`);
      }

      // Parse JSON
      let obj: unknown;
      try {
        obj = JSON.parse(line);
      } catch {
        console.warn(`  [WARN] Skipping malformed JSON line ${stats.rowsLoaded} in ${basename(filePath)}`);
        stats.rowsDropped++;
        continue;
      }

      if (obj === null || typeof obj !== 'object') {
        console.warn(`  [WARN] Skipping non-object line ${stats.rowsLoaded} in ${basename(filePath)}`);
        stats.rowsDropped++;
        continue;
      }

      const record = obj as Record<string, unknown>;
      const text = record[textKey];
      if (!text || typeof text !== 'string') {
        console.warn(`  [WARN] Missing "${textKey}" field on line ${stats.rowsLoaded} in ${basename(filePath)}`);
        stats.rowsDropped++;
        continue;
      }

      // Pre-filter
      const filtered = prefilter(text, maxLength);
      if (!filtered.ok) {
        stats.rowsDropped++;
        continue;
      }

      // Reuse "source" field if present, otherwise use CLI arg
      let docSource = source;
      if (record.source && typeof record.source === 'string') {
        docSource = record.source;
      }

      // Compute features (including id)
      const { id, value } = computeRecord(text, docSource, vocab);

      addToBatch(id, value);

      // Flush when batch is full
      if (batch.size >= BATCH_SIZE) {
        await flushBatch();
      }
    }
  }

  // Flush remaining batch
  await flushBatch();

  return stats;
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const { inputs, source, dbPath, textKey, maxLength, vocabPath } = parseArgs();

  // Sort for deterministic processing order
  inputs.sort();

  // Load the wordlist for the dictHit signal. If it's missing, the signal is
  // simply stored as null rather than aborting the whole import.
  let vocab: Set<string> | undefined;
  try {
    vocab = await loadVocabFromFile(vocabPath);
  } catch (err) {
    console.warn(`  [WARN] Could not load vocab (${vocabPath}): dictHit will be null.`);
  }

  console.log(`Found ${inputs.length} input file(s)`);
  console.log(`Source: ${source}`);
  console.log(`Text key: ${textKey}`);
  console.log(`Min length: ${MIN_LENGTH}`);
  console.log(`Max length: ${maxLength}`);
  console.log(`LevelDB: ${dbPath}`);
  console.log(`Vocab: ${vocab ? `${vocab.size} words (${vocabPath})` : 'none'}`);

  const db = new ClassicLevel<string, DocValue>(dbPath, {
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await db.open();

  let grandLoaded = 0;
  let grandDropped = 0;
  let grandDuplicate = 0;
  let grandIndexed = 0;

  try {
    for (let fi = 0; fi < inputs.length; fi++) {
      const filePath = inputs[fi];
      const fileType = detectFileType(filePath);

      if (fileType === 'jsonl') {
        console.log(`\n${'='.repeat(60)}`);
        console.log(`Processing: ${filePath}`);
        console.log('='.repeat(60));
      }

      const stats =
        fileType === 'text'
          ? await processTextFile(filePath, source, db, maxLength, vocab)
          : await processFile(filePath, source, db, textKey, maxLength, vocab);

      if (fileType === 'jsonl') {
        printSummary(stats, maxLength);
      }

      grandLoaded += stats.rowsLoaded;
      grandDropped += stats.rowsDropped;
      grandDuplicate += stats.rowsDuplicate;
      grandIndexed += stats.rowsIndexed;

      // Periodic progress for text files
      if (fileType === 'text' && (fi + 1) % 25 === 0) {
        console.log(`  ... processed ${fi + 1}/${inputs.length} files (${grandIndexed} indexed so far)`);
      }
    }
  } finally {
    await db.close();
  }

  // Final summary
  console.log(`\n${'='.repeat(60)}`);
  console.log('SUMMARY');
  console.log('='.repeat(60));
  console.log(
    `  Grand loaded:     ${String(grandLoaded).padStart(12)}\n` +
      `  Grand dropped:    ${String(grandDropped).padStart(12)}  (quality filter)\n` +
      `  Grand duplicates: ${String(grandDuplicate).padStart(12)}  (id collision)\n` +
      `  Grand indexed:    ${String(grandIndexed).padStart(12)}`
  );
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
