#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// dataset/indexing.ts — JSONL → Redis indexer (Bun)
//
// Reads .jsonl files, computes per-document features, and stores them in Redis
// as JSON objects keyed by a SHA-512/256 hash of the text.
// ──────────────────────────────────────────────────────────────────────────────

import { RedisClient } from 'bun';
import { createReadStream } from 'node:fs';
import { basename } from 'node:path';

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

const MIN_LENGTH = 10;
const MAX_LENGTH = 32_000;
const MIN_UNIQUE_CHARS = 10;
const MAX_UNIQUE_CHARS = 255;
const BATCH_SIZE = 256; // Redis batch flush size
// Sentence boundary regex adapted from fields.py:
// Matches .!? followed by whitespace + uppercase, or double newline + lowercase
const SENTENCE_RE = new RegExp('((?:[.!?][\\"\']?)\\s+(?=[A-Z\\"\'])|(?:[\\n\\r]{2,}\\s*(?=[a-zA-Z\\"\'])))', 'g');

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs(): {
  inputs: string[];
  source: string;
  redisUrl: string;
  textKey: string;
} {
  const args = process.argv.slice(2);
  let inputs: string[] = [];
  let source = 'cli';
  let redisUrl = '';
  let textKey = 'text';

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-i' || arg === '--input') && i + 1 < args.length) {
      inputs.push(args[++i]);
    } else if (arg === '-s' || arg === '--source') {
      source = args[++i];
    } else if ((arg === '-r' || arg === '--redis-url') && i + 1 < args.length) {
      redisUrl = args[++i];
    } else if ((arg === '-k' || arg === '--text-key') && i + 1 < args.length) {
      textKey = args[++i];
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun run dataset/indexing.ts [options]

Options:
  -i, --input <glob>        JSONL file path or glob pattern (required)
  -s, --source <label>      Source label for all documents (default: "cli")
  -r, --redis-url <url>     Redis connection URL (default: $REDIS_URL / $VALKEY_URL)
  -k, --text-key <key>      JSON field name for text (default: "text")
  -h, --help                Show this help`);
      process.exit(0);
    }
  }

  if (inputs.length === 0) {
    console.error('Error: --input is required. Use -h for help.');
    process.exit(1);
  }

  return { inputs, source, redisUrl, textKey };
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
// Feature computation
// ──────────────────────────────────────────────────────────────────────────────

function countWords(text: string): number {
  const tokens = text.split(/\s+/).filter(t => t.length > 0);
  return tokens.length;
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
// Document type
// ──────────────────────────────────────────────────────────────────────────────

interface DocRecord {
  id: string;
  text: string;
  source: string;
  length: number;
  uniqueChars: number;
  words: number;
  sentences: number;
  qualityScore: number;
  compressionRatio: number;
  entropy: number;
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

function printSummary(stats: FileStats): void {
  console.log(
    `  Loaded:     ${String(stats.rowsLoaded).padStart(12)}\n` +
      `  Dropped:    ${String(stats.rowsDropped).padStart(12)}  (len ≤ ${MIN_LENGTH} or > ${MAX_LENGTH} or uniqueChars out of range)\n` +
      `  Duplicates: ${String(stats.rowsDuplicate).padStart(12)}  (already indexed)\n` +
      `  Indexed:    ${String(stats.rowsIndexed).padStart(12)}`
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Pre-filter: length + unique chars
// ──────────────────────────────────────────────────────────────────────────────

function prefilter(text: string): boolean {
  const length = text.length;
  if (length <= MIN_LENGTH || length > MAX_LENGTH) return false;
  const uniqueChars = new Set(text).size;
  if (uniqueChars <= MIN_UNIQUE_CHARS || uniqueChars > MAX_UNIQUE_CHARS) return false;
  return true;
}

// ──────────────────────────────────────────────────────────────────────────────
// Compute all features for a document
// ──────────────────────────────────────────────────────────────────────────────

function computeFeatures(doc: { text: string }, source: string): DocRecord {
  const { text } = doc;
  const normalized = text.split(/\W+/).join(' ');
  const id = generateId(normalized);
  const length = normalized.length;
  const uniqueChars = new Set(normalized).size;
  const words = countWords(normalized);
  const sentences = countSentences(normalized);

  const entropy = charEntropy(text);
  const qualityScore_ = qualityScore(text);
  const compressionRatio_ = compressionRatio(text);

  return {
    id,
    text,
    source,
    length,
    uniqueChars,
    words,
    sentences,
    entropy,
    qualityScore: qualityScore_,
    compressionRatio: compressionRatio_,
  };
}

// ──────────────────────────────────────────────────────────────────────────────
// Process a single JSONL file
// ──────────────────────────────────────────────────────────────────────────────

async function processFile(filePath: string, source: string, client: RedisClient, textKey: string): Promise<FileStats> {
  const stats: FileStats = {
    path: filePath,
    rowsLoaded: 0,
    rowsDropped: 0,
    rowsDuplicate: 0,
    rowsIndexed: 0,
  };

  // Track seen IDs for within-file dedup
  const seenIds = new Set<string>();
  // Batch buffer for Redis writes
  const batch: { key: string; value: string }[] = [];

  async function flushBatch(): Promise<void> {
    if (batch.length === 0) return;

    // Build pipeline commands
    const commands: [string, string[]][] = batch.map(b => ['SET', [b.key, b.value]]);
    await client.send('MULTI', []);
    for (const [cmd, args] of commands) {
      await client.send(cmd, args);
    }
    await client.send('EXEC', []);

    batch.length = 0;
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
      if (line.length === 0) continue;

      stats.rowsLoaded++;

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
      const filtered = prefilter(text);
      if (!filtered) {
        console.warn(`  [WARN] Prefilter dropped line ${stats.rowsLoaded} in ${basename(filePath)} (len: ${text.length}, uniqueChars: ${new Set(text).size})`);
        stats.rowsDropped++;
        continue;
      }

      // Compute features (including ID)
      const doc = computeFeatures({ text }, source);

      // Within-file dedup
      if (seenIds.has(doc.id)) {
        console.debug(`  [DEBUG] Duplicate ID ${doc.id} on line ${stats.rowsLoaded} in ${basename(filePath)}`);
        stats.rowsDuplicate++;
        continue;
      } else {
        seenIds.add(doc.id);
      }

      // Redis dedup: check if already indexed
      const redisKey = `doc:${doc.id}`;
      const exists = await client.exists(redisKey);
      if (exists) {
        stats.rowsDuplicate++;
        continue;
      }

      // Add to batch
      batch.push({ key: redisKey, value: JSON.stringify(doc) });

      // Flush when batch is full
      if (batch.length >= BATCH_SIZE) {
        await flushBatch();
      }
    }
  }

  // Flush remaining batch
  await flushBatch();

  stats.rowsIndexed = stats.rowsLoaded - stats.rowsDropped - stats.rowsDuplicate;

  return stats;
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const { inputs, source, redisUrl, textKey } = parseArgs();

  if (inputs.length === 0) {
    console.error('Error: No input files found.');
    process.exit(1);
  }

  // Sort for deterministic processing order
  inputs.sort();

  console.log(`Found ${inputs.length} input file(s)`);
  console.log(`Source: ${source}`);
  console.log(`Text key: ${textKey}`);
  console.log(`Redis URL: ${redisUrl || '(default from env)'}`);

  // Connect to Redis
  const client = new RedisClient(redisUrl || undefined);
  await client.connect();

  let grandLoaded = 0;
  let grandDropped = 0;
  let grandDuplicate = 0;
  let grandIndexed = 0;

  for (const filePath of inputs) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Processing: ${filePath}`);
    console.log('='.repeat(60));

    const stats = await processFile(filePath, source, client, textKey);
    printSummary(stats);

    grandLoaded += stats.rowsLoaded;
    grandDropped += stats.rowsDropped;
    grandDuplicate += stats.rowsDuplicate;
    grandIndexed += stats.rowsIndexed;
  }

  // Final summary
  console.log(`\n${'='.repeat(60)}`);
  console.log('SUMMARY');
  console.log('='.repeat(60));
  console.log(
    `  Grand loaded:     ${String(grandLoaded).padStart(12)}\n` +
      `  Grand dropped:    ${String(grandDropped).padStart(12)}  (len ≤ ${MIN_LENGTH} or > ${MAX_LENGTH})\n` +
      `  Grand duplicates: ${String(grandDuplicate).padStart(12)}  (already indexed)\n` +
      `  Grand indexed:    ${String(grandIndexed).padStart(12)}`
  );

  client.close();
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
