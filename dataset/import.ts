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
import {
  computeRecord,
  DEFAULT_MAX_LENGTH,
  type DocValue,
  globalScore,
  loadVocabFromFile,
  MAX_UNIQUE_CHARS,
  MIN_LENGTH,
  MIN_UNIQUE_CHARS,
} from './features.ts';

const IS_WORKER = typeof Bun !== 'undefined' && Bun.isMainThread === false;

// Re-export the shared vocab builders so existing importers of import.ts keep working.
export { buildVocabFromFiles, loadVocabFromFile } from './features.ts';

// Re-export prefilter constants so tests can use them without replicating logic.
export { DEFAULT_MAX_LENGTH, MAX_UNIQUE_CHARS, MIN_LENGTH, MIN_UNIQUE_CHARS } from './features.ts';

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

const BATCH_SIZE = 1024; // LevelDB batch flush size
const LINES_PER_WORKER_BATCH = 256; // lines per postMessage round-trip
const DEFAULT_WORKERS = Math.max(1, Math.min(16, navigator.hardwareConcurrency ?? 4));

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
  extraFields: string[];
  offset: number;
  limit: number;
  workers: number;
} {
  const args = process.argv.slice(2);
  let inputs: string[] = [];
  let source = 'cli';
  let dbPath = './levelDB';
  let textKey = 'text';
  let maxLength = DEFAULT_MAX_LENGTH;
  // Vocabulary for the dictHit signal. Defaults to vocab.json next to this script
  // so the path is correct regardless of the current working directory.
  let vocabPath = './vocab2.json';
  let extraFields: string[] = [];
  let offset = 0;
  let limit = 0;
  let workers = DEFAULT_WORKERS;

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
    } else if (arg === '--vocab' && i + 1 < args.length) {
      vocabPath = args[++i];
    } else if ((arg === '-e' || arg === '--extra-fields') && i + 1 < args.length) {
      extraFields = args[++i]
        .split(',')
        .map(f => f.trim())
        .filter(Boolean);
    } else if ((arg === '-o' || arg === '--offset') && i + 1 < args.length) {
      offset = parseInt(args[++i], 10);
      if (isNaN(offset) || offset < 0) {
        console.error('Error: --offset must be a non-negative integer.');
        process.exit(1);
      }
    } else if ((arg === '-l' || arg === '--limit') && i + 1 < args.length) {
      limit = parseInt(args[++i], 10);
      if (isNaN(limit) || limit < 0) {
        console.error('Error: --limit must be a non-negative integer (0 = no limit).');
        process.exit(1);
      }
    } else if ((arg === '-w' || arg === '--workers') && i + 1 < args.length) {
      workers = parseInt(args[++i], 10);
      if (isNaN(workers) || workers < 0) {
        console.error('Error: --workers must be a non-negative integer (0 = in-process, no workers).');
        process.exit(1);
      }
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun run import.ts [options] [inputs...]

Options:
  -i, --input <glob>        JSONL file path (repeatable, required)
  -s, --source <label>      Source label for all documents (default: "cli")
  -d, --db <path>           LevelDB directory (default: "./levelDB")
  -k, --text-key <key>      JSON field name for text (default: "text")
  -m, --max-length <n>      Max character length to import (default: ${DEFAULT_MAX_LENGTH})
  -v, --vocab <path>        Wordlist JSON for the dictHit signal (default: vocab.json next to import.ts)
  -e, --extra-fields <list> Comma-separated extra field names to preserve from input JSONL (default: none)
  -o, --offset <n>          Fast-forward: skip the first N non-empty lines before processing (default: 0)
  -l, --limit <n>           Stop after indexing N records (default: 0 = no limit)
  -w, --workers <n>         Parallel feature-extraction workers (default: ${DEFAULT_WORKERS}, 0/1 = in-process)
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

  return {
    inputs,
    source,
    dbPath,
    textKey,
    offset,
    limit,
    maxLength,
    vocabPath,
    extraFields,
    workers,
  };
}

// ──────────────────────────────────────────────────────────────────────────────
// Conflict resolution
//
// Called whenever two documents resolve to the same id (within a run or against
// a value already stored in the DB). Returns the value that should be kept in
// the database.
// ──────────────────────────────────────────────────────────────────────────────

function onConflict(oldValue: DocValue, newValue: DocValue): DocValue | null {
  if (oldValue.source !== 'cli' && oldValue.text === newValue.text) {
    return null;
  }
  if (newValue.source !== 'cli' && oldValue.source === 'cli') {
    // Always prefer non-CLI sources over CLI
    oldValue.source = newValue.source;
  }
  return globalScore(newValue) > globalScore(oldValue) ? newValue : null;
}

// ──────────────────────────────────────────────────────────────────────────────
// Per-file stats
// ──────────────────────────────────────────────────────────────────────────────

export interface FileStats {
  path: string;
  rowsLoaded: number;
  rowsDropped: number;
  rowsSkipped: number;
  rowsLimited: number;
  rowsDuplicate: number;
  rowsIndexed: number;
}

function printSummary(stats: FileStats, maxLength: number): void {
  const lines: string[] = [
    `  Loaded:     ${String(stats.rowsLoaded).padStart(12)}`,
    `  Dropped:    ${String(stats.rowsDropped).padStart(12)}  (len ≤ ${MIN_LENGTH} or > ${maxLength} or uniqueChars out of range)`,
    `  Skipped:    ${String(stats.rowsSkipped).padStart(12)}  (offset fast-forward)`,
  ];
  if (stats.rowsLimited > 0) {
    lines.push(`  Limited:    ${String(stats.rowsLimited).padStart(12)}  (limit reached, records not indexed)`);
  }
  lines.push(
    `  Duplicates: ${String(stats.rowsDuplicate).padStart(12)}  (id collision)`,
    `  Indexed:    ${String(stats.rowsIndexed).padStart(12)}`
  );
  console.log(lines.join('\n'));
}

// ──────────────────────────────────────────────────────────────────────────────
// Pre-filter: length + unique chars
// ──────────────────────────────────────────────────────────────────────────────

// Whitespace-per-/\s/ lookup table, built from the regex itself so the class is
// identical by construction.
const SPACE_TABLE = new Uint8Array(65536);
{
  const re = /^\s$/;
  for (let c = 0; c < 65536; c++) {
    if (re.test(String.fromCharCode(c))) SPACE_TABLE[c] = 1;
  }
}

// Generation-stamped scratch table for distinct-codepoint counting: bumping the
// generation invalidates all entries without a 256KB memset per call.
const UNIQ_STAMP = new Int32Array(65536);
let uniqGeneration = 0;

// Distinct codepoints in `text` — exactly what `new Set(text).size` computes
// (surrogate pairs count once, lone surrogates count as themselves).
function countUniqueCodepoints(text: string): number {
  const gen = ++uniqGeneration;
  let uniq = 0;
  let astral: Set<number> | null = null;
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
      if (UNIQ_STAMP[c] !== gen) {
        UNIQ_STAMP[c] = gen;
        uniq++;
      }
    } else {
      if (astral === null) astral = new Set();
      if (!astral.has(c)) {
        astral.add(c);
        uniq++;
      }
    }
  }
  return uniq;
}

// Count whitespace-separated words, stopping as soon as `stopAt` are found —
// the prefilter only needs to know whether there are more than 2.
function countWords(text: string, stopAt: number): number {
  let words = 0;
  let inToken = false;
  const n = text.length;
  for (let i = 0; i < n; i++) {
    if (SPACE_TABLE[text.charCodeAt(i)] === 0) {
      if (!inToken) {
        inToken = true;
        if (++words >= stopAt) return words;
      }
    } else {
      inToken = false;
    }
  }
  return words;
}

export function prefilter(text: string, maxLength: number): Record<string, any> {
  const length = text.length;
  if (length <= MIN_LENGTH || length > maxLength) return { ok: false, length };
  const uniqueChars = countUniqueCodepoints(text);
  if (uniqueChars <= MIN_UNIQUE_CHARS || uniqueChars > MAX_UNIQUE_CHARS) {
    return { ok: false, uniqueChars };
  }
  const words = countWords(text, 3);
  if (words <= 2) return { ok: false, words };
  return { ok: true };
}

// ──────────────────────────────────────────────────────────────────────────────
// Worker pool — parallel per-line feature extraction
//
// The heavy per-record work (JSON.parse → prefilter → computeRecord) is pure
// and order-independent, so it can fan out across CPU cores. The main thread
// keeps full control of ORDER: it dispatches line batches round-robin and
// consumes the results strictly in input order, so stats, warnings, conflict
// resolution, offset/limit semantics and the resulting DB are identical to the
// sequential path. This very file is the worker script (guarded by
// Bun.isMainThread), so the worker uses the exact same computeRecord/prefilter.
// ──────────────────────────────────────────────────────────────────────────────

interface WorkerConfig {
  textKey: string;
  maxLength: number;
  source: string;
  extraFields: string[];
  vocab: Set<string> | null;
}

// Per-line outcome codes; keep these tiny — they cross the postMessage boundary.
const enum LineKind {
  BadJson = 0,
  NonObject = 1,
  NoText = 2,
  Prefiltered = 3,
  Ok = 4,
}

type LineResult = { k: LineKind; id?: string; value?: DocValue };

function processLine(line: string, cfg: WorkerConfig): LineResult {
  let obj: unknown;
  try {
    obj = JSON.parse(line);
  } catch {
    return { k: LineKind.BadJson };
  }
  if (obj === null || typeof obj !== 'object') return { k: LineKind.NonObject };

  const record = obj as Record<string, unknown>;
  const text = record[cfg.textKey];
  if (!text || typeof text !== 'string') return { k: LineKind.NoText };

  if (!prefilter(text, cfg.maxLength).ok) return { k: LineKind.Prefiltered };

  // Reuse "source" field if present, otherwise use CLI arg
  let docSource = cfg.source;
  if (record.source && typeof record.source === 'string') {
    docSource = record.source;
  }

  // Extract optional extra fields (only those that exist on the record)
  let extra: Record<string, unknown> | undefined;
  for (const f of cfg.extraFields) {
    if (f in record && record[f] != null) {
      if (!extra) extra = {};
      extra[f] = record[f];
    }
  }

  const { id, value } = computeRecord(text, docSource, cfg.vocab as unknown as Set<string>, extra);
  return { k: LineKind.Ok, id, value };
}

if (IS_WORKER) {
  let cfg: WorkerConfig | null = null;
  // @ts-expect-error — worker global
  self.onmessage = (event: MessageEvent) => {
    const msg = event.data;
    if (msg.type === 'init') {
      cfg = msg.cfg as WorkerConfig;
    } else if (msg.type === 'batch') {
      const lines: string[] = msg.lines;
      const results = new Array<LineResult>(lines.length);
      for (let i = 0; i < lines.length; i++) {
        results[i] = processLine(lines[i], cfg!);
      }
      // @ts-expect-error — worker global
      self.postMessage({ seq: msg.seq, results });
    }
  };
}

export class WorkerPool {
  private workers: Worker[] = [];
  private next = 0;
  private pending = new Map<number, { resolve: (r: LineResult[]) => void; reject: (e: unknown) => void }>();
  private seq = 0;
  private failure: unknown = null;

  constructor(size: number, cfg: WorkerConfig) {
    for (let i = 0; i < size; i++) {
      const w = new Worker(import.meta.url);
      w.onmessage = (event: MessageEvent) => {
        const { seq, results } = event.data;
        const entry = this.pending.get(seq);
        if (entry) {
          this.pending.delete(seq);
          entry.resolve(results);
        }
      };
      w.addEventListener('error', (event: ErrorEvent) => {
        this.fail(event.error ?? new Error(event.message || 'worker error'));
      });
      w.postMessage({ type: 'init', cfg });
      this.workers.push(w);
    }
  }

  private fail(err: unknown): void {
    this.failure = err;
    for (const { reject } of this.pending.values()) reject(err);
    this.pending.clear();
  }

  get size(): number {
    return this.workers.length;
  }

  run(lines: string[]): Promise<LineResult[]> {
    if (this.failure !== null) return Promise.reject(this.failure);
    const seq = this.seq++;
    return new Promise((resolve, reject) => {
      this.pending.set(seq, { resolve, reject });
      this.workers[this.next].postMessage({ type: 'batch', seq, lines });
      this.next = (this.next + 1) % this.workers.length;
    });
  }

  terminate(): void {
    for (const w of this.workers) w.terminate();
    this.workers = [];
    this.pending.clear();
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// GZIP file support (Bun native gunzip)
// ──────────────────────────────────────────────────────────────────────────────

const GZIP_EXT = '.gz';

function isGzipped(filePath: string): boolean {
  return filePath.endsWith(GZIP_EXT);
}

function stripGz(filePath: string): string {
  return filePath.slice(0, -GZIP_EXT.length);
}

/** Read and optionally decompress a file to a UTF-8 string. */
async function readFileContent(filePath: string): Promise<string> {
  if (isGzipped(filePath)) {
    const compressed = await Bun.file(filePath).bytes();
    const decompressed = Bun.gunzipSync(compressed);
    return new TextDecoder().decode(decompressed);
  }
  return readFileSync(filePath, 'utf8');
}

/**
 * Create an async iterable from a file path. Gzip files are transparently
 * decompressed in memory and yielded as a single chunk.
 */
function fileToAsyncIterable(filePath: string): AsyncIterable<string> {
  if (isGzipped(filePath)) {
    // For gzip files, decompress everything first, then yield as a single chunk.
    // Bun.gunzipSync is fast; the existing line-splitting loop handles the rest.
    let loaded = false;
    let content = '';
    const iterable: AsyncIterable<string> = {
      [Symbol.asyncIterator]() {
        return {
          async next(): Promise<IteratorResult<string>> {
            if (!loaded) {
              loaded = true;
              const compressed = await Bun.file(filePath).bytes();
              const decompressed = Bun.gunzipSync(compressed);
              content = new TextDecoder().decode(decompressed);
              return { value: content, done: false };
            }
            return { value: undefined as unknown as string, done: true };
          },
        };
      },
    };
    return iterable;
  }
  return createReadStream(filePath, { encoding: 'utf8' });
}

// ──────────────────────────────────────────────────────────────────────────────
// File type detection
// ──────────────────────────────────────────────────────────────────────────────

const JSONL_EXTENSIONS = new Set(['.json', '.jsonl', '.ndjson']);
const TEXT_EXTENSIONS = new Set(['.txt', '.md']);

function detectFileType(filePath: string): 'jsonl' | 'text' {
  // Strip .gz suffix before checking the underlying file extension
  const stripped = isGzipped(filePath) ? stripGz(filePath) : filePath;
  const ext = extname(stripped).toLowerCase();
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
  vocab: Set<string> | undefined,
  offset: number,
  limit: number
): Promise<{ stats: FileStats; remainingOffset: number; remainingLimit: number }> {
  const stats: FileStats = {
    path: filePath,
    rowsLoaded: 1,
    rowsDropped: 0,
    rowsSkipped: 0,
    rowsLimited: 0,
    rowsDuplicate: 0,
    rowsIndexed: 0,
  };

  const text = await readFileContent(filePath);

  const filtered = prefilter(text, maxLength);
  if (!filtered.ok) {
    delete filtered.ok;
    console.log(`  Skipping ${basename(filePath)}: failed pre-filter (${JSON.stringify(filtered)})`);
    stats.rowsDropped = 1;
    return { stats, remainingOffset: offset, remainingLimit: limit };
  }

  // Offset skip for text files
  if (offset > 0) {
    stats.rowsSkipped = 1;
    return { stats, remainingOffset: offset - 1, remainingLimit: limit };
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

  return {
    stats,
    remainingOffset: 0,
    remainingLimit: limit > 0 ? limit - 1 : 0,
  };
}

// ──────────────────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────────────────
// Process a single JSONL file
// ──────────────────────────────────────────────────────────────────────────────

// Pending writes keyed by id. Using a Map dedups within-batch collisions and
// lets us resolve them via onConflict before they ever hit the DB.
function makeBatcher(db: ClassicLevel<string, DocValue>, stats: FileStats) {
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

    // console.log(`  Flushed ${keys.length} keys (${ops.length} writes) to LevelDB...`);
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

  return { batch, flushBatch, addToBatch };
}

export async function processFile(
  filePath: string,
  source: string,
  db: ClassicLevel<string, DocValue>,
  textKey: string,
  maxLength: number,
  vocab: Set<string>,
  extraFields: string[],
  offset: number,
  limit: number,
  pool?: WorkerPool
): Promise<{ stats: FileStats; remainingOffset: number; remainingLimit: number }> {
  if (pool) {
    return processFileParallel(filePath, source, db, textKey, maxLength, vocab, extraFields, offset, limit, pool);
  }

  const stats: FileStats = {
    path: filePath,
    rowsLoaded: 0,
    rowsDropped: 0,
    rowsSkipped: 0,
    rowsLimited: 0,
    rowsDuplicate: 0,
    rowsIndexed: 0,
  };
  let skipRemaining = offset;
  let limitRemaining = limit;

  const { batch, flushBatch, addToBatch } = makeBatcher(db, stats);

  // Stream the file line by line (gzip files are decompressed on-the-fly)
  const fileStream = fileToAsyncIterable(filePath);
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

      // Offset skip: fast-forward N non-empty lines without any JSON parsing
      // or pre-filtering. This is O(1) per line (just a counter decrement) vs.
      // the full JSON parse + prefilter + computeRecord pipeline.
      if (skipRemaining > 0) {
        skipRemaining--;
        stats.rowsSkipped++;
        continue;
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

      // Limit check: cap the number of records that get fully indexed.
      // Only enforced when limit was explicitly set (> 0). A limit of 0 means
      // "no limit" so we never enter this branch.
      if (limit > 0 && limitRemaining === 0) {
        stats.rowsLimited++;
        // Flush whatever is in the batch and bail out of both loops.
        await flushBatch();
        return { stats, remainingOffset: skipRemaining, remainingLimit: 0 };
      }
      if (limit > 0) limitRemaining--;

      // Reuse "source" field if present, otherwise use CLI arg
      let docSource = source;
      if (record.source && typeof record.source === 'string') {
        docSource = record.source;
      }

      // Extract optional extra fields (only those that exist on the record)
      let extra: Record<string, unknown> | undefined;
      if (extraFields.length > 0) {
        for (const f of extraFields) {
          if (f in record && record[f] != null) {
            if (!extra) extra = {};
            extra[f] = record[f];
          }
        }
      }

      // Compute features (including id)
      const { id, value } = computeRecord(text, docSource, vocab, extra);

      addToBatch(id, value);

      // Flush when batch is full
      if (batch.size >= BATCH_SIZE) {
        await flushBatch();
      }
    }
  }

  // Flush remaining batch
  await flushBatch();

  return {
    stats,
    remainingOffset: skipRemaining,
    remainingLimit: limitRemaining,
  };
}

// ──────────────────────────────────────────────────────────────────────────────
// Parallel variant of processFile
//
// Same observable behavior as the sequential path: line batches fan out to the
// worker pool, but results are consumed strictly in input order, so stats,
// warnings, within-batch conflict resolution and offset/limit handling are
// identical — and so is the resulting database.
// ──────────────────────────────────────────────────────────────────────────────

async function processFileParallel(
  filePath: string,
  source: string,
  db: ClassicLevel<string, DocValue>,
  textKey: string,
  maxLength: number,
  vocab: Set<string>,
  extraFields: string[],
  offset: number,
  limit: number,
  pool: WorkerPool
): Promise<{ stats: FileStats; remainingOffset: number; remainingLimit: number }> {
  const stats: FileStats = {
    path: filePath,
    rowsLoaded: 0,
    rowsDropped: 0,
    rowsSkipped: 0,
    rowsLimited: 0,
    rowsDuplicate: 0,
    rowsIndexed: 0,
  };
  let skipRemaining = offset; // consumed at dispatch: offset lines never reach a worker
  let limitRemaining = limit;

  const { batch, flushBatch, addToBatch } = makeBatcher(db, stats);

  // One dispatched unit: `layout` holds one entry per qualifying line in input
  // order (SKIPPED for offset fast-forward, PROCESSED for lines sent to the
  // pool); `lines` holds only the PROCESSED lines.
  const SKIPPED = 0;
  const PROCESSED = 1;
  interface Dispatched {
    layout: number[];
    resultsPromise: Promise<LineResult[]>;
  }

  const inflight: Dispatched[] = [];
  const maxInflight = pool.size * 4;

  let layout: number[] = [];
  let lines: string[] = [];

  function dispatch(): void {
    if (layout.length === 0) return;
    const resultsPromise = lines.length > 0 ? pool.run(lines) : Promise.resolve([]);
    inflight.push({ layout, resultsPromise });
    layout = [];
    lines = [];
  }

  // Consume the oldest dispatched batch, replaying its lines in input order.
  // Returns true when the limit was hit and processing must stop.
  async function consumeHead(): Promise<boolean> {
    const head = inflight.shift()!;
    const results = await head.resultsPromise;
    let ri = 0;
    for (const kind of head.layout) {
      stats.rowsLoaded++;

      if (stats.rowsLoaded % 100_000 === 0) {
        console.log(`  Processed ${stats.rowsLoaded} lines...`);
      }

      if (kind === SKIPPED) {
        stats.rowsSkipped++;
        continue;
      }

      const r = results[ri++];
      if (r.k === LineKind.BadJson) {
        console.warn(`  [WARN] Skipping malformed JSON line ${stats.rowsLoaded} in ${basename(filePath)}`);
        stats.rowsDropped++;
      } else if (r.k === LineKind.NonObject) {
        console.warn(`  [WARN] Skipping non-object line ${stats.rowsLoaded} in ${basename(filePath)}`);
        stats.rowsDropped++;
      } else if (r.k === LineKind.NoText) {
        console.warn(`  [WARN] Missing "${textKey}" field on line ${stats.rowsLoaded} in ${basename(filePath)}`);
        stats.rowsDropped++;
      } else if (r.k === LineKind.Prefiltered) {
        stats.rowsDropped++;
      } else {
        if (limit > 0 && limitRemaining === 0) {
          stats.rowsLimited++;
          await flushBatch();
          return true;
        }
        if (limit > 0) limitRemaining--;

        addToBatch(r.id!, r.value!);
        if (batch.size >= BATCH_SIZE) {
          await flushBatch();
        }
      }
    }
    return false;
  }

  // Stream the file line by line (gzip files are decompressed on-the-fly)
  const fileStream = fileToAsyncIterable(filePath);
  let lineBuffer = '';

  for await (const chunk of fileStream) {
    lineBuffer += chunk;

    let newlineIdx: number;
    while ((newlineIdx = lineBuffer.indexOf('\n')) !== -1) {
      const line = lineBuffer.slice(0, newlineIdx).trim();
      lineBuffer = lineBuffer.slice(newlineIdx + 1);
      if (line.length <= 10) continue;

      if (skipRemaining > 0) {
        skipRemaining--;
        layout.push(SKIPPED);
      } else {
        layout.push(PROCESSED);
        lines.push(line);
      }

      if (layout.length >= LINES_PER_WORKER_BATCH) {
        dispatch();
        if (inflight.length >= maxInflight) {
          if (await consumeHead()) {
            return { stats, remainingOffset: skipRemaining, remainingLimit: 0 };
          }
        }
      }
    }
  }

  // Dispatch the tail and drain everything in order.
  dispatch();
  while (inflight.length > 0) {
    if (await consumeHead()) {
      return { stats, remainingOffset: skipRemaining, remainingLimit: 0 };
    }
  }

  await flushBatch();

  return {
    stats,
    remainingOffset: skipRemaining,
    remainingLimit: limitRemaining,
  };
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const { inputs, source, dbPath, textKey, maxLength, vocabPath, extraFields, offset, limit, workers } = parseArgs();

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
  console.log(`Extra fields: ${extraFields.length > 0 ? extraFields.join(', ') : 'none'}`);
  console.log(`Min length: ${MIN_LENGTH}`);
  console.log(`Max length: ${maxLength}`);
  console.log(`Offset: ${offset}`);
  console.log(`Limit: ${limit === 0 ? 'none' : limit}`);
  console.log(`LevelDB: ${dbPath}`);
  console.log(`Vocabulary: ${vocab ? `${vocab.size} words (${vocabPath})` : 'none'}`);
  console.log(`Workers: ${workers > 1 ? workers : 'none (in-process)'}`);

  const db = new ClassicLevel<string, DocValue>(dbPath, {
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await db.open();

  // Spin up the worker pool only if some input actually needs it (JSONL files).
  // (Passive extension check — unsupported extensions still error out at the
  // same point in the file loop as before.)
  const hasJsonl = inputs.some(p => JSONL_EXTENSIONS.has(extname(isGzipped(p) ? stripGz(p) : p).toLowerCase()));
  const pool =
    workers > 1 && hasJsonl
      ? new WorkerPool(workers, {
          textKey,
          maxLength,
          source,
          extraFields,
          vocab: vocab ?? null,
        })
      : undefined;

  let grandLoaded = 0;
  let grandDropped = 0;
  let grandSkipped = 0;
  let grandLimited = 0;
  let grandDuplicate = 0;
  let grandIndexed = 0;
  let remainingOffset = offset;
  let remainingLimit = limit;

  try {
    for (let fi = 0; fi < inputs.length; fi++) {
      const filePath = inputs[fi];
      const fileType = detectFileType(filePath);

      if (fileType === 'jsonl') {
        console.log(`\n${'='.repeat(60)}`);
        console.log(`Processing: ${filePath}`);
        console.log('='.repeat(60));
      }

      const result =
        fileType === 'text'
          ? await processTextFile(filePath, source, db, maxLength, vocab, remainingOffset, remainingLimit)
          : await processFile(filePath, source, db, textKey, maxLength, vocab!, extraFields, remainingOffset, remainingLimit, pool);

      const { stats } = result;
      remainingOffset = result.remainingOffset;
      remainingLimit = result.remainingLimit;

      if (fileType === 'jsonl') {
        printSummary(stats, maxLength);
      }

      grandLoaded += stats.rowsLoaded;
      grandDropped += stats.rowsDropped;
      grandSkipped += stats.rowsSkipped;
      grandLimited += stats.rowsLimited;
      grandDuplicate += stats.rowsDuplicate;
      grandIndexed += stats.rowsIndexed;

      // Stop early if limit reached
      if (remainingLimit === 0 && limit > 0) {
        console.log(`\n  Limit of ${limit} record(s) reached — stopping.`);
        break;
      }

      // Periodic progress for text files
      if (fileType === 'text' && (fi + 1) % 25 === 0) {
        console.log(`  ... processed ${fi + 1}/${inputs.length} files (${grandIndexed} indexed so far)`);
      }
    }
  } finally {
    pool?.terminate();
    await db.close();
  }

  // Final summary
  console.log(`\n${'='.repeat(60)}`);
  console.log('SUMMARY');
  console.log('='.repeat(60));
  const summaryLines = [
    `  Grand loaded:     ${String(grandLoaded).padStart(12)}`,
    `  Grand dropped:    ${String(grandDropped).padStart(12)}  (quality filter)`,
    `  Grand skipped:    ${String(grandSkipped).padStart(12)}  (offset fast-forward)`,
  ];
  if (grandLimited > 0) {
    summaryLines.push(`  Grand limited:    ${String(grandLimited).padStart(12)}  (limit reached, not indexed)`);
  }
  summaryLines.push(
    `  Grand duplicates: ${String(grandDuplicate).padStart(12)}  (id collision)`,
    `  Grand indexed:    ${String(grandIndexed).padStart(12)}`
  );
  console.log(summaryLines.join('\n'));
}

if (import.meta.main && !IS_WORKER) {
  main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
}
