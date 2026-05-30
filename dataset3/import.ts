#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// dataset3/import.ts — JSONL → LevelDB importer (Bun / Deno)
//
// Reads a JSON-lines file and stores each document in a LevelDB database using
// the document's `id` field as the key. Designed to stream tens / hundreds of
// millions of rows without loading the whole file into memory.
//
// Conflict handling: if a key already exists, the document with more fields wins.
// ──────────────────────────────────────────────────────────────────────────────

import { createReadStream } from 'node:fs';
import { basename } from 'node:path';
import { ClassicLevel } from 'classic-level';

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

const BATCH_SIZE = 10_000; // documents buffered before a single LevelDB batch write
const PROGRESS_EVERY = 100_000; // log progress every N lines

type Doc = Record<string, unknown>;

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs(): { input: string; dbPath: string; idKey: string } {
  const args = process.argv.slice(2);
  let input = '';
  let dbPath = './levelDB';
  let idKey = 'id';

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-i' || arg === '--input') && i + 1 < args.length) {
      input = args[++i];
    } else if ((arg === '-d' || arg === '--db') && i + 1 < args.length) {
      dbPath = args[++i];
    } else if ((arg === '-k' || arg === '--id-key') && i + 1 < args.length) {
      idKey = args[++i];
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun run dataset3/import.ts [options]

Options:
  -i, --input <file>   JSONL file path (required)
  -d, --db <path>      LevelDB directory (default: "./levelDB")
  -k, --id-key <key>   JSON field name used as the key (default: "id")
  -h, --help           Show this help`);
      process.exit(0);
    }
  }

  if (!input) {
    console.error('Error: --input is required. Use -h for help.');
    process.exit(1);
  }

  return { input, dbPath, idKey };
}

// ──────────────────────────────────────────────────────────────────────────────
// Conflict resolution
//
// Decides which document to keep when the same `id` appears more than once
// (either already in the DB or earlier in the same file).
//
// Default policy: prefer the document with more fields. On a tie, keep the
// incoming document (last write wins).
//
// NOTE: Custom merge / priority logic goes here later. For example, you might:
//   - merge fields from both documents instead of picking one
//   - prefer a specific `source` over another
//   - compare a `quality` / `updatedAt` field
// ──────────────────────────────────────────────────────────────────────────────

function resolveConflict(existing: Doc, incoming: Doc): Doc | null {
  const existingFields = Object.keys(existing).length;
  const incomingFields = Object.keys(incoming).length;

  if (existing.text === incoming.text) {
    return null;
  }
  if (incoming.source === 'cli' && existing.source !== 'cli') {
    return null;
  }

  // Prefer more fields; ties keep the incoming document.
  return incomingFields >= existingFields ? incoming : existing;
}

// ──────────────────────────────────────────────────────────────────────────────
// Import
// ──────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const { input, dbPath, idKey } = parseArgs();

  // valueEncoding 'json' lets us put/get plain objects directly.
  const db = new ClassicLevel<string, Doc>(dbPath, {
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await db.open();

  let rowsLoaded = 0;
  let rowsDropped = 0;
  let rowsConflict = 0;

  // Buffer the latest document per id for the current batch. Using a Map also
  // dedups ids that repeat within the same batch before we hit LevelDB.
  const buffer = new Map<string, Doc>();

  async function flush(): Promise<void> {
    if (buffer.size === 0) return;

    const ids = [...buffer.keys()];
    // Single round-trip to fetch all existing values for this batch.
    const existing = await db.getMany(ids);

    const batch = db.batch();
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      const incoming = buffer.get(id)!;
      const prev = existing[i];

      let winner: Doc | null = incoming;
      if (prev !== undefined) {
        winner = resolveConflict(prev, incoming);
        if (winner === null) {
          // Documents are identical, no update needed.
          continue;
        }
        rowsConflict++;
      }
      batch.put(id, winner);
    }
    await batch.write();
    buffer.clear();
  }

  // Stream the file line by line to keep memory flat regardless of file size.
  const fileStream = createReadStream(input, { encoding: 'utf8' });
  let lineBuffer = '';

  for await (const chunk of fileStream) {
    lineBuffer += chunk;

    let newlineIdx: number;
    while ((newlineIdx = lineBuffer.indexOf('\n')) !== -1) {
      const line = lineBuffer.slice(0, newlineIdx).trim();
      lineBuffer = lineBuffer.slice(newlineIdx + 1);
      if (line.length < 10) continue;

      rowsLoaded++;
      if (rowsLoaded % PROGRESS_EVERY === 0) {
        console.log(`  Processed ${rowsLoaded} lines (${rowsConflict} conflicts resolved)...`);
      }

      let doc: unknown;
      try {
        doc = JSON.parse(line);
      } catch {
        console.warn(`  [WARN] Skipping malformed JSON on line ${rowsLoaded} in ${basename(input)}`);
        rowsDropped++;
        continue;
      }

      if (doc === null || typeof doc !== 'object' || Array.isArray(doc)) {
        console.warn(`  [WARN] Skipping non-object on line ${rowsLoaded} in ${basename(input)}`);
        rowsDropped++;
        continue;
      }

      const record = doc as Doc;
      const id = record[idKey];
      if (id === undefined || id === null || (typeof id !== 'string' && typeof id !== 'number')) {
        console.warn(`  [WARN] Missing/invalid "${idKey}" on line ${rowsLoaded} in ${basename(input)}`);
        rowsDropped++;
        continue;
      }
      if (!record.text || typeof record.text !== 'string' || record.text.trim().length === 0) {
        console.warn(`  [WARN] Missing/invalid "text" on line ${rowsLoaded} in ${basename(input)}`);
        rowsDropped++;
        continue;
      }

      const key = String(id);
      // Resolve in-batch duplicates immediately so the buffer holds one doc per id.
      const pending = buffer.get(key);
      buffer.set(key, pending === undefined ? record : resolveConflict(pending, record));

      if (buffer.size >= BATCH_SIZE) {
        await flush();
      }
    }
  }

  // Handle a trailing line without a newline.
  const lastLine = lineBuffer.trim();
  if (lastLine.length > 0) {
    rowsLoaded++;
    try {
      const record = JSON.parse(lastLine) as Doc;
      const id = record?.[idKey];
      if (id !== undefined && id !== null && (typeof id === 'string' || typeof id === 'number')) {
        const key = String(id);
        const pending = buffer.get(key);
        buffer.set(key, pending === undefined ? record : resolveConflict(pending, record));
      } else {
        rowsDropped++;
        console.warn(`  [WARN] Missing/invalid "${idKey}" on final line in ${basename(input)}`);
      }
    } catch {
      rowsDropped++;
      console.warn(`  [WARN] Skipping malformed JSON on final line in ${basename(input)}`);
    }
  }

  await flush();
  await db.close();

  console.log('───────────────────────────────────────────────');
  console.log(`Done. Loaded ${rowsLoaded}, dropped ${rowsDropped}, conflicts resolved ${rowsConflict}.`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
