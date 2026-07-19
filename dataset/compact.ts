#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// dataset/compact.ts — LevelDB manual compaction tool (Bun / Deno)
//
// Walks the whole keyspace and triggers db.compactRange(start, end) in batches,
// so a large database is compacted incrementally rather than in one huge sweep.
//
//   bun dataset/compact.ts                 Compact the entire DB in batches
//   bun dataset/compact.ts -d ./my-DB      Compact a specific DB directory
//   bun dataset/compact.ts -b 50000        Use a custom batch size (keys/range)
//
// From the classic-level docs:
//   db.compactRange(start, end)
//     Manually trigger a database compaction in the range [start..end].
//     Returns a promise.
// ──────────────────────────────────────────────────────────────────────────────

import { ClassicLevel } from 'classic-level';

type Doc = Record<string, unknown>;

const DEFAULT_BATCH_SIZE = 50_000;

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs(): { dbPath: string; batchSize: number } {
  const args = process.argv.slice(2);
  let dbPath = './levelDB';
  let batchSize = DEFAULT_BATCH_SIZE;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-d' || arg === '--db') && i + 1 < args.length) {
      dbPath = args[++i];
    } else if ((arg === '-b' || arg === '--batch') && i + 1 < args.length) {
      batchSize = parseInt(args[++i], 10);
      if (isNaN(batchSize) || batchSize <= 0) {
        console.error('Error: --batch must be a positive integer.');
        process.exit(1);
      }
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun dataset/compact.ts [options]

Manually trigger LevelDB compaction across the whole keyspace, in batches.

Options:
  -d, --db <path>          LevelDB directory (default: "./levelDB")
  -b, --batch <n>          Keys per compaction range (default: ${DEFAULT_BATCH_SIZE})
  -h, --help               Show this help

Examples:
  bun dataset/compact.ts
  bun dataset/compact.ts -d ./Gutenberg-DB
  bun dataset/compact.ts -d ./Gutenberg-DB -b 100000`);
      process.exit(0);
    } else {
      console.error(`Error: Unknown argument "${arg}".`);
      console.log('Use -h for help.');
      process.exit(1);
    }
  }

  return { dbPath, batchSize };
}

// ──────────────────────────────────────────────────────────────────────────────
// Compaction — split the keyspace into batches and compact each range
// ──────────────────────────────────────────────────────────────────────────────

async function compact(db: ClassicLevel<string, Doc>, batchSize: number): Promise<void> {
  // First pass: collect batch boundaries (every `batchSize`-th key) using a
  // keys-only iterator so we never load values into memory.
  const boundaries: string[] = [];
  let total = 0;

  for await (const key of db.keys()) {
    if (total % batchSize === 0) {
      boundaries.push(key);
    }
    total++;
  }

  if (total === 0) {
    console.log('Database is empty — nothing to compact.');
    return;
  }

  // The last key marks the upper bound of the final range. `\xff` is appended so
  // the range [start..end] is inclusive of the true last key.
  let lastKey = '';
  for await (const key of db.keys({ reverse: true, limit: 1 })) {
    lastKey = key;
  }

  const numBatches = boundaries.length;
  console.log(`Compacting ${total} key(s) in ${numBatches} batch(es) of up to ${batchSize}...\n`);

  for (let i = 0; i < numBatches; i++) {
    const start = boundaries[i];
    // End of this batch is the start of the next, or just past the last key.
    const end = i + 1 < numBatches ? boundaries[i + 1] : lastKey + '\xff';

    const t0 = performance.now();
    await db.compactRange(start, end);
    const secs = ((performance.now() - t0) / 1000).toFixed(1);

    console.log(`  [${i + 1}/${numBatches}] compacted [${start} .. ${end}] in ${secs}s`);
  }

  console.log(`\nDone. Compacted ${total} key(s) across ${numBatches} range(s).`);
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const { dbPath, batchSize } = parseArgs();

  const db = new ClassicLevel<string, Doc>(dbPath, {
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await db.open();

  try {
    await compact(db, batchSize);
  } finally {
    await db.close();
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
