#!/usr/bin/env bun

import { ClassicLevel } from 'classic-level';

const BATCH_SIZE = 2048;

function parseArgs() {
  const args = process.argv.slice(2);
  let dbPath = './levelDB';

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-d' || arg === '--db') && i + 1 < args.length) {
      dbPath = args[++i];
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun run mapProc.ts [options]

Options:
  -d, --db <path>           LevelDB directory (default: "./levelDB")
  -h, --help                Show this help`);
      process.exit(0);
    }
  }

  return { dbPath };
}

// ──────────────────────────────────────────────────────────────────────────────
// UPDATE PROCESS FUNCTION
// ──────────────────────────────────────────────────────────────────────────────
// This acts as a map + process.
// - Return the modified object to save/update it in the database.
// - Return null or undefined to skip the record (no-op, keeps original).
function processRecord(key: string, record: any): any | null {
  if (record.source === 'BritishLibrary') {
    record.source = 'British';
    return record;
  } else if (record.source === 'LOC-PC') {
    record.source = 'LOC-PD';
    return record;
  }

  // Nothing changed for this record: return null so it is NOT rewritten.
  // (Rewriting every record in a multi-GB DB triggers a compaction storm and
  // looks like an infinite hang.)
  return null;
}

async function main() {
  const { dbPath } = parseArgs();
  console.log(`Open LevelDB at ${dbPath}`);
  const db = new ClassicLevel<string, any>(dbPath, {
    keyEncoding: 'utf8',
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await db.open();

  let batch = db.batch();
  let batchCount = 0;
  let processedCount = 0;
  let updatedCount = 0;

  console.log('Iterating over LevelDB records...');

  for await (const [key, value] of db.iterator()) {
    processedCount++;

    const updatedValue = processRecord(key, value);

    if (updatedValue !== null && updatedValue !== undefined) {
      batch.put(key, updatedValue);
      batchCount++;
      updatedCount++;
    }

    if (batchCount >= BATCH_SIZE) {
      await batch.write();
      batch = db.batch();
      batchCount = 0;
    }

    if (processedCount % 100_000 === 0) {
      console.log(`  Processed: ${processedCount}, Updated so far: ${updatedCount}`);
    }
  }

  if (batchCount > 0) {
    await batch.write();
  }

  console.log(`Done.\n  Total Processed: ${processedCount}\n  Total Updated: ${updatedCount}`);
  await db.close();
}

main().catch(console.error);
