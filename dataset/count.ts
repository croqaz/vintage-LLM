#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// count.ts — LevelDB document counter (Bun / Deno)
//
// Scans all documents in a LevelDB database (created by import.ts), filters by a
// JS expression, and prints how many documents match out of the total.
//
//   bun count.ts "<expr>"      Count documents matching the expression
// ──────────────────────────────────────────────────────────────────────────────

import { ClassicLevel } from 'classic-level';
import type { DocValue } from './features.ts';
import { globalScore } from './features.ts';

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs(): { expr: string; dbPath: string } {
  const args = process.argv.slice(2);
  let dbPath = './levelDB';

  // Consume flags first so positional args stay clean.
  const positional: string[] = [];
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-d' || arg === '--db') && i + 1 < args.length) {
      dbPath = args[++i];
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun count.ts "<expression>" [options]

Scans all documents and prints how many match the JS expression out of the total.

Options:
  -d, --db <path>          LevelDB directory (default: "./levelDB")
  -h, --help               Show this help

Examples:
  bun count.ts "doc.len < 100 && doc.entropy >= 2"
  bun count.ts 'doc.source === "British" && doc.tokens === 1'
  bun count.ts "doc.quality < 0"`);
      process.exit(0);
    } else {
      positional.push(arg);
    }
  }

  if (positional.length < 1) {
    console.error('Error: <expression> is required.');
    console.log('Use -h for help.');
    process.exit(1);
  }

  return { expr: positional[0], dbPath };
}

// ──────────────────────────────────────────────────────────────────────────────
// Count mode — iterate + filter + tally
// ──────────────────────────────────────────────────────────────────────────────

async function countDocs(db: ClassicLevel<string, DocValue>, expr: string): Promise<void> {
  // Compile the expression into a filter function.
  let filter: (doc: DocValue) => boolean;
  try {
    filter = new Function('doc', `return (${expr});`) as (doc: DocValue) => boolean;
  } catch (err) {
    console.error(`Error: Invalid expression — ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  }

  let scanned = 0;
  let matched = 0;

  for await (const [key, doc] of db.iterator()) {
    scanned++;

    // Compute derived score so expressions can reference doc.score.
    // Quality larger than 100 is good, smaller is bad.
    // All the others have to be close to 100 to be good.
    // Perfect score > 100 (all seven = 100).
    doc.score = globalScore(doc);

    try {
      if (filter(doc)) {
        matched++;
      }
    } catch (err) {
      console.error(`  [WARN] Expression error on ${key}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  const pct = scanned > 0 ? ((matched / scanned) * 100).toFixed(3) : '0.00';
  console.log(`\n${matched.toLocaleString('en-US')} / ${scanned.toLocaleString('en-US')} documents match (${pct}%).\n`);
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const { expr, dbPath } = parseArgs();

  const db = new ClassicLevel<string, DocValue>(dbPath, {
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await db.open();

  try {
    await countDocs(db, expr);
  } finally {
    await db.close();
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
