#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// dataset/query.ts — LevelDB document query tool (Bun / Deno)
//
// Modes:
//   bun query.ts id <id>              Fetch a single document by ID
//   bun query.ts query "<expr>"       Scan & filter documents by JS expression
// ──────────────────────────────────────────────────────────────────────────────

import { ClassicLevel } from 'classic-level';

type Doc = Record<string, unknown>;

const EXAMPLE: Doc = {
  id: 'abc123',
  text: 'Hello world',
  source: 'A',
  len: 100,
  uniqChar: 32,
  tokens: 25,
  sentences: 3,
  entropy: 111.25,
  quality: 123.36,
  compress: 99.85,
  dictHit: 92.5,
  alpha: 95.0,
  vowel: 88.0,
  ascii: 99.5,
  score: -10,
};

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs():
  | { mode: 'id'; id: string; dbPath: string }
  | {
      mode: 'query';
      expr: string;
      limit: number;
      dbPath: string;
    } {
  const args = process.argv.slice(2);
  let dbPath = './levelDB';
  let limit = 100;

  // Consume flags first so positional args stay clean.
  const positional: string[] = [];
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-d' || arg === '--db') && i + 1 < args.length) {
      dbPath = args[++i];
    } else if ((arg === '-l' || arg === '--limit') && i + 1 < args.length) {
      limit = parseInt(args[++i], 10);
      if (isNaN(limit) || limit <= 0) {
        console.error('Error: --limit must be a positive integer.');
        process.exit(1);
      }
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun dataset/query.ts <mode> [options]

Modes:
  id <id>                  Fetch a single document by its ID
  query "<expression>"     Scan all documents and filter by JS expression

Options:
  -d, --db <path>          LevelDB directory (default: "./levelDB")
  -l, --limit <n>          Max results for query mode (default: 100)
  -h, --help               Show this help

Examples:
  bun dataset/query.ts id 1234
  bun dataset/query.ts query "doc.len < 100 && doc.entropy >= 2"
  bun dataset/query.ts query 'doc.source === "British" && doc.tokens === 1'
  bun dataset/query.ts query "doc.quality < 0" --limit 10`);
      process.exit(0);
    } else {
      positional.push(arg);
    }
  }

  if (positional.length < 1) {
    console.error('Error: <mode> is required. Use "id" or "query".');
    console.log('Use -h for help.');
    process.exit(1);
  }

  const mode = positional[0];

  if (mode === 'id') {
    if (positional.length < 2) {
      console.error('Error: <id> is required for "id" mode.');
      process.exit(1);
    }
    return { mode: 'id', id: positional[1], dbPath };
  }

  if (mode === 'query') {
    if (positional.length < 2) {
      console.error('Error: <expression> is required for "query" mode.');
      process.exit(1);
    }
    return { mode: 'query', expr: positional[1], limit, dbPath };
  }

  console.error(`Error: Unknown mode "${mode}". Use "id" or "query".`);
  console.log('Use -h for help.');
  process.exit(1);
}

// ──────────────────────────────────────────────────────────────────────────────
// ID mode — fetch a single document
// ──────────────────────────────────────────────────────────────────────────────

async function fetchById(db: ClassicLevel<string, Doc>, id: string): Promise<void> {
  const doc = await db.get(id);
  if (doc === undefined) {
    console.log(`Key "${id}" not found.`);
    return;
  }
  console.log(JSON.stringify(doc, null, 2));
}

// ──────────────────────────────────────────────────────────────────────────────
// Query mode — iterate + filter
// ──────────────────────────────────────────────────────────────────────────────

async function queryDocs(db: ClassicLevel<string, Doc>, expr: string, limit: number): Promise<void> {
  // Compile the expression into a filter function.
  let filter: (doc: Doc) => boolean;
  try {
    filter = new Function('doc', `return (${expr});`) as (doc: Doc) => boolean;
    filter(EXAMPLE); // catch syntax errors early
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
    // Perfect score > 0 (all seven = 100).
    doc.score = -(
      ((doc.quality as number) ?? 0) -
      100 +
      Math.abs(((doc.compress as number) ?? 0) - 100) +
      Math.abs(((doc.entropy as number) ?? 0) - 100) +
      Math.abs(((doc.dictHit as number) ?? 0) - 100) +
      Math.abs(((doc.alpha as number) ?? 0) - 100) +
      Math.abs(((doc.vowel as number) ?? 0) - 100) +
      Math.abs(((doc.ascii as number) ?? 0) - 100)
    );

    try {
      if (filter(doc)) {
        matched++;
        if (doc.text.length > 16_000) {
          doc.text = doc.text.slice(0, 8_000) + '\n... [TRUNCATED] ...\n' + doc.text.slice(-8_000);
        }
        console.log(`\n--- Match #${matched}/${scanned} (${key}) ---`);
        console.log(JSON.stringify(doc, null, 2));

        if (matched >= limit) {
          console.warn(`\n[Limit reached: ${limit} results]\n`);
          return;
        }
      }
    } catch (err) {
      console.error(`  [WARN] Expression error on ${key}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  console.log(`\nDone. Scanned ${scanned} documents, ${matched} match(es).\n`);
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const parsed = parseArgs();

  const db = new ClassicLevel<string, Doc>(parsed.dbPath, {
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await db.open();

  try {
    if (parsed.mode === 'id') {
      await fetchById(db, parsed.id);
    } else {
      await queryDocs(db, parsed.expr, parsed.limit);
    }
  } finally {
    await db.close();
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
