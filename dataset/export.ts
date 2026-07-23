#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// export.ts — LevelDB document export tool (Bun / Deno)
//
// Iterates over all documents in a LevelDB database (created by import.ts),
// filters by JS expression, and exports matching documents as JSONL to stdout.
//
// Options:
//   -f, --fields <list>   Comma-separated list of fields to include (default: all)
//   -l, --limit <n>       Max documents to export (default: all)
//   -d, --db <path>       LevelDB directory (default: "./levelDB")
//   -o, --output <file>   Write JSONL to this file (default: stdout)
// ──────────────────────────────────────────────────────────────────────────────

import { ClassicLevel } from 'classic-level';
import type { DocValue } from './features.ts';
import { globalScore } from './features.ts';

type Doc = Record<string, unknown>;

const EXAMPLE: DocValue = {
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
  score: 75,
};

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs(): {
  expr: string;
  fields: string[] | null;
  limit: number;
  dbPath: string;
  output: string | null;
} {
  const args = process.argv.slice(2);
  let expr = '';
  let fields: string[] | null = null;
  let limit = 0; // 0 = no limit
  let dbPath = './levelDB';
  let output: string | null = null;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-d' || arg === '--db') && i + 1 < args.length) {
      dbPath = args[++i];
    } else if ((arg === '-o' || arg === '--output') && i + 1 < args.length) {
      output = args[++i];
    } else if ((arg === '-l' || arg === '--limit') && i + 1 < args.length) {
      limit = parseInt(args[++i], 10);
      if (isNaN(limit) || limit <= 0) {
        console.error('Error: --limit must be a positive integer.');
        process.exit(1);
      }
    } else if ((arg === '-f' || arg === '--fields') && i + 1 < args.length) {
      fields = args[++i]
        .split(',')
        .map(f => f.trim())
        .filter(f => f.length > 0);
      if (fields.length === 0) {
        console.error('Error: --fields requires at least one field name.');
        process.exit(1);
      }
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun export.ts [expression] [options]

Scans all documents in LevelDB, optionally filters by JS expression, and exports
matching documents as JSONL to stdout.

Options:
  -d, --db <path>          LevelDB directory (default: "./levelDB")
  -l, --limit <n>          Max documents to export (default: all)
  -f, --fields <list>      Comma-separated list of fields to include (default: all)
  -o, --output <file>      Write JSONL to this file (default: stdout)
  -h, --help               Show this help

Examples:
  bun export.ts
  bun export.ts "doc.quality < 0"
  bun export.ts 'doc.source === "British"' --limit 10
  bun export.ts "doc.entropy >= 2" --fields id,text,entropy
  bun export.ts --fields source,tokens,entropy --limit 100`);
      process.exit(0);
    } else if (!arg.startsWith('-')) {
      if (!expr) expr = arg;
    }
  }

  if (!expr) {
    expr = 'true';
  }

  return { expr, fields, limit, dbPath, output };
}

// ──────────────────────────────────────────────────────────────────────────────
// Field selector — returns a function that picks only the requested fields
// ──────────────────────────────────────────────────────────────────────────────

function makeFieldSelector(fieldNames: string[]): (doc: Doc) => Doc {
  return (doc: Doc): Doc => {
    const picked: Doc = {};
    for (const name of fieldNames) {
      if (name in doc) {
        picked[name] = doc[name];
      }
    }
    return picked;
  };
}

// ──────────────────────────────────────────────────────────────────────────────
// Export mode — iterate + filter + export
// ──────────────────────────────────────────────────────────────────────────────

async function exportDocs(
  db: ClassicLevel<string, DocValue>,
  expr: string,
  fields: string[] | null,
  limit: number,
  output: string | null
): Promise<void> {
  // Compile the expression into a filter function
  let filter: (doc: DocValue) => boolean;
  try {
    filter = new Function('doc', `return (${expr});`) as (doc: DocValue) => boolean;
    filter(EXAMPLE); // catch syntax errors early
  } catch (err) {
    console.error(`Error: Invalid expression — ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  }

  // Build the field selector (identity if no fields specified)
  const selectFields = fields ? makeFieldSelector(fields) : null;

  // Set up the output sink: a file writer (flushed every 500 writes) or stdout.
  const FLUSH_EVERY = 500;
  const sink = output ? Bun.file(output).writer() : null;
  let sinceFlush = 0;

  const writeLine = (line: string): void => {
    if (sink) {
      sink.write(line + '\n');
      if (++sinceFlush >= FLUSH_EVERY) {
        sink.flush();
        sinceFlush = 0;
      }
    } else {
      console.log(line);
    }
  };

  let scanned = 0;
  let exported = 0;

  for await (const [key, doc] of db.iterator()) {
    scanned++;

    // Quality larger than 100 is good, smaller is bad.
    // All the others have to be close to 100 to be good.
    doc.score = globalScore(doc);

    try {
      if (filter!(doc)) {
        // Build the output record
        // const out: Record<string, unknown> = { id: key };
        const out: Record<string, unknown> = {};

        // Merge doc fields (filtered if --fields specified)
        if (selectFields) {
          Object.assign(out, selectFields(doc));
        } else {
          Object.assign(out, doc);
        }

        // Write as JSONL
        writeLine(JSON.stringify(out));

        exported++;

        if (limit > 0 && exported >= limit) {
          console.error(`\n[Limit reached: ${limit} documents]\n`);
          if (sink) await sink.end();
          return;
        }
      }
    } catch (err) {
      console.error(`  [WARN] Expression error on ${key}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  if (sink) await sink.end();

  console.error(`\nDone. Scanned ${scanned} documents, exported ${exported}.\n`);
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const parsed = parseArgs();

  const db = new ClassicLevel<string, DocValue>(parsed.dbPath, {
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await db.open();

  try {
    await exportDocs(db, parsed.expr, parsed.fields, parsed.limit, parsed.output);
  } finally {
    await db.close();
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
