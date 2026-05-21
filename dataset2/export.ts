#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// dataset/export.ts — Redis doc:* export tool (Bun)
//
// Scans all "m:*" keys, filters by JS expression, and exports matching
// documents as JSONL to stdout.
//
// Modes:
//   bun export.ts "<expr>"    Scan & export documents matching expr
//
// Options:
//   --fields <list>   Comma-separated list of fields to include (default: all)
//   --limit <n>       Max documents to export (default: all)
// ──────────────────────────────────────────────────────────────────────────────

import { RedisClient } from 'bun';

const SCAN_BATCH = 1000;

const EXAMPLE = {
  source: 'A',
  length: 100,
  uniqueChars: 32,
  words: 25,
  sentences: 3,
  entropy: 111.25,
  quality: 103.36,
  compress: 105.85,
};

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs(): {
  expr: string;
  fields: string[] | null;
  limit: number;
  redisUrl: string;
} {
  const args = process.argv.slice(2);
  let expr = '';
  let fields: string[] | null = null;
  let limit = 0; // 0 = no limit
  let redisUrl = '';

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-r' || arg === '--redis-url') && i + 1 < args.length) {
      redisUrl = args[++i];
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
      console.log(`Usage: bun dataset/export.ts query "<expression>" [options]

Scans all "m:*" keys in Redis, filters by JS expression, and exports
matching documents as JSONL to stdout.

Options:
  -r, --redis-url <url>    Redis connection URL (default: $REDIS_URL / $VALKEY_URL)
  -l, --limit <n>          Max documents to export (default: all)
  -f, --fields <list>      Comma-separated list of fields to include (default: all)
  -h, --help               Show this help

Examples:
  bun dataset/export.ts "quality < 0"
  bun dataset/export.ts 'source == "British"' --limit 10
  bun dataset/export.ts "entropy >= 2" --fields id,text,entropy
  bun dataset/export.ts "words > 10" --fields source,words,entropy --limit 100`);
      process.exit(0);
    }
  }

  if (args.length < 1) {
    console.error('Error: <expression> is required.');
    console.log('Use -h for help.');
    process.exit(1);
  }

  // First non-flag argument is the expression
  expr = args[0];

  return { expr, fields, limit, redisUrl };
}

// ──────────────────────────────────────────────────────────────────────────────
// Field selector — returns a function that picks only the requested fields
// ──────────────────────────────────────────────────────────────────────────────

function makeFieldSelector(fieldNames: string[]): (doc: Record<string, unknown>) => Record<string, unknown> {
  return (doc: Record<string, unknown>): Record<string, unknown> => {
    const picked: Record<string, unknown> = {};
    for (const name of fieldNames) {
      if (name in doc) {
        picked[name] = doc[name];
      }
    }
    return picked;
  };
}

// ──────────────────────────────────────────────────────────────────────────────
// Query mode — SCAN + filter + export
// ──────────────────────────────────────────────────────────────────────────────

async function exportDocs(client: RedisClient, expr: string, fields: string[] | null, limit: number): Promise<void> {
  // Compile the expression into a filter function
  let filter: (doc: Record<string, unknown>) => boolean;
  try {
    filter = new Function('doc', `return (${expr});`) as (doc: Record<string, unknown>) => boolean;
    filter(EXAMPLE); // Test the filter against the example document to catch syntax errors early
  } catch (err) {
    console.error(`Error: Invalid expression — ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  }

  // Build the field selector (identity if no fields specified)
  const selectFields = fields ? makeFieldSelector(fields) : null;

  let cursor = 0;
  let scanned = 0;
  let exported = 0;

  while (true) {
    const result = await client.send('SCAN', [String(cursor), 'MATCH', 'm:*', 'COUNT', String(SCAN_BATCH)]);
    cursor = parseInt(result[0], 10);
    const keys = result[1] as string[];

    if (keys.length > 0) {
      const rawValues = await client.send('MGET', keys);

      for (let i = 0; i < keys.length; i++) {
        const raw = rawValues[i];
        if (raw === null || raw === undefined) continue;

        let doc: Record<string, unknown>;
        try {
          doc = JSON.parse(raw);
        } catch {
          continue;
        }

        scanned++;

        try {
          if (filter(doc)) {
            // Build the output record
            const out: Record<string, unknown> = { id: keys[i].slice(2) };

            // Fetch the text blob from t:* key
            const textKey = `t:${keys[i].slice(2)}`;
            // Always include text in the output
            out.text = await client.get(textKey);

            // Merge doc fields (filtered if --fields specified)
            if (selectFields) {
              Object.assign(out, selectFields(doc));
            } else {
              Object.assign(out, doc);
            }

            // Write as JSONL
            console.log(JSON.stringify(out));
            console.log();
            exported++;

            if (limit > 0 && exported >= limit) {
              console.error(`\n[Limit reached: ${limit} documents]\n`);
              client.close();
              return;
            }
          }
        } catch (err) {
          console.error(`  [WARN] Expression error on ${keys[i]}: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    }

    if (cursor === 0) break;
  }

  console.error(`\nDone. Scanned ${scanned} documents, exported ${exported}.\n`);
  client.close();
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const parsed = parseArgs();

  const client = new RedisClient(parsed.redisUrl || undefined, {
    connectionTimeout: 2500,
    maxRetries: 3,
  });
  await client.connect();

  await exportDocs(client, parsed.expr, parsed.fields, parsed.limit);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
