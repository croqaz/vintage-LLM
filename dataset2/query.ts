#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// dataset/query.ts — Redis doc:* query tool (Bun)
//
// Modes:
//   bun query.ts id <id>              Fetch a single document by ID
//   bun query.ts query "<expr>"       Scan & filter documents by JS expression
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

function parseArgs():
  | { mode: 'id'; id: string; redisUrl: string }
  | {
      mode: 'query';
      expr: string;
      limit: number;
      redisUrl: string;
    } {
  const args = process.argv.slice(2);
  let redisUrl = '';
  let limit = 100;

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
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun dataset/query.ts <mode> [options]

Modes:
  id <id>                  Fetch a single document by its ID
  query "<expression>"     Scan all doc:* keys and filter by JS expression

Options:
  -r, --redis-url <url>    Redis connection URL (default: $REDIS_URL / $VALKEY_URL)
  -l, --limit <n>          Max results for query mode (default: 100)
  -h, --help               Show this help

Examples:
  bun dataset/query.ts id 1234
  bun dataset/query.ts query "doc.length < 100 && doc.entropy >= 2"
  bun dataset/query.ts query 'doc.source == "British" && doc.words == 1'
  bun dataset/query.ts query "doc.quality < 0" --limit 10`);
      process.exit(0);
    }
  }

  if (args.length < 1) {
    console.error('Error: <mode> is required. Use "id" or "query".');
    console.log('Use -h for help.');
    process.exit(1);
  }

  const mode = args[0];

  if (mode === 'id') {
    if (args.length < 2) {
      console.error('Error: <id> is required for "id" mode.');
      process.exit(1);
    }
    return { mode: 'id', id: args[1], redisUrl };
  }

  if (mode === 'query') {
    if (args.length < 2) {
      console.error('Error: <expression> is required for "query" mode.');
      process.exit(1);
    }
    return { mode: 'query', expr: args[1], limit, redisUrl };
  }

  console.error(`Error: Unknown mode "${mode}". Use "id" or "query".`);
  console.log('Use -h for help.');
  process.exit(1);
}

// ──────────────────────────────────────────────────────────────────────────────
// ID mode — fetch a single document
// ──────────────────────────────────────────────────────────────────────────────

async function fetchById(client: RedisClient, id: string): Promise<void> {
  const meta = await client.get(`m:${id}`);

  if (meta === null) {
    console.log(`Key "${id}" not found.`);
    return;
  }

  let doc: Record<string, unknown>;
  try {
    doc = JSON.parse(meta);
  } catch {
    console.error(meta);
    return;
  }

  doc.text = await client.get(`t:${id}`);
  console.log(JSON.stringify(doc, null, 2));
}

// ──────────────────────────────────────────────────────────────────────────────
// Query mode — SCAN + filter
// ──────────────────────────────────────────────────────────────────────────────

async function queryDocs(client: RedisClient, expr: string, limit: number): Promise<void> {
  // Compile the expression into a filter function
  let filter: (doc: Record<string, unknown>) => boolean;
  try {
    // The expression is evaluated with `doc` as the implicit variable name
    filter = new Function('doc', `return (${expr});`) as (doc: Record<string, unknown>) => boolean;
    filter(EXAMPLE); // Test the filter against the example document to catch syntax errors early
  } catch (err) {
    console.error(`Error: Invalid expression — ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  }

  let cursor = 0;
  let scanned = 0;
  let matched = 0;

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
            matched++;
            console.log(`\n--- Match #${matched}/${scanned} (${keys[i]}) ---`);

            // Fetch the text blob from t:* key
            const textKey = `t:${keys[i].slice(2)}`;
            // Always include text in the output
            doc.text = await client.get(textKey);
            console.log(JSON.stringify(doc, null, 2));

            if (matched >= limit) {
              console.warn(`\n[Limit reached: ${limit} results]\n`);
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

  console.log(`\nDone. Scanned ${scanned} documents, ${matched} match(es).\n`);
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

  if (parsed.mode === 'id') {
    await fetchById(client, parsed.id);
  } else {
    await queryDocs(client, parsed.expr, parsed.limit);
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
