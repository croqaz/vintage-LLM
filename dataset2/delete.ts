#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// dataset/delete.ts — Redis doc:* delete tool (Bun)
//
// Modes:
//   bun delete.ts id <id>              Delete a single document by ID
//   bun delete.ts query "<expr>"       Scan & delete documents matching expr
//
// Options:
//   --dry-run   Only show what would be deleted, don't actually delete
// ──────────────────────────────────────────────────────────────────────────────

import { RedisClient } from 'bun';

const SCAN_BATCH = 1000;

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs():
  | { mode: 'id'; id: string; redisUrl: string; dryRun: boolean }
  | {
      mode: 'query';
      expr: string;
      limit: number;
      redisUrl: string;
      dryRun: boolean;
    } {
  const args = process.argv.slice(2);
  let redisUrl = '';
  let limit = 100;
  let dryRun = false;

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
    } else if (arg === '--dry-run') {
      dryRun = true;
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun run dataset/delete.ts <mode> [options]

Modes:
  id <id>                  Delete a single document by its ID
  query "<expression>"     Scan all doc:* keys and delete matching documents

Options:
  -r, --redis-url <url>    Redis connection URL (default: $REDIS_URL / $VALKEY_URL)
  -l, --limit <n>          Max documents to process in query mode (default: 100)
      --dry-run            Only show what would be deleted, don't actually delete
  -h, --help               Show this help

Examples:
  bun run dataset/delete.ts id 1234
  bun run dataset/delete.ts query "qualityScore < 0" --dry-run
  bun run dataset/delete.ts query "source == 'spam'" --limit 1000
  bun run dataset/delete.ts query "entropy >= 2" --dry-run --limit 50`);
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
    return { mode: 'id', id: args[1], redisUrl, dryRun };
  }

  if (mode === 'query') {
    if (args.length < 2) {
      console.error('Error: <expression> is required for "query" mode.');
      process.exit(1);
    }
    return { mode: 'query', expr: args[1], limit, redisUrl, dryRun };
  }

  console.error(`Error: Unknown mode "${mode}". Use "id" or "query".`);
  console.log('Use -h for help.');
  process.exit(1);
}

// ──────────────────────────────────────────────────────────────────────────────
// ID mode — delete a single document
// ──────────────────────────────────────────────────────────────────────────────

async function deleteById(client: RedisClient, id: string, dryRun: boolean): Promise<void> {
  let key = `m:${id}`;

  if (dryRun) {
    const meta = await client.get(key);
    if (meta === null) {
      console.log(`Key "${key}" not found — nothing to delete.`);
      return;
    }
    console.log(`[DRY-RUN] Would delete: ${key}`);
    console.log(`  Value: ${JSON.stringify(JSON.parse(meta), null, 2)}`);
    return;
  }

  const result = await client.del(`m:${id}`, `t:${id}`);
  if (result === 1) {
    console.log(`Deleted: ${key}`);
  } else {
    console.log(`Key "${key}" not found — nothing to delete.`);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Query mode — SCAN + filter + delete
// ──────────────────────────────────────────────────────────────────────────────

async function queryAndDelete(client: RedisClient, expr: string, limit: number, dryRun: boolean): Promise<void> {
  // Compile the expression into a filter function
  let filter: (doc: Record<string, unknown>) => boolean;
  try {
    filter = new Function('doc', `return (${expr});`) as (doc: Record<string, unknown>) => boolean;
  } catch (err) {
    console.error(`Error: Invalid expression — ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  }

  let cursor = 0;
  let scanned = 0;
  let matched = 0;
  const matchedKeys: string[] = [];

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
            matchedKeys.push(keys[i]);
            if (matched >= limit) {
              break;
            }
          }
        } catch (err) {
          console.error(`  [WARN] Expression error on ${keys[i]}: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    }

    if (matched >= limit || cursor === 0) break;
  }

  // ── Print all matched IDs ────────────────────────────────────────────────

  console.log(`\nFound ${matched} document(s) matching the expression.`);
  console.log(`\n--- IDs to be ${dryRun ? 'deleted (DRY-RUN)' : 'deleted'} ---`);
  for (const key of matchedKeys) {
    console.log(key);
  }
  console.log(`\nTotal: ${matchedKeys.length} document(s)`);

  // ── Actually delete ──────────────────────────────────────────────────────

  if (dryRun) {
    console.log(`\n[DRY-RUN] No documents were deleted.`);
    return;
  }

  if (matchedKeys.length === 0) {
    console.log(`\nNothing to delete.`);
    return;
  }

  // Batch delete with limit
  cursor = 0;
  let deleted = 0;
  while (cursor < matchedKeys.length) {
    const batchKeys = matchedKeys.slice(cursor, cursor + SCAN_BATCH);
    const deleteResult = await client.send('DEL', batchKeys);
    // Also delete corresponding t:* keys
    const tKeys = batchKeys.map(k => 't:' + k.slice(2));
    await client.send('DEL', tKeys);
    console.log(`Deleted batch of ${batchKeys.length} document(s).`);
    deleted += typeof deleteResult === 'number' ? deleteResult : parseInt(String(deleteResult), 10);
    cursor += SCAN_BATCH;
  }

  // ── Print statistics ─────────────────────────────────────────────────────

  console.log(`\n${'='.repeat(60)}`);
  console.log('STATISTICS');
  console.log('='.repeat(60));
  console.log(`  Documents scanned:  ${scanned}`);
  console.log(`  Documents matched:  ${matchedKeys.length}`);
  console.log(`  Documents deleted:  ${deleted}`);
  console.log();
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
    await deleteById(client, parsed.id, parsed.dryRun);
  } else {
    await queryAndDelete(client, parsed.expr, parsed.limit, parsed.dryRun);
  }

  client.close();
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
