// #!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// dataset3/delete.ts — LevelDB document delete tool (Bun / Deno)
//
// Modes:
//   bun delete.ts id <id>              Delete a single document by ID
//   bun delete.ts query "<expr>"       Scan & delete documents matching expr
//
// Options:
//   --dry-run   Only show what would be deleted, don't actually delete
// ──────────────────────────────────────────────────────────────────────────────

import { ClassicLevel } from 'classic-level';

type Doc = Record<string, unknown>;

const BATCH_SIZE = 512;

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs():
  | { mode: 'id'; id: string; dbPath: string; dryRun: boolean }
  | {
      mode: 'query';
      expr: string;
      limit: number;
      dbPath: string;
      dryRun: boolean;
    } {
  const args = process.argv.slice(2);
  let dbPath = './levelDB';
  let limit = 100;
  let dryRun = false;

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
    } else if (arg === '--dry-run') {
      dryRun = true;
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun dataset3/delete.ts <mode> [options]

Modes:
  id <id>                  Delete a single document by its ID
  query "<expression>"     Scan all documents and delete matching ones

Options:
  -d, --db <path>          LevelDB directory (default: "./levelDB")
  -l, --limit <n>          Max documents to process in query mode (default: 100)
      --dry-run            Only show what would be deleted, don't actually delete
  -h, --help               Show this help

Examples:
  bun dataset3/delete.ts id 1234
  bun dataset3/delete.ts query "doc.qualityScore < 0" --dry-run
  bun dataset3/delete.ts query "doc.source === 'spam'" --limit 1000
  bun dataset3/delete.ts query "doc.entropy >= 2" --dry-run --limit 50`);
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
    return { mode: 'id', id: positional[1], dbPath, dryRun };
  }

  if (mode === 'query') {
    if (positional.length < 2) {
      console.error('Error: <expression> is required for "query" mode.');
      process.exit(1);
    }
    return { mode: 'query', expr: positional[1], limit, dbPath, dryRun };
  }

  console.error(`Error: Unknown mode "${mode}". Use "id" or "query".`);
  console.log('Use -h for help.');
  process.exit(1);
}

// ──────────────────────────────────────────────────────────────────────────────
// ID mode — delete a single document
// ──────────────────────────────────────────────────────────────────────────────

async function deleteById(db: ClassicLevel<string, Doc>, id: string, dryRun: boolean): Promise<void> {
  let doc: Doc | undefined;
  try {
    doc = await db.get(id);
  } catch {
    console.log(`Key "${id}" not found — nothing to delete.`);
    return;
  }

  if (dryRun) {
    console.log(`[DRY-RUN] Would delete: ${id}`);
    console.log(`  Value: ${JSON.stringify(doc, null, 2)}`);
    return;
  }

  await db.del(id);
  console.log(`Deleted: ${id}`);
}

// ──────────────────────────────────────────────────────────────────────────────
// Query mode — iterate + filter + delete
// ──────────────────────────────────────────────────────────────────────────────

async function queryAndDelete(db: ClassicLevel<string, Doc>, expr: string, limit: number, dryRun: boolean): Promise<void> {
  // Compile the expression into a filter function.
  let filter: (doc: Doc) => boolean;
  try {
    filter = new Function('doc', `return (${expr});`) as (doc: Doc) => boolean;
  } catch (err) {
    console.error(`Error: Invalid expression — ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  }

  let scanned = 0;
  let matched = 0;
  const matchedKeys: string[] = [];

  for await (const [key, doc] of db.iterator()) {
    scanned++;

    try {
      if (filter(doc)) {
        matched++;
        matchedKeys.push(key);
        if (matched >= limit) break;
      }
    } catch (err) {
      console.error(`  [WARN] Expression error on ${key}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  // ── Print all matched IDs ──────────────────────────────────────────────

  console.log(`\nFound ${matched} document(s) matching the expression.`);
  console.log(`\n--- IDs to be ${dryRun ? 'deleted (DRY-RUN)' : 'deleted'} ---`);
  for (const key of matchedKeys) {
    console.log(key);
  }
  console.log(`\nTotal: ${matchedKeys.length} document(s)`);

  // ── Actually delete ────────────────────────────────────────────────────

  if (dryRun) {
    console.log(`\n[DRY-RUN] No documents were deleted.`);
    return;
  }

  if (matchedKeys.length === 0) {
    console.log(`\nNothing to delete.`);
    return;
  }

  // Batch delete
  let deleted = 0;
  for (let i = 0; i < matchedKeys.length; i += BATCH_SIZE) {
    const batchKeys = matchedKeys.slice(i, i + BATCH_SIZE);
    const batch = db.batch();
    for (const key of batchKeys) {
      batch.del(key);
    }
    await batch.write();
    deleted += batchKeys.length;
    console.log(`Deleted batch of ${batchKeys.length} document(s).`);
  }

  // ── Print statistics ───────────────────────────────────────────────────

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

  const db = new ClassicLevel<string, Doc>(parsed.dbPath, {
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await db.open();

  try {
    if (parsed.mode === 'id') {
      await deleteById(db, parsed.id, parsed.dryRun);
    } else {
      await queryAndDelete(db, parsed.expr, parsed.limit, parsed.dryRun);
    }
  } finally {
    await db.close();
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
