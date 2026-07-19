#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// dataset/source.ts — Overwrite the `source` field on every document (Bun / Deno)
//
// Scans a LevelDB database and sets `doc.source = <value>` on EVERY record,
// writing the change back in batches.
//
//   bun dataset/source.ts --source gutenberg
//   bun dataset/source.ts -d ./Gutenberg-DB --source gutenberg
//   bun dataset/source.ts -d ./Vintage-short-DB -s british --yes
//
// Options:
//   -d, --db <path>          LevelDB directory (default: "./levelDB")
//   -s, --source <value>     New source value to write into every record (required)
//   -y, --yes                Skip the confirmation prompt
//   -h, --help               Show this help
//
// ⚠️  DESTRUCTIVE — READ THIS FIRST ⚠️
//   This OVERWRITES the `source` field of ALL records in the selected DB.
//   Any previous source values are lost FOREVER. There is NO UNDO.
//   Make sure -d points at the right database before you confirm.
// ──────────────────────────────────────────────────────────────────────────────

import { ClassicLevel } from 'classic-level';

type Doc = Record<string, unknown>;

const BATCH_SIZE = 512;

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs(): { dbPath: string; source: string; yes: boolean } {
  const args = process.argv.slice(2);
  let dbPath = './levelDB';
  let source: string | undefined;
  let yes = false;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-d' || arg === '--db') && i + 1 < args.length) {
      dbPath = args[++i];
    } else if ((arg === '-s' || arg === '--source') && i + 1 < args.length) {
      source = args[++i];
    } else if (arg === '-y' || arg === '--yes') {
      yes = true;
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun dataset/source.ts [options] --source <value>

Overwrite the \`source\` field on EVERY record in the selected DB.

Options:
  -d, --db <path>          LevelDB directory (default: "./levelDB")
  -s, --source <value>     New source value to write into every record (required)
  -y, --yes                Skip the confirmation prompt
  -h, --help               Show this help

Examples:
  bun dataset/source.ts --source gutenberg
  bun dataset/source.ts -d ./Gutenberg-DB --source gutenberg
  bun dataset/source.ts -d ./Vintage-short-DB -s british --yes`);
      process.exit(0);
    } else {
      console.error(`Error: Unknown argument "${arg}".`);
      console.log('Use -h for help.');
      process.exit(1);
    }
  }

  if (source === undefined || source === '') {
    console.error('Error: --source <value> is required.');
    console.log('Use -h for help.');
    process.exit(1);
  }

  return { dbPath, source: source as string, yes };
}

// ──────────────────────────────────────────────────────────────────────────────
// Confirmation — make the destructive nature unmistakable
// ──────────────────────────────────────────────────────────────────────────────

async function confirm(dbPath: string, source: string): Promise<boolean> {
  console.warn('');
  console.warn('  ' + '⚠'.repeat(30));
  console.warn('  ⚠  DESTRUCTIVE OPERATION — THERE IS NO UNDO  ⚠');
  console.warn('  ' + '⚠'.repeat(30));
  console.warn('');
  console.warn(`  You are about to OVERWRITE the "source" field of`);
  console.warn(`  EVERY record in this database:`);
  console.warn('');
  console.warn(`      DB:      ${dbPath}`);
  console.warn(`      source → "${source}"`);
  console.warn('');
  console.warn(`  The previous source of ALL records will be lost FOREVER.`);
  console.warn(`  This CANNOT be undone.`);
  console.warn('');

  process.stdout.write(`  Type the new source ("${source}") to confirm: `);

  const line: string = await new Promise(resolve => {
    process.stdin.resume();
    process.stdin.once('data', (data: { toString(): string }) => {
      process.stdin.pause();
      resolve(data.toString());
    });
  });

  return line.trim() === source;
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const { dbPath, source, yes } = parseArgs();

  if (!yes) {
    const ok = await confirm(dbPath, source);
    if (!ok) {
      console.error('\nConfirmation failed — aborting. No records were changed.');
      process.exit(1);
    }
  }

  const db = new ClassicLevel<string, Doc>(dbPath, {
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await db.open();

  try {
    let scanned = 0;
    let updated = 0;
    let batch = db.batch();
    let pending = 0;

    for await (const [key, doc] of db.iterator()) {
      scanned++;

      try {
        doc.source = source;
        batch.put(key, doc);
        pending++;

        if (pending >= BATCH_SIZE) {
          await batch.write();
          updated += pending;
          console.log(`Updated ${updated} record(s)...`);
          batch = db.batch();
          pending = 0;
        }
      } catch (err) {
        console.error(`  [WARN] Error on ${key}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    if (pending > 0) {
      await batch.write();
      updated += pending;
    }

    console.log(`\n${'='.repeat(60)}`);
    console.log('STATISTICS');
    console.log('='.repeat(60));
    console.log(`  DB path:            ${dbPath}`);
    console.log(`  Records scanned:    ${scanned}`);
    console.log(`  Records updated:    ${updated}`);
    console.log(`  New source value:   "${source}"`);
    console.log();
  } finally {
    await db.close();
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
