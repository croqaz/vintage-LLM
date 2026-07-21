#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// dataset/source.ts — rewrite the `source` field on documents (Bun / Deno)
//
// Two mutually-exclusive modes:
//
//   BULK      Set `source` on EVERY record to one value.
//               bun dataset/source.ts -d ./Gutenberg-DB --source gutenberg
//
//   SELECTIVE Rename specific source values only (normalize/correct labels).
//             Records whose source doesn't match a pair are left untouched.
//               bun dataset/source.ts -d ./hmd-lwm-DB \
//                 -p 'British Library Living with Machines Project=LWM' \
//                 -p 'British Library Heritage Made Digital Newspapers=HMD'
//
// Options:
//   -d, --db <path>          LevelDB directory (default: "./levelDB")
//   -s, --source <value>     BULK mode: value written into every record
//   -p, --source-pairs A=B   SELECTIVE mode: rename source A → B (repeatable)
//       --dry-run            Report what would change without writing anything
//   -y, --yes                Skip the confirmation prompt
//   -h, --help               Show this help
//
// Only records that actually change are written back, so unmatched records are
// never rewritten (avoids a needless compaction storm).
//
// ⚠️  DESTRUCTIVE — there is NO UNDO. In BULK mode every previous source value
//     is lost forever. Double-check -d before you confirm.
// ──────────────────────────────────────────────────────────────────────────────

import { ClassicLevel } from 'classic-level';
import { basename } from 'node:path';

type Doc = Record<string, unknown>;

const BATCH_SIZE = 512;

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

type ParsedArgs =
  | {
      mode: 'bulk';
      dbPath: string;
      source: string;
      yes: boolean;
      dryRun: boolean;
    }
  | {
      mode: 'pairs';
      dbPath: string;
      pairs: Map<string, string>;
      yes: boolean;
      dryRun: boolean;
    };

function parseArgs(): ParsedArgs {
  const args = process.argv.slice(2);
  let dbPath = './levelDB';
  let source: string | undefined;
  const pairs = new Map<string, string>();
  let yes = false;
  let dryRun = false;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-d' || arg === '--db') && i + 1 < args.length) {
      dbPath = args[++i];
    } else if ((arg === '-s' || arg === '--source') && i + 1 < args.length) {
      source = args[++i];
    } else if ((arg === '-p' || arg === '--source-pairs') && i + 1 < args.length) {
      const raw = args[++i];
      const eq = raw.indexOf('=');
      if (eq <= 0 || eq === raw.length - 1) {
        console.error(`Error: --source-pairs must be "OLD=NEW" (got "${raw}").`);
        process.exit(1);
      }
      const from = raw.slice(0, eq);
      const to = raw.slice(eq + 1);
      if (pairs.has(from)) {
        console.error(`Error: duplicate rename for source "${from}".`);
        process.exit(1);
      }
      pairs.set(from, to);
    } else if (arg === '--dry-run') {
      dryRun = true;
    } else if (arg === '-y' || arg === '--yes') {
      yes = true;
    } else if (arg === '-h' || arg === '--help') {
      console.log(
        `Usage: bun dataset/source.ts -d <db> (--source <value> | --source-pairs OLD=NEW ...)

Rewrite the \`source\` field on documents in a LevelDB database.

Modes (mutually exclusive):
  -s, --source <value>       BULK: set every record's source to <value>
  -p, --source-pairs OLD=NEW SELECTIVE: rename source OLD → NEW (repeatable);
                             records not matching any pair are left untouched

Options:
  -d, --db <path>            LevelDB directory (default: "./levelDB")
      --dry-run              Report what would change without writing anything
  -y, --yes                  Skip the confirmation prompt
  -h, --help                 Show this help

Examples:
  bun dataset/source.ts -d ./Gutenberg-DB --source gutenberg
  bun dataset/source.ts -d ./hmd-lwm-DB \\
    -p 'British Library Living with Machines Project=LWM' \\
    -p 'British Library Heritage Made Digital Newspapers=HMD' --dry-run`
      );
      process.exit(0);
    } else {
      console.error(`Error: Unknown argument "${arg}".`);
      console.log('Use -h for help.');
      process.exit(1);
    }
  }

  const hasSource = source !== undefined && source !== '';
  const hasPairs = pairs.size > 0;

  if (hasSource && hasPairs) {
    console.error('Error: --source and --source-pairs are mutually exclusive. Use one mode.');
    process.exit(1);
  }
  if (!hasSource && !hasPairs) {
    console.error('Error: provide either --source <value> or --source-pairs OLD=NEW.');
    console.log('Use -h for help.');
    process.exit(1);
  }

  return hasSource ? { mode: 'bulk', dbPath, source: source as string, yes, dryRun } : { mode: 'pairs', dbPath, pairs, yes, dryRun };
}

// ──────────────────────────────────────────────────────────────────────────────
// Confirmation
// ──────────────────────────────────────────────────────────────────────────────

async function readLine(): Promise<string> {
  return new Promise(resolve => {
    process.stdin.resume();
    process.stdin.once('data', (data: { toString(): string }) => {
      process.stdin.pause();
      resolve(data.toString());
    });
  });
}

// BULK mode: prove intent by typing the new source value (destructive: rewrites all).
async function confirmBulk(dbPath: string, source: string): Promise<boolean> {
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
  console.warn('');
  process.stdout.write(`  Type the new source ("${source}") to confirm: `);
  return (await readLine()).trim() === source;
}

// SELECTIVE mode: only matching records change, so guard mainly against the
// wrong DB — confirm by typing the DB directory name.
async function confirmPairs(dbPath: string, pairs: Map<string, string>): Promise<boolean> {
  const dbName = basename(dbPath.replace(/\/+$/, '')) || dbPath;
  console.warn('');
  console.warn(`  About to rename source labels in: ${dbPath}`);
  console.warn('');
  for (const [from, to] of pairs) {
    console.warn(`      "${from}"  →  "${to}"`);
  }
  console.warn('');
  console.warn(`  Only records whose source matches an entry above will change.`);
  console.warn('');
  process.stdout.write(`  Type the DB name ("${dbName}") to confirm: `);
  return (await readLine()).trim() === dbName;
}

// ──────────────────────────────────────────────────────────────────────────────
// Bulk mode
// ──────────────────────────────────────────────────────────────────────────────

async function runBulk(db: ClassicLevel<string, Doc>, source: string, dryRun: boolean): Promise<void> {
  let scanned = 0;
  let changed = 0;
  let unchanged = 0;
  let batch = db.batch();
  let pending = 0;

  for await (const [key, doc] of db.iterator()) {
    scanned++;
    if (doc.source === source) {
      unchanged++;
      continue;
    }
    changed++;
    if (dryRun) continue;

    doc.source = source;
    batch.put(key, doc);
    pending++;
    if (pending >= BATCH_SIZE) {
      await batch.write();
      console.log(`  Updated ${changed} record(s)...`);
      batch = db.batch();
      pending = 0;
    }
  }

  if (pending > 0) await batch.write();

  console.log(`\n${'='.repeat(60)}`);
  console.log(dryRun ? 'STATISTICS — DRY-RUN (nothing written)' : 'STATISTICS');
  console.log('='.repeat(60));
  console.log(`  Records scanned:    ${scanned}`);
  console.log(`  ${dryRun ? 'Would change:     ' : 'Changed:          '}  ${changed}`);
  console.log(`  Already correct:      ${unchanged}`);
  console.log(`  New source value:   "${source}"`);
  console.log();
}

// ──────────────────────────────────────────────────────────────────────────────
// Selective (pairs) mode
// ──────────────────────────────────────────────────────────────────────────────

async function runPairs(db: ClassicLevel<string, Doc>, pairs: Map<string, string>, dryRun: boolean): Promise<void> {
  let scanned = 0;
  let renamed = 0;
  const perPair = new Map<string, number>(); // "from → to" → count
  const unmatched = new Map<string, number>(); // untouched source value → count
  let batch = db.batch();
  let pending = 0;

  for await (const [key, doc] of db.iterator()) {
    scanned++;
    const cur = typeof doc.source === 'string' ? doc.source : String(doc.source);
    const to = pairs.get(cur);

    if (to === undefined) {
      unmatched.set(cur, (unmatched.get(cur) ?? 0) + 1);
      continue;
    }

    renamed++;
    perPair.set(`${cur} → ${to}`, (perPair.get(`${cur} → ${to}`) ?? 0) + 1);
    if (dryRun) continue;

    doc.source = to;
    batch.put(key, doc);
    pending++;
    if (pending >= BATCH_SIZE) {
      await batch.write();
      console.log(`  Renamed ${renamed} record(s)...`);
      batch = db.batch();
      pending = 0;
    }
  }

  if (pending > 0) await batch.write();

  console.log(`\n${'='.repeat(60)}`);
  console.log(dryRun ? 'STATISTICS — DRY-RUN (nothing written)' : 'STATISTICS');
  console.log('='.repeat(60));
  console.log(`  Records scanned:    ${scanned}`);
  console.log(`  ${dryRun ? 'Would rename:     ' : 'Renamed:          '}  ${renamed}`);

  console.log('\n  Per mapping:');
  for (const [from] of pairs) {
    const to = pairs.get(from)!;
    const label = `${from} → ${to}`;
    console.log(`    ${label}: ${perPair.get(label) ?? 0}`);
  }

  if (unmatched.size > 0) {
    console.log('\n  Untouched sources:');
    const sorted = [...unmatched.entries()].sort((a, b) => b[1] - a[1]);
    for (const [src, n] of sorted) {
      console.log(`    ${src}: ${n}`);
    }
  }
  console.log();
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const parsed = parseArgs();

  // Confirmation (skipped by --yes and in --dry-run, which writes nothing).
  if (!parsed.yes && !parsed.dryRun) {
    const ok = parsed.mode === 'bulk' ? await confirmBulk(parsed.dbPath, parsed.source) : await confirmPairs(parsed.dbPath, parsed.pairs);
    if (!ok) {
      console.error('\nConfirmation failed — aborting. No records were changed.');
      process.exit(1);
    }
  }

  const db = new ClassicLevel<string, Doc>(parsed.dbPath, {
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await db.open();

  try {
    if (parsed.mode === 'bulk') {
      await runBulk(db, parsed.source, parsed.dryRun);
    } else {
      await runPairs(db, parsed.pairs, parsed.dryRun);
    }
  } finally {
    await db.close();
  }
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
