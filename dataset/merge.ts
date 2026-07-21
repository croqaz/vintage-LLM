#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// dataset/merge.ts — quality-aware LevelDB merge tool (Bun / Deno)
//
// Folds one or more SOURCE databases into a single TARGET database. The target
// may already exist from a previous merge, so this is safe to run repeatedly as
// the sources gain new entries.
//
//   bun dataset/merge.ts -t ./Merged-DB ./Gutenberg-DB ./Vintage-short-DB
//   bun dataset/merge.ts --target ./Merged-DB --dry-run ./Corpus-1800-1875-DB
//
// ── How conflicts are resolved ──────────────────────────────────────────────
// The record key is a SHA-512/256 hash of the *normalized* text So when the
// same ID appears in two DBs, the two records are the same underlying text
// — but their stored `text` and metrics can differ. On collision we keep the
// copy with the higher global score (see globalScore).
// Ties keep whatever is already in the target, which makes re-runs idempotent.
// The source DB identity does NOT influence the winner — it is purely quality.
// ──────────────────────────────────────────────────────────────────────────────

import { ClassicLevel } from 'classic-level';
import type { DocValue } from './features.ts';
import { globalScore } from './features.ts';

const BATCH_SIZE = 512; // LevelDB batch flush size

// Returns the value that should win. Strict `>` so ties keep `existing`.
function resolveConflict(existing: DocValue, incoming: DocValue): DocValue {
  return globalScore(incoming) > globalScore(existing) ? incoming : existing;
}

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs(): { target: string; sources: string[]; dryRun: boolean } {
  const args = process.argv.slice(2);
  let target = '';
  let dryRun = false;
  const sources: string[] = [];

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-t' || arg === '--target') && i + 1 < args.length) {
      target = args[++i];
    } else if (arg === '--dry-run') {
      dryRun = true;
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun dataset/merge.ts -t <target> [options] <source...>

Merge one or more source LevelDB databases into a target database, keeping the
highest-quality copy whenever the same document ID appears more than once.

Arguments:
  <source...>              One or more source DB directories to merge in.

Options:
  -t, --target <path>      Target DB directory (created if missing; required).
      --dry-run            Report what would change without writing anything.
  -h, --help               Show this help.

Examples:
  bun dataset/merge.ts -t ./Merged-DB ./Gutenberg-DB ./Vintage-short-DB
  bun dataset/merge.ts -t ./Merged-DB --dry-run ./Corpus-1800-1875-DB`);
      process.exit(0);
    } else if (!arg.startsWith('-')) {
      sources.push(arg);
    } else {
      console.error(`Error: Unknown argument "${arg}".`);
      console.log('Use -h for help.');
      process.exit(1);
    }
  }

  if (!target) {
    console.error('Error: -t/--target <path> is required.');
    console.log('Use -h for help.');
    process.exit(1);
  }
  if (sources.length === 0) {
    console.error('Error: at least one source DB directory is required.');
    console.log('Use -h for help.');
    process.exit(1);
  }

  // Guard against merging a DB into itself.
  const norm = (p: string) => p.replace(/\/+$/, '');
  for (const s of sources) {
    if (norm(s) === norm(target)) {
      console.error(`Error: source "${s}" is the same as the target — refusing to merge a DB into itself.`);
      process.exit(1);
    }
  }

  return { target, sources, dryRun };
}

// ──────────────────────────────────────────────────────────────────────────────
// Per-source statistics
// ──────────────────────────────────────────────────────────────────────────────

interface SourceStats {
  path: string;
  scanned: number; // records read from this source
  inserted: number; // new IDs added to target
  replaced: number; // existing IDs where this source's copy won on quality
  kept: number; // existing IDs where the target's copy won (or tied)
  collapsed: number; // within-source duplicate IDs collapsed before writing
  qualityGain: number; // Σ (scoreNew − scoreOld) over replacements
}

function newSourceStats(path: string): SourceStats {
  return {
    path,
    scanned: 0,
    inserted: 0,
    replaced: 0,
    kept: 0,
    collapsed: 0,
    qualityGain: 0,
  };
}

// ──────────────────────────────────────────────────────────────────────────────
// Count the records currently in a DB (keys-only, no value decode)
// ──────────────────────────────────────────────────────────────────────────────

async function countKeys(db: ClassicLevel<string, DocValue>): Promise<number> {
  let n = 0;
  for await (const _ of db.keys()) n++;
  return n;
}

// ──────────────────────────────────────────────────────────────────────────────
// Merge a single source DB into the target
// ──────────────────────────────────────────────────────────────────────────────

async function mergeSource(
  source: ClassicLevel<string, DocValue>,
  target: ClassicLevel<string, DocValue>,
  stats: SourceStats,
  dryRun: boolean
): Promise<void> {
  // Pending writes keyed by id. The Map collapses within-source duplicate ids
  // (resolving them by quality) before we ever touch the target.
  const batch = new Map<string, DocValue>();

  async function flushBatch(): Promise<void> {
    if (batch.size === 0) return;

    const keys = [...batch.keys()];
    const existing = await target.getMany(keys); // one round-trip
    const ops: { type: 'put'; key: string; value: DocValue }[] = [];

    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      const incoming = batch.get(key)!;
      const old = existing[i];

      if (old === undefined) {
        stats.inserted++;
        ops.push({ type: 'put', key, value: incoming });
      } else {
        const winner = resolveConflict(old, incoming);
        if (winner === incoming) {
          stats.replaced++;
          stats.qualityGain += globalScore(incoming) - globalScore(old);
          ops.push({ type: 'put', key, value: incoming });
        } else {
          stats.kept++;
        }
      }
    }

    if (ops.length > 0 && !dryRun) {
      await target.batch(ops);
    }
    batch.clear();
  }

  function addToBatch(id: string, value: DocValue): void {
    const existing = batch.get(id);
    if (existing === undefined) {
      batch.set(id, value);
    } else {
      // Two records in this source share an id — collapse by quality.
      stats.collapsed++;
      batch.set(id, resolveConflict(existing, value));
    }
  }

  let lastLogged = 0;
  for await (const [id, doc] of source.iterator()) {
    stats.scanned++;
    addToBatch(id, doc);

    if (batch.size >= BATCH_SIZE) {
      await flushBatch();
      // Log only right after a flush, when the pending Map is empty — so the
      // reported counts (new/replaced/kept) fully account for everything
      // scanned so far, with no records still buffered and uncounted.
      if (stats.scanned - lastLogged >= 100_000) {
        lastLogged = stats.scanned;
        console.log(`  ... scanned ${stats.scanned} records (${stats.inserted} new, ${stats.replaced} replaced so far)`);
      }
    }
  }

  await flushBatch();
}

// ──────────────────────────────────────────────────────────────────────────────
// Reporting
// ──────────────────────────────────────────────────────────────────────────────

function pad(n: number, w = 12): string {
  return String(n).padStart(w);
}

function printSummary(target: string, dryRun: boolean, perSource: SourceStats[], targetBefore: number, elapsedMs: number): void {
  const sum = (f: (s: SourceStats) => number) => perSource.reduce((a, s) => a + f(s), 0);
  const scanned = sum(s => s.scanned);
  const inserted = sum(s => s.inserted);
  const replaced = sum(s => s.replaced);
  const kept = sum(s => s.kept);
  const collapsed = sum(s => s.collapsed);
  const qualityGain = sum(s => s.qualityGain);
  const collisions = replaced + kept;
  const targetAfter = targetBefore + inserted; // only inserts grow the count
  const secs = elapsedMs / 1000;
  const rate = secs > 0 ? Math.round(scanned / secs) : scanned;

  console.log(`\n${'='.repeat(72)}`);
  console.log(dryRun ? 'MERGE SUMMARY  —  DRY-RUN (nothing was written)' : 'MERGE SUMMARY');
  console.log('='.repeat(72));

  // Per-source table
  console.log('\nPer-source:');
  console.log(`  ${'source'.padEnd(30)}${'scanned'.padStart(10)}${'new'.padStart(10)}${'replaced'.padStart(10)}${'kept'.padStart(10)}`);
  console.log(`  ${'-'.repeat(70)}`);
  for (const s of perSource) {
    const name = s.path.length > 30 ? '…' + s.path.slice(-29) : s.path.padEnd(30);
    console.log(`  ${name.padEnd(30)}${pad(s.scanned, 10)}${pad(s.inserted, 10)}${pad(s.replaced, 10)}${pad(s.kept, 10)}`);
    if (s.collapsed > 0) {
      console.log(`  ${' '.repeat(30)}(collapsed ${s.collapsed} within-source duplicate id(s))`);
    }
  }

  // Totals
  console.log('\nTotals:');
  console.log(`  Target DB:              ${target}`);
  console.log(`  Sources merged:         ${pad(perSource.length)}`);
  console.log(`  Records scanned:        ${pad(scanned)}`);
  console.log(`  New inserted:           ${pad(inserted)}`);
  console.log(`  ID collisions:          ${pad(collisions)}  (already present in target)`);
  console.log(`    ├─ replaced:          ${pad(replaced)}  (source copy had higher quality)`);
  console.log(`    └─ kept:              ${pad(kept)}  (target copy won or tied)`);
  console.log(`  Within-source dupes:    ${pad(collapsed)}  (collapsed before writing)`);

  // Quality effect of replacements
  const avgGain = replaced > 0 ? qualityGain / replaced : 0;
  console.log('\nQuality:');
  console.log(`  Total quality gained:   ${pad(+qualityGain.toFixed(2))}  (Σ score improvement from replacements)`);
  console.log(`  Avg gain / replacement: ${pad(+avgGain.toFixed(2))}`);

  // Size
  console.log('\nTarget size:');
  console.log(`  Records before:         ${pad(targetBefore)}`);
  console.log(`  Records ${dryRun ? 'projected:      ' : 'after:          '}${pad(targetAfter)}`);
  console.log(`  Net growth:             ${pad(inserted)}`);

  // Throughput
  console.log('\nTiming:');
  console.log(`  Elapsed:                ${pad(+secs.toFixed(1))}s`);
  console.log(`  Throughput:             ${pad(rate)} records/s`);
  console.log();
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const { target, sources, dryRun } = parseArgs();
  const startedAt = performance.now();

  console.log(`Merging ${sources.length} source(s) into: ${target}`);
  if (dryRun) console.log('*** DRY-RUN: no writes will be performed ***');
  for (const s of sources) console.log(`  source: ${s}`);

  const targetDb = new ClassicLevel<string, DocValue>(target, {
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await targetDb.open();

  const perSource: SourceStats[] = [];
  let targetBefore = 0;

  try {
    console.log('\nCounting existing target records...');
    targetBefore = await countKeys(targetDb);
    console.log(`  Target currently holds ${targetBefore} record(s).`);

    for (const srcPath of sources) {
      console.log(`\n${'─'.repeat(72)}`);
      console.log(`Merging source: ${srcPath}`);
      console.log('─'.repeat(72));

      const srcDb = new ClassicLevel<string, DocValue>(srcPath, {
        valueEncoding: 'json',
        maxFileSize: 1_000_000_000,
      });
      await srcDb.open();

      const stats = newSourceStats(srcPath);
      try {
        await mergeSource(srcDb, targetDb, stats, dryRun);
      } finally {
        await srcDb.close();
      }

      perSource.push(stats);
      console.log(`  Done: ${stats.scanned} scanned, ${stats.inserted} new, ${stats.replaced} replaced, ${stats.kept} kept.`);
    }
  } finally {
    await targetDb.close();
  }

  printSummary(target, dryRun, perSource, targetBefore, performance.now() - startedAt);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
