#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// dataset3/stats.ts — LevelDB document statistics collector (Bun / Deno)
//
// Iterates over every document in a LevelDB database (created by import.ts),
// extracts per-document metrics, and prints a summary table with
// mean, min, P5, P95, max plus a per-source row count.
//
// Memory note: with up to ~100M rows we cannot keep every value in memory to
// compute percentiles. Mean / min / max are accumulated exactly in a single
// streaming pass, while P5 / P95 are estimated from a fixed-size reservoir
// sample per metric (bounded memory, independent of total row count).
// ──────────────────────────────────────────────────────────────────────────────

import { ClassicLevel } from 'classic-level';

// Number of samples kept per metric for percentile estimation.
const RESERVOIR_SIZE = 1_000_000;

type Doc = Record<string, unknown>;

interface DocRecord {
  source?: string;
  length?: number;
  uniqueChars?: number;
  words?: number;
  sentences?: number;
  quality?: number;
  compress?: number;
  entropy?: number;
}

// ──────────────────────────────────────────────────────────────────────────────
// Reservoir sampler — keeps a uniform random sample of a stream in O(k) memory.
// (Algorithm R, Vitter 1985.)
// ──────────────────────────────────────────────────────────────────────────────

class Reservoir {
  private readonly capacity: number;
  private readonly sample: number[];
  private count = 0; // total values seen
  min = Infinity;
  max = -Infinity;
  private sum = 0;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.sample = [];
  }

  add(value: number): void {
    this.count++;
    this.sum += value;
    if (value < this.min) this.min = value;
    if (value > this.max) this.max = value;

    if (this.sample.length < this.capacity) {
      this.sample.push(value);
    } else {
      // Replace an existing sample with decreasing probability.
      const j = Math.floor(Math.random() * this.count);
      if (j < this.capacity) this.sample[j] = value;
    }
  }

  get mean(): number {
    return this.count > 0 ? this.sum / this.count : 0;
  }

  // Linear-interpolation percentile over the sorted reservoir sample.
  percentile(p: number): number {
    const n = this.sample.length;
    if (n === 0) return 0;
    if (n === 1) return this.sample[0];

    const sorted = this.sample.slice().sort((a, b) => a - b);
    const rank = (p / 100) * (n - 1);
    const lo = Math.floor(rank);
    const hi = Math.ceil(rank);
    if (lo === hi) return sorted[lo];
    const frac = rank - lo;
    return sorted[lo] * (1 - frac) + sorted[hi] * frac;
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs(): { dbPath: string; limit: number } {
  const args = process.argv.slice(2);
  let dbPath = './levelDB';
  let limit = 0; // 0 = no limit

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
      console.log(`Usage: bun run dataset3/stats.ts [options]

Options:
  -d, --db <path>   LevelDB directory (default: "./levelDB")
  -l, --limit <n>   Number of documents to scan (default: all)
  -h, --help        Show this help`);
      process.exit(0);
    }
  }

  return { dbPath, limit };
}

// ──────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ──────────────────────────────────────────────────────────────────────────────

function fmtNum(n: number, decimals: number): string {
  return n.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtInt(n: number): string {
  return n.toLocaleString('en-US');
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const { dbPath, limit } = parseArgs();

  const db = new ClassicLevel<string, Doc>(dbPath, {
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await db.open();

  // One reservoir per metric (also tracks exact mean/min/max).
  const reservoirs: Record<string, Reservoir> = {
    length: new Reservoir(RESERVOIR_SIZE),
    uniqueChars: new Reservoir(RESERVOIR_SIZE),
    words: new Reservoir(RESERVOIR_SIZE),
    sentences: new Reservoir(RESERVOIR_SIZE),
    quality: new Reservoir(RESERVOIR_SIZE),
    compress: new Reservoir(RESERVOIR_SIZE),
    entropy: new Reservoir(RESERVOIR_SIZE),
  };

  // Per-source counts (bounded by the number of distinct sources, not rows).
  const sourceCounts = new Map<string, number>();

  let totalRows = 0;

  // Stream values only; never materialise the whole DB in memory.
  for await (const value of db.values()) {
    const doc = value as DocRecord;
    totalRows++;

    if (doc.source) {
      sourceCounts.set(doc.source, (sourceCounts.get(doc.source) ?? 0) + 1);
    }

    reservoirs.length.add(doc.length ?? 0);
    reservoirs.uniqueChars.add(doc.uniqueChars ?? 0);
    reservoirs.words.add(doc.words ?? 0);
    reservoirs.sentences.add(doc.sentences ?? 0);
    reservoirs.quality.add(doc.quality ?? 0);
    reservoirs.compress.add(doc.compress ?? 0);
    reservoirs.entropy.add(doc.entropy ?? 0);

    if (limit > 0 && totalRows >= limit) break;
  }

  await db.close();

  // ── Compute stats ────────────────────────────────────────────────────────

  const metricDefs = [
    { name: 'Char length', res: reservoirs.length, decimals: 1 },
    { name: 'Unique chars', res: reservoirs.uniqueChars, decimals: 2 },
    { name: 'Word count', res: reservoirs.words, decimals: 1 },
    { name: 'Sentence count', res: reservoirs.sentences, decimals: 2 },
    { name: 'Quality score', res: reservoirs.quality, decimals: 2 },
    { name: 'Compression', res: reservoirs.compress, decimals: 2 },
    { name: 'Entropy', res: reservoirs.entropy, decimals: 2 },
  ];

  // ── Print table ──────────────────────────────────────────────────────────

  console.log('|');
  console.log(`| Scanned rows: ${fmtInt(totalRows)}`);
  console.log('|');

  // Sources, most frequent first.
  const sources = [...sourceCounts.entries()].sort((a, b) => b[1] - a[1]);
  if (sources.length > 0) {
    console.log(`| Sources (${fmtInt(sources.length)}):`);
    for (const [name, count] of sources) {
      console.log(`| - ${name}: ${fmtInt(count)}`);
    }
    console.log('|');
  }

  if (limit > 0) {
    console.log(`| Limited to ${fmtInt(limit)} rows\n|`);
  }

  console.log(`| Metric             |    Mean |     Min |      P5 |     P95 |      Max |`);
  console.log(`|--------------------|---------|---------|---------|---------|----------|`);

  for (const m of metricDefs) {
    const r = m.res;
    const min = r.min === Infinity ? 0 : r.min;
    const max = r.max === -Infinity ? 0 : r.max;
    const label = m.name.padEnd(18);
    let row = `| ${label} | ${fmtNum(r.mean, m.decimals).padStart(7)}`;
    row += ` | ${fmtNum(min, m.decimals).padStart(7)}`;
    row += ` | ${fmtNum(r.percentile(5), m.decimals).padStart(7)}`;
    row += ` | ${fmtNum(r.percentile(95), m.decimals).padStart(7)}`;
    row += ` | ${fmtNum(max, m.decimals).padStart(8)} |`;
    console.log(row);
  }

  console.log();
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
