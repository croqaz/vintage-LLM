#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// dataset/stats.ts — Redis doc:* statistics collector (Bun)
//
// Iterates over all "doc:*" keys via SCAN, extracts per-document metrics,
// and prints a summary table with mean, min, P5, P95, max.
// ──────────────────────────────────────────────────────────────────────────────

import { RedisClient } from 'bun';

const SCAN_BATCH = 1000;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

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

interface MetricStats {
  name: string;
  values: number[];
  mean: number;
  min: number;
  p5: number;
  p95: number;
  max: number;
  decimals: number;
}

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs(): { limit: number; redisUrl: string } {
  const args = process.argv.slice(2);
  let limit = 0; // 0 = no limit
  let redisUrl = '';

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-l' || arg === '--limit') && i + 1 < args.length) {
      limit = parseInt(args[++i], 10);
      if (isNaN(limit) || limit <= 0) {
        console.error('Error: --limit must be a positive integer.');
        process.exit(1);
      }
    } else if ((arg === '-r' || arg === '--redis-url') && i + 1 < args.length) {
      redisUrl = args[++i];
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun dataset/stats.ts [options]

Options:
  -l, --limit <n>       Number of doc:* keys to sample (default: all)
  -r, --redis-url <url> Redis connection URL (default: $REDIS_URL / $VALKEY_URL)
  -h, --help            Show this help`);
      process.exit(0);
    }
  }

  return { limit, redisUrl };
}

// ──────────────────────────────────────────────────────────────────────────────
// Percentile helper (linear interpolation)
// ──────────────────────────────────────────────────────────────────────────────

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];

  const rank = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  const frac = rank - lo;

  if (lo === hi) return sorted[lo];
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
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
  const { limit, redisUrl } = parseArgs();

  // Connect to Redis
  const client = new RedisClient(redisUrl || undefined, {
    connectionTimeout: 2500,
    maxRetries: 3,
  });
  await client.connect();

  // ── Collect all sources ──────────────────────────────────────────────────

  const sources = new Set<string>();
  // Per-metric accumulators
  const values: Record<string, number[]> = {
    length: [],
    uniqueChars: [],
    words: [],
    sentences: [],
    quality: [],
    compress: [],
    entropy: [],
  };

  let cursor = 0;
  let scanned = 0;
  let totalRows = 0;

  while (true) {
    const result = await client.send('SCAN', [String(cursor), 'MATCH', 'm:*', 'COUNT', String(SCAN_BATCH)]);
    cursor = parseInt(result[0], 10);
    const keys = result[1] as string[];

    if (keys.length > 0) {
      // MGET all values in this batch
      const mgetArgs = keys.map(k => k);
      const rawValues = await client.send('MGET', mgetArgs);

      for (let i = 0; i < keys.length; i++) {
        const raw = rawValues[i];
        if (raw === null || raw === undefined) continue;

        let doc: DocRecord;
        try {
          doc = JSON.parse(typeof raw === 'string' ? raw : raw.toString());
        } catch {
          continue; // skip malformed
        }

        totalRows++;

        // Collect source
        if (doc.source) sources.add(doc.source);

        // Collect metric values (default to 0 if missing)
        values.length.push(doc.length ?? 0);
        values.uniqueChars.push(doc.uniqueChars ?? 0);
        values.words.push(doc.words ?? 0);
        values.sentences.push(doc.sentences ?? 0);
        values.quality.push(doc.quality ?? 0);
        values.compress.push(doc.compress ?? 0);
        values.entropy.push(doc.entropy ?? 0);
      }
    }

    scanned += keys.length;

    // Check limit
    if (limit > 0 && totalRows >= limit) {
      break;
    }

    if (cursor === 0) break; // scan complete
  }

  // Trim to limit if we over-scanned (SCAN may return more than limit)
  if (limit > 0) {
    for (const key of Object.keys(values)) {
      values[key] = values[key].slice(0, limit);
    }
  }

  // ── Compute stats ────────────────────────────────────────────────────────

  const metricDefs = [
    { name: 'Char length', arr: values.length, decimals: 1 },
    { name: 'Unique chars', arr: values.uniqueChars, decimals: 2 },
    { name: 'Word count', arr: values.words, decimals: 1 },
    { name: 'Sentence count', arr: values.sentences, decimals: 2 },
    { name: 'Quality score', arr: values.quality, decimals: 2 },
    { name: 'Compression ratio', arr: values.compress, decimals: 2 },
    { name: 'Entropy', arr: values.entropy, decimals: 2 },
  ];

  const metrics: MetricStats[] = metricDefs.map(d => {
    const sorted = d.arr.slice().sort((a, b) => a - b);
    return {
      name: d.name,
      values: d.arr,
      mean: sorted.reduce((s, v) => s + v, 0) / sorted.length,
      min: sorted[0],
      p5: percentile(sorted, 5),
      p95: percentile(sorted, 95),
      max: sorted[sorted.length - 1],
      decimals: d.decimals,
    };
  });

  // ── Print table ──────────────────────────────────────────────────────────

  const sourceList = Array.from(sources).sort().join(', ');

  console.log('|');
  console.log(`| Scanned rows: ${fmtInt(totalRows)}`);
  console.log(`| Sources: ${sourceList}`);
  console.log('|');

  if (limit > 0) {
    console.log(`| Limited to ${limit > 0 ? fmtInt(limit) : 'all'} rows\n|`);
  }

  // Header
  console.log(`| Metric             |    Mean |     Min |      P5 |     P95 |     Max |`);
  console.log(`|--------------------|---------|---------|---------|---------|---------|`);

  // Rows
  for (const m of metrics) {
    const label = m.name.padEnd(18);
    let row = `| ${label} | ${fmtNum(m.mean, m.decimals).padStart(7)}`;
    row += ` | ${fmtNum(m.min, m.decimals).padStart(7)}`;
    row += ` | ${fmtNum(m.p5, m.decimals).padStart(7)}`;
    row += ` | ${fmtNum(m.p95, m.decimals).padStart(7)}`;
    row += ` | ${fmtNum(m.max, m.decimals).padStart(7)} |`;
    console.log(row);
  }

  console.log();
  client.close();
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
