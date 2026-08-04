import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { ClassicLevel } from 'classic-level';

import { exportDocs, parseArgs, shardName } from '../export.ts';
import type { DocValue } from '../features.ts';

// ── helpers ──────────────────────────────────────────────────────────────────

let _seq = 0;

function makeDoc(): DocValue {
  const n = ++_seq;
  return {
    text: `doc number ${n}`,
    source: 'test',
    len: 20,
    uniqChar: 10,
    tokens: 5,
    sentences: 1,
    entropy: 100,
    quality: 100,
    compress: 100,
    dictHit: 100,
    alpha: 100,
    vowel: 100,
    ascii: 100,
    score: 100,
  };
}

function makeTempDir(): string {
  return mkdtempSync(join(tmpdir(), 'export-test-'));
}

async function openDb(dir: string): Promise<ClassicLevel<string, DocValue>> {
  const db = new ClassicLevel<string, DocValue>(dir, { valueEncoding: 'json' });
  await db.open();
  return db;
}

async function seedDb(db: ClassicLevel<string, DocValue>, count: number): Promise<void> {
  const batch = db.batch();
  for (let i = 0; i < count; i++) {
    batch.put(`doc-${i}`, makeDoc());
  }
  await batch.write();
}

// ══════════════════════════════════════════════════════════════════════════════
// shardName
// ══════════════════════════════════════════════════════════════════════════════

describe('shardName', () => {
  test('appends zero-padded shard number before .jsonl', () => {
    expect(shardName('out.jsonl', 1)).toBe('out-0001.jsonl');
    expect(shardName('out.jsonl', 42)).toBe('out-0042.jsonl');
    expect(shardName('out.jsonl', 9999)).toBe('out-9999.jsonl');
  });

  test('works with directory path', () => {
    expect(shardName('/tmp/data/out.jsonl', 3)).toBe('/tmp/data/out-0003.jsonl');
  });

  test('handles filenames without .jsonl extension', () => {
    expect(shardName('out', 1)).toBe('out-0001.jsonl');
    expect(shardName('data.bin', 7)).toBe('data.bin-0007.jsonl');
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// parseArgs (happy path only — error cases tested via CLI subprocess)
// ══════════════════════════════════════════════════════════════════════════════

describe('parseArgs', () => {
  const savedArgv = process.argv;

  afterAll(() => {
    process.argv = savedArgv;
  });

  test('parses --shard and --output', () => {
    process.argv = ['bun', 'export.ts', '-o', 'out.jsonl', '--shard', '1000', 'true'];
    const parsed = parseArgs();
    expect(parsed.shardSize).toBe(1000);
    expect(parsed.output).toBe('out.jsonl');
    expect(parsed.expr).toBe('true');
  });

  test('shardSize defaults to 0 when --shard is omitted', () => {
    process.argv = ['bun', 'export.ts'];
    const parsed = parseArgs();
    expect(parsed.shardSize).toBe(0);
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// Integration tests — real exportDocs with LevelDB
// ══════════════════════════════════════════════════════════════════════════════

describe('exportDocs with sharding', () => {
  let dbDir: string;
  let outDir: string;
  let db: ClassicLevel<string, DocValue>;

  beforeAll(async () => {
    dbDir = makeTempDir();
    outDir = makeTempDir();
    db = await openDb(dbDir);
    await seedDb(db, 7);
  });

  afterAll(async () => {
    await db.close();
    rmSync(dbDir, { recursive: true, force: true });
    rmSync(outDir, { recursive: true, force: true });
  });

  test('shardSize=3 splits 7 records into 3 files (3+3+1)', async () => {
    const base = join(outDir, 'out.jsonl');
    await exportDocs(db, 'true', null, 0, base, 3);

    const files = readdirSync(outDir).sort();
    expect(files).toEqual(['out-0001.jsonl', 'out-0002.jsonl', 'out-0003.jsonl']);

    // Each shard has exactly the right number of lines
    const lines1 = readFileSync(join(outDir, 'out-0001.jsonl'), 'utf-8').trim().split('\n');
    const lines2 = readFileSync(join(outDir, 'out-0002.jsonl'), 'utf-8').trim().split('\n');
    const lines3 = readFileSync(join(outDir, 'out-0003.jsonl'), 'utf-8').trim().split('\n');

    expect(lines1).toHaveLength(3);
    expect(lines2).toHaveLength(3);
    expect(lines3).toHaveLength(1);

    // All lines are valid JSON
    for (const line of [...lines1, ...lines2, ...lines3]) {
      expect(() => JSON.parse(line)).not.toThrow();
    }
  });

  test('shardSize=0 (no sharding) writes single file', async () => {
    const dir = makeTempDir();
    const base = join(dir, 'single.jsonl');

    await exportDocs(db, 'true', null, 0, base, 0);

    const files = readdirSync(dir);
    expect(files).toEqual(['single.jsonl']);

    const lines = readFileSync(base, 'utf-8').trim().split('\n');
    expect(lines).toHaveLength(7);

    rmSync(dir, { recursive: true, force: true });
  });

  test('shardSize larger than exported count → single shard file', async () => {
    const dir = makeTempDir();
    const base = join(dir, 'big.jsonl');

    await exportDocs(db, 'true', null, 0, base, 100);

    const files = readdirSync(dir);
    expect(files).toEqual(['big-0001.jsonl']);

    const lines = readFileSync(join(dir, 'big-0001.jsonl'), 'utf-8').trim().split('\n');
    expect(lines).toHaveLength(7);

    rmSync(dir, { recursive: true, force: true });
  });

  test('shardSize=1 produces one file per record', async () => {
    const dir = makeTempDir();
    const base = join(dir, 'each.jsonl');

    await exportDocs(db, 'true', null, 0, base, 1);

    const files = readdirSync(dir).sort();
    expect(files).toHaveLength(7);
    expect(files[0]).toBe('each-0001.jsonl');
    expect(files[6]).toBe('each-0007.jsonl');

    // Each file has exactly 1 line
    for (const f of files) {
      const lines = readFileSync(join(dir, f), 'utf-8').trim().split('\n');
      expect(lines).toHaveLength(1);
    }

    rmSync(dir, { recursive: true, force: true });
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// CLI validation
// ══════════════════════════════════════════════════════════════════════════════

describe('CLI', () => {
  test('--help mentions --shard', async () => {
    const proc = Bun.spawn(['bun', 'run', './export.ts', '--help'], {
      stdout: 'pipe',
      stderr: 'pipe',
    });
    const out = await new Response(proc.stdout).text();
    expect(out).toContain('--shard');
    expect(out).toContain('-s, --shard');
  });

  test('--shard without --output prints error', async () => {
    const proc = Bun.spawn(['bun', 'run', './export.ts', '--shard', '1000'], {
      stdout: 'pipe',
      stderr: 'pipe',
    });
    const stderr = await new Response(proc.stderr).text();
    expect(stderr).toContain('--shard requires --output');
  });

  test('--shard 0 prints error', async () => {
    const proc = Bun.spawn(['bun', 'run', './export.ts', '--shard', '0', '-o', 'out.jsonl'], {
      stdout: 'pipe',
      stderr: 'pipe',
    });
    const stderr = await new Response(proc.stderr).text();
    expect(stderr).toContain('must be a positive integer');
  });

  test('--shard abc prints error', async () => {
    const proc = Bun.spawn(['bun', 'run', './export.ts', '--shard', 'abc', '-o', 'out.jsonl'], {
      stdout: 'pipe',
      stderr: 'pipe',
    });
    const stderr = await new Response(proc.stderr).text();
    expect(stderr).toContain('must be a positive integer');
  });
});