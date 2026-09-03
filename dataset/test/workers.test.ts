import { afterAll, describe, expect, test } from 'bun:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { ClassicLevel } from 'classic-level';

import { DEFAULT_MAX_LENGTH, processFile, WorkerPool } from '../import.ts';
import type { DocValue } from '../features.ts';

// ──────────────────────────────────────────────────────────────────────────────
// The parallel (worker pool) path must produce the exact same database and the
// exact same stats as the sequential path — including offset/limit handling and
// duplicate resolution.
// ──────────────────────────────────────────────────────────────────────────────

let _seq = 0;
function validLine(text?: string): string {
  return JSON.stringify({
    text:
      text ??
      `[${++_seq}] The quick brown fox jumps over the lazy dog. ` +
        'Pack my box with five dozen liquor jugs. ' +
        'How vexingly quick daft zebras jump!',
  });
}

const cleanups: string[] = [];
afterAll(() => {
  for (const p of cleanups) rmSync(p, { recursive: true, force: true });
});

function tmpPath(): string {
  const p = mkdtempSync(join(tmpdir(), 'import-worker-test-'));
  cleanups.push(p);
  return p;
}

async function openDb(dir: string): Promise<ClassicLevel<string, DocValue>> {
  const db = new ClassicLevel<string, DocValue>(dir, { valueEncoding: 'json' });
  await db.open();
  return db;
}

async function writeJsonl(lines: string[]): Promise<string> {
  const path = tmpPath() + '.jsonl';
  cleanups.push(path);
  await Bun.write(path, lines.join('\n') + '\n');
  return path;
}

async function dumpDb(db: ClassicLevel<string, DocValue>): Promise<string> {
  const entries: string[] = [];
  for await (const [k, v] of db.iterator()) {
    entries.push(k + '\t' + JSON.stringify(v));
  }
  return entries.join('\n');
}

async function runBothPaths(lines: string[], offset: number, limit: number): Promise<void> {
  const jsonlPath = await writeJsonl(lines);
  const vocab = new Set(['the', 'quick', 'brown', 'fox', 'lazy', 'dog']);

  const seqDir = tmpPath();
  const seqDb = await openDb(seqDir);
  const seq = await processFile(jsonlPath, 'test', seqDb, 'text', DEFAULT_MAX_LENGTH, vocab, [], offset, limit);
  const seqDump = await dumpDb(seqDb);
  await seqDb.close();

  const pool = new WorkerPool(2, { textKey: 'text', maxLength: DEFAULT_MAX_LENGTH, source: 'test', extraFields: [], vocab });
  try {
    const parDir = tmpPath();
    const parDb = await openDb(parDir);
    const par = await processFile(jsonlPath, 'test', parDb, 'text', DEFAULT_MAX_LENGTH, vocab, [], offset, limit, pool);
    const parDump = await dumpDb(parDb);
    await parDb.close();

    expect(par.stats).toEqual({ ...seq.stats, path: par.stats.path });
    expect(par.remainingOffset).toBe(seq.remainingOffset);
    expect(par.remainingLimit).toBe(seq.remainingLimit);
    expect(parDump).toBe(seqDump);
  } finally {
    pool.terminate();
  }
}

describe('worker pool path === sequential path', () => {
  test('plain import', async () => {
    await runBothPaths(
      Array.from({ length: 700 }, () => validLine()),
      0,
      0
    );
  }, 30_000);

  test('with duplicates and garbage', async () => {
    const dup = validLine();
    const lines = [dup, 'garbage {{{', dup, validLine(), JSON.stringify({ nope: 1 }), dup, validLine()];
    await runBothPaths(lines, 0, 0);
  }, 30_000);

  test('with offset and limit', async () => {
    await runBothPaths(
      Array.from({ length: 600 }, () => validLine()),
      37,
      111
    );
  }, 30_000);

  test('offset beyond file', async () => {
    await runBothPaths(
      Array.from({ length: 20 }, () => validLine()),
      9999,
      0
    );
  }, 30_000);
});
