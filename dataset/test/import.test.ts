import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { ClassicLevel } from 'classic-level';

import { DEFAULT_MAX_LENGTH, type FileStats, MAX_UNIQUE_CHARS, MIN_LENGTH, MIN_UNIQUE_CHARS, prefilter, processFile } from '../import.ts';
import type { DocValue } from '../features.ts';

// ── helpers ──────────────────────────────────────────────────────────────────

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

const GARBAGE = 'not valid json at all {{{';
const NO_TEXT = JSON.stringify({ foo: 'bar' });

function makeTempDb(): string {
  return mkdtempSync(join(tmpdir(), 'import-test-'));
}

async function openDb(dir: string): Promise<ClassicLevel<string, DocValue>> {
  const db = new ClassicLevel<string, DocValue>(dir, { valueEncoding: 'json' });
  await db.open();
  return db;
}

// Write lines to a temp JSONL file
function writeJsonl(lines: string[]): string {
  const dir = makeTempDb();
  // We use the dir for the DB but need a file for the JSONL input.
  // Use a temp file approach: write to a .jsonl in the same temp area.
  const path = dir + '.jsonl';
  Bun.write(path, lines.join('\n') + '\n');
  return path;
}

// ──────────────────────────────────────────────────────────────────────────────
// Simulated pipeline — mirrors processFile loop but importable and LevelDB-free.
// Uses the REAL prefilter from import.ts.
// ──────────────────────────────────────────────────────────────────────────────

interface SimResult {
  loaded: number;
  dropped: number;
  skipped: number;
  limited: number;
  indexed: number;
  remainingOffset: number;
  remainingLimit: number;
}

function simulate(
  lines: string[],
  offset: number,
  limit: number = 0,
  maxLength: number = DEFAULT_MAX_LENGTH,
  textKey: string = 'text'
): SimResult {
  const result: SimResult = {
    loaded: 0,
    dropped: 0,
    skipped: 0,
    limited: 0,
    indexed: 0,
    remainingOffset: offset,
    remainingLimit: limit,
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.length <= 10) continue;

    result.loaded++;

    // ── offset fast-forward (line-level, no parse) ──
    if (result.remainingOffset > 0) {
      result.remainingOffset--;
      result.skipped++;
      continue;
    }

    // ── JSON parse ──
    let obj: unknown;
    try {
      obj = JSON.parse(trimmed);
    } catch {
      result.dropped++;
      continue;
    }

    if (obj === null || typeof obj !== 'object') {
      result.dropped++;
      continue;
    }

    const record = obj as Record<string, unknown>;
    const text = record[textKey];
    if (!text || typeof text !== 'string') {
      result.dropped++;
      continue;
    }

    // ── prefilter (REAL, imported from ../import.ts) ──
    if (!prefilter(text, maxLength).ok) {
      result.dropped++;
      continue;
    }

    // ── limit check ──
    if (limit > 0 && result.remainingLimit === 0) {
      result.limited++;
      continue;
    }
    if (limit > 0) result.remainingLimit--;

    // ── indexed ──
    result.indexed++;
  }

  return result;
}

// ══════════════════════════════════════════════════════════════════════════════
// prefilter (real import)
// ══════════════════════════════════════════════════════════════════════════════

describe('prefilter', () => {
  test('rejects text <= MIN_LENGTH', () => {
    expect(prefilter('x'.repeat(MIN_LENGTH), DEFAULT_MAX_LENGTH).ok).toBe(false);
  });

  test('accepts text just above MIN_LENGTH with sufficient chars/words', () => {
    const text =
      'The quick brown fox jumps over the lazy dog. ' +
      'Pack my box with five dozen liquor jugs. ' +
      'How vexingly quick daft zebras jump!';
    expect(prefilter(text, DEFAULT_MAX_LENGTH).ok).toBe(true);
  });

  test('rejects text > maxLength', () => {
    expect(prefilter('x'.repeat(201), 200).ok).toBe(false);
  });

  test('rejects text with too few unique chars', () => {
    const text = 'ababababab ababababab ababababab ababababab ababababab ' + 'ababababab ababababab ababababab ababababab ababababab ab';
    expect(prefilter(text, DEFAULT_MAX_LENGTH).ok).toBe(false);
  });

  test('rejects text with <= 2 words', () => {
    const t = 'abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuv';
    expect(prefilter(t, DEFAULT_MAX_LENGTH).ok).toBe(false);
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// offset (line-level fast-forward)
// ══════════════════════════════════════════════════════════════════════════════

describe('offset', () => {
  test('offset=0 processes all valid', () => {
    const r = simulate([validLine(), validLine(), validLine()], 0);
    expect(r.skipped).toBe(0);
    expect(r.indexed).toBe(3);
  });

  test('offset=1 skips first line, processes rest', () => {
    const r = simulate([validLine(), validLine(), validLine()], 1);
    expect(r.skipped).toBe(1);
    expect(r.indexed).toBe(2);
    expect(r.remainingOffset).toBe(0);
  });

  test('offset=5 with only 3 lines skips all, none indexed', () => {
    const r = simulate([validLine(), validLine(), validLine()], 5);
    expect(r.loaded).toBe(3);
    expect(r.skipped).toBe(3);
    expect(r.indexed).toBe(0);
    expect(r.remainingOffset).toBe(2);
  });

  test('offset=9999 with only 10 lines — all skipped, offset carries over', () => {
    const lines = Array.from({ length: 10 }, () => validLine());
    const r = simulate(lines, 9999);
    expect(r.loaded).toBe(10);
    expect(r.skipped).toBe(10);
    expect(r.indexed).toBe(0);
    expect(r.remainingOffset).toBe(9989);
  });

  test('offset skips garbage lines without parsing them', () => {
    const lines = [GARBAGE, GARBAGE, validLine(), validLine()];
    const r = simulate(lines, 2);
    expect(r.skipped).toBe(2);
    expect(r.dropped).toBe(0); // never parsed → never dropped
    expect(r.indexed).toBe(2);
  });

  test('offset carries over across files (two simulate calls)', () => {
    const f1 = [validLine(), validLine()];
    const f2 = [validLine(), validLine(), validLine()];

    const r1 = simulate(f1, 3);
    expect(r1.skipped).toBe(2);
    expect(r1.remainingOffset).toBe(1);

    const r2 = simulate(f2, r1.remainingOffset);
    expect(r2.skipped).toBe(1);
    expect(r2.indexed).toBe(2);
    expect(r2.remainingOffset).toBe(0);
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// limit
// ══════════════════════════════════════════════════════════════════════════════

describe('limit', () => {
  test('limit=0 means no limit — all indexed', () => {
    const r = simulate([validLine(), validLine(), validLine()], 0, 0);
    expect(r.indexed).toBe(3);
    expect(r.limited).toBe(0);
  });

  test('limit=1 indexes exactly 1 record', () => {
    const r = simulate([validLine(), validLine(), validLine()], 0, 1);
    expect(r.indexed).toBe(1);
    expect(r.limited).toBe(2);
  });

  test('limit=3 with 3 records indexes all, none limited', () => {
    const r = simulate([validLine(), validLine(), validLine()], 0, 3);
    expect(r.indexed).toBe(3);
    expect(r.limited).toBe(0);
    expect(r.remainingLimit).toBe(0);
  });

  test('limit=5 with only 3 records — all indexed, remainingLimit=2', () => {
    const r = simulate([validLine(), validLine(), validLine()], 0, 5);
    expect(r.indexed).toBe(3);
    expect(r.limited).toBe(0);
    expect(r.remainingLimit).toBe(2);
  });

  test('limit does not count dropped records', () => {
    const lines = [GARBAGE, validLine(), NO_TEXT, validLine(), GARBAGE, validLine(), validLine()];
    const r = simulate(lines, 0, 2);
    expect(r.dropped).toBe(3);
    expect(r.indexed).toBe(2);
    expect(r.limited).toBe(2);
  });

  test('limit carries over across files', () => {
    const r1 = simulate([validLine(), validLine(), validLine()], 0, 4);
    expect(r1.indexed).toBe(3);
    expect(r1.remainingLimit).toBe(1);

    const r2 = simulate([validLine(), validLine()], 0, r1.remainingLimit);
    expect(r2.indexed).toBe(1);
    expect(r2.limited).toBe(1);
    expect(r2.remainingLimit).toBe(0);
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// offset + limit combined
// ══════════════════════════════════════════════════════════════════════════════

describe('offset + limit', () => {
  test('offset=2 limit=2 — skip 2 lines, index next 2, limit the rest', () => {
    const lines = [validLine(), validLine(), validLine(), validLine(), validLine(), validLine()];
    const r = simulate(lines, 2, 2);
    expect(r.skipped).toBe(2);
    expect(r.indexed).toBe(2);
    expect(r.limited).toBe(2);
    expect(r.remainingOffset).toBe(0);
    expect(r.remainingLimit).toBe(0);
  });

  test('offset=5 limit=99999 with only 10 lines', () => {
    const lines = Array.from({ length: 10 }, () => validLine());
    const r = simulate(lines, 5, 99999);
    expect(r.skipped).toBe(5);
    expect(r.indexed).toBe(5);
    expect(r.limited).toBe(0);
    expect(r.remainingLimit).toBe(99994);
  });

  test('offset=99999 limit=5 with only 10 lines — all skipped, limit untouched', () => {
    const lines = Array.from({ length: 10 }, () => validLine());
    const r = simulate(lines, 99999, 5);
    expect(r.skipped).toBe(10);
    expect(r.indexed).toBe(0);
    expect(r.limited).toBe(0);
    expect(r.remainingLimit).toBe(5);
  });

  test('offset=5 limit=0 — skip 5, no limit, index everything after', () => {
    const lines = Array.from({ length: 10 }, () => validLine());
    const r = simulate(lines, 5, 0);
    expect(r.skipped).toBe(5);
    expect(r.indexed).toBe(5);
    expect(r.limited).toBe(0);
  });

  test('offset=0 limit=0 — process everything (defaults)', () => {
    const lines = Array.from({ length: 5 }, () => validLine());
    const r = simulate(lines, 0, 0);
    expect(r.skipped).toBe(0);
    expect(r.indexed).toBe(5);
    expect(r.limited).toBe(0);
  });

  test('offset=3 limit=1 with garbage mixed in', () => {
    const lines = [GARBAGE, validLine(), GARBAGE, validLine(), validLine(), GARBAGE, validLine()];
    const r = simulate(lines, 3, 1);
    expect(r.skipped).toBe(3);
    expect(r.indexed).toBe(1);
    expect(r.limited).toBe(2);
    expect(r.dropped).toBe(1); // GARBAGE after offset, limit already 0
  });

  test('garbage after offset but before limit: dropped, do not consume limit', () => {
    const lines = [validLine(), GARBAGE, NO_TEXT, validLine(), validLine(), validLine()];
    const r = simulate(lines, 1, 2);
    expect(r.skipped).toBe(1);
    expect(r.dropped).toBe(2);
    expect(r.indexed).toBe(2);
    expect(r.limited).toBe(1);
  });

  test('empty file (all lines ≤10 chars) with offset and limit', () => {
    const r = simulate(['', '   ', 'short', '1234567890'], 100, 100);
    expect(r.loaded).toBe(0);
    expect(r.skipped).toBe(0);
    expect(r.indexed).toBe(0);
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// Integration tests — real processFile against temp LevelDB
// ══════════════════════════════════════════════════════════════════════════════

describe('integration (real processFile + LevelDB)', () => {
  test('offset=0 limit=0 indexes all valid records', async () => {
    const jsonlPath = writeJsonl([validLine(), validLine(), validLine()]);
    const dbDir = makeTempDb();
    const db = await openDb(dbDir);

    try {
      const { stats } = await processFile(jsonlPath, 'test', db, 'text', DEFAULT_MAX_LENGTH, new Set(), [], 0, 0);
      expect(stats.rowsLoaded).toBe(3);
      expect(stats.rowsSkipped).toBe(0);
      expect(stats.rowsLimited).toBe(0);
      expect(stats.rowsDropped).toBe(0);
      expect(stats.rowsIndexed).toBe(3);

      // Verify data actually landed in the DB
      const keys = await db.keys().all();
      expect(keys.length).toBe(3);
    } finally {
      await db.close();
      rmSync(dbDir, { recursive: true, force: true });
      rmSync(jsonlPath, { force: true });
    }
  });

  test('offset skips lines in real processFile', async () => {
    const lines = [validLine(), validLine(), validLine(), validLine(), validLine()];
    const jsonlPath = writeJsonl(lines);
    const dbDir = makeTempDb();
    const db = await openDb(dbDir);

    try {
      const { stats, remainingOffset } = await processFile(jsonlPath, 'test', db, 'text', DEFAULT_MAX_LENGTH, new Set(), [], 3, 0);
      expect(remainingOffset).toBe(0);
      expect(stats.rowsLoaded).toBe(5);
      expect(stats.rowsSkipped).toBe(3);
      expect(stats.rowsIndexed).toBe(2);

      const keys = await db.keys().all();
      expect(keys.length).toBe(2);
    } finally {
      await db.close();
      rmSync(dbDir, { recursive: true, force: true });
      rmSync(jsonlPath, { force: true });
    }
  });

  test('limit caps indexed records in real processFile', async () => {
    const lines = [validLine(), validLine(), validLine(), validLine(), validLine()];
    const jsonlPath = writeJsonl(lines);
    const dbDir = makeTempDb();
    const db = await openDb(dbDir);

    try {
      const { stats, remainingLimit } = await processFile(jsonlPath, 'test', db, 'text', DEFAULT_MAX_LENGTH, new Set(), [], 0, 2);
      expect(remainingLimit).toBe(0);
      expect(stats.rowsIndexed).toBe(2);
      expect(stats.rowsLimited).toBeGreaterThanOrEqual(1); // at least 1 record was limited

      const keys = await db.keys().all();
      expect(keys.length).toBe(2);
    } finally {
      await db.close();
      rmSync(dbDir, { recursive: true, force: true });
      rmSync(jsonlPath, { force: true });
    }
  });

  test('offset + limit together in real processFile', async () => {
    const lines = [
      GARBAGE, // skipped by offset (line 1)
      validLine(), // skipped by offset (line 2)
      validLine(), // indexed #1
      GARBAGE, // dropped (bad JSON)
      validLine(), // indexed #2
      validLine(), // limited (limit exhausted)
    ];
    const jsonlPath = writeJsonl(lines);
    const dbDir = makeTempDb();
    const db = await openDb(dbDir);

    try {
      const { stats, remainingOffset, remainingLimit } = await processFile(
        jsonlPath,
        'test',
        db,
        'text',
        DEFAULT_MAX_LENGTH,
        new Set(),
        [],
        2,
        2
      );
      expect(remainingOffset).toBe(0);
      expect(remainingLimit).toBe(0);
      expect(stats.rowsSkipped).toBe(2);
      expect(stats.rowsDropped).toBe(1); // GARBAGE at line 4
      expect(stats.rowsIndexed).toBe(2);
      expect(stats.rowsLimited).toBeGreaterThanOrEqual(1);

      const keys = await db.keys().all();
      expect(keys.length).toBe(2);
    } finally {
      await db.close();
      rmSync(dbDir, { recursive: true, force: true });
      rmSync(jsonlPath, { force: true });
    }
  });

  test('offset > total lines — nothing indexed', async () => {
    const lines = [validLine(), validLine()];
    const jsonlPath = writeJsonl(lines);
    const dbDir = makeTempDb();
    const db = await openDb(dbDir);

    try {
      const { stats, remainingOffset } = await processFile(jsonlPath, 'test', db, 'text', DEFAULT_MAX_LENGTH, new Set(), [], 9999, 0);
      expect(remainingOffset).toBe(9997);
      expect(stats.rowsSkipped).toBe(2);
      expect(stats.rowsIndexed).toBe(0);
    } finally {
      await db.close();
      rmSync(dbDir, { recursive: true, force: true });
      rmSync(jsonlPath, { force: true });
    }
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// CLI validation
// ══════════════════════════════════════════════════════════════════════════════

describe('CLI', () => {
  test('--help mentions --offset and --limit', async () => {
    const proc = Bun.spawn(['bun', 'run', './import.ts', '--help'], {
      stdout: 'pipe',
      stderr: 'pipe',
    });
    const out = await new Response(proc.stdout).text();
    expect(out).toContain('--offset');
    expect(out).toContain('-o, --offset');
    expect(out).toContain('--limit');
    expect(out).toContain('-l, --limit');
  });

  test('--offset -1 exits with error', async () => {
    const proc = Bun.spawn(['bun', 'run', './import.ts', '--offset', '-1', '-i', 'test/test_me.jsonl'], { stdout: 'pipe', stderr: 'pipe' });
    const stderr = await new Response(proc.stderr).text();
    expect(stderr).toContain('non-negative');
  });

  test('--limit -1 exits with error', async () => {
    const proc = Bun.spawn(['bun', 'run', './import.ts', '--limit', '-1', '-i', 'test/test_me.jsonl'], { stdout: 'pipe', stderr: 'pipe' });
    const stderr = await new Response(proc.stderr).text();
    expect(stderr).toContain('non-negative');
  });

  test('--limit 0 is valid (means no limit)', async () => {
    const proc = Bun.spawn(['bun', 'run', './import.ts', '--limit', '0', '-i', 'test/test_me.jsonl', '--db', '/tmp/test_offset_limit_db'], {
      stdout: 'pipe',
      stderr: 'pipe',
    });
    const stderr = await new Response(proc.stderr).text();
    expect(stderr).not.toContain('non-negative');
    expect(stderr).not.toContain('Error: --limit');
  });
});
