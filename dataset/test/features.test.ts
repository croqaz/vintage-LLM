import { describe, expect, test } from 'bun:test';

import { computeRecord } from '../features.ts';
import { DEFAULT_MAX_LENGTH, prefilter } from '../import.ts';
import golden from './features_golden.json';

// ──────────────────────────────────────────────────────────────────────────────
// Golden regression tests for computeRecord.
//
// The fixture (test/features_golden.json) was generated with the ORIGINAL,
// pre-optimization implementation of features.ts. These tests pin the exact
// ids and metric values, so any optimization of computeRecord that changes a
// single stored byte — and would therefore produce a different LevelDB —
// fails here.
// ──────────────────────────────────────────────────────────────────────────────

describe('computeRecord golden values', () => {
  const vocab = new Set(golden.vocab as string[]);

  for (let i = 0; i < golden.cases.length; i++) {
    const c = golden.cases[i];
    test(`case ${i}: ${JSON.stringify(c.text.slice(0, 40))}...`, () => {
      const { id, value } = computeRecord(c.text, 'golden', vocab);
      expect(id).toBe(c.id);
      // Compare via JSON to also pin key order — the DB stores JSON text, so
      // key order is part of on-disk compatibility.
      expect(JSON.stringify(value)).toBe(JSON.stringify(c.value));
    });
  }
});

// ──────────────────────────────────────────────────────────────────────────────
// prefilter must agree with its reference (naive) definition.
// ──────────────────────────────────────────────────────────────────────────────

function referencePrefilter(text: string, maxLength: number, minLength: number, minUniq: number, maxUniq: number): Record<string, any> {
  const length = text.length;
  if (length <= minLength || length > maxLength) return { ok: false, length };
  const uniqueChars = new Set(text).size;
  if (uniqueChars <= minUniq || uniqueChars > maxUniq) return { ok: false, uniqueChars };
  const words = text.split(/\s+/).filter(t => t.length > 0).length;
  if (words <= 2) return { ok: false, words };
  return { ok: true };
}

describe('prefilter matches reference semantics', () => {
  const texts = [
    'The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly!',
    'x'.repeat(150), // too few unique chars
    'ab '.repeat(60), // few unique chars
    'wordone wordtwo' + 'x'.repeat(120), // 2 words (fails)
    'one two three ' + 'abcdefghijklmnopqrstuvwxyz0123456789 '.repeat(4),
    'unicode spaces　count as separators ' + 'padding '.repeat(20),
    'emoji 😀😀 and astral 𝔸𝕭 chars mixed in some longer text ' + 'filler '.repeat(15),
    'z'.repeat(99), // too short
    [...Array(200)].map((_, i) => String.fromCodePoint(0x100 + i)).join('') + ' spread out words here', // too many unique
  ];

  for (let i = 0; i < texts.length; i++) {
    test(`text ${i}`, () => {
      const got = prefilter(texts[i], DEFAULT_MAX_LENGTH);
      const want = referencePrefilter(texts[i], DEFAULT_MAX_LENGTH, 100, 10, 132);
      // ok must always agree; on failure the reported reason must agree too,
      // except `words`, where prefilter may stop counting early (at 3).
      expect(got.ok).toBe(want.ok);
      if (!want.ok) {
        if (want.length !== undefined) expect(got.length).toBe(want.length);
        if (want.uniqueChars !== undefined) expect(got.uniqueChars).toBe(want.uniqueChars);
        if (want.words !== undefined) expect(got.words).toBe(want.words);
      }
    });
  }
});
