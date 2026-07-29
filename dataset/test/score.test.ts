import { expect, test } from 'bun:test';
import type { DocValue } from '../features.ts';
import { globalScore } from '../features.ts';

test('scores', () => {
  let doc1: DocValue = {
    text: '...',
    len: 3,
    uniqChar: 3,
    tokens: 1,
    sentences: 1,
    entropy: 100,
    quality: 100,
    compress: 100,
    dictHit: 100,
    alpha: 100,
    vowel: 100,
    ascii: 100,
  };
  expect(globalScore(doc1)).toBe(100);

  doc1 = {
    text: '...',
    len: 3,
    uniqChar: 3,
    tokens: 1,
    sentences: 1,
    entropy: 100,
    quality: 0,
    compress: 100,
    dictHit: 100,
    alpha: 100,
    vowel: 100,
    ascii: 100,
  };
  expect(globalScore(doc1)).toBe(0);

  doc1 = {
    text: '...',
    len: 3,
    uniqChar: 3,
    tokens: 1,
    sentences: 1,
    entropy: 100,
    quality: 123.36,
    compress: 100,
    dictHit: 100,
    alpha: 100,
    vowel: 100,
    ascii: 100,
  };
  expect(globalScore(doc1)).toBe(123.36);

  // Three scores deviate from 100,
  // all three are less than 100
  let doc2: DocValue = {
    text: '...',
    len: 3,
    uniqChar: 3,
    tokens: 1,
    sentences: 1,
    entropy: 99,
    quality: 123.36,
    compress: 99,
    dictHit: 99,
    alpha: 100,
    vowel: 100,
    ascii: 100,
  };
  expect(globalScore(doc2)).toBe(123.36 - 3);

  // Three scores deviate from 100,
  // all three are more than 100
  doc2 = {
    text: '...',
    len: 3,
    uniqChar: 3,
    tokens: 1,
    sentences: 1,
    entropy: 101,
    quality: 123.36,
    compress: 101,
    dictHit: 101,
    alpha: 100,
    vowel: 100,
    ascii: 100,
  };
  expect(globalScore(doc2)).toBe(123.36 - 3);

  doc2 = {
    text: '...',
    len: 3,
    uniqChar: 3,
    tokens: 1,
    sentences: 1,
    entropy: 100,
    quality: 123.36,
    compress: 100,
    dictHit: 100,
    alpha: 99,
    vowel: 99,
    ascii: 99,
  };
  expect(globalScore(doc2)).toBe(123.36 - 3);

  doc2 = {
    text: '...',
    len: 3,
    uniqChar: 3,
    tokens: 1,
    sentences: 1,
    entropy: 100,
    quality: 123.36,
    compress: 100,
    dictHit: 100,
    alpha: 101,
    vowel: 101,
    ascii: 101,
  };
  expect(globalScore(doc2)).toBe(123.36 - 3);

  doc2 = {
    text: '...',
    len: 3,
    uniqChar: 3,
    tokens: 1,
    sentences: 1,
    quality: 100,
    entropy: 101,
    compress: 99,
    dictHit: 101,
    alpha: 99,
    vowel: 101,
    ascii: 99,
  };
  expect(globalScore(doc2)).toBe(100 - 6);
});
