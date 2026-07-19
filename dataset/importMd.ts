#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// importMd.ts — Markdown → LevelDB indexer (Bun)
//
// Reads one or more .md files, strips YAML frontmatter (if present), computes
// per-document features, and stores each file as a single entry in a LevelDB
// database (via classic-level). The key is a SHA-512/256 hash of the normalized
// text (the document ID); the value is a single JSON object holding the metadata
// together with the clean Markdown text.
//
//   key:   <id>
//   value: { source, length, uniqueChars, words, sentences,
//            entropy, quality, compress, text }
// ──────────────────────────────────────────────────────────────────────────────

import { ClassicLevel } from 'classic-level';
import { readFileSync } from 'node:fs';
import { basename } from 'node:path';
import {
  computeRecord,
  DEFAULT_MAX_LENGTH,
  type DocValue,
  loadVocabFromFile,
  MAX_UNIQUE_CHARS,
  MIN_LENGTH,
  MIN_UNIQUE_CHARS,
} from './features.ts';

// Re-export the shared vocab builders so existing importers of import.ts keep working.
export { buildVocabFromFiles, loadVocabFromFile } from './features.ts';

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

/** Maximum number of files to process before printing a progress line. */
const PROGRESS_INTERVAL = 100;

// ──────────────────────────────────────────────────────────────────────────────
// YAML frontmatter detection & stripping
//
// Detects a block delimited by "---" on its own line at the very beginning of
// the file. If found, everything between the opening and closing "---" (or the
// rest of the file if no closing delimiter is found) is removed, leaving only
// the clean Markdown body.
// ──────────────────────────────────────────────────────────────────────────────

const FRONTMATTER_RE = /^---\s*\n(?:[\s\S]*?\n)?---\s*\n*/;

/**
 * Strip YAML frontmatter from a Markdown document.
 *
 * Recognises:
 *   ---
 *   title: "..."
 *   ...
 *   ---
 *
 * If the opening `---` is not followed by a closing `---`, the function strips
 * everything from the opening line onward (i.e. treats the whole file as
 * frontmatter and returns an empty string). This is intentionally conservative.
 *
 * @param text - Raw file content.
 * @returns Clean Markdown text without the frontmatter block.
 */
function stripFrontmatter(text: string): string {
  const match = text.match(FRONTMATTER_RE);
  if (match) {
    // Remove the entire matched block (including trailing newlines).
    return text.slice(match[0].length);
  }
  return text;
}

// ──────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────────────────────────

function parseArgs(): {
  inputs: string[];
  source: string;
  dbPath: string;
  maxLength: number;
  vocabPath: string;
} {
  const args = process.argv.slice(2);
  let inputs: string[] = [];
  let source = 'cli';
  let dbPath = './levelDB';
  let maxLength = DEFAULT_MAX_LENGTH;
  // Vocabulary for the dictHit signal. Defaults to vocab.json next to this script
  // so the path is correct regardless of the current working directory.
  let vocabPath = './vocab.json';

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-i' || arg === '--input') && i + 1 < args.length) {
      inputs.push(args[++i]);
    } else if ((arg === '-s' || arg === '--source') && i + 1 < args.length) {
      source = args[++i];
    } else if ((arg === '-d' || arg === '--db') && i + 1 < args.length) {
      dbPath = args[++i];
    } else if ((arg === '-m' || arg === '--max-length' || arg === '--maxLength') && i + 1 < args.length) {
      maxLength = parseInt(args[++i], 10);
      if (isNaN(maxLength) || maxLength <= MIN_LENGTH) {
        console.error(`Error: --max-length must be an integer greater than ${MIN_LENGTH}.`);
        process.exit(1);
      }
    } else if (arg === '--vocab' && i + 1 < args.length) {
      vocabPath = args[++i];
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun run importMd.ts [options] [inputs...]

Options:
  -i, --input <file>        Markdown file path (repeatable, required)
  -s, --source <label>      Source label for all documents (default: "cli")
  -d, --db <path>           LevelDB directory (default: "./levelDB")
  -m, --max-length <n>      Max character length to import (default: ${DEFAULT_MAX_LENGTH})
  -v, --vocab <path>        Wordlist JSON for the dictHit signal (default: vocab.json next to importMd.ts)
  -h, --help                Show this help

Notes:
  - Each input .md file becomes one entry in the database.
  - YAML frontmatter (delimited by ---) is automatically stripped.
  - The --text-key flag from import.ts is NOT available; Markdown files
    are plain text, not JSON lines.`);
      process.exit(0);
    } else if (!arg.startsWith('-')) {
      inputs.push(arg);
    }
  }

  if (inputs.length === 0) {
    console.error('Error: at least one input file is required. Use -h for help.');
    process.exit(1);
  }

  return { inputs, source, dbPath, maxLength, vocabPath };
}

// ──────────────────────────────────────────────────────────────────────────────
// Conflict resolution
// ──────────────────────────────────────────────────────────────────────────────

function calcScore(value: DocValue): number {
  return (
    value.quality -
    100 +
    (value.compress > 100 ? 100 - value.compress : value.compress - 100) +
    (value.entropy > 100 ? 100 - value.entropy : value.entropy - 100)
  );
}

function onConflict(oldValue: DocValue, newValue: DocValue): DocValue | null {
  if (oldValue.source !== 'cli' && oldValue.text === newValue.text) {
    return null;
  }
  if (newValue.source !== 'cli' && oldValue.source === 'cli') {
    // Always prefer non-CLI sources over CLI
    oldValue.source = newValue.source;
  }
  return calcScore(newValue) >= calcScore(oldValue) ? newValue : oldValue;
}

// ──────────────────────────────────────────────────────────────────────────────
// Pre-filter: length + unique chars
// ──────────────────────────────────────────────────────────────────────────────

function prefilter(text: string, maxLength: number): Record<string, any> {
  const length = text.length;
  if (length <= MIN_LENGTH || length > maxLength) return { ok: false, length };
  const uniqueChars = new Set(text).size;
  if (uniqueChars <= MIN_UNIQUE_CHARS || uniqueChars > MAX_UNIQUE_CHARS) {
    return { ok: false, uniqueChars };
  }
  const toks = text.split(/\s+/).filter(t => t.length > 0);
  const words = toks.length;
  if (words <= 2) return { ok: false, words };
  return { ok: true };
}

// ──────────────────────────────────────────────────────────────────────────────
// Per-file stats
// ──────────────────────────────────────────────────────────────────────────────

interface FileStats {
  path: string;
  rowsLoaded: number;
  rowsDropped: number;
  rowsDuplicate: number;
  rowsIndexed: number;
  frontmatterStripped: boolean;
  strippedLength: number;
}

function printSummary(stats: FileStats, maxLength: number): void {
  const frontmatterNote = stats.frontmatterStripped ? `  Frontmatter: stripped (${stats.strippedLength} chars removed)\n` : '';
  console.log(
    frontmatterNote +
      `  Loaded:       ${String(stats.rowsLoaded).padStart(12)}\n` +
      `  Dropped:      ${String(stats.rowsDropped).padStart(12)}  (len ≤ ${MIN_LENGTH} or > ${maxLength} or uniqueChars out of range)\n` +
      `  Duplicates:   ${String(stats.rowsDuplicate).padStart(12)}  (id collision)\n` +
      `  Indexed:      ${String(stats.rowsIndexed).padStart(12)}`
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Process a single Markdown file
// ──────────────────────────────────────────────────────────────────────────────

async function processMdFile(
  filePath: string,
  source: string,
  db: ClassicLevel<string, DocValue>,
  maxLength: number,
  vocab?: Set<string>
): Promise<FileStats> {
  const stats: FileStats = {
    path: filePath,
    rowsLoaded: 1,
    rowsDropped: 0,
    rowsDuplicate: 0,
    rowsIndexed: 0,
    frontmatterStripped: false,
    strippedLength: 0,
  };

  // Read the raw file.
  const rawText = readFileSync(filePath, 'utf8');
  const rawLength = rawText.length;

  // Strip YAML frontmatter.
  const strippedText = stripFrontmatter(rawText);
  const strippedLength = rawLength - strippedText.length;
  if (strippedLength > 0) {
    stats.frontmatterStripped = true;
    stats.strippedLength = strippedLength;
  }

  const text = strippedText;

  // Pre-filter.
  const filtered = prefilter(text, maxLength);
  if (!filtered.ok) {
    delete filtered.ok;
    console.log(`  Skipping ${basename(filePath)}: failed pre-filter (${JSON.stringify(filtered)})`);
    stats.rowsDropped = 1;
    return stats;
  }

  // Compute features (including id).
  const { id, value } = computeRecord(text, source, vocab);

  // Check for existing entry in the DB.
  const existing = await db.getMany([id]);
  if (existing[0] !== undefined) {
    stats.rowsDuplicate = 1;
    const resolved = onConflict(existing[0], value);
    if (resolved) {
      await db.put(id, resolved);
      stats.rowsIndexed = 1;
      stats.rowsDuplicate = 0;
    }
  } else {
    await db.put(id, value);
    stats.rowsIndexed = 1;
  }

  return stats;
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const { inputs, source, dbPath, maxLength, vocabPath } = parseArgs();

  // Sort for deterministic processing order.
  inputs.sort();

  // Load the wordlist for the dictHit signal. If it's missing, the signal is
  // simply stored as null rather than aborting the whole import.
  let vocab: Set<string> | undefined;
  try {
    vocab = await loadVocabFromFile(vocabPath);
  } catch (err) {
    console.warn(`  [WARN] Could not load vocab (${vocabPath}): dictHit will be null.`);
  }

  console.log(`Found ${inputs.length} Markdown file(s)`);
  console.log(`Source: ${source}`);
  console.log(`Min length: ${MIN_LENGTH}`);
  console.log(`Max length: ${maxLength}`);
  console.log(`LevelDB: ${dbPath}`);
  console.log(`Vocabulary: ${vocab ? `${vocab.size} words (${vocabPath})` : 'none'}`);
  console.log(`Frontmatter: auto-stripped`);

  const db = new ClassicLevel<string, DocValue>(dbPath, {
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await db.open();

  let grandLoaded = 0;
  let grandDropped = 0;
  let grandDuplicate = 0;
  let grandIndexed = 0;

  try {
    for (let fi = 0; fi < inputs.length; fi++) {
      const filePath = inputs[fi];

      console.log(`\n${'='.repeat(60)}`);
      console.log(`[${fi + 1}/${inputs.length}] Processing: ${filePath}`);
      console.log('='.repeat(60));

      const stats = await processMdFile(filePath, source, db, maxLength, vocab);

      printSummary(stats, maxLength);

      grandLoaded += stats.rowsLoaded;
      grandDropped += stats.rowsDropped;
      grandDuplicate += stats.rowsDuplicate;
      grandIndexed += stats.rowsIndexed;

      // Periodic progress for large batches.
      if ((fi + 1) % PROGRESS_INTERVAL === 0) {
        console.log(`  ... processed ${fi + 1}/${inputs.length} files (${grandIndexed} indexed so far)`);
      }
    }
  } finally {
    await db.close();
  }

  // Final summary.
  console.log(`\n${'='.repeat(60)}`);
  console.log('SUMMARY');
  console.log('='.repeat(60));
  console.log(
    `  Grand loaded:     ${String(grandLoaded).padStart(12)}\n` +
      `  Grand dropped:    ${String(grandDropped).padStart(12)}  (quality filter)\n` +
      `  Grand duplicates: ${String(grandDuplicate).padStart(12)}  (id collision)\n` +
      `  Grand indexed:    ${String(grandIndexed).padStart(12)}`
  );
}

if (import.meta.main) {
  main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
}
