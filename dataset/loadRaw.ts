#!/usr/bin/env bun

import { ClassicLevel } from 'classic-level';
import { createReadStream } from 'node:fs';
import { extname } from 'node:path';

function generateId(text: string): string {
  const hasher = new Bun.CryptoHasher('sha512-256');
  hasher.update(text);
  return hasher.digest('hex');
}

const BATCH_SIZE = 512;

function parseArgs() {
  const args = process.argv.slice(2);
  let inputs: string[] = [];
  let dbPath = './levelDB';

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '-i' || arg === '--input') && i + 1 < args.length) {
      inputs.push(args[++i]);
    } else if ((arg === '-d' || arg === '--db') && i + 1 < args.length) {
      dbPath = args[++i];
    } else if (arg === '-h' || arg === '--help') {
      console.log(`Usage: bun run loadRaw.ts [options] [inputs...]

Options:
  -i, --input <glob>        JSON/JSONL file path (repeatable, required)
  -d, --db <path>           LevelDB directory (default: "./levelDB")
  -h, --help                Show this help`);
      process.exit(0);
    } else if (!arg.startsWith('-')) {
      inputs.push(arg);
    }
  }

  if (inputs.length === 0) {
    console.error('Error: input files are required. Use -h for help.');
    process.exit(1);
  }
  return { inputs, dbPath };
}

async function processArray(items: any[], db: ClassicLevel<string, any>, stats: any) {
  let batch = db.batch();
  let batchCount = 0;

  for (const item of items) {
    if (typeof item !== 'object' || item === null) {
      stats.dropped++;
      continue;
    }

    let id = item.id;
    const text = item.text;

    if (!id) {
      if (!text || typeof text !== 'string') {
        stats.dropped++;
        continue;
      }
      id = generateId(text);
    }

    if (!item.id) {
      item.id = id;
    }

    batch.put(id, item);
    batchCount++;
    stats.indexed++;

    if (batchCount >= BATCH_SIZE) {
      await batch.write();
      batch = db.batch();
      batchCount = 0;
    }
  }

  if (batchCount > 0) {
    await batch.write();
  }
}

async function processFile(filePath: string, db: ClassicLevel<string, any>) {
  console.log(`Processing ${filePath}...`);
  const stats = { loaded: 0, dropped: 0, indexed: 0 };
  const ext = extname(filePath).toLowerCase();

  if (ext === '.json') {
    try {
      const content = await Bun.file(filePath).text();
      const parsed = JSON.parse(content);
      const items = Array.isArray(parsed) ? parsed : [parsed];
      stats.loaded = items.length;
      await processArray(items, db, stats);
    } catch (e: any) {
      console.error(`  [ERROR] Failed to process ${filePath}: ${e.message}`);
    }
  } else {
    const fileStream = createReadStream(filePath, { encoding: 'utf8' });
    let lineBuffer = '';
    let batch = db.batch();
    let batchCount = 0;

    for await (const chunk of fileStream) {
      lineBuffer += chunk;
      let newlineIdx: number;
      while ((newlineIdx = lineBuffer.indexOf('\n')) !== -1) {
        const line = lineBuffer.slice(0, newlineIdx).trim();
        lineBuffer = lineBuffer.slice(newlineIdx + 1);
        if (!line) continue;

        stats.loaded++;
        let item: any;
        try {
          item = JSON.parse(line);
        } catch {
          stats.dropped++;
          continue;
        }

        if (typeof item !== 'object' || item === null) {
          stats.dropped++;
          continue;
        }

        let id = item.id;
        const text = item.text;

        if (!id) {
          if (!text || typeof text !== 'string') {
            stats.dropped++;
            continue;
          }
          id = generateId(text);
        }

        if (!item.id) {
          item.id = id;
        }

        batch.put(id, item);
        batchCount++;
        stats.indexed++;

        if (batchCount >= BATCH_SIZE) {
          await batch.write();
          batch = db.batch();
          batchCount = 0;
        }
      }
    }

    if (lineBuffer.trim()) {
      stats.loaded++;
      try {
        const item = JSON.parse(lineBuffer.trim());
        if (typeof item === 'object' && item !== null) {
          let id = item.id;
          const text = item.text;

          if (!id && text && typeof text === 'string') {
            id = generateId(text);
            item.id = id;
          }

          if (id || (text && typeof text === 'string')) {
            if (id) {
              batch.put(id, item);
              batchCount++;
              stats.indexed++;
            }
          } else {
            stats.dropped++;
          }
        } else {
          stats.dropped++;
        }
      } catch {
        stats.dropped++;
      }
    }

    if (batchCount > 0) {
      await batch.write();
    }
  }

  console.log(`  Loaded:  ${stats.loaded}`);
  console.log(`  Dropped: ${stats.dropped}`);
  console.log(`  Indexed: ${stats.indexed}`);
}

async function main() {
  const { inputs, dbPath } = parseArgs();
  console.log(`Open LevelDB at ${dbPath}`);
  const db = new ClassicLevel<string, any>(dbPath, {
    keyEncoding: 'utf8',
    valueEncoding: 'json',
    maxFileSize: 1_000_000_000,
  });
  await db.open();

  for (const file of inputs) {
    if (!(await Bun.file(file).exists())) {
      console.warn(`File not found: ${file}`);
      continue;
    }
    await processFile(file, db);
  }

  await db.close();
  console.log('Done.');
}

main().catch(console.error);
