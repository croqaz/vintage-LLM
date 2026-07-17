#!/usr/bin/env bun
// ──────────────────────────────────────────────────────────────────────────────
// explorer.ts — browser explorer for a LevelDB text-corpus DB (Bun)
//
//   bun explorer.ts -d ./Gutenberg-DB [-p 8080] [--vocab ./vocab.json]
//
// Opens ONE classic-level DB (read-write) and serves a single-page browser UI
// plus a small JSON API. Designed for DBs with up to ~100M records:
//   • loads instantly with CHEAP stats only (never a full scan)
//   • pages record-by-record using sorted-key iterators (Next/Prev, big steps,
//     Random, jump-to-id)
//   • lets you edit a record's text (re-hashes → re-keys) or delete it
//   • exact record count is opt-in (streamed, cancellable)
//
// Keys are SHA-512/256 hex digests → uniformly distributed, which is what makes
// the cheap count estimate and O(1)-ish random access possible.
// ──────────────────────────────────────────────────────────────────────────────

import { ClassicLevel } from 'classic-level';
import { readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { computeRecord, type DocValue, loadVocabFromFile } from './features.ts';

// ──────────────────────────────────────────────────────────────────────────────
// CLI args
// ──────────────────────────────────────────────────────────────────────────────

function firstDbDir(): string {
  try {
    const dirs = readdirSync('.', { withFileTypes: true })
      .filter(e => e.isDirectory() && e.name.endsWith('-DB'))
      .map(e => e.name)
      .sort();
    if (dirs.length) return './' + dirs[0];
  } catch {}
  return './levelDB';
}

function parseArgs(): { dbPath: string; port: number; vocabPath: string } {
  const args = process.argv.slice(2);
  let dbPath = '';
  let port = 8080;
  let vocabPath = './vocab.json';
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if ((a === '-d' || a === '--db') && i + 1 < args.length) dbPath = args[++i];
    else if ((a === '-p' || a === '--port') && i + 1 < args.length) {
      port = parseInt(args[++i], 10);
    } else if (a === '--vocab' && i + 1 < args.length) vocabPath = args[++i];
    else if (a === '-h' || a === '--help') {
      console.log(`Usage: bun explorer.ts [options]

Options:
  -d, --db <path>      LevelDB directory to explore (default: first *-DB dir in cwd)
  -p, --port <n>       HTTP port (default: 8080)
  --vocab <path>       Vocabulary JSON for metric recompute on edit (default: ./vocab.json)
  -h, --help           Show this help`);
      process.exit(0);
    }
  }
  if (!dbPath) dbPath = firstDbDir();
  if (isNaN(port) || port <= 0) {
    console.error('Error: --port must be a positive integer.');
    process.exit(1);
  }
  return { dbPath, port, vocabPath };
}

// ──────────────────────────────────────────────────────────────────────────────
// Cheap helpers over the sorted keyspace
// ──────────────────────────────────────────────────────────────────────────────

// Recursively sum on-disk byte size of the DB directory (fast; no DB access).
function dirSize(path: string): number {
  let total = 0;
  let entries;
  try {
    entries = readdirSync(path, { withFileTypes: true });
  } catch {
    return 0;
  }
  for (const e of entries) {
    const full = join(path, e.name);
    try {
      if (e.isDirectory()) total += dirSize(full);
      else total += statSync(full).size;
    } catch {}
  }
  return total;
}

// '000' → '001', '00' → '01', 'ff' → '100' (only ever called on all-zero prefixes here)
function hexPrefixUpper(lo: string): string {
  const n = parseInt(lo, 16) + 1;
  return n.toString(16).padStart(lo.length, '0');
}

function randomHexKey(len = 64): string {
  const bytes = new Uint8Array(len / 2);
  crypto.getRandomValues(bytes);
  let out = '';
  for (const b of bytes) out += b.toString(16).padStart(2, '0');
  return out;
}

async function keysSlice(db: ClassicLevel<string, DocValue>, opts: any): Promise<string[]> {
  return await db.keys(opts).all();
}

// Estimate total record count by counting exactly within a narrow all-zero hex
// prefix (a uniform 1/16^w slice of the SHA keyspace) and extrapolating.
async function estimateCount(db: ClassicLevel<string, DocValue>): Promise<{
  estimate: number;
  sampleSize: number;
  fraction: string;
  capped: boolean;
  exact: boolean;
}> {
  const CAP = 500_000;
  for (const w of [3, 2, 1]) {
    const lo = '0'.repeat(w);
    const hi = hexPrefixUpper(lo);
    const keys = await keysSlice(db, { gte: lo, lt: hi, limit: CAP });
    const factor = Math.pow(16, w);
    if (keys.length >= 100) {
      return {
        estimate: keys.length * factor,
        sampleSize: keys.length,
        fraction: `1/${factor}`,
        capped: keys.length >= CAP,
        exact: false,
      };
    }
  }
  // The 1/16 slice held < 100 keys ⇒ with uniform hash keys the whole DB is
  // tiny, so an exact count is cheap and far more useful than a noisy estimate.
  const TINY_CAP = 100_000;
  const all = await keysSlice(db, { limit: TINY_CAP });
  return {
    estimate: all.length,
    sampleSize: all.length,
    fraction: '1/1',
    capped: all.length >= TINY_CAP,
    exact: true,
  };
}

// Resolve a navigation direction to the target key (or undefined at the edges).
async function navKey(db: ClassicLevel<string, DocValue>, from: string | null, dir: string): Promise<string | undefined> {
  const BIG = 10;
  switch (dir) {
    case 'first': {
      return (await keysSlice(db, { limit: 1 }))[0];
    }
    case 'last': {
      return (await keysSlice(db, { reverse: true, limit: 1 }))[0];
    }
    case 'next': {
      if (from == null) return (await keysSlice(db, { limit: 1 }))[0];
      return (await keysSlice(db, { gt: from, limit: 1 }))[0];
    }
    case 'prev': {
      if (from == null) {
        return (await keysSlice(db, { reverse: true, limit: 1 }))[0];
      }
      return (await keysSlice(db, { lt: from, reverse: true, limit: 1 }))[0];
    }
    case 'next10': {
      const ks = await keysSlice(db, from == null ? { limit: BIG } : { gt: from, limit: BIG });
      return ks[ks.length - 1];
    }
    case 'prev10': {
      const ks = await keysSlice(db, from == null ? { reverse: true, limit: BIG } : { lt: from, reverse: true, limit: BIG });
      return ks[ks.length - 1];
    }
    case 'random': {
      const rnd = randomHexKey();
      const hit = (await keysSlice(db, { gte: rnd, limit: 1 }))[0];
      // Past the last key → wrap to the first.
      return hit ?? (await keysSlice(db, { limit: 1 }))[0];
    }
    default:
      return undefined;
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// JSON response helpers
// ──────────────────────────────────────────────────────────────────────────────

const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': 'application/json' },
  });

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

const { dbPath, port, vocabPath } = parseArgs();

const db = new ClassicLevel<string, DocValue>(dbPath, {
  valueEncoding: 'json',
  maxFileSize: 1_000_000_000,
});
await db.open();

let vocab = new Set<string>();
try {
  vocab = await loadVocabFromFile(vocabPath);
} catch (err) {
  console.warn(`[warn] Could not load vocab from ${vocabPath}: ${err instanceof Error ? err.message : err}`);
  console.warn('[warn] Edits will recompute dictHit against an EMPTY vocabulary (dictHit → 0).');
}

async function handle(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const path = url.pathname;

  // ── UI ──────────────────────────────────────────────────────────────────────
  if (req.method === 'GET' && (path === '/' || path === '/index.html')) {
    return new Response(PAGE, {
      headers: { 'content-type': 'text/html; charset=utf-8' },
    });
  }

  // ── Cheap DB info ────────────────────────────────────────────────────────────
  if (req.method === 'GET' && path === '/api/info') {
    const [first, last, est] = await Promise.all([
      keysSlice(db, { limit: 1 }),
      keysSlice(db, { reverse: true, limit: 1 }),
      estimateCount(db),
    ]);
    return json({
      dbPath,
      sizeBytes: dirSize(dbPath),
      firstKey: first[0] ?? null,
      lastKey: last[0] ?? null,
      estimate: est,
    });
  }

  // ── Fetch a record by id ──────────────────────────────────────────────────────
  if (req.method === 'GET' && path.startsWith('/api/record/')) {
    const id = decodeURIComponent(path.slice('/api/record/'.length));
    const value = await db.get(id);
    if (value === undefined) return json({ error: 'not found', id }, 404);
    return json({ id, value });
  }

  // ── Navigate ───────────────────────────────────────────────────────────────
  if (req.method === 'GET' && path === '/api/nav') {
    const dir = url.searchParams.get('dir') ?? 'next';
    const from = url.searchParams.get('from');
    const key = await navKey(db, from, dir);
    if (key === undefined) return json({ id: null, value: null });
    const value = await db.get(key);
    if (value === undefined) return json({ id: null, value: null });
    return json({ id: key, value });
  }

  // ── Edit / save (re-key on hash change) ───────────────────────────────────────
  if (req.method === 'PUT' && path.startsWith('/api/record/')) {
    const oldId = decodeURIComponent(path.slice('/api/record/'.length));
    const old = await db.get(oldId);
    if (old === undefined) return json({ error: 'not found', id: oldId }, 404);
    let body: { text?: unknown };
    try {
      body = await req.json();
    } catch {
      return json({ error: 'invalid JSON body' }, 400);
    }
    if (typeof body.text !== 'string') {
      return json({ error: 'body.text (string) required' }, 400);
    }

    const { id: newId, value } = computeRecord(body.text, old.source, vocab);
    if (newId === oldId) {
      await db.put(newId, value);
      return json({ id: newId, value, rekeyed: false });
    }
    // Different text → different key. Refuse to clobber a distinct existing record.
    const clash = (await db.getMany([newId]))[0];
    if (clash !== undefined) {
      return json(
        {
          error: 'collision',
          message: 'A different record already has the new content hash.',
          id: newId,
        },
        409
      );
    }
    await db.batch([
      { type: 'del', key: oldId },
      { type: 'put', key: newId, value },
    ]);
    return json({ id: newId, value, rekeyed: true, oldId });
  }

  // ── Delete ────────────────────────────────────────────────────────────────
  if (req.method === 'DELETE' && path.startsWith('/api/record/')) {
    const id = decodeURIComponent(path.slice('/api/record/'.length));
    const existed = (await db.getMany([id]))[0] !== undefined;
    if (!existed) return json({ error: 'not found', id }, 404);
    // Find the neighbour to jump to after deletion.
    const next = (await keysSlice(db, { gt: id, limit: 1 }))[0] ?? (await keysSlice(db, { lt: id, reverse: true, limit: 1 }))[0] ?? null;
    await db.del(id);
    return json({ deleted: true, id, next });
  }

  // ── Exact count (opt-in, streamed, cancellable) ───────────────────────────────
  if (req.method === 'GET' && path === '/api/count/exact') {
    const stream = new ReadableStream({
      async start(controller) {
        const enc = new TextEncoder();
        const send = (obj: unknown) => controller.enqueue(enc.encode(`data: ${JSON.stringify(obj)}\n\n`));
        let n = 0;
        try {
          for await (const _key of db.keys()) {
            n++;
            if (n % 100_000 === 0) {
              send({ count: n, done: false });
              if (req.signal.aborted) break;
            }
          }
          if (!req.signal.aborted) send({ count: n, done: true });
        } catch (err) {
          send({
            error: err instanceof Error ? err.message : String(err),
            done: true,
          });
        }
        controller.close();
      },
    });
    return new Response(stream, {
      headers: {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      },
    });
  }

  return new Response('Not found', { status: 404 });
}

const server = Bun.serve({
  port,
  idleTimeout: 0, // allow long-lived exact-count streams
  fetch(req) {
    return handle(req).catch(err => {
      console.error('Request error:', err);
      return json({ error: err instanceof Error ? err.message : String(err) }, 500);
    });
  },
});

console.log(`\nLevelDB explorer running:`);
console.log(`  DB:    ${dbPath}`);
console.log(`  URL:   http://localhost:${server.port}`);

process.on('SIGINT', async () => {
  console.log('\nClosing DB…');
  await db.close();
  process.exit(0);
});

// ──────────────────────────────────────────────────────────────────────────────
// Single-page UI (vanilla, no build step)
// ──────────────────────────────────────────────────────────────────────────────

const PAGE = /* html */ `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LevelDB Corpus Explorer</title>
<style>
  :root { color-scheme: light dark; --bg:#0f1115; --panel:#171a21; --fg:#e6e6e6; --muted:#8b93a1; --accent:#5aa9e6; --border:#262b36; --danger:#e6675a; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; background:var(--bg); color:var(--fg); }
  header { padding:10px 16px; background:var(--panel); border-bottom:1px solid var(--border); position:sticky; top:0; z-index:5; }
  .stats { display:flex; flex-wrap:wrap; gap:8px 22px; align-items:baseline; }
  .stats b { color:var(--accent); }
  .stats .kv { color:var(--muted); }
  .stats .kv span { color:var(--fg); }
  main { max-width:1000px; margin:0 auto; padding:16px; }
  .nav { display:flex; flex-wrap:wrap; gap:6px; margin:12px 0; align-items:center; }
  button { font:inherit; background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:6px; padding:6px 10px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  button.primary { background:var(--accent); color:#04121f; border-color:var(--accent); font-weight:bold; }
  button.danger { color:var(--danger); border-color:var(--danger); }
  button:disabled { opacity:.4; cursor:not-allowed; }
  input[type=text] { font:inherit; background:var(--bg); color:var(--fg); border:1px solid var(--border); border-radius:6px; padding:6px 8px; }
  #jump { width:min(60vw,520px); }
  .idline { color:var(--muted); word-break:break-all; margin:6px 0; }
  .idline code { color:var(--accent); }
  table.meta { border-collapse:collapse; margin:12px 0; }
  table.meta td { border:1px solid var(--border); padding:3px 10px; }
  table.meta td.k { color:var(--muted); }
  table.meta td.v { text-align:right; color:var(--fg); }
  textarea { width:100%; min-height:45vh; font:inherit; background:var(--bg); color:var(--fg); border:1px solid var(--border); border-radius:6px; padding:10px; resize:vertical; }
  .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:8px 0; }
  .msg { padding:6px 10px; border-radius:6px; margin:8px 0; display:none; }
  .msg.show { display:block; }
  .msg.ok { background:#123021; color:#7fd6a2; }
  .msg.err { background:#301818; color:#ffb0a6; }
  .msg.warn { background:#302a12; color:#e8d58a; }
  .dirty textarea { border-color:var(--danger); }
  .muted { color:var(--muted); }
</style>
</head>
<body>
<header>
  <div class="stats" id="stats"><span class="muted">Loading DB info…</span></div>
</header>
<main>
  <div class="nav">
    <button data-dir="first">⟨⟨ First</button>
    <button data-dir="prev10">⟨⟨ Prev 10</button>
    <button data-dir="prev">⟨ Prev</button>
    <button data-dir="next">Next ⟩</button>
    <button data-dir="next10">Next 10 ⟩⟩</button>
    <button data-dir="last">Last ⟩⟩</button>
    <button data-dir="random">🎲 Random</button>
  </div>
  <div class="nav">
    <input type="text" id="jump" placeholder="Paste an id (64-hex) and press Enter to jump…" spellcheck="false">
    <button id="jumpBtn">Go</button>
  </div>

  <div class="msg" id="msg"></div>

  <div id="record" style="display:none">
    <div class="idline">id: <code id="recId"></code></div>
    <table class="meta" id="metaTable"></table>
    <div class="row">
      <strong>text</strong>
      <span class="muted" id="textInfo"></span>
    </div>
    <div id="textWrap"></div>
    <div class="row">
      <button class="primary" id="saveBtn" disabled>Save (recompute + re-key)</button>
      <button id="revertBtn" disabled>Revert</button>
      <button class="danger" id="deleteBtn">Delete</button>
    </div>
  </div>
  <div id="empty" class="muted" style="display:none">No record loaded. Use the navigation above.</div>
</main>

<script>
const $ = s => document.querySelector(s);
const fmtBytes = n => { const u=['B','KB','MB','GB','TB']; let i=0; while(n>=1024&&i<u.length-1){n/=1024;i++;} return n.toFixed(i?1:0)+' '+u[i]; };
const fmtNum = n => n.toLocaleString('en-US');
const LARGE_TEXT = 300000; // chars beyond which we defer rendering the textarea

let current = null;   // { id, value }
let original = '';    // original text for dirty/revert
let countES = null;

function showMsg(kind, text) {
  const m = $('#msg'); m.className = 'msg show ' + kind; m.textContent = text;
}
function clearMsg() { $('#msg').className = 'msg'; }

// score derived exactly like query.ts:150
function scoreOf(d) {
  return -(((d.quality??0)-100) + Math.abs((d.compress??0)-100) + Math.abs((d.entropy??0)-100)
    + Math.abs((d.dictHit??0)-100) + Math.abs((d.alpha??0)-100) + Math.abs((d.vowel??0)-100) + Math.abs((d.ascii??0)-100));
}

async function loadInfo() {
  try {
    const r = await fetch('/api/info'); const info = await r.json();
    const est = info.estimate;
    $('#stats').innerHTML =
      '<b>DB Explorer</b>' +
      '<span class="kv">db: <span>'+info.dbPath+'</span></span>' +
      '<span class="kv">size: <span>'+fmtBytes(info.sizeBytes)+'</span></span>' +
      '<span class="kv">records: <span>'+(est.exact?'':'≈ ')+fmtNum(est.estimate)+'</span> <span class="muted">('+(est.exact?'exact':est.fraction+' sample'+(est.capped?', capped':''))+')</span></span>' +
      '<span class="kv" id="exactWrap"><button id="exactBtn">Compute exact count</button> <span id="exactOut" class="muted"></span></span>';
    $('#exactBtn').onclick = toggleExact;
  } catch(e) { $('#stats').innerHTML = '<span class="msg err show">Failed to load info: '+e+'</span>'; }
}

function toggleExact() {
  const btn = $('#exactBtn'), out = $('#exactOut');
  if (countES) { countES.close(); countES=null; btn.textContent='Compute exact count'; out.textContent += ' (cancelled)'; return; }
  btn.textContent = 'Cancel';
  out.textContent = 'scanning…';
  countES = new EventSource('/api/count/exact');
  countES.onmessage = ev => {
    const d = JSON.parse(ev.data);
    if (d.error) { out.textContent = 'error: '+d.error; }
    else if (d.done) { out.textContent = 'exact: '+fmtNum(d.count); countES.close(); countES=null; btn.textContent='Compute exact count'; }
    else { out.textContent = 'scanned '+fmtNum(d.count)+'…'; }
  };
  countES.onerror = () => { if(countES){countES.close();countES=null;} btn.textContent='Compute exact count'; };
}

function renderMeta(v) {
  const keys = ['source','len','uniqChar','tokens','sentences','quality','compress','entropy','dictHit','alpha','vowel','ascii'];
  let rows = '';
  for (const k of keys) rows += '<tr><td class="k">'+k+'</td><td class="v">'+(v[k]??'')+'</td></tr>';
  rows += '<tr><td class="k">score</td><td class="v">'+scoreOf(v).toFixed(2)+'</td></tr>';
  $('#metaTable').innerHTML = rows;
}

function renderText(v) {
  const wrap = $('#textWrap');
  $('#textInfo').textContent = fmtNum(v.text.length)+' chars';
  if (v.text.length > LARGE_TEXT) {
    wrap.innerHTML = '<div class="msg warn show">Large text ('+fmtNum(v.text.length)+' chars). Rendering deferred to keep the browser responsive.</div>'
      + '<button id="loadTextBtn">Load & edit text</button>';
    $('#loadTextBtn').onclick = () => mountTextarea(v.text);
    $('#saveBtn').disabled = true; $('#revertBtn').disabled = true;
  } else {
    mountTextarea(v.text);
  }
}

function mountTextarea(text) {
  original = text;
  $('#textWrap').innerHTML = '<textarea id="text" spellcheck="false"></textarea>';
  const ta = $('#text');
  ta.value = text;
  ta.oninput = () => {
    const dirty = ta.value !== original;
    $('#saveBtn').disabled = !dirty; $('#revertBtn').disabled = !dirty;
    $('#record').classList.toggle('dirty', dirty);
  };
}

function render(rec) {
  current = rec;
  clearMsg();
  $('#empty').style.display = 'none';
  $('#record').style.display = 'block';
  $('#record').classList.remove('dirty');
  $('#recId').textContent = rec.id;
  renderMeta(rec.value);
  renderText(rec.value);
  $('#saveBtn').disabled = true; $('#revertBtn').disabled = true;
}

async function loadId(id, {push=true}={}) {
  if (!id) return;
  try {
    const r = await fetch('/api/record/'+encodeURIComponent(id));
    if (r.status === 404) { showMsg('err','Record not found: '+id); return; }
    const rec = await r.json();
    render(rec);
    if (push && location.hash !== '#/id/'+rec.id) location.hash = '#/id/'+rec.id;
  } catch(e) { showMsg('err','Load failed: '+e); }
}

async function nav(dir) {
  const from = current ? current.id : null;
  try {
    const r = await fetch('/api/nav?dir='+dir+(from?'&from='+encodeURIComponent(from):''));
    const rec = await r.json();
    if (!rec.id) { showMsg('warn','No record in that direction (edge of DB or empty DB).'); return; }
    render(rec);
    location.hash = '#/id/'+rec.id;
  } catch(e) { showMsg('err','Navigation failed: '+e); }
}

async function save() {
  const ta = $('#text'); if (!ta || !current) return;
  $('#saveBtn').disabled = true;
  try {
    const r = await fetch('/api/record/'+encodeURIComponent(current.id), {
      method:'PUT', headers:{'content-type':'application/json'}, body: JSON.stringify({ text: ta.value })
    });
    const d = await r.json();
    if (r.status === 409) { showMsg('err','Save blocked: '+d.message); return; }
    if (!r.ok) { showMsg('err','Save failed: '+(d.error||r.status)); return; }
    render({ id: d.id, value: d.value });
    location.hash = '#/id/'+d.id;
    showMsg('ok', d.rekeyed ? 'Saved — text changed, record re-keyed to a new id.' : 'Saved (id unchanged).');
  } catch(e) { showMsg('err','Save failed: '+e); }
}

async function del() {
  if (!current) return;
  if (!confirm('Delete this record permanently?\\n\\nid: '+current.id)) return;
  try {
    const r = await fetch('/api/record/'+encodeURIComponent(current.id), { method:'DELETE' });
    const d = await r.json();
    if (!r.ok) { showMsg('err','Delete failed: '+(d.error||r.status)); return; }
    if (d.next) { await loadId(d.next); showMsg('ok','Deleted. Jumped to the next record.'); }
    else { current=null; $('#record').style.display='none'; $('#empty').style.display='block'; location.hash=''; showMsg('ok','Deleted. DB is now empty.'); }
  } catch(e) { showMsg('err','Delete failed: '+e); }
}

// wire up
document.querySelectorAll('.nav button[data-dir]').forEach(b => b.onclick = () => nav(b.dataset.dir));
$('#jumpBtn').onclick = () => loadId($('#jump').value.trim());
$('#jump').addEventListener('keydown', e => { if (e.key==='Enter') loadId($('#jump').value.trim()); });
$('#saveBtn').onclick = save;
$('#revertBtn').onclick = () => { if(current) mountTextarea(original), $('#record').classList.remove('dirty'), $('#saveBtn').disabled=true, $('#revertBtn').disabled=true; };
$('#deleteBtn').onclick = del;

window.addEventListener('hashchange', () => {
  const m = location.hash.match(/^#\\/id\\/([0-9a-fA-F]+)/);
  if (m && (!current || current.id !== m[1])) loadId(m[1], {push:false});
});

// boot
loadInfo();
const m = location.hash.match(/^#\\/id\\/([0-9a-fA-F]+)/);
if (m) loadId(m[1], {push:false});
else { $('#empty').style.display='block'; nav('first'); }
</script>
</body>
</html>`;
