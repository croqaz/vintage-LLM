"""Fast, parallel, resumable pre-1900 filter for very large JSONL files.

Same scoring as `detect.py --filter`, but:
  * multiprocessing across CPU cores,
  * a fast anachronism matcher (token lookups instead of a giant regex),
  * a checkpoint written every ~10s to <output>.progress so an interrupted
    run resumes in place, losing only the last few seconds of work.

New in this version (vs the original filter_bulk):
  * --preserve-schema  : emit the FULL original record instead of just
                         {"text": ..., <score fields>} — output keeps the
                         input's exact JSON structure.
  * --drop-log FILE    : write every dropped entry to FILE (jsonl) with the
                         full record and the exact reason(s).
  * --normalize-unicode PATH : unicode-normalize all string values (ftfy +
                         bad-char removal) before scoring, using a unicode.py
                         module like Synthetic-Archive/filtered/unicode.py.
  * --batch DIR / multiple inputs : process many files, one output + drop log
                         each (same base name in --out-dir / --drop-log-dir),
                         plus a _filter_summary.json.
  * Parse errors, non-object lines and empty texts are now logged as drops
    (with reason) instead of being silently skipped.

Usage (matches detect.py knobs):
    python filter_bulk.py split-long-docs2.jsonl \
        --out Institutional-Books.jsonl \
        --style-weight 0.015 --marker-weight 0.015 --threshold 0.5

    # keep the original JSON schema + log drops (e.g. {"text":..., "extra":...})
    python filter_bulk.py data.jsonl --out kept.jsonl \
        --preserve-schema \
        --drop-log dropped.jsonl \
        --normalize-unicode unicode.py

    # batch: every *.jsonl in data/ -> filtered/ + filtered/dropped/
    python filter_bulk.py --batch data/ --out-dir filtered/ --drop-log-dir filtered/dropped/ \
        --preserve-schema --normalize-unicode unicode.py --restart

    # if it stops for any reason, just run the SAME command again -> it resumes.
    # to start over instead of resuming:  add  --restart
"""

import argparse
import json
import os
import re
import sys
import time
import glob
from multiprocessing import Pool

from detect import Scorer, tokenize, drop_reasons, load_normalizer

_S = None
_NORM = None
_REPLACE = None
_CFG = None


def _init(cfg):
    global _S, _NORM, _REPLACE, _CFG
    _CFG = cfg
    _S = Scorer()
    _NORM = load_normalizer(cfg['normalize_path']) if cfg.get('normalize_path') else None
    # compile old:new word replacements once per worker
    # Handles singular and plural forms (optional 's' suffix) to match the
    # banned-terms matcher's suffix logic, and preserves the original casing.
    _REPLACE = []
    for old, new in (cfg.get('replace_terms') or {}).items():
        rx = re.compile(r'\b' + re.escape(old) + r'(s?)\b', re.I)
        _REPLACE.append((rx, new))


def _replace_value(v):
    """Recursively apply --replace-terms to every string in a value."""
    if isinstance(v, str):
        for rx, new in _REPLACE:
            def _repl(m):
                suffix = m.group(1)
                base = new
                if m.group(0).isupper():
                    base = new.upper()
                elif m.group(0)[:1].isupper():
                    base = new[:1].upper() + new[1:]
                return base + suffix
            v = rx.sub(_repl, v)
        return v
    if isinstance(v, dict):
        return {k: _replace_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_replace_value(x) for x in v]
    return v


def _norm_value(v):
    """Recursively unicode-normalize every string in a value."""
    if isinstance(v, str):
        return _NORM(v) if _NORM else v
    if isinstance(v, dict):
        return {k: _norm_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_norm_value(x) for x in v]
    return v


def _drop_entry(record, norm_record, reason_tag, details, r=None, raw=None):
    """Build one drop-log line."""
    entry = {
        'record': norm_record if norm_record is not None else record,
        'reason': reason_tag,
        'details': details,
    }
    if r is not None:
        entry.update({
            'p_pre1900': r['p_pre1900'],
            'english_frac': r['english_frac'],
            'n_tokens': r['n_tokens'],
            'banned_hits': r['banned_hits'],
            'features': r['features'],
        })
    if raw is not None:
        entry['raw'] = raw
    return json.dumps(entry, ensure_ascii=False) + '\n'


def _work(raw):
    """raw: one input line as bytes (incl. trailing newline). Returns
    (rawlen, keep, payload_bytes_or_None)."""
    rawlen = len(raw)
    try:
        record = json.loads(raw)
    except Exception as e:
        # Unreadable line -> drop (parse_error), if a drop log is requested.
        if not _CFG.get('drop_log'):
            return rawlen, False, None
        payload = _drop_entry(None, None, 'parse_error',
                              [[str(type(e).__name__), str(e)[:200]]],
                              raw=raw.decode('utf-8', errors='replace')[:10_000])
        return rawlen, False, payload.encode('utf-8')

    if not isinstance(record, dict):
        if not _CFG.get('drop_log'):
            return rawlen, False, None
        payload = _drop_entry(record, None, 'not_object', [['not_object', 'JSON value is not an object']])
        return rawlen, False, payload.encode('utf-8')

    # ---- pre-filter word replacement (--replace-terms) ----
    if _REPLACE:
        record = _replace_value(record)

    text = str(record.get(_CFG['field'], '') or '')
    if not text:
        if not _CFG.get('drop_log'):
            return rawlen, False, None
        payload = _drop_entry(record, record, 'empty_text', [['empty_text', f"field '{_CFG['field']}' is empty or missing"]])
        return rawlen, False, payload.encode('utf-8')

    # ---- normalize (ftfy + bad-char removal), then score the clean text ----
    norm_record = _norm_value(record) if _CFG.get('normalize_path') else record
    norm_text = norm_record.get(_CFG['field'], '') or text
    toks = tokenize(norm_text)
    r = _S.score(norm_text, toks, style_weight=_CFG['style'], marker_weight=_CFG['marker'])

    keep = r['p_pre1900'] >= _CFG['threshold'] and r['english_frac'] >= _CFG['min_english'] and r['n_tokens'] >= _CFG['min_tokens']
    if keep:
        if _CFG.get('preserve_schema'):
            payload = json.dumps(norm_record, ensure_ascii=False)
        else:
            r['keep'] = True
            payload = json.dumps({'text': norm_text, **r}, ensure_ascii=False)
        return rawlen, True, (payload + '\n').encode('utf-8')

    if not _CFG.get('drop_log'):
        return rawlen, False, None
    reasons = drop_reasons(r, _CFG['threshold'], _CFG['min_english'], _CFG['min_tokens'])
    payload = _drop_entry(record, norm_record, reasons[0][0] if reasons else 'unknown', reasons, r=r)
    return rawlen, False, payload.encode('utf-8')


def _raw_lines(path, start):
    with open(path, 'rb') as f:
        f.seek(start)
        for raw in f:
            yield raw


def process_file(input_path, output_path, cfg, drop_log_path=None,
                 restart=False, workers=15, chunksize=16, checkpoint_secs=10.0):
    """Process one JSONL file. Returns (rows_read, rows_kept, rows_dropped).

    Resumable: a checkpoint is written to <output_path>.progress every
    `checkpoint_secs`; re-running the same call resumes in place. Pass
    restart=True to ignore any checkpoint. On a clean finish the checkpoint
    file is removed.
    """
    total_bytes = os.path.getsize(input_path)
    prog_path = output_path + '.progress'

    # per-call copy so workers know whether to emit drop-log lines
    cfg = dict(cfg, drop_log=drop_log_path is not None)

    # ---- resume or start fresh ----
    start_off = read = kept = dropped = 0
    out = drop_out = None
    if not restart and os.path.exists(prog_path) and os.path.exists(output_path):
        try:
            p = json.load(open(prog_path))
            start_off, read, kept, dropped = p['input_offset'], p['rows_read'], p['rows_kept'], p['rows_dropped']
            out = open(output_path, 'r+b')
            out.truncate(p['output_size'])
            out.seek(p['output_size'])
            if drop_log_path:
                drop_out = open(drop_log_path, 'r+b')
                drop_out.truncate(p.get('drop_log_size', 0))
                drop_out.seek(p.get('drop_log_size', 0))
            print(
                f'resuming at byte {start_off:,} / {total_bytes:,} ({100 * start_off / total_bytes:.1f}%), '
                f'{kept:,} kept, {dropped:,} dropped so far',
                file=sys.stderr,
            )
        except Exception:
            start_off = read = kept = dropped = 0
            out = drop_out = None

    if out is None:
        out = open(output_path, 'wb')
        if drop_log_path:
            drop_out = open(drop_log_path, 'wb')
        print(f'starting fresh -> {output_path}', file=sys.stderr)

    offset = start_off
    last_ckpt = time.time()
    t0 = time.time()
    read0 = read

    def checkpoint():
        out.flush()
        os.fsync(out.fileno())
        if drop_out is not None:
            drop_out.flush()
            os.fsync(drop_out.fileno())
        tmp = prog_path + '.tmp'
        with open(tmp, 'w') as pf:
            json.dump({
                'input_offset': offset,
                'output_size': out.tell(),
                'drop_log_size': drop_out.tell() if drop_out is not None else 0,
                'rows_read': read,
                'rows_kept': kept,
                'rows_dropped': dropped,
                'total_bytes': total_bytes,
                'ts': time.time(),
            }, pf)
        os.replace(tmp, prog_path)

    pool = Pool(workers, initializer=_init, initargs=(cfg,))
    try:
        for rawlen, keep, payload in pool.imap(_work, _raw_lines(input_path, start_off), chunksize=chunksize):
            offset += rawlen
            read += 1
            if keep:
                out.write(payload)
                kept += 1
            elif payload is not None:  # a drop-log line
                drop_out.write(payload)
                dropped += 1
            now = time.time()
            if now - last_ckpt >= checkpoint_secs:
                checkpoint()
                last_ckpt = now
                rate = (read - read0) / max(1e-9, now - t0)
                pct = 100 * offset / total_bytes
                eta = (total_bytes - offset) / max(1, offset - start_off) * (now - t0)
                print(
                    f'  {pct:5.1f}% | {read:,} read, {kept:,} kept, {dropped:,} dropped | '
                    f'{rate:.0f} rows/s | ETA {eta / 3600:.1f}h',
                    file=sys.stderr, flush=True
                )
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print('\ninterrupted — checkpointing before exit...', file=sys.stderr)
        pool.terminate()
    finally:
        checkpoint()
        out.close()
        if drop_out is not None:
            drop_out.close()

    print(f'\ndone: {read:,} read, {kept:,} kept, {dropped:,} dropped -> {output_path}', file=sys.stderr)
    if offset >= total_bytes:
        if os.path.exists(prog_path):
            os.remove(prog_path)  # finished cleanly
        print('(complete; removed .progress)', file=sys.stderr)
    return read, kept, dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='*', help='input jsonl file(s); or use --batch DIR')
    ap.add_argument('--batch', metavar='DIR', help='process every *.jsonl in DIR')
    ap.add_argument('--out', help='output file (single-input mode)')
    ap.add_argument('--out-dir', default='.', help='output directory (batch mode; default ".")')
    ap.add_argument('--drop-log', help='file to write dropped entries + reasons (single-input mode)')
    ap.add_argument('--drop-log-dir', default='dropped', help='dir for per-file drop logs (batch mode; default "dropped")')
    ap.add_argument('--field', default='text')
    ap.add_argument('--style-weight', type=float, default=1.0)
    ap.add_argument('--marker-weight', type=float, default=1.0)
    ap.add_argument('--threshold', type=float, default=0.75)
    ap.add_argument('--min-english', type=float, default=0.0)
    ap.add_argument('--min-tokens', type=int, default=0)
    ap.add_argument('--preserve-schema', action='store_true', help='emit the full original record (input schema) instead of {"text", score...}')
    ap.add_argument('--normalize-unicode', metavar='PATH', help='unicode-normalize all strings via this unicode.py module (ftfy + bad-char strip)')
    ap.add_argument('--replace-terms', metavar='PAIRS', help="pre-filter word replacements, comma-separated old:new pairs, applied to every string before scoring (e.g. 'infrastructure:structure'); case-insensitive, whole-word")
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--chunksize', type=int, default=16, help='lines per task batch')
    ap.add_argument('--checkpoint-secs', type=float, default=10.0)
    ap.add_argument('--restart', action='store_true', help='ignore any checkpoint and start over')
    args = ap.parse_args(argv)

    # ---- resolve inputs ----
    if args.batch:
        input_files = sorted(glob.glob(os.path.join(args.batch, '*.jsonl')))
        if not input_files:
            print(f'no *.jsonl files found in {args.batch}', file=sys.stderr)
            sys.exit(1)
    else:
        input_files = list(args.inputs)
    if not input_files:
        ap.print_help()
        sys.exit(1)

    cfg = {
        'field': args.field,
        'style': args.style_weight,
        'marker': args.marker_weight,
        'threshold': args.threshold,
        'min_english': args.min_english,
        'min_tokens': args.min_tokens,
        'preserve_schema': args.preserve_schema,
        'normalize_path': args.normalize_unicode,
    }
    if args.replace_terms:
        for pair in args.replace_terms.split(','):
            if ':' in pair:
                old, new = pair.split(':', 1)
                cfg.setdefault('replace_terms', {})[old.strip()] = new.strip()

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.batch:
        # single-input mode: --out defaults to <input>-filtered.jsonl
        if args.out is None:
            base = os.path.basename(input_files[0])
            stem, _ = os.path.splitext(base)
            args.out = os.path.join(args.out_dir, f'{stem}-filtered.jsonl')

    total_read = total_kept = total_dropped = 0
    t0 = time.time()
    for i, input_path in enumerate(input_files):
        base = os.path.basename(input_path)
        stem, _ = os.path.splitext(base)
        if len(input_files) == 1 and args.out:
            out_path = args.out
        else:
            out_path = os.path.join(args.out_dir, base)
        if len(input_files) == 1 and args.drop_log:
            drop_log_path = args.drop_log
        elif args.drop_log or args.batch:
            os.makedirs(args.drop_log_dir, exist_ok=True)
            drop_log_path = os.path.join(args.drop_log_dir, f'{stem}.dropped.jsonl')
        else:
            drop_log_path = None

        print(f'[{i+1}/{len(input_files)}] {base}', file=sys.stderr, flush=True)
        read, kept, dropped = process_file(
            input_path, out_path, cfg,
            drop_log_path=drop_log_path,
            restart=args.restart,
            workers=args.workers,
            chunksize=args.chunksize,
            checkpoint_secs=args.checkpoint_secs,
        )
        total_read += read
        total_kept += kept
        total_dropped += dropped
        print(f'  -> {base}: {read:,} read, {kept:,} kept ({100*kept/max(1,read):.1f}%), '
              f'{dropped:,} dropped ({100*dropped/max(1,read):.1f}%)', file=sys.stderr, flush=True)

    elapsed = time.time() - t0
    print('\n=== SUMMARY ===', file=sys.stderr)
    print(f'  files:   {len(input_files)}', file=sys.stderr)
    print(f'  read:    {total_read:,}', file=sys.stderr)
    print(f'  kept:    {total_kept:,} ({100*total_kept/max(1,total_read):.1f}%)', file=sys.stderr)
    print(f'  dropped: {total_dropped:,} ({100*total_dropped/max(1,total_read):.1f}%)', file=sys.stderr)
    print(f'  elapsed: {elapsed/60:.1f} min', file=sys.stderr)

    summary = {
        'files': len(input_files),
        'total_read': total_read,
        'total_kept': total_kept,
        'total_dropped': total_dropped,
        'keep_pct': round(100 * total_kept / max(1, total_read), 2),
        'drop_pct': round(100 * total_dropped / max(1, total_read), 2),
        'threshold': args.threshold,
        'style_weight': args.style_weight,
        'marker_weight': args.marker_weight,
        'min_english': args.min_english,
        'min_tokens': args.min_tokens,
        'preserve_schema': args.preserve_schema,
        'normalize_unicode': args.normalize_unicode,
        'elapsed_sec': round(elapsed, 1),
        'elapsed_min': round(elapsed / 60, 1),
    }
    summary_path = os.path.join(args.out_dir, '_filter_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'summary -> {summary_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
