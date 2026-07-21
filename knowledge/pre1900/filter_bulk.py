"""Fast, parallel, resumable pre-1900 filter for very large JSONL files.

Same scoring as `detect.py --filter`, but:
  * multiprocessing across CPU cores,
  * a fast anachronism matcher (token lookups instead of a giant regex),
  * a checkpoint written every ~10s to  <output>.progress  so an interrupted
    run resumes in place, losing only the last few seconds of work.

Usage (matches your detect.py knobs):
    python filter_bulk.py split-long-docs2.jsonl \
        --out Institutional-Books.jsonl \
        --style-weight 0.015 --marker-weight 0.015 --threshold 0.5

    # if it stops for any reason, just run the SAME command again -> it resumes.
    # to start over instead of resuming:  add  --restart
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

from detect import Scorer, tokenize

_S = None
_CFG = None


def _init(cfg):
    global _S, _CFG
    _CFG = cfg
    _S = Scorer()


def _work(raw):
    """raw: one input line as bytes (incl. trailing newline). Returns
    (rawlen, keep, payload_str_or_None)."""
    rawlen = len(raw)
    try:
        row = json.loads(raw)
        text = str(row.get(_CFG['field'], '') or '')
        if not text:
            return rawlen, False, None
        toks = tokenize(text)
        r = _S.score(text, toks, style_weight=_CFG['style'], marker_weight=_CFG['marker'])
        keep = r['p_pre1900'] >= _CFG['threshold'] and r['english_frac'] >= _CFG['min_english'] and r['n_tokens'] >= _CFG['min_tokens']
        if not keep:
            return rawlen, False, None
        r['keep'] = True
        return rawlen, True, json.dumps({'text': text, **r}, ensure_ascii=False)
    except Exception:
        return rawlen, False, None  # skip malformed lines rather than crash


def _raw_lines(path, start):
    with open(path, 'rb') as f:
        f.seek(start)
        for raw in f:
            yield raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--out', required=True)
    ap.add_argument('--field', default='text')
    ap.add_argument('--style-weight', type=float, default=1.0)
    ap.add_argument('--marker-weight', type=float, default=1.0)
    ap.add_argument('--threshold', type=float, default=0.75)
    ap.add_argument('--min-english', type=float, default=0.0)
    ap.add_argument('--min-tokens', type=int, default=0)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--chunksize', type=int, default=16, help='lines per task batch')
    ap.add_argument('--checkpoint-secs', type=float, default=10.0)
    ap.add_argument('--restart', action='store_true', help='ignore any checkpoint and start over')
    args = ap.parse_args()

    prog_path = args.out + '.progress'
    total_bytes = os.path.getsize(args.input)

    # ---- resume or start fresh ----
    start_off = read = kept = 0
    if not args.restart and os.path.exists(prog_path) and os.path.exists(args.out):
        p = json.load(open(prog_path))
        start_off, read, kept = p['input_offset'], p['rows_read'], p['rows_kept']
        out = open(args.out, 'r+b')
        out.truncate(p['output_size'])  # drop anything written after the checkpoint
        out.seek(p['output_size'])
        print(
            f'resuming at byte {start_off:,} / {total_bytes:,} ({100 * start_off / total_bytes:.1f}%), {kept:,} kept so far',
            file=sys.stderr,
        )
    else:
        out = open(args.out, 'wb')
        print(f'starting fresh -> {args.out}', file=sys.stderr)

    cfg = {
        'field': args.field,
        'style': args.style_weight,
        'marker': args.marker_weight,
        'threshold': args.threshold,
        'min_english': args.min_english,
        'min_tokens': args.min_tokens,
    }

    offset = start_off
    last_ckpt = time.time()
    t0 = time.time()
    read0 = read

    def checkpoint():
        out.flush()
        os.fsync(out.fileno())
        tmp = prog_path + '.tmp'
        with open(tmp, 'w') as pf:
            json.dump(
                {
                    'input_offset': offset,
                    'output_size': out.tell(),
                    'rows_read': read,
                    'rows_kept': kept,
                    'total_bytes': total_bytes,
                    'ts': time.time(),
                },
                pf,
            )
        os.replace(tmp, prog_path)

    pool = Pool(args.workers, initializer=_init, initargs=(cfg,))
    try:
        for rawlen, keep, payload in pool.imap(_work, _raw_lines(args.input, start_off), chunksize=args.chunksize):
            offset += rawlen
            read += 1
            if keep:
                out.write(payload.encode('utf-8'))
                out.write(b'\n')
                kept += 1
            now = time.time()
            if now - last_ckpt >= args.checkpoint_secs:
                checkpoint()
                last_ckpt = now
                rate = (read - read0) / max(1e-9, now - t0)
                pct = 100 * offset / total_bytes
                eta = (total_bytes - offset) / max(1, offset - start_off) * (now - t0)
                print(
                    f'  {pct:5.1f}% | {read:,} read, {kept:,} kept | {rate:.0f} rows/s | ETA {eta / 3600:.1f}h', file=sys.stderr, flush=True
                )
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print('\ninterrupted — checkpointing before exit...', file=sys.stderr)
        pool.terminate()
    finally:
        checkpoint()
        out.close()

    print(f'\ndone: {read:,} read, {kept:,} kept -> {args.out}', file=sys.stderr)
    if offset >= total_bytes:
        os.path.exists(prog_path) and os.remove(prog_path)  # finished cleanly
        print('(complete; removed .progress)', file=sys.stderr)


if __name__ == '__main__':
    main()
