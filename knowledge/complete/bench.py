#!/usr/bin/env python3
"""Throughput/latency benchmark for an OpenAI-compatible completions server.

Drives the local vLLM / SGLang server the same way sample.py does (raw
/v1/completions, no chat template — the TypeWriter model is a base model), but
records the numbers sample.py throws away: per-request latency and the server's
reported `usage.completion_tokens`, so we can compute real decode throughput.

Sweeps one or more concurrency levels against the SAME running server and prints
one JSON line per level (plus a human-readable summary). Intended to be run
INSIDE the serving container so it only ever talks to 127.0.0.1.
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def load_seeds(path: str, limit: int, words: int):
    seeds = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if words > 0:
                line = ' '.join(line.split()[:words])
            if line:
                seeds.append(line)
            if len(seeds) >= limit:
                break
    return seeds


def one_request(api_url, model, prompt, i, max_tokens, temperature, top_p, top_k, min_p, rep_pen, timeout):
    payload = {
        'prompt': prompt,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'min_p': min_p,
        'repetition_penalty': rep_pen,
        'seed': 42 + i,
        'stream': False,
    }
    if model:
        payload['model'] = model
    t0 = time.perf_counter()
    resp = requests.post(f'{api_url}/v1/completions', json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    dt = time.perf_counter() - t0
    usage = data.get('usage') or {}
    comp = usage.get('completion_tokens')
    if comp is None:  # server didn't report usage — approximate by whitespace
        comp = len(data['choices'][0].get('text', '').split())
    return dt, comp


def run_level(api_url, model, seeds, concurrency, gen, timeout):
    n = len(seeds)
    latencies = []
    total_tokens = 0
    errors = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [
            pool.submit(one_request, api_url, model, seeds[i], i,
                        gen['max_tokens'], gen['temperature'], gen['top_p'],
                        gen['top_k'], gen['min_p'], gen['rep_pen'], timeout)
            for i in range(n)
        ]
        for f in as_completed(futs):
            try:
                dt, comp = f.result()
                latencies.append(dt)
                total_tokens += comp
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f'[bench]   request failed: {exc}', file=sys.stderr)
    wall = time.perf_counter() - t0
    ok = len(latencies)
    latencies.sort()

    def pct(p):
        if not latencies:
            return 0.0
        k = min(len(latencies) - 1, int(round(p / 100 * (len(latencies) - 1))))
        return latencies[k]

    return {
        'concurrency': concurrency,
        'requests': n,
        'ok': ok,
        'errors': errors,
        'max_tokens': gen['max_tokens'],
        'wall_s': round(wall, 2),
        'total_completion_tokens': total_tokens,
        'output_tok_per_s': round(total_tokens / wall, 1) if wall else 0.0,
        'requests_per_s': round(ok / wall, 3) if wall else 0.0,
        'mean_latency_s': round(statistics.mean(latencies), 2) if latencies else 0.0,
        'p50_latency_s': round(pct(50), 2),
        'p95_latency_s': round(pct(95), 2),
        'mean_tok_per_req': round(total_tokens / ok, 1) if ok else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('seeds_file')
    ap.add_argument('--api-url', default='http://127.0.0.1:1234')
    ap.add_argument('--model', default=None)
    ap.add_argument('--limit', type=int, default=64, help='number of seeds/requests per level')
    ap.add_argument('--seed-words', type=int, default=3, help='keep first N words of each seed (0=whole line)')
    ap.add_argument('--concurrency', default='1,8,32,64', help='comma-separated concurrency levels to sweep')
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=1.25)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--top-k', type=int, default=60)
    ap.add_argument('--min-p', type=float, default=0.033)
    ap.add_argument('--repetition-penalty', type=float, default=1.075)
    ap.add_argument('--timeout', type=int, default=1200)
    ap.add_argument('--label', default='', help='free-text tag echoed into each result row (e.g. engine+flags)')
    ap.add_argument('--warmup', type=int, default=2, help='throwaway requests before timing')
    ap.add_argument('-o', '--output', default=None, help='append JSON result rows to this file')
    args = ap.parse_args()

    seeds = load_seeds(args.seeds_file, args.limit, args.seed_words)
    if not seeds:
        print('ERROR: no seeds loaded', file=sys.stderr)
        sys.exit(1)
    # Pad/truncate to exactly `limit` requests by cycling seeds.
    while len(seeds) < args.limit:
        seeds += seeds
    seeds = seeds[: args.limit]

    gen = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'min_p': args.min_p,
        'rep_pen': args.repetition_penalty,
    }

    if args.warmup > 0:
        print(f'[bench] warming up with {args.warmup} request(s) ...', file=sys.stderr)
        try:
            for i in range(args.warmup):
                one_request(args.api_url, args.model, seeds[i % len(seeds)], i,
                            gen['max_tokens'], gen['temperature'], gen['top_p'],
                            gen['top_k'], gen['min_p'], gen['rep_pen'], args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f'[bench] warmup failed: {exc}', file=sys.stderr)

    out_fh = open(args.output, 'a', encoding='utf-8') if args.output else None
    levels = [int(x) for x in args.concurrency.split(',') if x.strip()]
    print(f'[bench] label={args.label!r}  requests/level={args.limit}  '
          f'max_tokens={args.max_tokens}  levels={levels}', file=sys.stderr)
    print(f'{"conc":>5} {"reqs":>5} {"wall_s":>8} {"out_tok/s":>10} {"req/s":>7} '
          f'{"mean_lat":>9} {"p95_lat":>8} {"tok/req":>8}', file=sys.stderr)
    for c in levels:
        row = run_level(args.api_url, args.model, seeds, c, gen, args.timeout)
        row['label'] = args.label
        print(f'{row["concurrency"]:>5} {row["ok"]:>5} {row["wall_s"]:>8} '
              f'{row["output_tok_per_s"]:>10} {row["requests_per_s"]:>7} '
              f'{row["mean_latency_s"]:>9} {row["p95_latency_s"]:>8} '
              f'{row["mean_tok_per_req"]:>8}', file=sys.stderr)
        line = json.dumps(row, ensure_ascii=False)
        print(line)
        if out_fh:
            out_fh.write(line + '\n')
            out_fh.flush()
    if out_fh:
        out_fh.close()


if __name__ == '__main__':
    main()
