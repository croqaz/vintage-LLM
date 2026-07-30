#!/usr/bin/env python3
"""Throughput/latency benchmark for a running vLLM OpenAI server.

Builds a fixed workload (N seeds x M prompt templates, chat mode) and fires it
at the endpoint with a bounded in-flight concurrency, then reports exact token
throughput from the API `usage` fields. Run against a server started separately
(e.g. `synth.sh serve`). Emits one JSON result line on stdout (for scripting)
plus a human summary on stderr.

    python bench_gemma.py prompts/continue-v2.txt prompts/narrative.txt \
        --base-url http://localhost:1234/v1 --num-seeds 40 \
        --chunk-tokens 700 --max-tokens 256 --concurrency 128
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import ocr_clean
import tiktoken
from openai import AsyncOpenAI

_ENC = tiktoken.get_encoding('cl100k_base')


def load_chunks(seeds_path: str, num_seeds: int, chunk_tokens: int) -> list[str]:
    chunks: list[str] = []
    with open(seeds_path, encoding='utf-8') as f:
        for line in f:
            if len(chunks) >= num_seeds:
                break
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get('text')
            except json.JSONDecodeError:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            text = ocr_clean.clean(text)
            toks = _ENC.encode(text)[:chunk_tokens]
            chunks.append(_ENC.decode(toks))
    return chunks


async def one(client, model, sem, messages, max_tokens, temperature, results):
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
            )
            dt = time.perf_counter() - t0
            u = r.usage
            results.append((dt, u.prompt_tokens, u.completion_tokens))
        except Exception as e:
            results.append((time.perf_counter() - t0, 0, 0))
            sys.stderr.write(f'[err] {e}\n')


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('prompts', nargs='+')
    ap.add_argument('--base-url', default='http://localhost:1234/v1')
    ap.add_argument('--api-key', default='not-needed')
    ap.add_argument('--system', default='systems/vintage_1850.txt')
    ap.add_argument('--seeds', default='seeds.jsonl')
    ap.add_argument('--num-seeds', type=int, default=40)
    ap.add_argument('--chunk-tokens', type=int, default=700)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--concurrency', type=int, default=128)
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    system = Path(args.system).read_text(encoding='utf-8').strip() if args.system else None
    templates = [Path(p).read_text(encoding='utf-8') for p in args.prompts]
    chunks = load_chunks(args.seeds, args.num_seeds, args.chunk_tokens)

    reqs = []
    for chunk in chunks:
        for tpl in templates:
            msgs = []
            if system:
                msgs.append({'role': 'system', 'content': system})
            msgs.append({'role': 'user', 'content': tpl.replace('{text}', chunk)})
            reqs.append(msgs)

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    model = (await client.models.list()).data[0].id
    sem = asyncio.Semaphore(args.concurrency)
    results: list[tuple[float, int, int]] = []

    # brief warmup so we time steady state, not cold cudagraph/first-token
    warm = reqs[: min(len(reqs), args.concurrency)]
    await asyncio.gather(*(one(client, model, sem, m, 32, args.temperature, []) for m in warm))

    t0 = time.perf_counter()
    await asyncio.gather(*(one(client, model, sem, m, args.max_tokens, args.temperature, results) for m in reqs))
    wall = time.perf_counter() - t0
    await client.close()

    lats = sorted(dt for dt, _, _ in results)
    ok = [(dt, p, c) for dt, p, c in results if c > 0]
    comp = sum(c for _, _, c in ok)
    prom = sum(p for _, p, c in ok)
    n = len(reqs)
    out = {
        'label': args.label,
        'model': model,
        'concurrency': args.concurrency,
        'requests': n,
        'ok': len(ok),
        'wall_s': round(wall, 2),
        'prompt_tokens': prom,
        'completion_tokens': comp,
        'out_tok_per_s': round(comp / wall, 1),
        'total_tok_per_s': round((comp + prom) / wall, 1),
        'req_per_s': round(len(ok) / wall, 2),
        'mean_lat_s': round(sum(lats) / len(lats), 2) if lats else 0,
        'p50_lat_s': round(lats[len(lats) // 2], 2) if lats else 0,
        'p99_lat_s': round(lats[min(len(lats) - 1, int(len(lats) * 0.99))], 2) if lats else 0,
        'max_tokens': args.max_tokens,
        'chunk_tokens': args.chunk_tokens,
    }
    print(json.dumps(out))
    sys.stderr.write(
        f'[{args.label}] c={args.concurrency} {out["out_tok_per_s"]} out-tok/s '
        f'({out["req_per_s"]} req/s, {out["ok"]}/{n} ok, wall {out["wall_s"]}s, '
        f'p50 {out["p50_lat_s"]}s p99 {out["p99_lat_s"]}s)\n'
    )


if __name__ == '__main__':
    asyncio.run(main())
