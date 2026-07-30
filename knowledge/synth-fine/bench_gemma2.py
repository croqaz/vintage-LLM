#!/usr/bin/env python3
"""Benchmark that reproduces the *production* synth-data workload exactly.

Unlike bench_gemma.py (short 900-token prompts, 256-token outputs), this mirrors
what the long-running generate.py job actually sends: real seed chunks built by
generate.py's own chunker, the real prompt templates, no system prompt,
max_tokens=1024, temperature=0.8. Reports req/s (directly comparable to the
`N/M done ... X/s` line generate.py prints) plus token throughput.

    python bench_real.py prompts/continue-v2.txt ... \
        --base-url http://localhost:4567/v1 --num-seeds 80 --concurrency 64
"""

import argparse
import asyncio
import json
import sys
import time

import generate as gen
from openai import AsyncOpenAI

PLACEHOLDER = '{text}'


def reorder(template: str) -> str:
    """Move the seed text to the FRONT of the prompt.

    All templates ship as `<instructions>\n{text}`, so every one of the N
    templates applied to the same chunk has a different prefix and vLLM's prefix
    cache can never reuse the (expensive, ~3000-token) chunk prefill. Putting the
    chunk first makes it a shared prefix across all N requests for that chunk.
    """
    head, _, tail = template.partition(PLACEHOLDER)
    return PLACEHOLDER + '\n\n' + head.strip() + tail.strip()


async def one(client, model, sem, prompt, args, results, errs):
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            u = r.usage
            results.append((time.perf_counter() - t0, u.prompt_tokens, u.completion_tokens))
        except Exception as e:
            msg = str(e)
            errs['ctx' if 'maximum context length' in msg else 'other'] += 1
            errs['last'] = msg[:160]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('prompts', nargs='+')
    ap.add_argument('--base-url', default='http://localhost:4567/v1')
    ap.add_argument('--api-key', default='not-needed')
    ap.add_argument('--seeds', default='seeds.jsonl')
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--num-seeds', type=int, default=80)
    ap.add_argument('--chunk-tokens', type=int, default=3000)
    ap.add_argument('--max-chunks', type=int, default=3)
    ap.add_argument('--max-tokens', type=int, default=1024)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--concurrency', type=int, default=192)
    ap.add_argument('--reorder', action='store_true', help='chunk-first prompts (prefix-cache friendly)')
    ap.add_argument('--clean', action='store_true')
    ap.add_argument('--warmup', type=int, default=32, help='warmup requests at max_tokens=32')
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    templates = gen.load_templates(args.prompts, None)
    if args.reorder:
        templates = {k: reorder(v) for k, v in templates.items()}
    seeds = list(gen.iter_seeds(args.seeds, args.offset, args.num_seeds))
    clean_opts = {'aggressive': False, 'keep_paragraphs': False} if args.clean else None
    tasks = list(gen.build_tasks(seeds, templates, args.chunk_tokens, args.max_chunks, set(), clean_opts))
    prompts = [gen.render(t.template, t.chunk) for t in tasks]

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600.0, max_retries=0)
    model = (await client.models.list()).data[0].id
    sem = asyncio.Semaphore(args.concurrency)
    errs = {'ctx': 0, 'other': 0, 'last': ''}

    warm = prompts[: min(len(prompts), args.warmup)]
    wargs = argparse.Namespace(**{**vars(args), 'max_tokens': 32})
    await asyncio.gather(*(one(client, model, sem, p, wargs, [], dict(errs)) for p in warm))

    results: list[tuple[float, int, int]] = []
    t0 = time.perf_counter()
    await asyncio.gather(*(one(client, model, sem, p, args, results, errs) for p in prompts))
    wall = time.perf_counter() - t0
    await client.close()

    lats = sorted(dt for dt, _, _ in results)
    comp = sum(c for _, _, c in results)
    prom = sum(p for _, p, _ in results)
    n = len(prompts)
    out = {
        'label': args.label,
        'concurrency': args.concurrency,
        'reorder': args.reorder,
        'requests': n,
        'ok': len(results),
        'err_ctx': errs['ctx'],
        'err_other': errs['other'],
        'wall_s': round(wall, 2),
        'req_per_s': round(len(results) / wall, 2),
        'out_tok_per_s': round(comp / wall, 1),
        'prompt_tok_per_s': round(prom / wall, 1),
        'mean_out_tok': round(comp / len(results)) if results else 0,
        'mean_prompt_tok': round(prom / len(results)) if results else 0,
        'p50_lat_s': round(lats[len(lats) // 2], 2) if lats else 0,
        'p99_lat_s': round(lats[min(len(lats) - 1, int(len(lats) * 0.99))], 2) if lats else 0,
        'max_tokens': args.max_tokens,
    }
    print(json.dumps(out))
    sys.stderr.write(
        f'[{args.label}] c={args.concurrency}{" reorder" if args.reorder else ""} '
        f'{out["req_per_s"]} req/s  {out["out_tok_per_s"]} out-tok/s  '
        f'({out["ok"]}/{n} ok, ctx-err {errs["ctx"]}, other {errs["other"]}, '
        f'wall {out["wall_s"]}s, p50 {out["p50_lat_s"]}s p99 {out["p99_lat_s"]}s)\n'
    )
    if errs['other']:
        sys.stderr.write(f'  last error: {errs["last"]}\n')


if __name__ == '__main__':
    asyncio.run(main())
