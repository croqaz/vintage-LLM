#!/usr/bin/env python3
"""Capture (or compare) generations for a config, to check quality regressions.

Two modes:
  --dump FILE   generate N real-workload prompts greedily (temperature 0) and
                save {prompt_hash, prompt_name, text} to FILE.
  --compare A B compare two dumps produced that way: character-level similarity
                and length stats per prompt template.

Greedy decoding makes the comparison meaningful — any divergence is the config
(fp8 weights, fp8 KV, backend), not sampling noise.
"""

import argparse
import asyncio
import difflib
import json
import statistics
import sys
from collections import defaultdict

import generate as gen
from openai import AsyncOpenAI

PROMPTS = [
    'prompts/continue-v2.txt',
    'prompts/diverse_qa_pairs.txt',
    'prompts/extract_knowledge.txt',
    'prompts/narrative.txt',
    'prompts/lost-manuscript-framing.txt',
]


async def dump(args) -> None:
    templates = gen.load_templates(PROMPTS, None)
    if args.reorder:
        from bench_real import reorder

        templates = {k: reorder(v) for k, v in templates.items()}
    seeds = list(gen.iter_seeds(args.seeds, args.offset, args.num_seeds))
    tasks = list(gen.build_tasks(seeds, templates, 3000, 1, set(), None))[: args.limit]
    client = AsyncOpenAI(base_url=args.base_url, api_key='not-needed', timeout=600.0, max_retries=0)
    model = (await client.models.list()).data[0].id
    sem = asyncio.Semaphore(16)
    out = [None] * len(tasks)

    async def go(i, t):
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{'role': 'user', 'content': gen.render(t.template, t.chunk)}],
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                )
                out[i] = {
                    'key': f'{t.prompt_name}:{t.seed_index}:{t.chunk_index}',
                    'prompt': t.prompt_name,
                    'text': r.choices[0].message.content or '',
                }
            except Exception as e:
                out[i] = {
                    'key': f'{t.prompt_name}:{t.seed_index}:{t.chunk_index}',
                    'prompt': t.prompt_name,
                    'text': '',
                    'error': str(e)[:120],
                }

    await asyncio.gather(*(go(i, t) for i, t in enumerate(tasks)))
    await client.close()
    with open(args.dump, 'w', encoding='utf-8') as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    ok = [r for r in out if r['text']]
    sys.stderr.write(f'[dump] {len(ok)}/{len(out)} generations -> {args.dump}\n')


def load(path):
    with open(path, encoding='utf-8') as f:
        return {r['key']: r for r in map(json.loads, f) if r.get('text')}


def compare(a_path, b_path) -> None:
    a, b = load(a_path), load(b_path)
    keys = sorted(set(a) & set(b))
    per = defaultdict(list)
    for k in keys:
        ta, tb = a[k]['text'], b[k]['text']
        per[a[k]['prompt']].append((difflib.SequenceMatcher(None, ta, tb).ratio(), len(ta), len(tb)))
    print(f'{"template":<26} {"n":>3} {"similarity":>10} {"len A":>8} {"len B":>8}')
    allr = []
    for name, rows in sorted(per.items()):
        r = [x[0] for x in rows]
        allr += r
        print(
            f'{name:<26} {len(rows):>3} {statistics.mean(r):>10.3f} '
            f'{statistics.mean(x[1] for x in rows):>8.0f} {statistics.mean(x[2] for x in rows):>8.0f}'
        )
    if allr:
        print(f'{"OVERALL":<26} {len(allr):>3} {statistics.mean(allr):>10.3f}')
        print(
            f'  identical: {sum(1 for r in allr if r > 0.999)}/{len(allr)}   '
            f'>0.9: {sum(1 for r in allr if r > 0.9)}   <0.5: {sum(1 for r in allr if r < 0.5)}'
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://127.0.0.1:4567/v1')
    ap.add_argument('--seeds', default='seeds.jsonl')
    ap.add_argument('--offset', type=int, default=1000)
    ap.add_argument('--num-seeds', type=int, default=12)
    ap.add_argument('--limit', type=int, default=40)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--reorder', action='store_true', help='chunk-first prompts')
    ap.add_argument('--dump')
    ap.add_argument('--compare', nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.dump:
        asyncio.run(dump(args))
    else:
        ap.error('need --dump or --compare')


if __name__ == '__main__':
    main()
