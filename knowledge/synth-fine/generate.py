#!/usr/bin/env python3
"""Synthetic-data generator for the vintage (~1850) LLM corpus.

Reads seeds from a JSONL file (each line: {"text": ...}), chunks the seed text
to a token budget, renders one or more prompt templates against each chunk, and
sends them to an OpenAI-compatible endpoint (default: the llama.cpp server on
localhost:1234). Results stream to a JSONL output file as they complete.

Prompt templates live in the ``prompts/`` folder as plain-text files. A template
is any text containing the placeholder ``{text}``, which is replaced (literal
substitution, so braces in the seed are safe) with the seed chunk. An optional
shared *system* prompt (``--system``) is prepended as the chat system message so
period-style constraints can be tuned independently of the task templates.

Example
-------
    python generate.py prompts/continue.txt prompts/summarize.txt \
        --system systems/vintage_1850.txt \
        --limit 100 --concurrency 8 --output out/synth.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

try:
    import tiktoken

    _ENC = tiktoken.get_encoding('cl100k_base')
except Exception:  # pragma: no cover - tiktoken is a soft dependency
    _ENC = None

import ocr_clean
from openai import AsyncOpenAI

PLACEHOLDER = '{text}'
ERA_PLACEHOLDER = '{era}'


def apply_era(text: str, era: str | None) -> str:
    """Fill the {era} placeholder (verbatim) in a template or system prompt."""
    if era is None:
        return text
    return text.replace(ERA_PLACEHOLDER, era)


# --------------------------------------------------------------------------- #
# Chunking                                                                    #
# --------------------------------------------------------------------------- #
def chunk_text(text: str, chunk_tokens: int, max_chunks: int) -> list[str]:
    """Split ``text`` into at most ``max_chunks`` windows of ~``chunk_tokens``.

    Uses tiktoken (cl100k_base) as an approximate tokenizer when available.
    Falcon uses a different vocab, so this is a budget estimate, not exact —
    which is fine, we only need to keep prompts comfortably under context.
    Falls back to a ~4-chars-per-token heuristic if tiktoken is unavailable.
    """
    text = text.strip()
    if not text:
        return []
    if _ENC is not None:
        toks = _ENC.encode(text)
        chunks = [_ENC.decode(toks[i : i + chunk_tokens]) for i in range(0, len(toks), chunk_tokens)]
    else:
        span = chunk_tokens * 4
        chunks = [text[i : i + span] for i in range(0, len(text), span)]
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks[:max_chunks]


def iter_seeds(path: str, offset: int, limit: int | None) -> Iterator[tuple[int, str]]:
    """Yield ``(seed_index, text)`` from a JSONL file, honoring offset/limit."""
    yielded = 0
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx < offset:
                continue
            if limit is not None and yielded >= limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get('text')
            if isinstance(text, str) and text.strip():
                yield idx, text
                yielded += 1


def render(template: str, chunk: str) -> str:
    """Literal-substitute the {text} placeholder (safe with braces in data)."""
    return template.replace(PLACEHOLDER, chunk)


# --------------------------------------------------------------------------- #
# Work items                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    prompt_name: str
    template: str
    seed_index: int
    chunk_index: int
    chunk: str


def build_tasks(
    seeds: Iterable[tuple[int, str]],
    templates: dict[str, str],
    chunk_tokens: int,
    max_chunks: int,
    done: set[tuple[str, int, int]],
    clean_opts: dict | None = None,
) -> Iterator[Task]:
    for seed_index, text in seeds:
        if clean_opts is not None:
            # Clean the whole seed before chunking, so token budgeting reflects
            # the text we actually send and chunk boundaries land on clean text.
            text = ocr_clean.clean(text, **clean_opts)
        chunks = chunk_text(text, chunk_tokens, max_chunks)
        for chunk_index, chunk in enumerate(chunks):
            for name, template in templates.items():
                if (name, seed_index, chunk_index) in done:
                    continue
                yield Task(name, template, seed_index, chunk_index, chunk)


def load_done(output_path: str) -> set[tuple[str, int, int]]:
    """Read an existing output file and collect completed work identities."""
    done: set[tuple[str, int, int]] = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (obj.get('prompt'), obj.get('seed_index'), obj.get('chunk_index'))
            if None not in key:
                done.add(key)  # type: ignore[arg-type]
    return done


# --------------------------------------------------------------------------- #
# Generation                                                                  #
# --------------------------------------------------------------------------- #
async def worker(
    client: AsyncOpenAI,
    model_id: str,
    task: Task,
    args: argparse.Namespace,
    system_prompt: str | None,
    out,
    lock: asyncio.Lock,
    counters: dict,
) -> None:
    rendered = render(task.template, task.chunk)

    for attempt in range(args.retries + 1):
        try:
            if args.mode == 'completion':
                # Raw text continuation: the model literally continues the
                # prompt, which eliminates chat-style meta-commentary such as
                # "This document discusses...". System prompt (if any) is
                # prepended as plain text.
                prompt = f'{system_prompt}\n\n{rendered}' if system_prompt else rendered
                resp = await client.completions.create(
                    model=model_id,
                    prompt=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
                text = (resp.choices[0].text or '').strip()
            else:
                messages = []
                if system_prompt:
                    messages.append({'role': 'system', 'content': system_prompt})
                messages.append({'role': 'user', 'content': rendered})
                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
                text = (resp.choices[0].message.content or '').strip()
            record = {
                'prompt': task.prompt_name,
                'model': model_id,
                'params': {
                    'mode': args.mode,
                    'temperature': args.temperature,
                    'top_p': args.top_p,
                    'max_tokens': args.max_tokens,
                    'clean': args.clean,
                },
                'source_preview': task.chunk[:999],
                'text': text,
                'seed_index': task.seed_index,
                'chunk_index': task.chunk_index,
            }
            async with lock:
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                out.flush()
                counters['ok'] += 1
                _progress(counters)
            return
        except Exception as e:
            if attempt >= args.retries:
                async with lock:
                    counters['err'] += 1
                    sys.stderr.write(f'\n[error] {task.prompt_name} seed={task.seed_index} chunk={task.chunk_index}: {e}\n')
                    _progress(counters)
                return
            await asyncio.sleep(min(2**attempt, 10))


def _progress(counters: dict) -> None:
    done = counters['ok'] + counters['err']
    total = counters['total']
    elapsed = time.time() - counters['start']
    rate = done / elapsed if elapsed > 0 else 0.0
    sys.stderr.write(f'\r{done}/{total} done  ok={counters["ok"]} err={counters["err"]}  {rate:.1f}/s')
    sys.stderr.flush()


async def run(args: argparse.Namespace) -> None:
    templates = load_templates(args.prompts, args.era)
    if not templates:
        sys.exit('No prompt templates given / none contained the {text} placeholder.')
    system_prompt = None
    if args.system:
        system_prompt = Path(args.system).read_text(encoding='utf-8').strip()
        if ERA_PLACEHOLDER in system_prompt and args.era is None:
            sys.exit(f'{args.system} uses {ERA_PLACEHOLDER} but no --era was given.')
        system_prompt = apply_era(system_prompt, args.era)

    clean_opts = None
    if args.clean:
        clean_opts = {
            'aggressive': args.aggressive_clean,
            'keep_paragraphs': args.keep_paragraphs,
        }

    # One or more comma-separated endpoints; auto-detect each server's model
    # unless an explicit --model is given (applied to all endpoints).
    base_urls = [u.strip() for u in args.base_url.split(',') if u.strip()]
    endpoints: list[tuple[AsyncOpenAI, str]] = []
    for url in base_urls:
        client = AsyncOpenAI(base_url=url, api_key=args.api_key)
        model_id = args.model or await autodetect_model(url, args.api_key)
        endpoints.append((client, model_id))

    done = load_done(args.output) if args.resume else set()
    if done:
        sys.stderr.write(f'[resume] skipping {len(done)} already-generated records\n')

    seeds = list(iter_seeds(args.seeds, args.offset, args.limit))
    tasks = list(build_tasks(seeds, templates, args.chunk_tokens, args.max_chunks, done, clean_opts))

    counters = {'ok': 0, 'err': 0, 'total': len(tasks), 'start': time.time()}
    sys.stderr.write(
        f'[plan] {len(seeds)} seeds x {len(templates)} prompts -> '
        f'{len(tasks)} generations, mode={args.mode}, clean={args.clean}, '
        f'endpoints={len(endpoints)}, concurrency={args.concurrency}\n'
    )
    if not tasks:
        for client, _ in endpoints:
            await client.close()
        return

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    async def guarded(i: int, task: Task, out) -> None:
        client, model_id = endpoints[i % len(endpoints)]  # round-robin
        async with sem:
            await worker(client, model_id, task, args, system_prompt, out, lock, counters)

    with open(args.output, 'a', encoding='utf-8') as out:
        await asyncio.gather(*(guarded(i, t, out) for i, t in enumerate(tasks)))
    for client, _ in endpoints:
        await client.close()
    sys.stderr.write('\n[done]\n')


def load_templates(paths: list[str], era: str | None = None) -> dict[str, str]:
    templates: dict[str, str] = {}
    for p in paths:
        text = Path(p).read_text(encoding='utf-8')
        if PLACEHOLDER not in text:
            sys.stderr.write(f'[warn] {p} has no {PLACEHOLDER} placeholder; skipping\n')
            continue
        if ERA_PLACEHOLDER in text and era is None:
            sys.exit(f'{p} uses {ERA_PLACEHOLDER} but no --era was given.')
        templates[Path(p).stem] = apply_era(text, era)
    return templates


async def autodetect_model(base_url: str, api_key: str) -> str:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    try:
        models = await client.models.list()
        model_id = models.data[0].id
        sys.stderr.write(f'[model] auto-detected: {model_id}\n')
        return model_id
    finally:
        await client.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prompts', nargs='+', help='One or more prompt template files.')
    p.add_argument('--seeds', default='seeds.jsonl', help='Input JSONL of seeds.')
    p.add_argument('--output', default='out/synth.jsonl', help='Output JSONL.')
    p.add_argument('--system', default=None, help='Optional shared system-prompt file.')
    p.add_argument(
        '--base-url',
        default='http://localhost:1234/v1',
        help='OpenAI-compatible endpoint(s). Comma-separate several to round-robin, '
        "e.g. 'http://localhost:1234/v1,http://localhost:4567/v1'.",
    )
    p.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY', 'not-needed'))
    p.add_argument('--model', default=None, help='Model id (auto-detected per endpoint if omitted).')
    p.add_argument(
        '--mode',
        choices=['chat', 'completion'],
        default='chat',
        help="'chat' uses /v1/chat/completions; 'completion' uses /v1/completions "
        '(raw text continuation — avoids chat-style meta-commentary).',
    )
    p.add_argument(
        '--era',
        default=None,
        help='Value substituted for the {era} placeholder in prompts and the '
        "system prompt, e.g. --era 'the year 1850' or --era 'mid-nineteenth-century'.",
    )
    p.add_argument('--clean', action='store_true', help='Conservatively clean OCR artefacts from seed text.')
    p.add_argument('--aggressive-clean', action='store_true', help='With --clean, also strip brace/angle glyphs.')
    p.add_argument('--keep-paragraphs', action='store_true', help='With --clean, preserve paragraph breaks.')
    p.add_argument('--chunk-tokens', type=int, default=3000)
    p.add_argument('--max-chunks', type=int, default=3, help='Max chunks per seed.')
    p.add_argument('--offset', type=int, default=0, help='Skip this many seeds.')
    p.add_argument('--limit', type=int, default=None, help='Max seeds to process.')
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=1024, help='Output token cap.')
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--resume', action='store_true', help='Skip records already in output.')
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
