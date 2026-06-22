#!/usr/bin/env python3
"""Extract 2-5 thematic keywords per poem via OpenRouter, for fine-tuning data.

Reads ``poems.jsonl`` ({title, author, text}) and, for each poem, asks an LLM
to produce a short list of *thematic* terms describing what the poem is about.
These terms are the conditioning signal for a later reverse-generation
fine-tuning task ("Write a poem about: love, deception" -> the original poem),
so they are deliberately *concepts*, not words copied from the text.

Output is written to ``poems_keywords.jsonl`` as the original record plus:
    "keywords":   ["love", "deception", ...]   (2-5 lowercase terms)
    "confidence": "high" | "low"               ("low" => abstract, review me)
    "model":      "<model slug used>"

Design notes
------------
- The *author* is intentionally NOT sent to the model (per the spec): we want
  terms grounded in the poem itself, not in what we know of the poet.
- Themes need not appear verbatim in the poem. For abstract poems the best
  term often does not occur in the text at all -- that is expected, and the
  model is told so. Genuinely hard/abstract poems are flagged confidence=low
  so they can be reviewed as a subset rather than trusted blindly.
- The run is *resumable*: poems already present in the output file (keyed by
  title+author) are skipped, so a crash or rate-limit interruption is cheap to
  recover from. Requests run concurrently with retry + backoff.

Usage
-----
    export OPENROUTER_API_KEY=sk-or-...
    python3 extract_keywords.py --limit 10        # small sample first
    python3 extract_keywords.py                   # full run (resumable)
    python3 extract_keywords.py --dry-run         # print prompt, no API calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'poems.jsonl'
OUTPUT = HERE / 'poems_keywords.jsonl'

# OpenRouter is OpenAI-compatible; point the SDK at its base URL.
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'

# Default model. Override with --model or the OPENROUTER_MODEL env var.
# NOTE: verify the exact slug for the model you want at
# https://openrouter.ai/models -- slugs change as new versions ship.
DEFAULT_MODEL = 'anthropic/claude-sonnet-4.6'


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a literary analyst. Given a poem's title and text, identify the 2 to 5 \
terms that best capture what the poem is *about* -- its central themes, \
subjects, and mood. These terms will be used as a writing prompt to regenerate \
a poem like this one, so they should read like what a person would type to ask \
for such a poem (e.g. "mortality", "unrequited love", "passage of time").

Rules:
- Return THEMES, SUBJECTS, and MOOD -- concepts and emotions, not words copied \
from the poem. The single best term often does NOT appear in the text at all; \
that is good. Infer the underlying meaning.
- Prefer broadly recognizable concepts (e.g. "grief", "loss of faith", \
"fleeting beauty") over obscure or poem-specific wording. Another reader given \
your terms should recognize the poem's territory.
- Each term is 1-3 words, lowercase, no trailing punctuation.
- Do NOT include: the poet's name, the poem's title, proper nouns specific to \
this poem, or form/style labels ("sonnet", "rhyme", "stanza").
- Order terms from most to least central. Give just 2 for a simple poem; up to \
5 for a thematically rich one.
- Set confidence to "low" when the poem is highly abstract and its theme had to \
be inferred rather than read off the text, so it can be reviewed later; \
otherwise "high"."""

USER_TEMPLATE = """\
Title: {title}

Poem:
{text}"""

# JSON schema for structured output. OpenRouter enforces this for models that
# support it (Anthropic models included), so we get back valid JSON.
RESPONSE_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'poem_keywords',
        'strict': True,
        'schema': {
            'type': 'object',
            'additionalProperties': False,
            'properties': {
                # NOTE: Anthropic's structured-output backend rejects
                # minItems/maxItems > 1, so the 2-5 count is enforced via the
                # prompt and clamped in code rather than in the schema.
                'keywords': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
                'confidence': {
                    'type': 'string',
                    'enum': ['high', 'low'],
                },
            },
            'required': ['keywords', 'confidence'],
        },
    },
}


def build_messages(poem: dict) -> list[dict]:
    """Build the chat messages for one poem (author deliberately omitted)."""
    user = USER_TEMPLATE.format(
        title=poem.get('title', '').strip(),
        text=poem.get('text', '').strip(),
    )
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user},
    ]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def poem_key(poem: dict) -> tuple[str, str]:
    """Stable identity for a poem, used to skip already-processed records."""
    return (poem.get('title', '').strip(), poem.get('author', '').strip())


def load_poems(path: Path) -> list[dict]:
    poems = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                poems.append(json.loads(line))
    return poems


def load_done_keys(path: Path) -> set[tuple[str, str]]:
    """Read keys already present in the output file (for resuming)."""
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(poem_key(rec))
    return done


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_one(
    client: OpenAI,
    model: str,
    poem: dict,
    *,
    max_retries: int = 5,
) -> dict:
    """Call the LLM for one poem and return the original record + keywords."""
    messages = build_messages(poem)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=RESPONSE_FORMAT,
                temperature=0.2,
            )
            content = resp.choices[0].message.content or '{}'
            data = json.loads(content)
            keywords = [str(k).strip().lower() for k in data.get('keywords', [])]
            keywords = [k for k in keywords if k][:5]  # prompt asks for 2-5
            return {
                **poem,
                'keywords': keywords,
                'confidence': data.get('confidence', 'high'),
                'model': model,
            }
        except Exception as e:  # network, rate limit, malformed JSON, etc.
            last_err = e
            # Exponential backoff: 1s, 2s, 4s, 8s, ...
            time.sleep(2**attempt)
    raise RuntimeError(f'failed after {max_retries} attempts for {poem.get("title")!r}: {last_err}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model',
        default=os.environ.get('OPENROUTER_MODEL', DEFAULT_MODEL),
        help=f'OpenRouter model slug (default: {DEFAULT_MODEL})',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='process at most N (not-yet-done) poems; useful for sampling',
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=4,
        help='number of concurrent requests (default: 4)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='print the prompt for the first poem and exit (no API calls)',
    )
    args = parser.parse_args()

    poems = load_poems(INPUT)

    if args.dry_run:
        msgs = build_messages(poems[0])
        print('=== SYSTEM ===\n' + msgs[0]['content'])
        print('\n=== USER ===\n' + msgs[1]['content'])
        print(f'\n=== MODEL ===\n{args.model}')
        print('\n(dry run -- no API calls made)')
        return

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        sys.exit('error: set OPENROUTER_API_KEY in the environment')

    done = load_done_keys(OUTPUT)
    # Deduplicate input: some poems appear multiple times with the same
    # title+author; process each unique poem only once.
    seen: set[tuple[str, str]] = set()
    unique_poems: list[dict] = []
    for p in poems:
        k = poem_key(p)
        if k not in seen:
            seen.add(k)
            unique_poems.append(p)
    todo = [p for p in unique_poems if poem_key(p) not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(
        f'{len(poems)} poems total, {len(done)} already done, '
        f'processing {len(todo)} now (model={args.model}, '
        f'concurrency={args.concurrency})'
    )
    if not todo:
        return

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    # Append results as they complete so progress survives interruption.
    # A lock serialises writes to the single output handle across threads.
    write_lock = threading.Lock()
    done_count = 0
    fail_count = 0

    with OUTPUT.open('a', encoding='utf-8') as out, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(extract_one, client, args.model, p): p for p in todo}
        for fut in as_completed(futures):
            poem = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:
                fail_count += 1
                print(f'  FAIL  {poem.get("title")!r}: {e}', file=sys.stderr)
                continue
            with write_lock:
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out.flush()
            done_count += 1
            if done_count % 25 == 0 or done_count == len(todo):
                print(f'  ...{done_count}/{len(todo)} done')

    print(f'finished: {done_count} written, {fail_count} failed -> {OUTPUT}')


if __name__ == '__main__':
    main()
