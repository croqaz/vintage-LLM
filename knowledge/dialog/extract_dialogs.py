#!/usr/bin/env python3
"""
Extract character dialogue from a chunked Gutenberg book using the OpenRouter API.

For one book (a directory of ``*_chunk_NNNN.txt`` files under ``gutenberg_chunks/``)
every chunk is sent to an LLM that returns the spoken lines as JSONL, one object
per quoted utterance:

    {"Napoleon": "Who is this good man who is staring at me?"}
    {"M. Myriel": "Sire,"}

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python3 extract_dialogs.py pg135-clean --model openai/gpt-4o-mini
    python3 extract_dialogs.py gutenberg_chunks/pg135-clean -m anthropic/claude-3.5-haiku

Output goes to ``dialogs/<book>/`` (one JSONL per chunk) plus a combined
``dialogs/<book>/<book>_dialogs.jsonl``. Re-running skips chunks already done
unless ``--overwrite`` is given, so an interrupted run resumes cheaply.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
OPENROUTER_MODELS_URL = 'https://openrouter.ai/api/v1/models'

# --- The prompt -------------------------------------------------------------
# Kept short and explicit: dumber/cheaper models need unambiguous rules and a
# couple of worked examples more than they need prose.

SYSTEM_PROMPT = """\
You extract spoken dialogue from a passage of a novel.

Output rules:
- Output JSONL: one JSON object per line, nothing else (no prose, no code fences, no array).
- Each object is {"SPEAKER": "SPOKEN TEXT"} where SPEAKER is the character's name.
- Emit one object per quoted speech segment, in the order it appears.
- If a single quote is split by a speech tag (e.g. "Sire," said X, "you ..."),
  output one object for each quoted segment, both attributed to the same speaker.
- Include ONLY the words actually inside quotation marks. Never include narration.
- Identify the speaker from the surrounding text. If genuinely unknown, use "Unknown".
- Use the name the narrative uses for the speaker (e.g. "M. Myriel", "Dorothea").
- If the passage contains no dialogue, output nothing.

Example input:
“I think she is,” said Celia, feeling afraid lest she should say
something that would not please her sister, and blushing as prettily as
possible above her necklace. “She likes giving up.”

Example output:
{"Celia": "I think she is,"}
{"Celia": "She likes giving up."}

Example input:
Napoleon turned round and said abruptly:—
"Who is this good man who is staring at me?"
"Sire," said M. Myriel, "you are looking at a good man, and I at a great man."

Example output:
{"Napoleon": "Who is this good man who is staring at me?"}
{"M. Myriel": "Sire,"}
{"M. Myriel": "you are looking at a good man, and I at a great man."}"""

USER_TEMPLATE = 'Extract the dialogue from this passage:\n\n{passage}'


def resolve_book_dir(book: str) -> Path:
    """Accept a book id ('pg135-clean'), a bare id ('pg135'), or a path."""
    p = Path(book)
    if p.is_dir():
        return p
    base = Path('gutenberg_chunks')
    for cand in (base / book, base / f'{book}-clean', Path(f'{book}-clean')):
        if cand.is_dir():
            return cand
    sys.exit(f'error: could not find a chunk directory for {book!r}')


def find_chunks(book_dir: Path) -> list[Path]:
    chunks = sorted(book_dir.glob('*_chunk_*.txt'))
    if not chunks:
        sys.exit(f"error: no '*_chunk_*.txt' files found in {book_dir}")
    return chunks


def supported_effort(api_key, model, effort, *, timeout=30.0):
    """Return ``effort`` if this model accepts it as a reasoning effort, else None.

    Per the OpenRouter docs, GET /api/v1/models exposes a per-model ``reasoning``
    object. ``supported_efforts`` lists the accepted levels (null means all gateway
    values are accepted); a missing ``reasoning`` object means the model has no
    effort selection, so we send nothing.
    """
    headers = {'Authorization': f'Bearer {api_key}'}
    try:
        resp = requests.get(OPENROUTER_MODELS_URL, headers=headers, timeout=timeout)
        resp.raise_for_status()
        models = resp.json().get('data', [])
    except (requests.RequestException, ValueError) as exc:
        print(f'warning: could not fetch model list ({exc}); sending request without reasoning effort', file=sys.stderr)
        return None

    entry = next((m for m in models if m.get('id') == model), None)
    if entry is None:
        print(f'warning: model {model!r} not found in model list; sending request without reasoning effort', file=sys.stderr)
        return None

    reasoning = entry.get('reasoning')
    if not isinstance(reasoning, dict):
        print(f'note: {model} does not expose reasoning effort; not sending it.')
        return None

    efforts = reasoning.get('supported_efforts')
    # null/absent supported_efforts => all gateway effort values are accepted.
    if efforts is None or effort in efforts:
        return effort

    print(f'note: {model} supports efforts {efforts}, not {effort!r}; not sending it.')
    return None


def call_openrouter(api_key, model, passage, *, max_tokens, temperature, timeout, max_retries, reasoning_effort=None):
    """Call OpenRouter and return the raw assistant text, retrying transient errors."""
    payload = {
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': USER_TEMPLATE.format(passage=passage)},
        ],
    }
    if reasoning_effort:
        payload['reasoning'] = {'effort': reasoning_effort}
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            wait = min(2**attempt, 30)
            print(f'    network error ({exc}); retry {attempt}/{max_retries} in {wait}s', file=sys.stderr)
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            data = resp.json()
            try:
                return data['choices'][0]['message']['content']
            except (KeyError, IndexError):
                raise RuntimeError(f'unexpected API response: {data}')

        # 429 = rate limit, 5xx = server hiccup -> back off and retry.
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = min(2**attempt, 30)
            print(f'    HTTP {resp.status_code}; retry {attempt}/{max_retries} in {wait}s', file=sys.stderr)
            time.sleep(wait)
            continue

        raise RuntimeError(f'OpenRouter HTTP {resp.status_code}: {resp.text[:500]}')

    raise RuntimeError(f'giving up after {max_retries} retries')


def parse_dialogs(raw: str) -> list[dict]:
    """Leniently turn the model's reply into a list of {speaker: text} dicts."""
    if not raw:
        return []

    # Strip ``` / ```json fences if the model added them anyway.
    text = re.sub(r'^```(?:json|jsonl)?\s*|\s*```$', '', raw.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()

    objects: list = []
    # Best case: the whole reply is valid JSON (array or single object).
    try:
        parsed = json.loads(text)
        objects = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        # Otherwise treat it as JSONL, and as a last resort scrape {...} spans.
        for line in text.splitlines():
            line = line.strip().rstrip(',')
            if not line:
                continue
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError:
                for m in re.finditer(r'\{[^{}]*\}', line):
                    try:
                        objects.append(json.loads(m.group()))
                    except json.JSONDecodeError:
                        pass

    out = []
    for obj in objects:
        if not isinstance(obj, dict) or not obj:
            continue
        # Accept the requested {Name: text} shape, or a {speaker, text} shape.
        if {'speaker', 'text'} <= obj.keys():
            speaker, dialog = obj.get('speaker'), obj.get('text')
        elif len(obj) == 1:
            ((speaker, dialog),) = obj.items()
        else:
            continue
        speaker = str(speaker).strip() if speaker is not None else ''
        dialog = str(dialog).strip() if dialog is not None else ''
        if speaker and dialog:
            out.append({speaker: dialog})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('book', help='book id (e.g. pg135-clean) or path to its chunk dir')
    ap.add_argument('-m', '--model', required=True, help='OpenRouter model id, e.g. openai/gpt-4o-mini')
    ap.add_argument('-o', '--output-dir', default='dialogs', help='root output directory (default: dialogs)')
    ap.add_argument('--max-tokens', type=int, default=16000, help='max completion tokens per chunk (default: 16000)')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--timeout', type=float, default=300.0, help='per-request timeout in seconds (default: 300)')
    ap.add_argument('--max-retries', type=int, default=5)
    ap.add_argument(
        '--reasoning-effort',
        default='medium',
        choices=['minimal', 'low', 'medium', 'high', 'xhigh', 'off'],
        help="reasoning effort to request when the model supports it (default: medium; 'off' to never send it)",
    )
    ap.add_argument('--overwrite', action='store_true', help='re-process chunks even if their output already exists')
    ap.add_argument('--dry-run', action='store_true', help='list chunks and approximate sizes without calling the API')
    args = ap.parse_args()

    book_dir = resolve_book_dir(args.book)
    book_name = book_dir.name
    chunks = find_chunks(book_dir)

    out_dir = Path(args.output_dir) / book_name
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / f'{book_name}_dialogs.jsonl'

    print(f'book: {book_name}  ({len(chunks)} chunks)  ->  {out_dir}')

    if args.dry_run:
        for c in chunks:
            words = len(c.read_text(encoding='utf-8', errors='replace').split())
            print(f'  {c.name}: ~{words:,} words (~{int(words * 1.3):,} tokens)')
        return

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        sys.exit('error: OPENROUTER_API_KEY is not set')

    reasoning_effort = None
    if args.reasoning_effort != 'off':
        reasoning_effort = supported_effort(api_key, args.model, args.reasoning_effort)
        if reasoning_effort:
            print(f'reasoning effort: {reasoning_effort}')

    all_entries: list[dict] = []
    total = 0
    for i, chunk in enumerate(chunks, 1):
        per_chunk_path = out_dir / f'{chunk.stem}.jsonl'
        if per_chunk_path.exists() and not args.overwrite:
            entries = [json.loads(l) for l in per_chunk_path.read_text(encoding='utf-8').splitlines() if l.strip()]
            print(f'[{i}/{len(chunks)}] {chunk.name}: skipped (cached, {len(entries)} lines)')
            all_entries.extend(entries)
            total += len(entries)
            continue

        passage = chunk.read_text(encoding='utf-8', errors='replace')
        print(f'[{i}/{len(chunks)}] {chunk.name}: ~{len(passage.split()):,} words ... ', end='', flush=True)

        raw = call_openrouter(
            api_key,
            args.model,
            passage,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            reasoning_effort=reasoning_effort,
        )
        entries = parse_dialogs(raw)

        with per_chunk_path.open('w', encoding='utf-8') as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
        print(f'{len(entries)} dialogue lines')
        all_entries.extend(entries)
        total += len(entries)

    with combined_path.open('w', encoding='utf-8') as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    print(f'\ndone: {total} dialogue lines -> {combined_path}')


if __name__ == '__main__':
    main()
