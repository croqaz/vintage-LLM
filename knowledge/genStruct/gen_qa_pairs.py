#!/usr/bin/env python3
"""
Generate Q&A dialogue pairs from JSON shard files by calling an LLM API.

Uses the OpenAI Python library with async concurrency (default 4 parallel
requests).  Supports multiple models and multiple prompt files — each shard is
processed for every (model × prompt) combination.

Supports:
  - OpenRouter API  (https://openrouter.ai/api/v1/chat/completions)
  - OpenAI-compatible local endpoints (Ollama, vLLM, llama.cpp server, etc.)

Usage:
    python gen_qa_pairs.py shards.json [shards2.json ...] [options]

Options:
    -o, --output FILE          Output JSON-lines file (default: <first_input>_qa.jsonl)
    -m, --model MODELS         Model name(s), separated by comma / semicolon / space
                               (default: google/gemma-4-31b-it)
    -e, --endpoint URL         Chat completions endpoint
                               (default: https://openrouter.ai/api/v1/chat/completions)
    -k, --api-key KEY          API key.  If not given, reads OPENROUTER_API_KEY env var.
    --max-tokens N             Max tokens in response (default: 4096)
    --temperature FLOAT        Sampling temperature (default: 0.7)
    --timeout N                Request timeout in seconds (default: 180)
    --retries N                Retries on failure (default: 3)
    --concurrency N            Max parallel API requests (default: 4)
    --delay FLOAT              Delay (seconds) after each API call (default: 0.5)
    --prompt FILES             System-prompt file(s), separated by comma / semicolon.
                               E.g. _prompt_facts.txt,_prompt_prose.txt  (REQUIRED)
    --book-title STR           Fills {title} in the prompt (used by prose/verse/scripture templates)
    --book-author STR          Fills {author} in the prompt (used by prose/verse/scripture templates)
    --start-at ID              Resume from a shard id (inclusive)
    -h, --help                 Show this help

Output format (one JSON object per line):
    {"shard_id": int, "window": str, "source_file": str,
     "messages": [{role, content}, ...],
     "claims_to_evidence": {claim: excerpt, ...},
     "usage": {...},
     "model": str, "prompt_file": str}
"""

import argparse
import asyncio
import json
import os
import re
import sys

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# System prompts now live in external files (one per genre), named with a
# leading "_prompt_" so they sort apart from the actual book .txt files:
#   _prompt_facts.txt      expository / factual prose (the original prompt)
#   _prompt_prose.txt      prose fiction (novels, novellas)
#   _prompt_verse.txt      poetry / verse
#   _prompt_scripture.txt  scripture
# Pass one or more with --prompt (required).  Prompts may contain the literal
# tokens {title} and {author}; they are filled from --book-title/--book-author.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Optional prompt fragment: let the student occasionally be wrong so the
# teacher can correct them.  Enabled with --allow-mistakes.
# ---------------------------------------------------------------------------
STUDENT_MISTAKE_PROMPT = """
OCCASIONAL STUDENT MISTAKES:
In roughly one dialogue out of four, and at most once per dialogue, let the student
make a genuine, plausible factual error in a question — a misattributed author,
a confused date, a muddled definition, a wrong title — and have the teacher gently
and clearly correct it before going on to answer the real substance.
- The correction must be grounded in the facts of the passage (no invented details).
- Keep the teacher courteous, never mocking ("Oh, you must be mixing the authors
  there — Hamlet is Shakespeare's, the English dramatist; La Fontaine was the French
  fabulist. Setting that right, ...").
- The mistake must be the student's, never the teacher's, and the teacher's final
  answer must still be correct.
- Do not force a mistake into every dialogue; most should proceed without error.""".strip()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_list_arg(s, lower=False):
    """Split a string on commas, semicolons, or spaces.

    Returns a list of non-empty trimmed items.  If *lower* is True each item
    is lower-cased (useful for model ids).
    """
    items = [x.strip() for x in re.split(r'[,; ]', s) if x.strip()]
    if lower:
        items = [x.lower() for x in items]
    return items


def load_shards(*paths):
    """Yield (source_file, chunk_file, shard_dict) tuples from one or more
    shard JSON files.  chunk_file is the basename of the input chunk file, so
    shard ids from different chunk files can be told apart."""
    for path in paths:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        src = data['source_file']
        chunk_file = os.path.basename(path)
        for shard in data['shards']:
            yield src, chunk_file, shard


def load_completed_ids(output_path):
    """Return set of (chunk_file, shard_id, model, prompt_file) tuples already
    present in the output JSON-lines file.  chunk_file is included so that
    multiple input chunk files with overlapping shard ids do not skip each
    other's work.  Old-format records without 'chunk_file' fall back to
    'source_file' so pre-existing single-chunk outputs still resume.
    Old-format records without 'prompt_file' are skipped so they get
    re-generated."""
    ids = set()
    if not os.path.exists(output_path):
        return ids
    with open(output_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only count a record as done if it has real messages and the new
            # required fields — skip failed placeholders and old-format rows.
            if rec.get('messages') and 'raw_output' not in rec and 'prompt_file' in rec:
                chunk_id = rec.get('chunk_file') or rec.get('source_file')
                ids.add((chunk_id, rec.get('shard_id'), rec.get('model'), rec.get('prompt_file')))
    return ids


def extract_json(text):
    """Try to pull a JSON object out of a (possibly markdown-wrapped) string."""
    text = text.strip()
    # Strip ```json ... ``` fences
    if text.startswith('```'):
        # find first newline
        nl = text.find('\n')
        if nl != -1:
            text = text[nl + 1 :]
        text = text.rstrip()
        if text.endswith('```'):
            text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: model emitted prose around the object. Grab the outermost
        # brace span and try that.
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


# Speaker labels recognised by the plain-text fallback parser.  Keys are
# lower-cased label prefixes; values are the message role to assign.
PLAIN_SPEAKER_ROLES = {
    'student': 'student',
    'reader': 'student',
    'user': 'student',
    'question': 'student',
    'teacher': 'teacher',
    'tutor': 'teacher',
    'assistant': 'teacher',
    'professor': 'teacher',
    'answer': 'teacher',
}


# Matches a speaker line such as "Student: ...", "Teacher: ..." or
# "Assistant (Teacher): ...".  The label may carry a qualifier in parentheses;
# the body may sit on the same line or on the lines that follow.  Bold
# markers are normalised away before matching (see parse_plain_dialogue).
PLAIN_LABEL_RE = re.compile(r'^\s*(?:\*{1,3}\s*)?(?P<label>[A-Za-z][A-Za-z ()\-]{0,40}?)\s*:\s*(?P<body>.*)$')


# Collapses bold speaker labels to plain "Label:" so the label regex never
# leaves stray asterisks in the captured body.  Handles both markdown
# conventions seen in the wild: "**Label:**" (stars wrap "Label:", as in
# "**User:** ...") and "**Label:** ..." with the stars before the colon, plus
# unclosed "**Label:".
PLAIN_BOLD_RE = re.compile(
    r'\*{1,3}([A-Za-z][A-Za-z ()\-]{0,40}?)\s*:\s*\*{1,3}'
    r'|\*{1,3}([A-Za-z][A-Za-z ()\-]{0,40}?)\s*:'
    r'|([A-Za-z][A-Za-z ()\-]{0,40}?)\s*:\s*\*{1,3}'
)


def _debold_label(line):
    """Replace bold speaker labels in *line* with plain ones."""
    return PLAIN_BOLD_RE.sub(lambda m: (m.group(1) or m.group(2) or m.group(3)) + ':', line)


def parse_plain_dialogue(text):
    """Best-effort recovery of a student/teacher dialogue from plain prose.

    Some models ignore the JSON schema and return markdown-formatted dialogue
    like::

        **Dialogue:**

        **Student:** What makes a person a poet?

        **Assistant (Teacher):** Ah, a most intriguing question! ...

    This finds speaker-label lines, groups the following lines into each
    speaker's turn, and returns a list of ``{'role', 'content'}`` messages.
    Returns None unless a plausible dialogue was recovered (>= 2 turns with
    at least one student and one teacher) — so a stray single-turn answer is
    not silently accepted.
    """
    turns = []  # [role, [body lines]]
    for line in text.splitlines():
        line = _debold_label(line)
        m = PLAIN_LABEL_RE.match(line)
        if m:
            label = m.group('label').strip().lower()
            role = next((r for prefix, r in PLAIN_SPEAKER_ROLES.items() if label.startswith(prefix)), None)
            if role is not None:
                turns.append([role, [m.group('body').strip()]])
                continue
        if turns:
            turns[-1][1].append(line.strip())

    messages = []
    for role, lines in turns:
        content = '\n'.join(l for l in lines if l).strip()
        # Drop markdown code fences and bold markers left in the body.
        content = re.sub(r'(?m)^[`~]{3,}.*$', '', content)
        content = re.sub(r'\*\*(.+?)\*\*', r'\1', content).strip()
        if content:
            messages.append({'role': role, 'content': content})

    has_student = any(m['role'] == 'student' for m in messages)
    has_teacher = any(m['role'] == 'teacher' for m in messages)
    if len(messages) >= 2 and has_student and has_teacher:
        return messages
    return None


def _norm_ws(s):
    """Collapse runs of whitespace so verbatim matching ignores OCR line breaks."""
    return re.sub(r'\s+', ' ', s).strip()


def find_unverified_claims(claims_to_evidence, source_text):
    """Return the claim keys whose evidence excerpt is NOT a verbatim
    (whitespace-normalised) substring of the source passage."""
    nsrc = _norm_ws(source_text)
    unverified = []
    for claim, ev in (claims_to_evidence or {}).items():
        if not isinstance(ev, str) or _norm_ws(ev) not in nsrc:
            unverified.append(claim)
    return unverified


def append_jsonl(path, record, lock=None):
    """Append a single JSON record as one line (thread/async-safe with lock)."""
    line = json.dumps(record, ensure_ascii=False) + '\n'
    # If no lock, just write (safe for single-thread)
    if lock is None:
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(line)
        return
    # Use synchronous lock for asyncio (lock is an asyncio.Lock, but we
    # write with sync I/O via run_in_executor in the caller).
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(line)


# ---------------------------------------------------------------------------
# Async API helpers
# ---------------------------------------------------------------------------


async def call_api_async(client, model, system_prompt, shard_text, max_tokens, temperature, timeout):
    """Call the chat-completions API asynchronously.  Returns the parsed
    JSON content string and usage dict, or raises."""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': shard_text},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={'type': 'json_object'},
        timeout=timeout,
    )
    choice = response.choices[0]
    raw_content = choice.message.content
    finish = choice.finish_reason
    usage = response.usage.model_dump() if response.usage else {}
    return raw_content, finish, usage


async def process_one(
    client,
    model,
    prompt_file,
    system_prompt,
    shard,
    src,
    chunk_file,
    max_tokens,
    temperature,
    timeout,
    output_path,
    write_lock,
    semaphore,
    delay,
    stats,
):
    """Process one (shard × model × prompt) combination."""
    sid = shard['id']

    async with semaphore:
        try:
            raw_content, finish, usage = await call_api_async(
                client,
                model,
                system_prompt,
                shard['text'],
                max_tokens,
                temperature,
                timeout,
            )
        except Exception as exc:
            print(
                f'  ERROR shard={sid} model={model} prompt={prompt_file}: {exc}',
                file=sys.stderr,
            )
            stats['errors'] += 1
            return

        # --- Validate response -------------------------------------------------
        if not raw_content:
            print(
                f'  ERROR shard={sid} model={model}: empty content (finish_reason={finish})',
                file=sys.stderr,
            )
            stats['errors'] += 1
            return

        if finish == 'length':
            print(
                f'  WARNING shard={sid} model={model}: truncated at max_tokens — skipping (raise --max-tokens)',
                file=sys.stderr,
            )
            stats['skipped'] += 1
            return

        # --- Parse JSON --------------------------------------------------------
        parsed = None
        parse_error = None
        try:
            parsed = extract_json(raw_content)
        except (json.JSONDecodeError, KeyError) as exc:
            parse_error = exc  # handled below via plain-text recovery

        if parsed is None or not parsed.get('messages'):
            # JSON parse failed, or the model returned valid JSON with the
            # wrong shape — try to salvage the dialogue from the plain text
            # we already paid for instead of dropping it.
            recovered = parse_plain_dialogue(raw_content)
            if recovered:
                parsed = {
                    'messages': recovered,
                    'claims_to_evidence': {},
                    'recovered_from_plain': True,
                }
            elif parsed is None:
                print(
                    f'  WARNING shard={sid} model={model}: could not parse LLM output as JSON: {parse_error}',
                    file=sys.stderr,
                )
                print(
                    f'  Raw output (first 300 chars): {raw_content[:300]}',
                    file=sys.stderr,
                )
                parsed = {
                    'messages': [],
                    'claims_to_evidence': {},
                    'raw_output': raw_content,
                }

        # --- Validate claims ---------------------------------------------------
        claims = parsed.get('claims_to_evidence', {})
        unverified = find_unverified_claims(claims, shard['text'])

        # --- Assemble & write record -------------------------------------------
        record = {
            'shard_id': sid,
            'window': shard['window'],
            'source_file': src.split('.txt')[0],
            'chunk_file': chunk_file,
            'start_word': shard['start_word'],
            'end_word': shard['end_word'],
            'word_count': shard['word_count'],
            'messages': parsed.get('messages', []),
            'claims_to_evidence': claims,
            'unverified_claims': unverified,
            'usage': usage,
            'model': model,
            'prompt_file': prompt_file,
        }
        if 'raw_output' in parsed:
            record['raw_output'] = parsed['raw_output']
        if parsed.get('recovered_from_plain'):
            record['recovered_from_plain'] = True

        # Thread-safe append via executor
        async with write_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, append_jsonl, output_path, record)

        # --- Log ---------------------------------------------------------------
        if 'raw_output' in parsed:
            print(f'  shard={sid} model={model} prompt={prompt_file}: (saved raw — JSON parse failed)')
            stats['errors'] += 1
        elif parsed.get('recovered_from_plain'):
            n_msgs = len(parsed.get('messages', []))
            print(f'  shard={sid} model={model} prompt={prompt_file}: OK  ({n_msgs} messages — recovered from plain text)')
            stats['ok'] += 1
            stats['recovered'] += 1
        else:
            n_msgs = len(parsed.get('messages', []))
            n_claims = len(claims)
            warn = f'  [!] {len(unverified)} non-verbatim claims' if unverified else ''
            print(f'  shard={sid} model={model} prompt={prompt_file}: OK  ({n_msgs} messages, {n_claims} claims){warn}')
            stats['ok'] += 1

        # Optional per-call delay (rate limiting)
        if delay:
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async(args):
    # -------------------------------------------------------------------
    # Resolve output path
    # -------------------------------------------------------------------
    output_path = args.output
    if output_path is None:
        base = os.path.splitext(args.inputs[0])[0]
        output_path = base + '_qa.jsonl'

    # -------------------------------------------------------------------
    # Parse model list
    # -------------------------------------------------------------------
    models = split_list_arg(args.model, lower=True)
    if not models:
        print('ERROR: at least one model is required', file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------
    # Parse prompt file list
    # -------------------------------------------------------------------
    prompt_files = split_list_arg(args.prompt, lower=False)
    if not prompt_files:
        print('ERROR: at least one --prompt FILE is required', file=sys.stderr)
        sys.exit(1)

    # Validate prompt files exist
    for pf in prompt_files:
        if not os.path.exists(pf):
            print(f'ERROR: prompt file not found: {pf}', file=sys.stderr)
            sys.exit(1)

    # -------------------------------------------------------------------
    # Load system prompts (one per file)
    # -------------------------------------------------------------------
    system_prompts = {}  # prompt_file_basename → full text
    for pf in prompt_files:
        with open(pf, 'r', encoding='utf-8') as fh:
            text = fh.read().strip()

        # Fill {title}/{author} placeholders
        for token, value, flag in (
            ('{title}', args.book_title, '--book-title'),
            ('{author}', args.book_author, '--book-author'),
        ):
            if token in text:
                if value:
                    text = text.replace(token, value)
                else:
                    print(
                        f'  WARNING: prompt {pf} uses {token} but {flag} was not given — continuing with a generic fallback',
                        file=sys.stderr,
                    )
                    fallback = 'this work' if token == '{title}' else 'the author'
                    text = text.replace(token, fallback)

        if args.allow_mistakes:
            text = text + '\n\n' + STUDENT_MISTAKE_PROMPT

        # Use basename as the short identifier in output
        key = os.path.basename(pf)
        system_prompts[key] = text

    # -------------------------------------------------------------------
    # Build OpenAI client
    # -------------------------------------------------------------------
    # Derive base_url from endpoint: strip trailing /chat/completions if present
    endpoint = args.endpoint.rstrip('/')
    if endpoint.endswith('/chat/completions'):
        base_url = endpoint[: -len('/chat/completions')]
    else:
        base_url = endpoint

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=args.api_key,
        max_retries=args.retries,
        timeout=args.timeout,
    )

    # -------------------------------------------------------------------
    # Load completed shard×model×prompt combos
    # -------------------------------------------------------------------
    done_combos = load_completed_ids(output_path)
    if done_combos:
        print(f'Found {len(done_combos)} already-processed combinations in {output_path}')

    # -------------------------------------------------------------------
    # Enumerate shards to process
    # -------------------------------------------------------------------
    todo_shards = []
    for src, chunk_file, shard in load_shards(*args.inputs):
        sid = shard['id']
        if args.start_at is not None and sid < args.start_at:
            continue
        # Filter out combinations already done for this shard (keyed by
        # chunk_file too, so shard ids from different input files don't collide)
        remaining_combos = []
        for model in models:
            for prompt_key, prompt_text in system_prompts.items():
                if (chunk_file, sid, model, prompt_key) not in done_combos:
                    remaining_combos.append((model, prompt_key, prompt_text))
        if remaining_combos:
            todo_shards.append((src, chunk_file, shard, remaining_combos))

    if not todo_shards:
        print('All shards already processed — nothing to do.')
        return

    total_combos = sum(len(combos) for _, _, _, combos in todo_shards)
    print(
        f'Will process {len(todo_shards)} shards × up to '
        f'{len(models)} models × {len(system_prompts)} prompts '
        f'= up to {total_combos} API calls → {output_path}'
    )
    print(f'Endpoint:  {endpoint}')
    print(f'Models:    {", ".join(models)}')
    print(f'Prompts:   {", ".join(system_prompts.keys())}')
    print(f'Concurrency: {args.concurrency}')
    print()

    # -------------------------------------------------------------------
    # Create tasks for all remaining combinations
    # -------------------------------------------------------------------
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    stats = {'ok': 0, 'errors': 0, 'skipped': 0, 'recovered': 0}

    tasks = []
    for src, chunk_file, shard, combos in todo_shards:
        for model, prompt_key, prompt_text in combos:
            task = asyncio.create_task(
                process_one(
                    client=client,
                    model=model,
                    prompt_file=prompt_key,
                    system_prompt=prompt_text,
                    shard=shard,
                    src=src,
                    chunk_file=chunk_file,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    output_path=output_path,
                    write_lock=write_lock,
                    semaphore=semaphore,
                    delay=args.delay,
                    stats=stats,
                )
            )
            tasks.append(task)

    # -------------------------------------------------------------------
    # Run all tasks (semaphore limits actual concurrency)
    # -------------------------------------------------------------------
    await asyncio.gather(*tasks, return_exceptions=True)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    recovered_note = f'  ({stats["recovered"]} recovered from plain text)' if stats['recovered'] else ''
    print(f'\nDone.  OK={stats["ok"]}  errors={stats["errors"]}  skipped={stats["skipped"]}{recovered_note}')
    print(f'Results in {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate Q&A dialogue pairs from sharded pre-1900 text via LLM.',
    )
    parser.add_argument('inputs', nargs='+', help='One or more shard JSON files')
    parser.add_argument('-o', '--output', help='Output JSON-lines file')
    parser.add_argument(
        '-m',
        '--model',
        default='google/gemma-4-31b-it',
        help='Model name(s), separated by comma/semicolon/space (default: google/gemma-4-31b-it)',
    )
    parser.add_argument(
        '-e',
        '--endpoint',
        default='https://openrouter.ai/api/v1/chat/completions',
        help='Chat completions endpoint',
    )
    parser.add_argument(
        '-k',
        '--api-key',
        default=os.environ.get('OPENROUTER_API_KEY', ''),
        help='API key (reads OPENROUTER_API_KEY env var if omitted)',
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=4096,
        help='Max tokens in response (default: 4096)',
    )
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument(
        '--concurrency',
        type=int,
        default=4,
        help='Max parallel API requests (default: 4)',
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay in seconds after each API call (default: 0.5)',
    )
    parser.add_argument(
        '--prompt',
        required=True,
        help='System-prompt file(s), separated by comma/semicolon. E.g. _prompt_facts.txt,_prompt_prose.txt (required)',
    )
    parser.add_argument(
        '--book-title',
        default=None,
        help='Book title; fills {title} in the prompt (e.g. prose/verse templates)',
    )
    parser.add_argument(
        '--book-author',
        default=None,
        help='Book author; fills {author} in the prompt (e.g. prose/verse templates)',
    )
    parser.add_argument(
        '--allow-mistakes',
        action='store_true',
        help='Occasionally let the student make a factual error for the teacher to correct',
    )
    parser.add_argument(
        '--start-at',
        type=int,
        default=None,
        help='Skip shards with id < this value',
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
