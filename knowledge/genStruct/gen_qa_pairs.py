#!/usr/bin/env python3
"""
Generate Q&A dialogue pairs from JSON shard files by calling an LLM API.

Supports:
  - OpenRouter API  (https://openrouter.ai/api/v1/chat/completions)
  - OpenAI-compatible local endpoints (Ollama, vLLM, llama.cpp server, etc.)

Usage:
    python gen_qa_pairs.py shards.json [shards2.json ...] [options]

Options:
    -o, --output FILE          Output JSON-lines file (default: <first_input>_qa.jsonl)
    -m, --model ID             Model name (default: google/gemma-4-31b-it)
    -e, --endpoint URL         Chat completions endpoint
                               (default: https://openrouter.ai/api/v1/chat/completions)
    -k, --api-key KEY          API key.  If not given, reads OPENROUTER_API_KEY env var.
    --max-tokens N             Max tokens in response (default: 2048)
    --temperature FLOAT        Sampling temperature (default: 0.7)
    --timeout N                Request timeout in seconds (default: 180)
    --retries N                Retries on failure (default: 3)
    --delay FLOAT              Delay between requests in seconds (default: 1.0)
    --prompt-file FILE         Path to a custom system-prompt file (overrides built-in)
    --start-at ID              Resume from a shard id (inclusive)
    -h, --help                 Show this help

Output format (one JSON object per line):
    {"shard_id": int, "window": str, "source_file": str,
     "messages": [{role, content}, ...],
     "claims_to_evidence": {claim: excerpt, ...},
     "usage": {...},
     "model": str}
"""

import argparse
import json
import os
import re
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """
You are creating training data for a language model whose knowledge stops at year 1900.
Use the supplied passage ONLY as your source of subject matter and facts. From it,
produce a multi-turn dialogue (2-5 turns) between a curious student and a knowledgeable
teacher having a general discussion about that subject.
The student should ask questions that arise naturally from the subject matter, and the
teacher should give informative answers grounded in the facts of the passage.
Begin each dialog by addressing the substance of the question directly. Do not
open with an exclamation, or flattery like: "Pray, sir,", "Ah, a most pertinent question",
and the like; reserve such flourishes for the rare moment that genuinely calls for one.

CRITICAL — the dialogue must stand entirely on its own, as if no passage existed:
- The speakers CANNOT see the supplied passage and must never refer to it. The passage
  is invisible context for you, not a shared document the speakers are looking at.
- FORBIDDEN: any reference to the source as a thing ("this guide", "this passage",
  "this text", "this analysis", "the chapter", "the excerpt", "the author says",
  "as the passage states", "the writer here", and the like). A reader who never saw
  the passage would be confused by such phrases — that is the test.
- ALLOWED: referring to specific, named external works or people that the subject is
  about, as a well-read person naturally would — e.g. "I was reading a play by
  Shakespeare and he wrote...", "I have been studying Mr. Webster's oration on...".
  Name the actual work or author, never the source passage itself.
- The conversation must start fresh, from scratch, with no invisible prior text to
  point back to. Open with a genuine question about the topic, not a reaction to
  something just read in "the text".

RULES:
- All factual claims must come from the supplied text exclusively.
- Speak as a person living in 1850s: clear, witty, courteous Victorian English;
  NOT fake-medieval ("thee/thou/forsooth" are forbidden).
- The student must refer back to earlier turns so later answers depend on them.
- No modern slang, no bulleted lists structure, no modern moral framing.
- No events, inventions, works, or people after 1900
  (e.g., no airplanes, no TV, no computers, no WWI/WWII).

Output valid JSON only, with this exact structure:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "claims_to_evidence": {
    "claim text": "exact excerpt from source",
    ...
  }
}""".strip()

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


def load_shards(*paths):
    """Yield (source_file, shard_dict) tuples from one or more shard JSON files."""
    for path in paths:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        src = data['source_file']
        for shard in data['shards']:
            yield src, shard


def load_completed_ids(output_path):
    """Return set of shard IDs already present in the output JSON-lines file."""
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
            # Only count a shard as done if it parsed into real messages.
            # Skip failed/empty records (and old raw_output fallbacks) so a
            # re-run retries them instead of treating them as complete.
            if rec.get('messages') and 'raw_output' not in rec:
                ids.add(rec.get('shard_id'))
    return ids


def build_payload(model, system_prompt, shard_text, max_tokens, temperature):
    """Return the JSON-serialisable request body for a chat completions API."""
    return {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': shard_text},
        ],
        'max_tokens': max_tokens,
        'temperature': temperature,
        'response_format': {'type': 'json_object'},
    }


def call_api(endpoint, api_key, payload, timeout, retries):
    """POST to the chat completions endpoint.  Returns the parsed JSON body or raises."""
    headers = {
        'Content-Type': 'application/json',
    }
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            wait = 2**attempt
            print(f'  [attempt {attempt}/{retries}] {exc} — retrying in {wait}s', file=sys.stderr)
            time.sleep(wait)

    raise last_exc


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


def append_jsonl(path, record):
    """Append a single JSON record as one line."""
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='Generate Q&A dialogue pairs from sharded pre-1900 text via LLM.')
    parser.add_argument('inputs', nargs='+', help='One or more shard JSON files')
    parser.add_argument('-o', '--output', help='Output JSON-lines file')
    parser.add_argument('-m', '--model', default='google/gemma-4-31b-it', help='Model name (default: google/gemma-4-31b-it)')
    parser.add_argument('-e', '--endpoint', default='https://openrouter.ai/api/v1/chat/completions', help='Chat completions endpoint')
    parser.add_argument(
        '-k', '--api-key', default=os.environ.get('OPENROUTER_API_KEY', ''), help='API key (reads OPENROUTER_API_KEY env var if omitted)'
    )
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--delay', type=float, default=1.0, help='Seconds between API calls (default: 1.0)')
    parser.add_argument('--prompt-file', help='Custom system prompt file (overrides built-in)')
    parser.add_argument(
        '--allow-mistakes',
        action='store_true',
        help='Occasionally let the student make a factual error for the teacher to correct',
    )
    parser.add_argument('--start-at', type=int, default=None, help='Skip shards with id < this value')
    args = parser.parse_args()

    # Resolve output path
    output_path = args.output
    if output_path is None:
        base = os.path.splitext(args.inputs[0])[0]
        output_path = base + '_qa.jsonl'

    # Load system prompt
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as fh:
            system_prompt = fh.read().strip()
    if args.allow_mistakes:
        system_prompt = system_prompt + '\n\n' + STUDENT_MISTAKE_PROMPT

    # Load which shards are already done
    done_ids = load_completed_ids(output_path)
    if done_ids:
        print(f'Found {len(done_ids)} already-processed shard IDs in {output_path}')

    # Enumerate shards to process
    todo = []
    for src, shard in load_shards(*args.inputs):
        sid = shard['id']
        if sid in done_ids:
            continue
        if args.start_at is not None and sid < args.start_at:
            continue
        todo.append((src, shard))

    if not todo:
        print('All shards already processed — nothing to do.')
        return

    print(f'Will process {len(todo)} shards → {output_path}')
    print(f'Endpoint: {args.endpoint}')
    print(f'Model:    {args.model}')
    print()

    for idx, (src, shard) in enumerate(todo, start=1):
        sid = shard['id']
        print(f'[{idx}/{len(todo)}] shard {sid}  ({shard["window"]}, {shard["word_count"]} words, src={src})')

        payload = build_payload(args.model, system_prompt, shard['text'], args.max_tokens, args.temperature)

        try:
            body = call_api(args.endpoint, args.api_key, payload, args.timeout, args.retries)
        except Exception as exc:
            print(f'  ERROR: {exc}', file=sys.stderr)
            continue

        # Validate the response envelope before touching it.  OpenRouter can
        # return HTTP 200 with an {"error": ...} body and no choices, or a
        # choice whose content is null — both would otherwise crash the run.
        choices = body.get('choices') or []
        if not choices:
            print(f'  ERROR: no choices in response: {body.get("error", body)}', file=sys.stderr)
            continue
        choice = choices[0]
        raw_content = (choice.get('message') or {}).get('content')
        finish = choice.get('finish_reason')

        if not raw_content:
            print(f'  ERROR: empty content (finish_reason={finish})', file=sys.stderr)
            continue

        if finish == 'length':
            # Truncated → the JSON is almost certainly invalid.  Skip without
            # saving so a re-run (with a larger --max-tokens) retries it.
            print('  WARNING: response truncated at max_tokens — skipping (raise --max-tokens)', file=sys.stderr)
            continue

        try:
            parsed = extract_json(raw_content)
        except (json.JSONDecodeError, KeyError) as exc:
            print(f'  WARNING: could not parse LLM output as JSON: {exc}', file=sys.stderr)
            print(f'  Raw output (first 300 chars): {raw_content[:300]}', file=sys.stderr)
            # Save the raw content so nothing is lost; the empty messages mean
            # load_completed_ids will retry this shard on the next run.
            parsed = {'messages': [], 'claims_to_evidence': {}, 'raw_output': raw_content}

        usage = body.get('usage', {})
        claims = parsed.get('claims_to_evidence', {})
        unverified = find_unverified_claims(claims, shard['text'])

        record = {
            'shard_id': sid,
            'window': shard['window'],
            'source_file': src.split('.txt')[0],
            'start_word': shard['start_word'],
            'end_word': shard['end_word'],
            'word_count': shard['word_count'],
            'messages': parsed.get('messages', []),
            'claims_to_evidence': claims,
            'unverified_claims': unverified,
            'usage': usage,
            'model': args.model.split('/')[-1],
        }
        if 'raw_output' in parsed:
            record['raw_output'] = parsed['raw_output']

        append_jsonl(output_path, record)
        done_ids.add(sid)

        if 'raw_output' in parsed:
            print('  (saved raw — JSON parse failed)')
        else:
            n_msgs = len(parsed.get('messages', []))
            n_claims = len(claims)
            warn = f'  [!] {len(unverified)} non-verbatim claims' if unverified else ''
            print(f'  OK  ({n_msgs} messages, {n_claims} claims){warn}')

        # Rate-limit
        if args.delay:
            time.sleep(args.delay)

    print(f'\nDone.  Results in {output_path}')


if __name__ == '__main__':
    main()
