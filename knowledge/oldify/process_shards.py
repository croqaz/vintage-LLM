#!/usr/bin/env python3
"""
LLM Shard Processor
Reads a shard JSON file, sends each shard's text to a local llama.cpp
API with a configurable prompt, and writes a JSON-lines output file
containing the original shard data and the LLM response.

Usage:
    python3 process_shards.py <input_shards.json> [--prompt PROMPT] [--output OUT.jsonl]
                             [--api URL] [--model MODEL] [--timeout SEC]
                             [--start N] [--limit N]

Example:
    python3 process_shards.py John-Bunyan_shards.json \\
        --prompt "Please summarize the following text:" \\
        --output John-Bunyan_summaries.jsonl

    # With OpenRouter:
    python3 process_shards.py Shakespeare_shards.json \\
        --api https://openrouter.ai/api/v1/chat/completions \\
        --model anthropic/claude-3.5-sonnet \\
        --api-key sk-or-v1-...
"""

import argparse
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request

# ──────────────────────────────────────────────
# Configuration defaults
# ──────────────────────────────────────────────

DEFAULT_PROMPT = """
Convert this old 1600s text into Modern English. Keep the translation faithful and use the exact same formatting. Don't comment or add explanations, just output the modern version of the text:
""".strip()
DEFAULT_API_URL = 'http://localhost:1234/v1/chat/completions'
DEFAULT_MODEL = 'Gemma4'
DEFAULT_TIMEOUT = 360
DEFAULT_OUTPUT_SUFFIX = '_processed.jsonl'

# API key sources (checked in order)
API_KEY_ENV_VARS = [
    'OPENROUTER_API_KEY',
    'OPENAI_API_KEY',
    'LLM_API_KEY',
]


# ──────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────


def call_llm(text, prompt, api_url, model, timeout, api_key=None):
    """
    Call the LLM API with the given text and prompt.
    If api_key is provided, sends Authorization: Bearer header.
    Returns a dict with the API response data.
    Raises on error.
    """
    # Build the full user message: prompt + text
    user_message = f'{prompt}\n\n{text}'

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': user_message}],
        'max_tokens': 4096,  # generous output window
        'temperature': 0.0,  # deterministic
    }

    data = json.dumps(payload).encode('utf-8')

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }

    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    req = urllib.request.Request(
        api_url,
        data=data,
        headers=headers,
        method='POST',
    )

    start_time = time.time()

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            elapsed = time.time() - start_time
            result = json.loads(raw.decode('utf-8'))
            result['_elapsed_seconds'] = round(elapsed, 3)
            return result
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start_time
        body = e.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'HTTP {e.code} from {api_url}: {body[:500]}') from e
    except urllib.error.URLError as e:
        elapsed = time.time() - start_time
        raise RuntimeError(f'Connection error to {api_url}: {e.reason}') from e
    except Exception as e:
        elapsed = time.time() - start_time
        raise RuntimeError(f'Unexpected error: {e}') from e


def extract_response(result):
    """
    Extract the relevant fields from the API response.
    Returns a clean dict with:
      - content: the assistant's message content
      - reasoning_content: the reasoning/thinking content (if any)
      - finish_reason: why generation stopped
      - usage: token usage info
      - elapsed_seconds: wall-clock time for the API call
    """
    choice = result.get('choices', [{}])[0]
    message = choice.get('message', {})

    return {
        'content': message.get('content', ''),
        'reasoning_content': message.get('reasoning_content', ''),
        'finish_reason': choice.get('finish_reason', ''),
        'usage': result.get('usage', {}),
        'model': result.get('model', ''),
        'elapsed_seconds': result.get('_elapsed_seconds', 0),
    }


def process_shards(
    input_path,
    output_path,
    prompt,
    api_url,
    model,
    timeout,
    api_key=None,
    start_index=0,
    limit=None,
):
    """
    Main processing loop. Reads shards, calls LLM, writes JSON lines.
    """
    # --- Load shards ---
    print(f'Loading: {input_path}')
    with open(input_path, 'r', encoding='utf-8') as f:
        shards = json.load(f)

    total_shards = len(shards)
    print(f'  Total shards: {total_shards:,}')

    # Apply start/limit
    end_index = min(start_index + limit, total_shards) if limit else total_shards
    shards_to_process = shards[start_index:end_index]

    if start_index > 0 or limit:
        print(f'  Processing range: [{start_index}, {end_index}) ({len(shards_to_process)} shards)')

    print(f'  API: {api_url}')
    print(f'  Model: {model}')
    print(f'  Auth: {"API key" if api_key else "none (local)"}')
    print(f'  Timeout: {timeout}s')
    print(f'  Prompt: {repr(prompt)}')
    print(f'  Output: {output_path}')
    print()

    # --- Process ---
    processed = 0
    errors = 0
    total_start = time.time()

    # Open output in append mode for crash-safety and seamless resume
    with open(output_path, 'a', encoding='utf-8') as out_f:
        for idx, shard in enumerate(shards_to_process):
            global_idx = start_index + idx
            shard_text = shard['text']
            shard_words = shard['words']
            shard_pos = shard['pos']

            # Progress indicator
            progress = f'[{global_idx + 1}/{end_index}]'
            print(f'{progress} pos={shard_pos} words={shard_words} ...', end=' ', flush=True)

            try:
                result = call_llm(shard_text, prompt, api_url, model, timeout, api_key)
                response = extract_response(result)

                record = {
                    'shard_index': global_idx,
                    'shard_pos': shard_pos,
                    'shard_words': shard_words,
                    'shard_text': shard_text,
                    'prompt': prompt,
                    'response': response,
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                out_f.flush()  # ensure streaming to disk

                content_len = len(response['content'])
                reasoning_len = len(response['reasoning_content'])
                elapsed = response['elapsed_seconds']
                print(f'OK ({elapsed:.1f}s, {content_len}c/{reasoning_len}r)')

                processed += 1

            except Exception as e:
                error_msg = f'{type(e).__name__}: {e}'
                print(f'ERROR: {error_msg}')

                # Write error record so no data is lost
                error_record = {
                    'shard_index': global_idx,
                    'shard_pos': shard_pos,
                    'shard_words': shard_words,
                    'shard_text': shard_text,
                    'prompt': prompt,
                    'error': error_msg,
                    'response': None,
                }
                out_f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
                out_f.flush()

                errors += 1

    # --- Summary ---
    total_elapsed = time.time() - total_start
    print()
    print('=' * 60)
    print(f'COMPLETE: {processed} processed, {errors} errors')
    print(f'Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}m)')
    if processed > 0:
        avg = total_elapsed / processed
        print(f'Average per shard: {avg:.1f}s')
    print(f'Output: {output_path}')
    print(f'Size: {os.path.getsize(output_path):,} bytes')
    print('=' * 60)

    return processed, errors


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Process shard JSON files through a local LLM API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 process_shards.py John-Bunyan_shards.json
  python3 process_shards.py Shakespeare_shards.json --prompt "Summarize:"
  python3 process_shards.py La-Fontaine_shards.json --start 10 --limit 5
  python3 process_shards.py in.json --api http://other:8080/v1/chat/completions
        """,
    )
    parser.add_argument(
        'input',
        type=str,
        help='Path to input shard JSON file',
    )
    parser.add_argument(
        '--prompt',
        '-p',
        type=str,
        default=DEFAULT_PROMPT,
        help=f'Prompt to prepend to each shard text (default: {DEFAULT_PROMPT!r})',
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Path to output JSON-lines file (default: <input>_processed.jsonl)',
    )
    parser.add_argument(
        '--api',
        type=str,
        default=DEFAULT_API_URL,
        help=f'LLM API endpoint URL (default: {DEFAULT_API_URL})',
    )
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Model name to use (default: {DEFAULT_MODEL})',
    )
    parser.add_argument(
        '--timeout',
        '-t',
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f'Request timeout in seconds (default: {DEFAULT_TIMEOUT})',
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start processing at this shard index (0-based)',
    )
    parser.add_argument(
        '--limit',
        '-n',
        type=int,
        default=None,
        help='Process at most N shards (default: all)',
    )
    parser.add_argument(
        '--api-key',
        '-k',
        type=str,
        default=None,
        help=('API key for cloud LLMs (OpenRouter, OpenAI, etc.). If not set, checks env vars: ' + ', '.join(API_KEY_ENV_VARS)),
    )

    args = parser.parse_args()

    # Resolve API key: CLI arg > env vars > None (local)
    api_key = args.api_key
    if api_key is None:
        for varname in API_KEY_ENV_VARS:
            api_key = os.environ.get(varname)
            if api_key:
                break

    # Validate input
    if not os.path.exists(args.input):
        print(f'Error: input file not found: {args.input}', file=sys.stderr)
        sys.exit(1)

    # Derive output path
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = base + DEFAULT_OUTPUT_SUFFIX

    # Auto-resume: if output exists, count completed lines and skip them.
    # Only applies when the user didn't explicitly set --start.
    auto_resume = args.start == 0  # default is 0, meaning "not explicitly set"
    if auto_resume and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            existing = sum(1 for _ in f)
        if existing > 0:
            print(f'Resuming: found {existing} existing entries in {args.output}')
            print(f'  Will skip shards 0-{existing - 1} and continue from shard {existing}')
            print()
            args.start = existing

    # Run
    try:
        processed, errors = process_shards(
            input_path=args.input,
            output_path=args.output,
            prompt=args.prompt,
            api_url=args.api,
            model=args.model,
            timeout=args.timeout,
            api_key=api_key,
            start_index=args.start,
            limit=args.limit,
        )
    except KeyboardInterrupt:
        print('\nInterrupted by user', file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f'Fatal error: {e}', file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    if errors > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
