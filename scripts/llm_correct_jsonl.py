"""
Corrects text fields in a JSON lines file using a language model.
Each entry must have a "text" field. The original text is preserved as "original"
and the corrected text replaces "text". All other fields are preserved.
"""

import argparse
import itertools
import json
import os
import random
import time

import requests

SYSTEM_PROMPT = """You are a helpful assistant that corrects badly scanned OCR text from old newspaper archives, year 1800.
Your task is to correct scanning errors and typos.
Don't judge, don't add any extra commentary or explanations.
Do NOT alter the original meaning, tone, or style, this text is a historical document and MUST be preserved!
Output *only* the corrected text."""

USER_PROMPT = """
Input example:
"j'\nr\nì\n' ILt\nr\nr-\nr-\no\nr\nA\nr\ngenerality of the privilege, by expressly declar.nng, thar every perion 'availing himlelf Of the liberty of the prels, ihould be repoaible for thr\nabufe of that liberty :. thus iecuring TO our\ncitizens the invaluable right os reputation\nagaina every malicious invader of it.\n\n\nPrinted publications attacking private cha.\nraeter, is conIidered with great reaion by the\nlaw AS very ATTRACTIONS offence. from its evi\ndent tendency to diturb the public peace-if\nmen find they can have no redrefs in cur\ncourts of Juice for fuch injuries, they will\nnaturally take fatisfaaion in their own way, in\nvolving perhaps their friends and families in\nthe confetti and leading evidently to duels,\nmurders, and perhaps Affirmation. ~, -\n\n\n"

Expected output example:
Generality of the privilege, by expressly declaring, that every person availing himself of the liberty of the press, should be responsible for the abuse of that liberty: thus securing to our citizens the invaluable right of reputation against every malicious invader of it.
Printed publications attacking private character, is considered with great reason by the law as a very atrocious offence, from its evident tendency to disturb the public peace—if men find they can have no redress in our courts of justice for such injuries, they will naturally take satisfaction in their own way, involving perhaps their friends and families in the conflict, and leading evidently to duels, murders, and perhaps animosities.
"""


def correct_text_batch(chunk, url, api_key=None, model=None, delay=0):
    if delay:
        if delay > 1:
            sign = random.choice([-1, 1])
            jitt = random.uniform(0, delay * 0.1)
            delay += sign * jitt
            print(f'  Waiting {delay:.2f}s before API call...')
        time.sleep(delay)
    if api_key:
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    else:
        headers = {'Content-Type': 'application/json'}

    data = {
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f'Correct the following OCR text:\n\n{chunk}'},
        ],
        'temperature': 0.1,
    }
    if model:
        data['model'] = model

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f'Error calling LLM: {e}')
        return chunk  # Fallback to original chunk on error


def correct_text(text, url, key_cycle, model=None, delay=0):
    api_key = next(key_cycle) if key_cycle is not None else None
    return correct_text_batch(text, url, api_key=api_key, model=model, delay=delay)


def load_processed_keys(output_file):
    """Return the set of deduplication keys already present in the output file."""
    processed = set()
    if not os.path.exists(output_file):
        return processed
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = entry.get('xxh64') or (entry.get('original') or entry.get('text', ''))[:64]
            if key:
                processed.add(key)
    return processed


def entry_key(entry):
    """Return a stable deduplication key for an input entry."""
    return entry.get('xxh64') or entry.get('text', '')[:64]


def main():
    parser = argparse.ArgumentParser(description='Correct OCR text in a JSON lines file using a local LLM.')
    parser.add_argument('input_file', help='Path to the input JSON lines file')
    parser.add_argument('-o', '--output_file', help='Path to the output JSON lines file (optional)')
    parser.add_argument('--api_url', default='http://127.1:1234/v1/chat/completions', help='LLM API URL')
    parser.add_argument(
        '--api_key',
        action='append',
        dest='api_keys',
        metavar='KEY',
        help='API key for authentication. Repeat to provide multiple keys; they will be rotated round-robin per request.',
    )
    parser.add_argument('--model', help='(Optional) Model name to use, e.g. "gemini-2.5-flash" or "gpt-4o-mini"')
    parser.add_argument(
        '--delay', type=float, default=0, help='Seconds to wait before each API call (e.g. 10 to call at most once per 10 s).'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return

    output_file = args.output_file or f'{args.input_file}.corrected.jsonl'

    key_cycle = itertools.cycle(args.api_keys) if args.api_keys else None
    if args.api_keys:
        print(f'Using {len(args.api_keys)} API key(s), rotating round-robin.')
    if args.delay:
        print(f'Delay between requests: {args.delay}s.')

    # Load already-processed entries so the script can be safely re-run to resume.
    processed_keys = load_processed_keys(output_file)
    if processed_keys:
        print(f'Resuming: {len(processed_keys)} entries already in output, will skip.')

    total_input = 0
    total_skipped_invalid = 0
    total_already_done = 0
    total_processed = 0

    # Open output in append mode so a crash never discards work already written.
    with open(args.input_file, 'r', encoding='utf-8') as fin, open(output_file, 'a', encoding='utf-8') as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            total_input += 1

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f'Skipping line {line_num}: invalid JSON ({e})')
                total_skipped_invalid += 1
                continue

            if 'text' not in entry:
                print(f'Skipping line {line_num}: no "text" field.')
                total_skipped_invalid += 1
                continue

            key = entry_key(entry)
            if key in processed_keys:
                print(f'Skipping entry {line_num}: already in output.')
                total_already_done += 1
                continue

            original_text = entry['text']
            orig_words = original_text.split()
            print(f'Will process entry {line_num} ({len(original_text)} chars, {len(orig_words)} words) ...')
            print(
                f'  Input  first/last word: {repr(orig_words[0]) if orig_words else "(empty)"} / {repr(orig_words[-1]) if orig_words else "(empty)"}'
            )

            corrected = correct_text(original_text, args.api_url, key_cycle=key_cycle, model=args.model, delay=args.delay)

            corr_words = corrected.split()
            print(
                f'  Output first/last word: {repr(corr_words[0]) if corr_words else "(empty)"} / {repr(corr_words[-1]) if corr_words else "(empty)"}'
            )
            print(f'  Words: input={len(orig_words)}, output={len(corr_words)} ({len(corr_words) - len(orig_words):+d})')

            entry['original'] = original_text
            entry['text'] = corrected

            fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
            fout.flush()  # Write immediately so no work is lost on crash.
            total_processed += 1

    # Validation report
    total_valid_input = total_input - total_skipped_invalid
    total_in_output = total_already_done + total_processed

    print('\n--- Summary ---')
    print(f'Input entries:       {total_input}')
    print(f'Invalid/skipped:     {total_skipped_invalid}')
    print(f'Already in output:   {total_already_done}')
    print(f'Processed this run:  {total_processed}')
    if total_in_output == total_valid_input:
        print(f'Validation OK: all {total_valid_input} valid entries are accounted for in the output.')
    else:
        missing = total_valid_input - total_in_output
        print(f'WARNING: {missing} entries are missing from the output! Check for errors above.')
    print(f'Output file: {output_file}')


if __name__ == '__main__':
    main()
