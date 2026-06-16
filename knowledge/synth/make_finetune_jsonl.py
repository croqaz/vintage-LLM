#!/usr/bin/env python3
"""
make_finetune_jsonl.py
======================

Convert the "rich" JSON-lines produced by `self_instruct_1900_poc.py`
(records shaped like {"messages": [system, user, assistant], "meta": {...}})
into a *standard, minimal* supervised fine-tuning format:

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

What this script does:
  - Reads one or more input .jsonl files.
  - For every record, pulls out the user turn and the assistant turn.
  - Decides what to do with the system turn (keep / replace / drop) -- a flag.
  - Runs a single `quality_ok()` gate before accepting the record.
  - Writes the surviving records, stripped down to the minimal schema,
     to the output file. The `meta` block from the input is dropped.

Usage examples:
  # keep whatever system prompt is already in the file (default)
  python3 make_finetune_jsonl.py self_instruct_1900.jsonl -o ft.jsonl

  # replace every system prompt with your own
  python3 make_finetune_jsonl.py in.jsonl -o ft.jsonl \
      --system replace --system-text "You are a helpful assistant."

  # drop the system message entirely (user/assistant only)
  python3 make_finetune_jsonl.py in.jsonl -o ft.jsonl --system drop

  # merge several inputs into one training file
  python3 make_finetune_jsonl.py a.jsonl b.jsonl c.jsonl -o ft.jsonl
"""

import argparse
import json
import sys


# ----------------------------------------------------------------------------
# Quality filtering
# ----------------------------------------------------------------------------
# This is the ONE place to add acceptance rules. It receives the three pieces
# of text from a record (system may be None) and returns (ok, reason).
# `reason` is only used for logging/stats when a record is rejected.
#
# Keep it returning early with a short reason string so the rejection stats
# at the end are informative.
def quality_ok(system_text, user_text, assistant_text):
    """Return (True, "") to accept a record, or (False, reason) to reject it."""

    # --- the prompt (user turn) must be reasonably long -------------------
    # "length of the input prompt must be larger than 20"
    if len(user_text) <= 20:
        return False, 'user_too_short'

    # --- the answer must have real lexical variety ------------------------
    # len(set(text)) > 10  -> catches degenerate answers like "aaaa...." or
    # very short/repetitive outputs that carry almost no training signal.
    if len(set(assistant_text)) <= 10:
        return False, 'assistant_low_char_unique'

    if assistant_text.startswith('I regret that'):
        return False, 'assistant_regret'
    if assistant_text.startswith('I apologize that'):
        return False, 'assistant_apologize'
    if assistant_text.startswith("I'm affraid I"):
        return False, 'assistant_affraid'
    if assistant_text.startswith("I'm sorry t"):
        return False, 'assistant_sorry'
    if 'I cannot answer ' in assistant_text:
        return False, 'assistant_cannot'
    if 'I cannot provide ' in assistant_text:
        return False, 'assistant_cannot_provide'
    if 'I have no knowledge of' in assistant_text:
        return False, 'assistant_no_knowledge'

    # -- TODO: add more rules here
    # Ideas you might want later (left as comments on purpose):
    #   - reject if answer just echoes prompt (ROUGE-L / substring overlap)
    #   - cap maximum length to drop runaway generations

    return True, ''


# ----------------------------------------------------------------------------
# Record-level helpers
# ----------------------------------------------------------------------------
def extract_turns(record):
    """Pull (system, user, assistant) text out of one input record.

    Returns a tuple (system_text_or_None, user_text, assistant_text).
    Raises ValueError if the record is missing a user or assistant turn,
    so the caller can count it as malformed.
    """
    messages = record.get('messages')
    if not isinstance(messages, list):
        raise ValueError('no messages list')

    # Take the *last* message of each role. The POC emits exactly one of each,
    # but using the last is robust if a record ever carries extra turns.
    system_text = None
    user_text = None
    assistant_text = None
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content', '')
        if role == 'system':
            system_text = content
        elif role == 'user':
            user_text = content
        elif role == 'assistant':
            assistant_text = content

    if user_text is None:
        raise ValueError('no user turn')
    if assistant_text is None:
        raise ValueError('no assistant turn')
    return system_text, user_text, assistant_text


def build_output(system_text, user_text, assistant_text, system_mode, system_replacement):
    """Assemble the minimal output record according to the system-message flag.

    system_mode is one of:
      - "keep":    use the record's own system message (if it had one)
      - "replace": use `system_replacement` for every record
      - "drop":    no system message at all
    """
    out_messages = []

    if system_mode == 'keep':
        if system_text:  # only add if the input actually had a non-empty one
            out_messages.append({'role': 'system', 'content': system_text})
    elif system_mode == 'replace':
        out_messages.append({'role': 'system', 'content': system_replacement})
    elif system_mode == 'drop':
        pass  # intentionally no system turn

    out_messages.append({'role': 'user', 'content': user_text})
    out_messages.append({'role': 'assistant', 'content': assistant_text})
    return {'messages': out_messages}


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Convert rich self-instruct JSONL into minimal fine-tuning JSONL.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('inputs', nargs='+', help='input .jsonl file(s)')
    parser.add_argument('-o', '--out', required=True, help='output .jsonl path')
    parser.add_argument(
        '--system',
        choices=['keep', 'replace', 'drop'],
        default='keep',
        help='what to do with the system message',
    )
    parser.add_argument(
        '--system-text',
        default='You are a helpful assistant.',
        help='system message used when --system replace',
    )
    args = parser.parse_args()

    # Stats so the run is transparent about what it threw away.
    stats = {
        'read': 0,
        'written': 0,
        'bad_json': 0,
        'malformed': 0,
    }
    rejected = {}  # reason -> count, from quality_ok

    with open(args.out, 'w', encoding='utf-8') as fout:
        for path in args.inputs:
            try:
                fin = open(path, 'r', encoding='utf-8')
            except OSError as e:
                print(f'! cannot open {path}: {e}', file=sys.stderr)
                continue

            with fin:
                for line_no, line in enumerate(fin, 1):
                    line = line.strip()
                    if not line:
                        continue  # skip blank lines
                    stats['read'] += 1

                    # --- parse ---------------------------------------------
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        stats['bad_json'] += 1
                        print(f'! {path}:{line_no} bad JSON, skipped', file=sys.stderr)
                        continue

                    # --- extract turns -------------------------------------
                    try:
                        system_text, user_text, assistant_text = extract_turns(record)
                    except ValueError as e:
                        stats['malformed'] += 1
                        print(f'! {path}:{line_no} {e}, skipped', file=sys.stderr)
                        continue

                    # --- quality gate --------------------------------------
                    ok, reason = quality_ok(system_text, user_text, assistant_text)
                    if not ok:
                        rejected[reason] = rejected.get(reason, 0) + 1
                        continue

                    # --- emit minimal record -------------------------------
                    out_record = build_output(
                        system_text,
                        user_text,
                        assistant_text,
                        args.system,
                        args.system_text,
                    )
                    fout.write(json.dumps(out_record, ensure_ascii=False) + '\n')
                    stats['written'] += 1

    # --- report ---------------------------------------------------------------
    print(f'read={stats["read"]}  written={stats["written"]}  bad_json={stats["bad_json"]}  malformed={stats["malformed"]}')
    if rejected:
        detail = '  '.join(f'{k}={v}' for k, v in sorted(rejected.items()))
        print(f'rejected by quality gate: {detail}')


if __name__ == '__main__':
    main()
