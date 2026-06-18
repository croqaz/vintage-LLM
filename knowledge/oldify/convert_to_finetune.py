#!/usr/bin/env python3
"""
Convert the raw datasets in this folder into standard fine-tuning JSONL files.

Goal of the dataset: teach an LLM to "old-ify" modern text into archaic / old speech.
  - The USER message contains the MODERN text.
  - The ASSISTANT message contains the OLD / archaic rendering.
  - A simple, generic SYSTEM prompt puts the model into "oldify" mode.

Two input shapes are handled:

1. The four `*_modern_Mistral.jsonl` files. Each line is a shard with:
     - shard_text       : the ORIGINAL old text (-> assistant)
     - response.content : the MODERN translation produced by Mistral (-> user)
   (We therefore REVERSE the original modern-isation direction.)

2. convert1.json : a JSON array of {original (modern), converted (old), changed}.
     - original  -> user
     - converted -> assistant

Output: one `<stem>_finetune.jsonl` per input, each line:
   {"messages": [{"role":"system",...},{"role":"user",...},{"role":"assistant",...}],
    "source": "<stem>"}
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SYSTEM_PROMPT = (
    'You are a master of old, archaic English. Rewrite the modern text the user '
    'gives you into old-style speech, preserving its meaning and formatting. '
    'Reply with only the converted text, and nothing else.'
)

# Short instruction prefixes prepended to each user message, cycled round-robin.
# The modern text itself is wrapped in quotes after the prefix.
USER_PREFIXES = [
    'Oldify this text:',
    'Rewrite to old-style speech:',
    'Make this sound like old-speech:',
    'Convert this into archaic English:',
    'Render this in old, archaic speech:',
    'Turn this into old-fashioned language:',
]

# Leading model preamble such as:
#   "Here is the modernized version of the text, keeping the exact same formatting:"
# optionally followed by a "---" separator line.
_PREAMBLE = re.compile(
    r"^\s*(?:here(?:\s+is|'s|’s)\b[^\n]*?"
    r'(?:version|text|translation)[^\n]*?:|'
    r'below is\b[^\n]*?:|'
    r'sure[,!.][^\n]*|certainly[,!.][^\n]*)\s*\n',
    re.IGNORECASE,
)


def clean_content(text: str) -> str:
    """Strip a leading model preamble line and its trailing '---' separator."""
    if text is None:
        return ''
    cleaned = _PREAMBLE.sub('', text, count=1)
    if cleaned != text:
        # remove a leading separator line (e.g. "---") left behind by the preamble
        cleaned = re.sub(r'^\s*-{3,}\s*\n', '', cleaned, count=1)
    return cleaned.strip()


def make_record(modern: str, old: str, source: str, index: int) -> dict:
    prefix = USER_PREFIXES[index % len(USER_PREFIXES)]
    user_content = f'{prefix} "{modern}"'
    return {
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': old},
        ],
        'source': source,
    }


def convert_mistral_jsonl(path: Path, source: str) -> list[dict]:
    """response.content (modern) -> user ; shard_text (old) -> assistant."""
    records = []
    skipped = 0
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            resp = d.get('response') or {}
            # Only keep complete generations.
            if resp.get('finish_reason') not in (None, 'stop'):
                skipped += 1
                continue
            modern = clean_content(str(resp.get('content', '')))
            old = str(d.get('shard_text', '')).strip()
            if not modern or not old:
                skipped += 1
                continue
            records.append(make_record(modern, old, source, len(records)))
    if skipped:
        print(f'  ({skipped} rows skipped)')
    return records


def convert_convert(path: Path, source: str) -> list[dict]:
    """original (modern) -> user ; converted (old) -> assistant."""
    records = []
    skipped = 0
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            modern = str(item.get('original', '')).strip()
            old = str(item.get('converted', '')).strip()
            if not modern or not old:
                skipped += 1
                continue
            records.append(make_record(modern, old, source, len(records)))
    if skipped:
        print(f'  ({skipped} rows skipped)')
    return records


def write_jsonl(records: list[dict], out_path: Path) -> None:
    with out_path.open('w', encoding='utf-8') as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')


def main() -> None:
    jobs = [
        ('John-Bunyan_modern_Mistral.jsonl', 'John-Bunyan', convert_mistral_jsonl),
        ('John-Milton_modern_Mistral.jsonl', 'John-Milton', convert_mistral_jsonl),
        ('La-Fontaine_modern_Mistral.jsonl', 'La-Fontaine', convert_mistral_jsonl),
        ('Shakespeare_modern_Mistral.jsonl', 'Shakespeare', convert_mistral_jsonl),
        ('convert.jsonl', 'convert', convert_convert),
    ]

    total = 0
    for fname, source, fn in jobs:
        in_path = HERE / fname
        if not in_path.exists():
            print(f'!! missing: {fname}', file=sys.stderr)
            continue
        print(f'Converting {fname} (source={source}) ...')
        records = fn(in_path, source)
        out_path = HERE / f'{source}_finetune.jsonl'
        write_jsonl(records, out_path)
        print(f'  -> {out_path.name}: {len(records)} examples')
        total += len(records)

    print(f'\nDone. {total} examples across {len(jobs)} files.')


if __name__ == '__main__':
    main()
