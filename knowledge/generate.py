"""
Generate knowledge in various formats (text, jsonl, or tokenizer chat template).
"""

import argparse
import json
import random
from pathlib import Path

from bible import BIBLE
from books import BOOKS
from fake import FAKE
from math_qa import MATH
from medieval_qa import MEDIEVAL
from memory import MEMORY
from quotes import QUOTES

TOK_VERSION = 't-v5'
DEFAULT_SEED = 42

# The point of generating this set from Python rather than scraping it is that
# sections can be switched on and off. Keyed by the name used on the command
# line; the order here is the order they are concatenated in.
SECTIONS = {
    'bible': BIBLE,
    'books': BOOKS,
    'math': MATH,
    'medieval': MEDIEVAL,
    'memory': MEMORY,
    'quotes': QUOTES,
    'fake': FAKE,
}


def select(include: list[str] | None, exclude: list[str] | None) -> tuple[list, dict[str, int]]:
    """Sections to use, plus what each contributed.

    --include wins outright when given, so `--include math --exclude math` is a
    contradiction rather than a silent empty set, and is refused below.
    """
    names = list(SECTIONS)
    unknown = sorted(set((include or []) + (exclude or [])) - set(names))
    if unknown:
        raise SystemExit(f'unknown section(s): {", ".join(unknown)}. Known: {", ".join(names)}')
    if include:
        chosen = [n for n in names if n in include]
    else:
        chosen = [n for n in names if n not in (exclude or [])]
    if not chosen:
        raise SystemExit('every section was excluded; nothing to generate')

    items: list = []
    counts = {}
    for name in chosen:
        section = list(SECTIONS[name])
        counts[name] = len(section)
        items.extend(section)
    return items, counts


def format_text(knowledge: list[dict[str, str] | list[dict[str, str]]]) -> list[str]:
    lines = []
    for item in knowledge:
        if isinstance(item, list):
            for qa in item:
                lines.append(f'Question: {qa["question"]}\nAnswer: {qa["answer"]}')
        else:
            lines.append(f'Question: {item["question"]}\nAnswer: {item["answer"]}')
    return lines


def format_jsonl(knowledge: list[dict[str, str] | list[dict[str, str]]], system_prompt: str | None = None) -> list[str]:
    lines = []
    for item in knowledge:
        if isinstance(item, list):
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            for qa in item:
                messages.extend([{'role': 'user', 'content': qa['question']}, {'role': 'assistant', 'content': qa['answer']}])
            lines.append(json.dumps({'messages': messages}))
        else:
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.extend([{'role': 'user', 'content': item['question']}, {'role': 'assistant', 'content': item['answer']}])
            lines.append(json.dumps({'messages': messages}))
    return lines


def format_chat_template(
    knowledge: list[dict[str, str] | list[dict[str, str]]], tok_version: str, system_prompt: str | None = None
) -> list[str]:
    from transformers import AutoTokenizer

    # Resolved against the repo root rather than the working directory, so this
    # works when run from knowledge/ as well as from the root. A path is also
    # accepted, because the chat template lives with the MODEL rather than with
    # the tokenizer: tokenizers/t-v5 carries no chat_template.jinja, the model
    # directories do.
    candidate = Path(tok_version)
    tokenizer_dir = candidate if candidate.is_dir() else Path(__file__).resolve().parent.parent / 'tokenizers' / tok_version
    if not tokenizer_dir.is_dir():
        raise SystemExit(f'no tokenizer at {tokenizer_dir}')
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    if tokenizer.chat_template is None:
        raise SystemExit(
            f'{tokenizer_dir} has no chat template, so --format template cannot render anything.\n'
            f'Pass a model directory instead, which is where the template lives:\n'
            f'    --tok-version training/Llama-141M/final-anneal-3h'
        )
    lines = []
    for item in knowledge:
        if isinstance(item, list):
            for qa in item:
                messages = []
                if system_prompt:
                    messages.append({'role': 'system', 'content': system_prompt})
                messages.extend([{'role': 'user', 'content': qa['question']}, {'role': 'assistant', 'content': qa['answer']}])
                lines.append(tokenizer.apply_chat_template(messages, tokenize=False))
        else:
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.extend([{'role': 'user', 'content': item['question']}, {'role': 'assistant', 'content': item['answer']}])
            lines.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return lines


def main():
    parser = argparse.ArgumentParser(description='Generate knowledge in various formats.')
    parser.add_argument(
        '--format',
        '-f',
        choices=['text', 'jsonl', 'template'],
        default='jsonl',
        help='Output format: text (Q&A), jsonl (default), or template (tokenizer chat template)',
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Disable shuffling (shuffling is on by default)',
    )
    parser.add_argument(
        '--seed',
        '-s',
        type=int,
        default=DEFAULT_SEED,
        help=f'Random seed for shuffling (default: {DEFAULT_SEED})',
    )
    parser.add_argument(
        '--tok-version',
        default=TOK_VERSION,
        help=f'Tokenizer for --format template: a name under tokenizers/ or a path to a model directory (default: {TOK_VERSION})',
    )
    parser.add_argument(
        '--system-prompt',
        default=None,
        help='Optional system prompt to include in the messages',
    )
    parser.add_argument(
        '--output',
        '-o',
        default=None,
        help='Output file path (default: stdout)',
    )
    parser.add_argument(
        '--include',
        nargs='+',
        metavar='SECTION',
        help=f'Only these sections. Choices: {", ".join(SECTIONS)}',
    )
    parser.add_argument(
        '--exclude',
        nargs='+',
        metavar='SECTION',
        help='All sections except these. Ignored when --include is given.',
    )
    parser.add_argument(
        '--list-sections',
        action='store_true',
        help='Print the sections and their entry counts, then exit.',
    )
    args = parser.parse_args()

    if args.list_sections:
        for name, section in SECTIONS.items():
            pairs = sum(len(i) if isinstance(i, list) else 1 for i in section)
            multi = sum(1 for i in section if isinstance(i, list))
            print(f'{name:10s} {len(section):6,} entries  {pairs:6,} QA pairs  {multi:5,} multi-turn')
        return

    knowledge, counts = select(args.include, args.exclude)

    if not args.no_shuffle:
        random.seed(args.seed)
        random.shuffle(knowledge)
    else:
        random.seed(0x511)  # Remember, remember...

    if args.format == 'text':
        lines = format_text(knowledge)
        output = '\n\n'.join(lines)
    elif args.format == 'jsonl':
        lines = format_jsonl(knowledge, args.system_prompt)
        output = '\n'.join(lines)
    else:  # template
        lines = format_chat_template(knowledge, args.tok_version, args.system_prompt)
        output = '\n'.join(lines)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        composition = ', '.join(f'{n} {c:,}' for n, c in counts.items())
        print(f'Written {len(lines)} items to {args.output}')
        print(f'Sections: {composition}')
    else:
        print(output)


if __name__ == '__main__':
    main()
