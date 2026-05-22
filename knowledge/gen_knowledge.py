import argparse
import json
import random

from base_knowledge import KNOWLEDGE
from transformers import AutoTokenizer

TOK_VERSION = 't-v3'
DEFAULT_SEED = 42


def format_text(knowledge: list[dict[str, str]]) -> list[str]:
    return [f'Question: {qa["question"]}\nAnswer: {qa["answer"]}' for qa in knowledge]


def format_jsonl(knowledge: list[dict[str, str]], system_prompt: str | None = None) -> list[str]:
    lines = []
    for qa in knowledge:
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.extend([{'role': 'user', 'content': qa['question']}, {'role': 'assistant', 'content': qa['answer']}])
        lines.append(json.dumps({'messages': messages}))
    return lines


def format_chat_template(knowledge: list[dict[str, str]], tok_version: str, system_prompt: str | None = None) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(f'tokenizers/{tok_version}')
    lines = []
    for qa in knowledge:
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.extend([{'role': 'user', 'content': qa['question']}, {'role': 'assistant', 'content': qa['answer']}])
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
        help=f'Tokenizer version to use with --format template (default: {TOK_VERSION})',
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
    args = parser.parse_args()

    knowledge = list(KNOWLEDGE)

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
        print(f'Written {len(lines)} items to {args.output}')
    else:
        print(output)


if __name__ == '__main__':
    main()
