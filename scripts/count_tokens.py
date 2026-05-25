"""
Small script that counts the number of tokens from a user provided text, or STDIN.
"""

import argparse
import sys

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description='Count tokens in text.')
    parser.add_argument('text', nargs='?', default=None, help='Text to tokenize (or pass via STDIN)')
    parser.add_argument('--tok', required=True, help='Path to the tokenizer directory')
    args = parser.parse_args()

    if args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read()

    if not text.strip():
        print('No text provided.')
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tok)
    except Exception as e:
        print(f'Error loading tokenizer: {e}')
        sys.exit(1)

    tokens = tokenizer.encode(text)
    print(tokens)
    print(len(tokens), 'tokens')


if __name__ == '__main__':
    main()
