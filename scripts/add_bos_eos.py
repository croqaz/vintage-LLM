"""
This script adds a beginning-of-sequence (BOS) token and an end-of-sequence (EOS) token to the start and end of text files matching a specified glob pattern. The BOS token is added at the beginning of the file, followed by a newline, and the EOS token is added at the end of the file, preceded by a newline. This is useful for preparing text data for training language models that require explicit sequence boundaries.
"""

import argparse
import glob
import os
import sys

BOS_TOKEN = '<|bos|>'
EOS_TOKEN = '<|eos|>'


def add_tokens_to_file(file_path: str):
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' does not exist.")
        sys.exit(1)

    print(f'Processing {file_path}...')
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    # Remove existing tokens if they are there, so we can cleanly add them with newlines
    if text.startswith(BOS_TOKEN):
        text = text[len(BOS_TOKEN) :].strip()
    if text.endswith(EOS_TOKEN):
        text = text[: -len(EOS_TOKEN)].strip()

    # Reconstruct the text with tokens and newlines
    new_text = f'{BOS_TOKEN}\n{text}\n{EOS_TOKEN}'

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_text)

    print(f'Successfully added {BOS_TOKEN} and {EOS_TOKEN} to {file_path}.')


def main():
    parser = argparse.ArgumentParser(
        description=f'Add {BOS_TOKEN} at the start and {EOS_TOKEN} at the end of text files matching a pattern.'
    )
    parser.add_argument('--files', required=True, help='Glob pattern for input text files (e.g., "data/*.txt")')
    args = parser.parse_args()

    matched_files = glob.glob(args.files)
    if not matched_files:
        print(f'No files matched the pattern: {args.files}')
        sys.exit(1)

    for file_path in matched_files:
        add_tokens_to_file(file_path)

    print(f'Finished processing {len(matched_files)} files.')


if __name__ == '__main__':
    main()
