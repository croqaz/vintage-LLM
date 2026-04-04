import argparse
from pathlib import Path

from transformers import PreTrainedTokenizerFast


def main():
    parser = argparse.ArgumentParser(description='Count tokens in text files or folders.')
    parser.add_argument('paths', nargs='+', type=Path, help='List of text files or folders to process')
    parser.add_argument('--tokenizer', '-t', type=Path, required=True, help='Path to the checkpoint directory containing the tokenizer')
    parser.add_argument('--ext', type=str, default='.txt', help='File extension to look for in folders (default: .txt)')

    args = parser.parse_args()

    if not args.tokenizer.is_dir():
        raise NotADirectoryError(f'The tokenizer directory {args.tokenizer} does not exist.')

    print(f'Loading tokenizer from {args.tokenizer}...')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    print(f'Vocab size: {tokenizer.vocab_size}\n')

    total_files = 0
    total_size = 0
    total_tokens = 0

    files_to_process = []
    for p in args.paths:
        if p.is_file():
            files_to_process.append(p)
        elif p.is_dir():
            files_to_process.extend(p.rglob(f'*{args.ext}'))
        else:
            print(f'Warning: {p} is neither a file nor a directory, skipping.')

    if not files_to_process:
        print('No files found to process.')
        return

    # Header for the output table
    print(f'{"File":<50} | {"Size (Bytes)":<15} | {"Tokens":<15}')
    print('-' * 85)

    for file_path in files_to_process:
        try:
            size = file_path.stat().st_size
            text = file_path.read_text(encoding='utf-8')
            tokens = tokenizer.encode(text)
            num_tokens = len(tokens)

            display_name = file_path.name
            if len(display_name) > 47:
                display_name = display_name[:44] + '...'

            print(f'{display_name:<50} | {size:<15,d} | {num_tokens:<15,d}')

            total_files += 1
            total_size += size
            total_tokens += num_tokens
        except Exception as e:
            print(f'Error processing {file_path}: {e}')

    print('-' * 85)
    print(f'Total Files : {total_files}')
    print(f'Total Size  : {total_size:,d} bytes')
    print(f'Total Tokens: {total_tokens:,d}')
    if total_files > 0:
        print(f'Bytes per Token: {total_size / total_tokens:.2f} b/tok')


if __name__ == '__main__':
    main()
