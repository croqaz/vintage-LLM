"""
This script decodes tokenized .arrow files back into human-readable text using a specified tokenizer.json file.
It outputs the decoded text into a .txt file with the same basename as the original file.
"""

import argparse
from pathlib import Path

from datasets import Dataset
from transformers import PreTrainedTokenizerFast


def main():
    parser = argparse.ArgumentParser(description='Decode tokenized .arrow files and output to .txt.')
    parser.add_argument('arrow_files', nargs='+', type=Path, help='Paths to the .arrow files (e.g., data-00000-of-00032.arrow)')
    parser.add_argument('--tokenizer', '-t', type=Path, required=True, help='Path to the tokenizer.json file')
    parser.add_argument(
        '--column', '-c', type=str, default='input_ids', help='Name of the column containing the tokens (default: input_ids)'
    )

    args = parser.parse_args()

    if not args.tokenizer.is_file():
        raise FileNotFoundError(f'The tokenizer file {args.tokenizer} does not exist.')

    # 1. Load the tokenizer
    print(f'Loading tokenizer from {args.tokenizer}...')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.tokenizer))
    print(f'Vocab size: {tokenizer.vocab_size}')

    # 2. Process each arrow file
    for arrow_path in args.arrow_files:
        if not arrow_path.is_file():
            print(f'\nSkipping {arrow_path}: File does not exist.')
            continue

        print(f'\n--- Loading {arrow_path} ---')
        try:
            ds = Dataset.from_file(str(arrow_path))
        except Exception as e:
            print(f'Error loading {arrow_path}: {e}')
            continue

        print(f'Columns present: {ds.column_names}')
        if args.column not in ds.column_names:
            print(f"Error: Column '{args.column}' not found in the dataset. Available columns: {ds.column_names}")
            continue

        print(f'Total rows in file: {len(ds):,}')

        # 3. Decode row by row to prevent memory overload and write to file
        output_file = arrow_path.with_suffix('.txt')
        print(f'Decoding to {output_file}...')

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for row_idx, row in enumerate(ds):
                    tokens = row[args.column]
                    if isinstance(tokens, int):
                        tokens = [tokens]

                    decoded_text = tokenizer.decode(tokens)
                    f.write(decoded_text)

                    # Print progress every 10,000 rows
                    if (row_idx + 1) % 10000 == 0:
                        print(f'  Processed {row_idx + 1:,} rows...')

            print(f'Successfully wrote the decoded text to: {output_file}')
        except Exception as e:
            print(f'Error writing to {output_file}: {e}')


if __name__ == '__main__':
    main()
