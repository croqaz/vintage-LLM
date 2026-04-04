import argparse
import glob
import os
from functools import partial
from pathlib import Path

from litdata import optimize
from litdata.streaming import TokensLoader
from litgpt.tokenizer import Tokenizer


def tokenize_file(fname: str, tokenizer: Tokenizer):
    with open(fname, encoding='utf-8') as file:
        text = file.read()
    text = text.strip()
    yield tokenizer.encode(text, bos=False, eos=False)


def tokenize_folder(input: str, output: str, tokenizer: Tokenizer, max_seq_length: int = 512) -> None:
    """
    Tokenizes a folder full of .txt files into .bin files without train/val splitting.

    Args:
        input: Path to the directory containing .txt files.
        output: Path where the tokenized .bin files will be saved.
        tokenizer: An initialized litgpt Tokenizer.
        max_seq_length: The block size for the tokens loader.
    """
    input_path = Path(input)
    output_path = Path(output)

    # Gather all text files
    text_files = sorted(glob.glob(str(input_path / '*.txt')))
    assert len(text_files) > 0, f'No .txt files found in {input_path}'

    if output_path.is_dir():
        print(f'Output directory {output_path} already exists. Please remove it or change the output path!')
        return

    # Use max available CPUs (leaving 1 free)
    num_workers = max(1, os.cpu_count() - 1)
    use_workers = min(num_workers, len(text_files))

    print(f'Starting tokenization of {len(text_files)} files using {use_workers} workers...')

    # Run the litdata optimization process
    optimize(
        fn=partial(tokenize_file, tokenizer=tokenizer),
        inputs=text_files,
        output_dir=str(output_path),
        num_workers=use_workers,
        chunk_bytes='50MB',
        item_loader=TokensLoader(block_size=max_seq_length + 1),  # +1 for the next token
    )

    print(f'Tokenization complete. Tokenized chunks saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenizes a folder full of .txt files into .bin files.')
    parser.add_argument('--tokenizer', type=str, default='model/tokenizer/', help='Path to the tokenizer directory.')
    parser.add_argument('--input', type=str, required=True, help='Path to the directory containing .txt files.')
    parser.add_argument('--output', type=str, required=True, help='Path where the tokenized .bin files will be saved.')
    parser.add_argument('--max-seq-length', type=int, default=512, help='The block size for the tokens loader.')

    args = parser.parse_args()

    tokenizer = Tokenizer(checkpoint_dir=args.tokenizer)
    tokenize_folder(args.input, args.output, tokenizer, args.max_seq_length)
