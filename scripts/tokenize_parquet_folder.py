import argparse
import glob
import os
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from litdata import optimize
from litdata.streaming import TokensLoader
from litgpt.tokenizer import Tokenizer


def tokenize_parquet_file(fname: str, tokenizer: Tokenizer, text_column: str):
    parquet_file = pq.ParquetFile(fname)
    for batch in parquet_file.iter_batches(columns=[text_column]):
        for value in batch.column(text_column):
            text = value.as_py()
            if text:
                yield tokenizer.encode(text, bos=False, eos=False)


def tokenize_parquet_folder(
    input: str,
    output: str,
    tokenizer: Tokenizer,
    text_column: str = 'text',
    max_seq_length: int = 512,
) -> None:
    """
    Tokenizes a folder of .parquet files into litdata .bin chunks.

    Args:
        input: Path to the directory containing .parquet files.
        output: Path where the tokenized .bin files will be saved.
        tokenizer: An initialized litgpt Tokenizer.
        text_column: Name of the column containing the text to tokenize.
        max_seq_length: The block size for the tokens loader.
    """
    input_path = Path(input)
    output_path = Path(output)

    parquet_files = sorted(glob.glob(str(input_path / '*.parquet')))
    assert len(parquet_files) > 0, f'No .parquet files found in {input_path}'

    if output_path.is_dir():
        print(f'Output directory {output_path} already exists. Please remove it or change the output path!')
        return

    num_workers = max(1, os.cpu_count() - 1)
    use_workers = min(num_workers, len(parquet_files))

    print(f'Starting tokenization of {len(parquet_files)} files using {use_workers} workers...')

    optimize(
        fn=partial(tokenize_parquet_file, tokenizer=tokenizer, text_column=text_column),
        inputs=parquet_files,
        output_dir=str(output_path),
        num_workers=use_workers,
        chunk_bytes='500MB',
        item_loader=TokensLoader(block_size=max_seq_length + 1),  # +1 for the next token
    )

    print(f'Tokenization complete. Tokenized chunks saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenizes a folder of .parquet files into .bin chunks.')
    parser.add_argument('--tokenizer', type=str, default='model/tokenizer/', help='Path to the tokenizer directory.')
    parser.add_argument('--input', type=str, required=True, help='Path to the directory containing .parquet files.')
    parser.add_argument('--output', type=str, required=True, help='Path where the tokenized .bin files will be saved.')
    parser.add_argument('--text-column', type=str, default='text', help='Name of the column containing text to tokenize.')
    parser.add_argument('--max-seq-length', type=int, default=512, help='The block size for the tokens loader.')

    args = parser.parse_args()

    tokenizer = Tokenizer(checkpoint_dir=args.tokenizer)
    tokenize_parquet_folder(args.input, args.output, tokenizer, args.text_column, args.max_seq_length)
