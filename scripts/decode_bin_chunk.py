"""
This script decodes a tokenized .bin Numpy file (commonly found in the 'val' folder of a tokenized dataset) back into human-readable text using the tokenizer from a specified checkpoint directory. It allows you to specify the data type of the .bin file and limits the number of tokens displayed for convenience. This is useful for verifying that your tokenization process is working correctly and for inspecting the contents of your tokenized dataset.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from litgpt.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description='Decode a tokenized .bin Numpy file and display the text.')
    parser.add_argument('bin_file', type=Path, help="Path to the tokenized .bin Numpy file (e.g., from the 'val' folder)")
    parser.add_argument('--tokenizer', '-t', type=Path, required=True, help='Path to the checkpoint directory containing the tokenizer')
    parser.add_argument(
        '--dtype', type=str, default='uint16', help='Numpy data type of the .bin file (default: uint16, common alternatives: uint32, int32)'
    )
    parser.add_argument('--max_tokens', '-m', type=int, default=1000, help='Maximum number of tokens to display (default: 1000)')

    args = parser.parse_args()

    if not args.bin_file.is_file():
        raise FileNotFoundError(f'The bin file {args.bin_file} does not exist.')
    if not args.tokenizer.is_dir():
        raise NotADirectoryError(f'The tokenizer directory {args.tokenizer} does not exist.')

    # 1. Load the tokenizer
    print(f'Loading tokenizer from {args.tokenizer}...')
    tokenizer = Tokenizer(args.tokenizer)
    print(f'Tokenizer backend: {tokenizer.backend}, Vocab size: {tokenizer.vocab_size}')

    # 2. Load the binary numpy array
    print(f'Loading binary token data from {args.bin_file} (dtype={args.dtype})...')
    try:
        # np.memmap is efficient for potentially massive tokenized corpora
        data = np.memmap(args.bin_file, dtype=args.dtype, mode='r')
    except ValueError as e:
        print(f'Error loading {args.bin_file}: {e}')
        return

    total_tokens = len(data)
    print(f'Total tokens in file: {total_tokens:,}')

    # 3. Take the requested chunk size
    tokens_to_decode = min(args.max_tokens, total_tokens)
    chunk = data[:tokens_to_decode]

    # Convert the numpy array to a torch.Tensor, matching what litgpt's decode() expects
    tensor_chunk = torch.tensor(chunk.astype(np.int32), dtype=torch.int)

    # 4. Decode the tokens back to text
    print(f'\nDecoding the first {tokens_to_decode} tokens...\n')
    decoded_text = tokenizer.decode(tensor_chunk)

    print('-' * 40)
    print(decoded_text)
    print('-' * 40)

    if tokens_to_decode < total_tokens:
        print(f'\n... (remaining {total_tokens - tokens_to_decode:,} tokens omitted)')


if __name__ == '__main__':
    main()
