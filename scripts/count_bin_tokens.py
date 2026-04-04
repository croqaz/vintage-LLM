"""
This script calculates the total number of tokens across all .bin files in a given directory.
It does this by reading the files as numpy memmap arrays and summing their lengths.
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Count tokens in tokenized .bin Numpy files.')
    parser.add_argument('directory', type=Path, help="Path to the directory containing .bin files (e.g., 'val/' or 'train/')")
    parser.add_argument(
        '--dtype',
        type=str,
        default='uint16',
        help='Numpy data type of the .bin files (default: uint16, common alternatives: uint32, int32)',
    )

    args = parser.parse_args()

    if not args.directory.is_dir():
        raise NotADirectoryError(f'The directory {args.directory} does not exist.')

    bin_files = list(args.directory.rglob('*.bin'))
    if not bin_files:
        print(f'No .bin files found in {args.directory}.')
        return

    print(f'Found {len(bin_files)} .bin files in {args.directory}. Using dtype {args.dtype}.\n')

    total_tokens = 0
    total_size = 0

    print(f'{"File":<50} | {"Size (Bytes)":<15} | {"Tokens":<15}')
    print('-' * 85)

    for bin_file in sorted(bin_files):
        try:
            # We can just get the size and divide by dtype size if we want to be fast,
            # or use np.memmap to be safe and let numpy do it.
            data = np.memmap(bin_file, dtype=args.dtype, mode='r')
            num_tokens = len(data)
            size = bin_file.stat().st_size

            display_name = bin_file.name
            if len(display_name) > 47:
                display_name = display_name[:44] + '...'

            print(f'{display_name:<50} | {size:<15,d} | {num_tokens:<15,d}')

            total_tokens += num_tokens
            total_size += size
        except Exception as e:
            print(f'Error processing {bin_file}: {e}')

    print('-' * 85)
    print(f'Total Files : {len(bin_files)}')
    print(f'Total Size  : {total_size:,d} bytes')
    print(f'Total Tokens: {total_tokens:,d}')


if __name__ == '__main__':
    main()
