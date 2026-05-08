import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

DEFAULT_TOKENIZER = './tokenizers/t-v3/'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Tokenize text file into train.bin and valid.bin.')
    parser.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    parser.add_argument('--input', type=Path, default=Path(__file__).with_name('tiny-shakespeare.txt'))
    parser.add_argument('--train-input', type=Path, help='Pre-split training text file. Requires --valid-input.')
    parser.add_argument('--valid-input', type=Path, help='Pre-split validation text file. Requires --train-input.')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent)
    parser.add_argument('--train-fraction', type=float, default=0.95)
    return parser.parse_args()


def encode_text(tokenizer, text: str) -> list[int]:
    if not text.lstrip().startswith(tokenizer.bos_token):
        text = tokenizer.bos_token + text.lstrip()
    if not text.rstrip().endswith(tokenizer.eos_token):
        text = text.rstrip() + tokenizer.eos_token
    return tokenizer.encode(text, add_special_tokens=False)


def token_dtype(train_ids: list[int], valid_ids: list[int]) -> type[np.uint16] | type[np.uint32]:
    max_token = max(train_ids + valid_ids, default=0)
    return np.uint16 if max_token < np.iinfo(np.uint16).max else np.uint32


def write_tokens(path: Path, ids: list[int], dtype: type[np.uint16] | type[np.uint32]) -> None:
    array = np.asarray(ids, dtype=dtype)
    array.tofile(path)
    print(f'Wrote {path} ({array.size:,} tokens, {dtype.__name__})')


def main() -> None:
    args = parse_args()

    has_custom_train = args.train_input is not None
    has_custom_valid = args.valid_input is not None
    if has_custom_train != has_custom_valid:
        raise ValueError('--train-input and --valid-input must be provided together')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    if has_custom_train and has_custom_valid:
        train_text = args.train_input.read_text(encoding='utf-8')
        valid_text = args.valid_input.read_text(encoding='utf-8')
        train_ids = encode_text(tokenizer, train_text)
        valid_ids = encode_text(tokenizer, valid_text)
        print(f'Using pre-split files: {args.train_input} and {args.valid_input}')
    else:
        if not 0 < args.train_fraction < 1:
            raise ValueError('--train-fraction must be between 0 and 1')
        text = args.input.read_text(encoding='utf-8')
        ids = encode_text(tokenizer, text)
        split_idx = int(len(ids) * args.train_fraction)
        train_ids = ids[:split_idx]
        valid_ids = ids[split_idx:]
        print(f'Using single input file: {args.input}')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = token_dtype(train_ids, valid_ids)
    write_tokens(args.output_dir / 'train.bin', train_ids, dtype)
    write_tokens(args.output_dir / 'valid.bin', valid_ids, dtype)

    print(f'Total tokens: {len(train_ids) + len(valid_ids):,}')
    print(f'Train tokens: {len(train_ids):,}')
    print(f'Valid tokens: {len(valid_ids):,}')


if __name__ == '__main__':
    main()
