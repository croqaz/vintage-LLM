#!/usr/bin/env python
"""
Split a plain-text book into overlapping word-level shards and output as JSON.

Usage:
    python shard_book.py <input.txt> [options]

Options:
    -o, --output FILE       Output JSON file (default: <input>_chunks.json)
    -s, --shard-size N      Words per shard (default: 4000)
    -f, --shift N           Offset for overlapping windows (default: 2000, half of shard-size)

Each shard entry contains:
    id, window ("aligned" | "shifted"),
    text,
    start_word, end_word (0-based inclusive),
    word_count,
    start_byte, end_byte (exclusive upper bound for slicing),
    approx_word_pos
"""

import argparse
import json
import os
import re


def shard_book(input_path, output_path, shard_size, shift):
    with open(input_path, 'r') as f:
        text = f.read()

    # Tokenize non-whitespace sequences and record their byte spans
    words = []
    for m in re.finditer(r'\S+', text):
        words.append((m.start(), m.end()))

    total_words = len(words)
    shards = []
    shard_id = 0

    # ----- Aligned windows: 0, shard_size, 2*shard_size, ... -----
    for start_word in range(0, total_words, shard_size):
        end_word = min(start_word + shard_size, total_words)
        if start_word >= end_word:
            break
        start_byte = words[start_word][0]
        end_byte = words[end_word - 1][1]
        shard_text = text[start_byte:end_byte]
        shards.append(
            {
                'id': shard_id,
                'window': 'aligned',
                'text': shard_text,
                'start_word': start_word,
                'end_word': end_word - 1,
                'word_count': end_word - start_word,
                'start_byte': start_byte,
                'end_byte': end_byte,
                'approx_word_pos': f'{start_word}–{end_word - 1}',
            }
        )
        shard_id += 1

    # ----- Shifted windows: shift, shift+shard_size, shift+2*shard_size, ... -----
    for start_word in range(shift, total_words, shard_size):
        end_word = min(start_word + shard_size, total_words)
        if start_word >= end_word:
            break
        start_byte = words[start_word][0]
        end_byte = words[end_word - 1][1]
        shard_text = text[start_byte:end_byte]
        shards.append(
            {
                'id': shard_id,
                'window': 'shifted',
                'text': shard_text,
                'start_word': start_word,
                'end_word': end_word - 1,
                'word_count': end_word - start_word,
                'start_byte': start_byte,
                'end_byte': end_byte,
                'approx_word_pos': f'{start_word}–{end_word - 1}',
            }
        )
        shard_id += 1

    out = {
        'source_file': os.path.basename(input_path),
        'total_words': total_words,
        'total_bytes': len(text),
        'shard_size_words': shard_size,
        'shift_words': shift,
        'num_shards': len(shards),
        'shards': shards,
    }

    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Summary to stdout
    aligned = sum(1 for s in shards if s['window'] == 'aligned')
    shifted = sum(1 for s in shards if s['window'] == 'shifted')
    print(f'Source:       {input_path}')
    print(f'Output:       {output_path}')
    print(f'Total words:  {total_words}')
    print(f'Total bytes:  {len(text)}')
    print(f'Shard size:   {shard_size} words')
    print(f'Shift:        {shift} words')
    print(f'Aligned shards: {aligned}')
    print(f'Shifted shards: {shifted}')
    print(f'Total shards:   {len(shards)}')
    print()
    for s in shards:
        print(
            f'  [{s["id"]:3}] {s["window"]:8s}  words {s["start_word"]:>6}–{s["end_word"]:<6}  '
            f'({s["word_count"]:>5} words)  bytes {s["start_byte"]:>8}–{s["end_byte"]}'
        )


def main():
    parser = argparse.ArgumentParser(description='Split a plain-text book into overlapping word-level shards (JSON output).')
    parser.add_argument('input', help='Path to the input .txt file')
    parser.add_argument('-o', '--output', help='Output JSON file path (default: <input>_chunks.json)')
    parser.add_argument('-s', '--shard-size', type=int, default=4000, help='Words per shard (default: 4000)')
    parser.add_argument('-f', '--shift', type=int, default=None, help='Offset for overlapping windows (default: half of shard-size)')
    args = parser.parse_args()

    if args.shift is None:
        args.shift = args.shard_size // 2

    if args.output is None:
        base, _ = os.path.splitext(args.input)
        args.output = base + '_chunks.json'

    shard_book(args.input, args.output, args.shard_size, args.shift)


if __name__ == '__main__':
    main()
