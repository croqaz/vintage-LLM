import argparse
import glob
import json
import os
import time
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from litdata import optimize
from litdata.streaming import TokensLoader
from litgpt.tokenizer import Tokenizer


def count_input_entries(files: list[str], text_column: str = 'text') -> dict:
    """Pre-scan input files and count entries per file for auditing.

    For .parquet files, counts total rows from metadata (some may be filtered
    during tokenization if the text column is empty/null).
    """
    stats = {
        'files': {},
        'total_entries': 0,
        'total_files': len(files),
        'by_type': {},
        'skipped_empty': 0,
    }

    for fname in files:
        ext = Path(fname).suffix.lower()
        count = 0
        skipped = 0

        if ext == '.parquet':
            metadata = pq.read_metadata(fname)
            total_rows = metadata.num_rows
            # Scan for non-empty text values to get an accurate count
            pf = pq.ParquetFile(fname)
            for batch in pf.iter_batches(columns=[text_column]):
                for value in batch.column(text_column):
                    text = value.as_py()
                    if text:
                        count += 1
                    else:
                        skipped += 1
        elif ext == '.jsonl':
            with open(fname, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    text = json.loads(line).get('text', '')
                    if text:
                        count += 1
                    else:
                        skipped += 1
        else:  # .txt and other text files
            with open(fname, encoding='utf-8') as f:
                text = f.read().strip()
            if text:
                count = 1

        file_size = os.path.getsize(fname)
        stats['files'][fname] = {'entries': count, 'skipped': skipped, 'type': ext, 'size_bytes': file_size}
        stats['total_entries'] += count
        stats['skipped_empty'] += skipped
        stats['by_type'].setdefault(ext, {'files': 0, 'entries': 0})
        stats['by_type'][ext]['files'] += 1
        stats['by_type'][ext]['entries'] += count

    return stats


def count_output_chunks(output_path: Path) -> dict:
    """Post-scan output directory for auditing."""
    stats = {
        'total_chunks': 0,
        'total_items': 0,
        'total_size_bytes': 0,
        'chunk_files': [],
    }

    # Read litdata index for item counts
    index_file = output_path / 'index.json'
    if index_file.exists():
        with open(index_file) as f:
            index_data = json.load(f)
        if 'chunks' in index_data:
            stats['total_chunks'] = len(index_data['chunks'])
            for chunk in index_data['chunks']:
                if 'chunk_size' in chunk:
                    stats['total_items'] += chunk['chunk_size']

    # Measure .bin file sizes
    for bin_file in sorted(output_path.glob('*.bin')):
        size = bin_file.stat().st_size
        stats['chunk_files'].append({'name': bin_file.name, 'size_bytes': size})
        stats['total_size_bytes'] += size

    return stats


def format_size(size_bytes: int | float) -> str:
    n = float(size_bytes)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} TB'


def print_audit_report(input_stats: dict, output_stats: dict, elapsed: float) -> None:
    sep = '-' * 60
    print(f'\n{sep}')
    print('TOKENIZATION AUDIT REPORT')
    print(sep)

    # Input summary
    print(f'\n  Input files:       {input_stats["total_files"]}')
    print(f'  Input entries:     {input_stats["total_entries"]}')
    if input_stats['skipped_empty'] > 0:
        print(f'  Skipped (empty):   {input_stats["skipped_empty"]}')

    for ext, type_stats in sorted(input_stats['by_type'].items()):
        print(f'    {ext:10s}  {type_stats["files"]:>4} files, {type_stats["entries"]:>8} entries')

    # Per-file breakdown
    print('\n  Per-file breakdown:')
    for fname, fstats in input_stats['files'].items():
        name = Path(fname).name
        print(f'    {name:40s}  {fstats["entries"]:>8} entries  ({format_size(fstats["size_bytes"])})')
        if fstats['skipped'] > 0:
            print(f'      {"":40s}  {fstats["skipped"]:>8} skipped (empty)')

    # Output summary
    print(f'\n  Output chunks:     {output_stats["total_chunks"]}')
    print(f'  Output items:      {output_stats["total_items"]}')
    print(f'  Output size:       {format_size(output_stats["total_size_bytes"])}')

    for cf in output_stats['chunk_files']:
        print(f'    {cf["name"]:40s}  {format_size(cf["size_bytes"])}')

    # Validation
    print(f'\n  Elapsed time:      {elapsed:.1f}s')
    if input_stats['total_entries'] == output_stats['total_items']:
        print('  Validation:        PASS (input entries == output items)')
    else:
        diff = input_stats['total_entries'] - output_stats['total_items']
        print(f'  Validation:        MISMATCH (input={input_stats["total_entries"]}, output={output_stats["total_items"]}, diff={diff})')

    print(sep)


def tokenize_file(fname: str, tokenizer: Tokenizer, text_column: str = 'text'):
    ext = Path(fname).suffix.lower()

    if ext == '.parquet':
        parquet_file = pq.ParquetFile(fname)
        for batch in parquet_file.iter_batches(columns=[text_column]):
            for value in batch.column(text_column):
                text = value.as_py()
                if text:
                    yield tokenizer.encode(text, bos=False, eos=False)
    elif ext == '.jsonl':
        with open(fname, encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get('text', '')
                if text:
                    yield tokenizer.encode(text, bos=False, eos=False)
    else:
        with open(fname, encoding='utf-8') as file:
            text = file.read()
        text = text.strip()
        if text:
            yield tokenizer.encode(text, bos=False, eos=False)


def tokenize_folder(
    input: str,
    output: str,
    tokenizer: Tokenizer,
    text_column: str = 'text',
    max_seq_length: int = 512,
) -> None:
    """
    Tokenizes a folder of .txt, .jsonl, and .parquet files into .bin files.

    Args:
        input: Path to the directory containing input files.
        output: Path where the tokenized .bin files will be saved.
        tokenizer: An initialized litgpt Tokenizer.
        text_column: Name of the text column in .parquet files.
        max_seq_length: The block size for the tokens loader.
    """
    input_path = Path(input)
    output_path = Path(output)

    # Gather all supported files
    all_files = sorted(
        glob.glob(str(input_path / '*.txt')) + glob.glob(str(input_path / '*.jsonl')) + glob.glob(str(input_path / '*.parquet'))
    )
    assert len(all_files) > 0, f'No .txt, .jsonl, or .parquet files found in {input_path}'

    if output_path.is_dir():
        print(f'Output directory {output_path} already exists. Please remove it or change the output path!')
        return

    # Pre-scan: count input entries
    print('Pre-scanning input files...')
    input_stats = count_input_entries(all_files, text_column=text_column)
    print(f'Found {input_stats["total_entries"]} entries across {input_stats["total_files"]} files')

    # Use max available CPUs (leaving 1 free)
    num_workers = max(1, os.cpu_count() - 1)
    use_workers = min(num_workers, len(all_files))

    print(f'Starting tokenization of {len(all_files)} files using {use_workers} workers...')
    start_time = time.monotonic()

    # Run the litdata optimization process
    optimize(
        fn=partial(tokenize_file, tokenizer=tokenizer, text_column=text_column),
        inputs=all_files,
        output_dir=str(output_path),
        num_workers=use_workers,
        chunk_bytes='500MB',
        item_loader=TokensLoader(block_size=max_seq_length + 1),  # +1 for the next token
    )

    elapsed = time.monotonic() - start_time

    # Post-scan: count output entries
    output_stats = count_output_chunks(output_path)

    # Print full audit report
    print_audit_report(input_stats, output_stats, elapsed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenizes a folder of .txt, .jsonl, or .parquet files into .bin files.')
    parser.add_argument('--tokenizer', type=str, default='model/tokenizer/', help='Path to the tokenizer directory.')
    parser.add_argument('--input', type=str, required=True, help='Path to the directory containing input files.')
    parser.add_argument('--output', type=str, required=True, help='Path where the tokenized .bin files will be saved.')
    parser.add_argument('--text-column', type=str, default='text', help='Name of the text column in .parquet files.')
    parser.add_argument('--max-seq-length', type=int, default=512, help='The block size for the tokens loader.')

    args = parser.parse_args()

    tokenizer = Tokenizer(checkpoint_dir=args.tokenizer)
    tokenize_folder(args.input, args.output, tokenizer, args.text_column, args.max_seq_length)
