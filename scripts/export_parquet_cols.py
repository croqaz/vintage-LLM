import argparse
import glob
import json
import sys

import pyarrow.dataset as ds

try:
    import orjson

    def _dumps(obj):
        return orjson.dumps(obj).decode()
except ImportError:
    _dumps = json.dumps


def export_parquet_glob_to_jsonl(
    parquet_glob,
    columns,
    output_jsonl_path='output.jsonl',
    batch_size=100_000,
):
    files = sorted(glob.glob(parquet_glob))
    if not files:
        print(f'No files found matching: {parquet_glob}')
        return 0

    print(f'Found {len(files)} files to process.')
    print(f'Exporting columns: {columns if columns else "ALL"}')
    exported = 0

    # 8 MB write buffer to reduce syscall overhead
    with open(output_jsonl_path, 'w', buffering=8 * 1024 * 1024) as output_file:
        for file_idx, file_path in enumerate(files, 1):
            print(f'Processing [{file_idx}/{len(files)}]: {file_path}', flush=True)
            try:
                dataset = ds.dataset(file_path, format='parquet')
                for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
                    col_data = batch.to_pydict()
                    col_names = list(col_data.keys())
                    rows = zip(*[col_data[c] for c in col_names])
                    output_file.write('\n'.join(_dumps(dict(zip(col_names, row))) for row in rows) + '\n')
                    exported += batch.num_rows
            except Exception as e:
                print(f'Error processing {file_path}: {e}', file=sys.stderr)
                continue

    print('\n--- Export Complete ---')
    print(f'Records exported: {exported:,}')
    return exported


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Parquet columns to a JSON Lines file.')
    parser.add_argument('parquet_glob', help='Glob pattern for input Parquet files (quote it to prevent shell expansion)')
    parser.add_argument(
        '-c',
        '--columns',
        nargs='+',
        default=None,
        help='Columns to export (default: all columns)',
    )
    parser.add_argument(
        '-o',
        '--output',
        default='output.jsonl',
        help='Output JSONL file path (default: output.jsonl)',
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=100_000,
        help='Rows per batch (default: 100000)',
    )
    args = parser.parse_args()
    export_parquet_glob_to_jsonl(
        args.parquet_glob,
        columns=args.columns,
        output_jsonl_path=args.output,
        batch_size=args.batch_size,
    )
