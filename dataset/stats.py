import argparse
import os

import lance
import numpy as np

STAT_COLUMNS = [
    ('length', 'Char length'),
    ('words', 'Word count'),
    ('sentences', 'Sentence count'),
    ('unique_chars', 'Unique chars'),
    ('quality_score', 'Quality score'),
    ('compression_ratio', 'Compression ratio'),
    ('char_entropy', 'Char entropy'),
]


def _fmt(v: float | np.floating) -> str:
    """Format a number: integers get commas, floats get 4 decimal places."""
    if abs(v) >= 1 and v == int(v):
        return f'{int(v):,}'
    if abs(v) >= 100:
        return f'{v:,.1f}'
    return f'{v:.4f}'


def _print_stats_table(ds: lance.LanceDataset) -> None:
    cols = {name: np.array(ds.to_table(columns=[name]).column(name).to_pylist(), dtype=np.float64) for name, _ in STAT_COLUMNS}

    metrics: list[tuple[str, np.ndarray]] = [(label, cols[col]) for col, label in STAT_COLUMNS]
    headers = ['Metric', 'Mean', 'Median', 'Std', 'Min', 'P5', 'P95', 'Max']
    rows: list[list[str]] = []
    for label, arr in metrics:
        rows.append(
            [
                label,
                _fmt(np.mean(arr)),
                _fmt(np.median(arr)),
                _fmt(np.std(arr)),
                _fmt(np.min(arr)),
                _fmt(np.percentile(arr, 5)),
                _fmt(np.percentile(arr, 95)),
                _fmt(np.max(arr)),
            ]
        )

    # Calculate column widths
    widths = [max(len(headers[c]), *(len(r[c]) for r in rows)) for c in range(len(headers))]

    # Build table
    hdr = '| ' + ' | '.join(h.ljust(widths[i]) if i == 0 else h.rjust(widths[i]) for i, h in enumerate(headers)) + ' |'
    sep = '|-' + '-|-'.join('-' * widths[i] if i == 0 else '-' * widths[i] for i in range(len(headers))) + '-|'
    # Right-align separator for numeric columns
    sep = '|-' + '-|-'.join('-' * widths[0] if i == 0 else '-' * (widths[i] - 1) + ':' for i in range(len(headers))) + '-|'

    print(f'\n{hdr}')
    print(sep)
    for row in rows:
        line = '| ' + ' | '.join(row[i].ljust(widths[i]) if i == 0 else row[i].rjust(widths[i]) for i in range(len(headers))) + ' |'
        print(line)
    print()


# ─────────────────────────────────────────────────────────
# Subcommand: stats
# ─────────────────────────────────────────────────────────
def cmd_stats(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.db_path):
        raise SystemExit(f'Dataset not found: {args.db_path}')

    ds = lance.dataset(args.db_path)
    print(f'Path: {args.db_path}')
    print(f'\nSchema:\n{ds.schema}')
    print(f'\nTotal rows:  {ds.count_rows():,}')
    print(f'Fragments:   {len(ds.get_fragments()):,}')
    print(f'Version:     {ds.version}')

    indices = ds.list_indices()
    if indices:
        print(f'\nIndexes ({len(indices)}):')
        for idx in indices:
            print(f'  {idx}')
    else:
        print('\nNo indexes.')

    _print_stats_table(ds)
