import argparse
import os

import lance


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
