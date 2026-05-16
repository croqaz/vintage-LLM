import argparse
import os

import lance
import orjson

BATCH_SIZE = 1_000_000


def cmd_export(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.db_path):
        raise SystemExit(f'Dataset not found: {args.db_path}')

    ds = lance.dataset(args.db_path)
    total_rows = ds.count_rows()
    print(f'Dataset: {args.db_path}  ({total_rows:,} rows)')

    # Determine columns to export (skip heavy vector columns by default)
    all_columns = [f.name for f in ds.schema]
    if args.columns:
        columns = [c.strip() for c in args.columns.split(',')]
    else:
        skip = {'minhash', 'lsh_bands', 'embed1'}
        columns = [c for c in all_columns if c not in skip]

    limit = args.limit if args.limit is not None else total_rows
    limit = min(limit, total_rows)

    out_path = args.output
    print(f'Exporting {limit:,} rows → {out_path}  (columns: {", ".join(columns)})')

    written = 0
    offset = 0
    with open(out_path, 'wb') as f:
        while written < limit:
            batch_limit = min(BATCH_SIZE, limit - written)
            tbl = ds.to_table(columns=columns, offset=offset, limit=batch_limit)
            rows = tbl.to_pylist()
            if not rows:
                break
            for row in rows:
                f.write(orjson.dumps(row) + b'\n')
            written += len(rows)
            offset += len(rows)
            print(f'  {written:,} / {limit:,} rows written')

    print(f'Done. {written:,} rows exported to {out_path}')
