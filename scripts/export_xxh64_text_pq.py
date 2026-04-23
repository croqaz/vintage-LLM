import glob
import sys

import lmdb
import pyarrow.dataset as ds


def export_parquet_glob_to_lmdb(parquet_glob, lmdb_dir, map_size_gb=16):
    # 1. Resolve the glob pattern to a list of files
    files = glob.glob(parquet_glob)
    if not files:
        print(f'No files found matching: {parquet_glob}')
        return

    print(f'Found {len(files)} files to process.')

    # 2. Open LMDB with a safer default map size (e.g., 16GB instead of 1TB)
    # If the DB exceeds this, LMDB will throw a specific MapFullError.
    map_size_bytes = map_size_gb * 1024 * 1024 * 1024
    env = lmdb.open(lmdb_dir, map_size=map_size_bytes, writemap=True)

    inserted = 0
    duplicates_skipped = 0

    # 3. Process files individually so we can track exactly where a crash happens
    for file_idx, file_path in enumerate(files, 1):
        print(f'Processing [{file_idx}/{len(files)}]: {file_path}')

        try:
            # Load only the specific file
            dataset = ds.dataset(file_path, format='parquet')

            # Force a maximum batch size to prevent RAM spikes (e.g., 50,000 rows at a time)
            for batch in dataset.to_batches(columns=['xxh64', 'text'], batch_size=50_000):
                keys = batch['xxh64'].to_pylist()
                texts = batch['text'].to_pylist()

                with env.begin(write=True) as txn:
                    for k, t in zip(keys, texts):
                        # Safely handle potential nulls/missing data
                        if k is None:
                            continue

                        key_bytes = str(k).encode('utf-8')
                        text_str = str(t) if t is not None else ''
                        val_bytes = text_str[:100].encode('utf-8', errors='ignore')

                        if txn.put(key_bytes, val_bytes, overwrite=False):
                            inserted += 1
                        else:
                            duplicates_skipped += 1

        except Exception as e:
            print(f'Error processing {file_path}: {e}', file=sys.stderr)
            # We continue to the next file instead of failing the whole job
            continue

    env.close()

    print('\n--- Export Complete ---')
    print(f'Unique records stored: {inserted:,}')
    print(f'Duplicates skipped:    {duplicates_skipped:,}')


if __name__ == '__main__':
    export_parquet_glob_to_lmdb('/data/AI-Dataset/Vintage-All/*.parquet', 'brit_dedupe.lmdb')
