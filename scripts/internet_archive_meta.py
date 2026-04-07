import argparse
import concurrent.futures
import json
import sys

import internetarchive as ia

DROP_KEYS = ['alternate_locations', 'workable_servers', 'server', 'd1', 'd2', 'd3', 'page_numbers', 'reviews', 'events']


def download_metadata(item_id):
    try:
        item = ia.get_item(item_id)
        if item.exists:
            metadata = item.item_metadata
            for key in DROP_KEYS:
                metadata.pop(key, None)
            return metadata, None
        else:
            return None, 'Item not found'
    except Exception as err:
        return None, str(err)


def main():
    parser = argparse.ArgumentParser(description='Download full metadata for archive.org items.')
    parser.add_argument('input_file', help='Text file with IDs, separated by newline or space')
    parser.add_argument('output_file', help='Output JSON lines file')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f'Error: Could not find input file: {args.input_file}', file=sys.stderr)
        sys.exit(1)

    ids = [i for i in content.split() if i]

    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_id = {executor.submit(download_metadata, item_id): item_id for item_id in ids}
            for future in concurrent.futures.as_completed(future_to_id):
                item_id = future_to_id[future]
                completed += 1
                try:
                    metadata, error = future.result()
                    if metadata:
                        out_f.write(json.dumps(metadata) + '\n')
                        print(f'[{completed}/{len(ids)}] Downloaded metadata for {item_id}', file=sys.stderr)
                    else:
                        print(f'[{completed}/{len(ids)}] Error downloading {item_id}: {error}', file=sys.stderr)
                except Exception as exc:
                    print(f'[{completed}/{len(ids)}] Error downloading {item_id}: {exc}', file=sys.stderr)


if __name__ == '__main__':
    main()
