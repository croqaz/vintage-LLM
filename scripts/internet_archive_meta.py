import argparse
import json
import sys

import internetarchive as ia


def main():
    parser = argparse.ArgumentParser(description='Download full metadata for archive.org items.')
    parser.add_argument('input_file', help='Text file with IDs, separated by newline or space')
    parser.add_argument('output_file', help='Output JSON lines file')

    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f'Error: Could not find input file: {args.input_file}', file=sys.stderr)
        sys.exit(1)

    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        ids = [i for i in content.split() if i]
        session = ia.get_session()
        for idx, item_id in enumerate(ids):
            try:
                item = session.get_item(item_id)
                if item.exists:
                    metadata = item.item_metadata
                    if 'alternate_locations' in metadata:
                        del metadata['alternate_locations']  # Remove to save space
                    if 'workable_servers' in metadata:
                        del metadata['workable_servers']  # Remove to save space
                    if 'server' in metadata:
                        del metadata['server']
                    if 'd1' in metadata:
                        del metadata['d1']
                    if 'd2' in metadata:
                        del metadata['d2']
                    if 'd3' in metadata:
                        del metadata['d3']
                    if 'page_numbers' in metadata:
                        del metadata['page_numbers']
                    if 'reviews' in metadata:
                        del metadata['reviews']
                    if 'events' in metadata:
                        del metadata['events']
                    out_f.write(json.dumps(metadata) + '\n')
                    print(f'[{idx + 1}/{len(ids)}] Downloaded metadata for {item_id}', file=sys.stderr)
                else:
                    print(f'[{idx + 1}/{len(ids)}] Item not found: {item_id}', file=sys.stderr)
            except Exception as e:
                print(f'[{idx + 1}/{len(ids)}] Error downloading {item_id}: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
