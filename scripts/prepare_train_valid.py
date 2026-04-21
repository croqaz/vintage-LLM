import argparse
import json
import os
import re

BOS_TOKEN = '<|bos|>'
EOS_TOKEN = '<|eos|>'


def find_nearest_sentence_boundary(text: str, target_index: int, window: int = 10000) -> int:
    """
    Finds the nearest sentence boundary to the target_index to avoid breaking sentences.
    Searches backwards and forwards within a window and picks the closest one.
    """
    if target_index <= 0:
        return 0
    if target_index >= len(text):
        return len(text)

    start_window = max(0, target_index - window)
    end_window = min(len(text), target_index + window)

    # Search backward
    backward_match = None
    for match in re.finditer(r'[.!?][ \n\r]+', text[start_window:target_index]):
        backward_match = start_window + match.end()

    # Search forward
    forward_match = re.search(r'[.!?][ \n\r]+', text[target_index:end_window])
    forward_pos = target_index + forward_match.end() if forward_match else None

    # Pick the closest one
    if backward_match and forward_pos:
        if (target_index - backward_match) <= (forward_pos - target_index):
            return backward_match
        else:
            return forward_pos
    elif backward_match:
        return backward_match
    elif forward_pos:
        return forward_pos

    # Fallback if no sentence boundary is found in the window
    return target_index


def is_jsonl(filepath: str) -> bool:
    """Return True if the first non-empty line of the file parses as a JSON object with a 'text' key."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    return isinstance(obj, dict) and 'text' in obj
                except (json.JSONDecodeError, ValueError):
                    return False
    return False


def process_text(input_file: str, output_dir: str, remove_original: bool) -> None:
    print(f'Reading {input_file}...')
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    total_len = len(text)
    if total_len == 0:
        print(f'Error: The file {input_file} is empty. Skipping...')
        return

    if len(text) > 50000 and not text.startswith(BOS_TOKEN):
        text = BOS_TOKEN + '\n' + text
    if len(text) > 50000 and not text.endswith(EOS_TOKEN):
        text = text + '\n' + EOS_TOKEN

    target_val_start = int(total_len * 0.45)
    target_val_end = int(total_len * 0.55)

    val_start = find_nearest_sentence_boundary(text, target_val_start)
    val_end = find_nearest_sentence_boundary(text, target_val_end)

    print('Extracting validation data from the middle...')
    val_text = text[val_start:val_end]
    train_text = text[:val_start] + text[val_end:]

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    train_path = os.path.join(output_dir, f'{base_name}-train.txt')
    val_path = os.path.join(output_dir, f'{base_name}-val.txt')

    print(f'Total length: {total_len:,} characters')
    print(f'Validation range: character {val_start:,} to {val_end:,} ({len(val_text):,} chars, {len(val_text) / total_len * 100:.2f}%)')
    print(f'Train split combines the start and end portions ({len(train_text):,} chars, {len(train_text) / total_len * 100:.2f}%)')

    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_text)
    print(f'--> Saved train file: {train_path}')

    with open(val_path, 'w', encoding='utf-8') as f:
        f.write(val_text)
    print(f'--> Saved validation file: {val_path}\n')

    if remove_original:
        os.remove(input_file)
        print(f'--> Removed original file: {input_file}')


def process_jsonl(input_file: str, output_dir: str, remove_original: bool) -> None:
    print(f'Reading {input_file} (JSON lines)...')
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    if total == 0:
        print(f'Error: The file {input_file} is empty. Skipping...')
        return

    for record in records:
        text = record.get('text', '')
        if len(text) > 50000:
            if not text.startswith(BOS_TOKEN):
                text = BOS_TOKEN + '\n' + text
            if not text.endswith(EOS_TOKEN):
                text = text + '\n' + EOS_TOKEN
            record['text'] = text

    val_start = int(total * 0.45)
    val_end = int(total * 0.55)

    print('Extracting validation data from the middle...')
    val_records = records[val_start:val_end]
    train_records = records[:val_start] + records[val_end:]

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    train_path = os.path.join(output_dir, f'{base_name}-train.jsonl')
    val_path = os.path.join(output_dir, f'{base_name}-val.jsonl')

    print(f'Total records: {total:,}')
    print(f'Validation range: record {val_start:,} to {val_end:,} ({len(val_records):,} records, {len(val_records) / total * 100:.2f}%)')
    print(f'Train split combines the start and end portions ({len(train_records):,} records, {len(train_records) / total * 100:.2f}%)')

    with open(train_path, 'w', encoding='utf-8') as f:
        for record in train_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f'--> Saved train file: {train_path}')

    with open(val_path, 'w', encoding='utf-8') as f:
        for record in val_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f'--> Saved validation file: {val_path}\n')

    if remove_original:
        os.remove(input_file)
        print(f'--> Removed original file: {input_file}')


def main():
    parser = argparse.ArgumentParser(description='Extract validation text (middle 10%) and train text (remaining 90%) from text files.')
    parser.add_argument('input_files', nargs='+', help='Paths to the input text files (e.g., data/book-1.txt data/books-2.jsonl)')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory where output files will be saved. Defaults to input file directory.',
    )
    parser.add_argument(
        '--remove-original',
        action='store_true',
        help='Remove the original input file after it was split into train and validation.',
    )

    args = parser.parse_args()

    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Error: '{input_file}' does not exist.")
            continue

        output_dir = args.output_dir if args.output_dir else os.path.dirname(input_file)
        if output_dir == '':
            output_dir = '.'

        if is_jsonl(input_file):
            process_jsonl(input_file, output_dir, args.remove_original)
        else:
            process_text(input_file, output_dir, args.remove_original)


if __name__ == '__main__':
    main()
