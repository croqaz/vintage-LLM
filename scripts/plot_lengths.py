import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

BINS = 128


def main():
    # Path to the Lengths.jsonl file
    current_dir = Path(__file__).parent
    lengths_file = current_dir.parent / 'Lengths.jsonl'

    print(f'Reading {lengths_file}...')

    # Aggregate lengths into a Counter to save memory
    length_counts = collections.Counter()

    total_lines = 0
    # Optional: get total size to show rough progress, though tqdm on file stream is better
    with open(lengths_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Processing lengths'):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data['len'] > 10_000:
                    continue
                length_counts[data['len']] += 1
                total_lines += 1
            except Exception:
                continue

    print(f'Processed {total_lines} valid lengths.')

    # Prepare data for plotting
    max_len = max(length_counts.keys()) if length_counts else 0
    min_len = min(length_counts.keys()) if length_counts else 0

    print(f'Min length: {min_len}, Max length: {max_len}')

    # Create bins for histogram (e.g. 100 bins)
    bin_width = (max_len - min_len) / BINS if BINS > 0 else 1
    if bin_width == 0:
        bin_width = 1

    # Bin the data
    lengths = []
    for length, count in length_counts.items():
        lengths.extend([length] * count)  # This might take memory if we used it in plt.hist, but wait plt.hist takes raw data

    print('Generating plot...')
    plt.figure(figsize=(12, 6))

    # If the lengths list is too large for plt.hist, we can use plt.bar by computing bins ourselves,
    # but 35M lengths in memory is only ~280MB, which matplotlib can handle.
    # To be safe and fast, we can plot a binned bar chart manually

    # Use numpy for fast histogram computing from the frequency dictionary
    items = list(length_counts.items())
    vals = np.array([x[0] for x in items])
    weights = np.array([x[1] for x in items])

    plt.hist(vals, bins=BINS, weights=weights, color='skyblue', edgecolor='black')
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    plot_file = current_dir.parent / 'lengths_distribution.png'
    plt.savefig(plot_file, dpi=600, bbox_inches='tight')
    print(f'Plot saved to {plot_file}')


if __name__ == '__main__':
    main()
