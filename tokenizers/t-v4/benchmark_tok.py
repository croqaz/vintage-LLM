"""Fertility / continued-words benchmark across tokenizer versions.

SmolLM-style metrics: words come from nltk.word_tokenize, each word is
encoded on its own with add_special_tokens=False; fertility is the mean
number of tokens per word, continued is the share of words split into
2+ tokens. Lower is better for both. Samples are drawn deterministically
from the original (uncleaned) dataset files.

Each metric is computed in two variants:
  bare     - the word exactly as SmolLM's algorithm encodes it ("profound")
  Ġ-prefix - with a leading space ("Ġprofound"), which is how nearly every
             word occurs in running text under a ByteLevel pre-tokenizer
"""

import os
import random
from collections import Counter

from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, PreTrainedTokenizerFast

SEED = 42
CHUNKS_PER_FILE = 64
CHUNK_BYTES = 64 * 1024
DATASETS = ['dataset-text1.txt', 'dataset-text2.txt', 'dataset-text3.txt', 'dataset-text4.txt']
TOKENIZERS = {'violet': 'violet', 'v2': 't-v2', 'v3': 't-v3', 'v4': 'fast_tok'}


def load_tokenizer(path):
    if path.endswith('.json'):
        return PreTrainedTokenizerFast(tokenizer_file=path)
    return AutoTokenizer.from_pretrained(path)


def sample_text(path):
    rng = random.Random(f'{SEED}:{path}')
    size = os.path.getsize(path)
    parts = []
    with open(path, 'rb') as f:
        for _ in range(CHUNKS_PER_FILE):
            f.seek(rng.randrange(0, max(1, size - CHUNK_BYTES)))
            raw = f.read(CHUNK_BYTES)
            head = raw.find(b'\n') + 1
            tail = raw.rfind(b'\n')
            if tail > head:
                parts.append(raw[head:tail].decode('utf-8', errors='ignore'))
    return '\n'.join(parts)


def compute_metrics(tokenizer, word_counts, prefix=''):
    """Weighted form of SmolLM's per-word-instance mean: identical result,
    but each unique word is encoded once."""
    encoded = tokenizer([prefix + w for w in word_counts], add_special_tokens=False)
    total_words = sum(word_counts.values())
    total_tokens = 0
    continued = 0
    for n, ids in zip(word_counts.values(), encoded['input_ids']):
        total_tokens += len(ids) * n
        if len(ids) >= 2:
            continued += n
    return total_tokens / total_words, continued / total_words


def print_report(tokenizers, per_file):
    header = f'{"dataset":<20} {"metric":<22}' + ''.join(f'{name:>10}' for name in tokenizers)
    print(header)
    print('-' * len(header))
    for path, counts in per_file.items():
        for prefix, tag in (('', 'bare'), (' ', 'Ġ-prefix')):
            results = {name: compute_metrics(tok, counts, prefix) for name, tok in tokenizers.items()}
            print(f'{path:<20} {"fertility " + tag:<22}' + ''.join(f'{results[n][0]:>10.4f}' for n in tokenizers))
            print(f'{"":<20} {"continued " + tag:<22}' + ''.join(f'{results[n][1]:>9.2%} ' for n in tokenizers))


def collect_word_counts():
    overall = Counter()
    per_file = {}
    for path in DATASETS:
        counts = Counter(word_tokenize(sample_text(path)))
        per_file[path] = counts
        overall += counts
    per_file['OVERALL'] = overall
    return per_file


def main():
    tokenizers = {name: load_tokenizer(path) for name, path in TOKENIZERS.items()}
    per_file = collect_word_counts()
    print_report(tokenizers, per_file)
    overall = per_file['OVERALL']
    print(f'\nwords sampled: {sum(overall.values()):,} ({len(overall):,} unique)')


if __name__ == '__main__':
    main()
