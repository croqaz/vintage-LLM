"""Single-token coverage of the most frequent words from words.json.

For each tokenizer: the share of the top-N words that encode to exactly
one token (higher is better), bare and with a leading space (Ġ form,
i.e. how the word appears in running text). 'weighted' weighs each word
by its corpus count, so it approximates the share of running-text word
occurrences that need only one token.
"""

import json

from benchmark_tok import TOKENIZERS, load_tokenizer

WORDS_FILE = 'words.json'
TOP_N = 100_000
CUTOFFS = [1_000, 10_000, 32_752, 100_000]


def single_token_flags(tokenizer, words, prefix):
    encoded = tokenizer([prefix + w for w in words], add_special_tokens=False)
    return [len(ids) == 1 for ids in encoded['input_ids']]


def main():
    counts = json.load(open(WORDS_FILE))
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N]
    words = [w for w, _ in top]
    weights = [c for _, c in top]

    header = f'{"tokenizer":<10} {"variant":<9}' + ''.join(f'{f"top {n // 1000}k":>10}' for n in CUTOFFS) + f'{"weighted":>10}'
    print(header)
    print('-' * len(header))
    for name, path in TOKENIZERS.items():
        tokenizer = load_tokenizer(path)
        for prefix, tag in (('', 'bare'), (' ', 'Ġ-prefix')):
            flags = single_token_flags(tokenizer, words, prefix)
            row = ''.join(f'{sum(flags[:n]) / n:>9.2%} ' for n in CUTOFFS)
            weighted = sum(w for f, w in zip(flags, weights) if f) / sum(weights)
            print(f'{name:<10} {tag:<9}' + row + f'{weighted:>9.2%} ')


if __name__ == '__main__':
    main()
