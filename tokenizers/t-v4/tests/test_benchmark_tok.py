import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_tok import compute_metrics


class FakeTokenizer:
    """Splits a word into one token per character-pair, e.g. 'abcd' -> 2 tokens."""

    def __init__(self, splits):
        self.splits = splits

    def __call__(self, words, add_special_tokens):
        assert add_special_tokens is False
        return {'input_ids': [list(range(self.splits[w])) for w in words]}


def test_metrics_match_per_instance_mean():
    # 'the' x3 -> 1 token each, 'darcy' x1 -> 2 tokens: 5 tokens / 4 words
    counts = Counter({'the': 3, 'darcy': 1})
    tok = FakeTokenizer({'the': 1, 'darcy': 2})
    fertility, continued = compute_metrics(tok, counts)
    assert fertility == pytest.approx(5 / 4)
    assert continued == pytest.approx(1 / 4)


def test_all_single_token_words():
    counts = Counter({'a': 10, 'b': 5})
    tok = FakeTokenizer({'a': 1, 'b': 1})
    fertility, continued = compute_metrics(tok, counts)
    assert fertility == 1.0
    assert continued == 0.0
