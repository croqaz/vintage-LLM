"""Tests for detect.py — the pre-1900 scorer (positive + negative)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect import Scorer, tokenize

THRESHOLD = 0.75

# Should be KEPT at the default threshold (genuinely pre-1900-sounding, clean)
KEEP = [
    'It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.',
    'Whither goest thou, fair maiden, ere the sun doth set upon the moor?',
    'The cavalry advanced at dawn, their muskets gleaming in the pale light.',
    'He took a steamship across the Atlantic and sent a telegraph upon arrival.',
    'The harvest was poor this year, and the villagers feared a hard winter.',
]

# Should be DROPPED (modern style, vocabulary, concepts, or post-1900 years)
DROP = [
    'I just posted a selfie online and it went viral on social media.',
    'The CPU overheated so I rebooted my laptop and reinstalled the software.',
    'World War II ended in 1945 after the atomic bombs were dropped.',
    'Scientists sequenced the DNA and analyzed it with a computer.',
    'Once upon a time, there was a little girl named Lily. She was very happy and liked to play.',
    'The factory moved to Libertyville in 1906.',  # year veto
]


def run():
    s = Scorer()
    fails = 0
    for t in KEEP:
        tokens = tokenize(t)
        p = s.score(t, tokens)['p_pre1900']
        if p < THRESHOLD:
            print(f'  FAIL (expected KEEP, p={p:.2f}): {t[:55]!r}')
            fails += 1
    for t in DROP:
        tokens = tokenize(t)
        p = s.score(t, tokens)['p_pre1900']
        if p >= THRESHOLD:
            print(f'  FAIL (expected DROP, p={p:.2f}): {t[:55]!r}')
            fails += 1

    # knob behaviour: a modern-PHRASED but concept-clean line is dropped by
    # default, but KEPT in concepts-only mode (style & marker weights = 0).
    timeless_modern = (
        'Once there was a little boy named Jack. One day he saw a big wheel in the park and ran to it, laughing as he turned it around.'
    )
    assert s.score(timeless_modern, tokenize(timeless_modern))['p_pre1900'] < THRESHOLD, 'should drop by default'
    relaxed = s.score(timeless_modern, tokenize(timeless_modern), style_weight=0.0, marker_weight=0.0)['p_pre1900']
    assert relaxed >= 0.6, f'concepts-only should keep it (got {relaxed:.2f})'

    # a concept hit cannot be rescued by relaxing the lexical knobs
    assert (
        s.score(
            'We launched a rocket in 1969.',
            tokenize('We launched a rocket in 1969.'),
            style_weight=0.0,
            marker_weight=0.0,
        )['p_pre1900']
        <= 0.03
    ), 'banned term / year must still veto in concepts-only mode'

    total = len(KEEP) + len(DROP)
    print(f'detect: {total - fails}/{total} cases passed (+ 3 knob assertions)')
    return fails


if __name__ == '__main__':
    sys.exit(1 if run() else 0)
