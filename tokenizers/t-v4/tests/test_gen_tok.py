import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gen_tok import clean_text


@pytest.mark.parametrize(
    ('raw', 'cleaned'),
    [
        # single-char runs collapse to exactly 3
        ('aaaaaaaaa', 'aaa'),
        ('~~~~~~~~', '~~~'),
        ('----------------', '---'),
        ('====', '==='),
        ('................', '...'),
        ('################', '###'),
        ('________________', '___'),
        # mixed table-border runs collapse to 3x the dominant char
        ('+--------', '---'),
        ('-----+', '---'),
        ('---+---+', '---'),
        ('+----+----+', '---'),
        ('.=.=.=.=', '...'),
        # repeated punctuation units collapse to one unit
        ('?!?!?!', '?!'),
        ('()()()', '()'),
        # runs up to 3 are kept as-is
        ('-', '-'),
        ('--', '--'),
        ('---', '---'),
        ('##', '##'),
        ('===', '==='),
        ('...', '...'),
        # runs never span lines, so per-line rules survive
        ('---\n---\n---\n---', '---\n---\n---\n---'),
    ],
)
def test_pattern_collapsing(raw, cleaned):
    assert clean_text(raw) == cleaned


@pytest.mark.parametrize(
    'text',
    [
        'Hello, how are you ?? :)',
        "Mr. Darcy said, 'No.'",
        'the U.S.A. in 1899, e.g. cost 1.50',
        'It was a dark and stormy night; the rain fell in torrents.',
        'CHAPTER III',
    ],
)
def test_normal_text_untouched(text):
    assert clean_text(text) == text


def test_runs_inside_sentences():
    assert clean_text('Total ...... 42') == 'Total ... 42'
    assert clean_text('word -------- word') == 'word --- word'
