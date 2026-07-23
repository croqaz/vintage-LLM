"""Tests for the conservative OCR cleaner."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ocr_clean as o


def test_joins_hyphen_linebreaks():
    assert o.clean('char-\nacter') == 'character'
    # chained splits need multiple passes
    assert o.clean('a-\nb-\nc') == 'abc'


def test_strips_junk_but_keeps_legit_typography():
    out = o.clean('Pro^mium and | pipes • bullets ¬ marks')
    assert '^' not in out and '|' not in out and '•' not in out and '¬' not in out
    # legitimate glyphs survive
    kept = o.clean('a fine dash — an é, a £5 note, œuvre, § 3')
    for ch in '—é£œ§':
        assert ch in kept


def test_fixes_spaced_punctuation():
    assert o.clean('itself ; and here , too .') == 'itself; and here, too.'


def test_collapses_whitespace_by_default():
    assert o.clean('one\n\ntwo   three\tfour') == 'one two three four'


def test_keep_paragraphs_preserves_breaks():
    out = o.clean('para one\nstill one\n\npara two', keep_paragraphs=True)
    assert out == 'para one still one\n\npara two'


def test_aggressive_strips_braces():
    assert '{' not in o.clean('a {b} c', aggressive=True)
    # non-aggressive leaves braces alone
    assert '{' in o.clean('a {b} c', aggressive=False)


def test_empty_and_whitespace():
    assert o.clean('') == ''
    assert o.clean('   \n\t ') == ''


def test_nfkc_normalisation():
    # ligature fi (U+FB01) folds to "fi"
    assert o.clean('ﬁne') == 'fine'


if __name__ == '__main__':
    import pytest

    raise SystemExit(pytest.main([__file__, '-q']))
