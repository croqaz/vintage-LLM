"""Tests for the generated-text quality scorer."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import score_sample as s


def test_meta_flag():
    assert s.flags('This document discusses various incidents.')['meta']
    assert s.flags('The passage argues that...')['meta']
    # a genuine continuation is not meta
    assert not s.flags('In the quiet halls, the orator continued his address.')['meta']


def test_ocr_flag_detects_junk_but_not_legit():
    assert s.flags('the same char^acter as before')['ocr']
    assert not s.flags('a fine dash — and an é, worth £5')['ocr']


def test_leak_and_anachronism():
    assert s.flags('As an AI, I cannot continue this text.')['leak']
    assert s.flags('He picked up the telephone and called.')['anach']
    assert not s.flags('He picked up his quill and wrote.')['anach']


def test_meta_catches_original_and_analysis_openings():
    # these were missed before broadening the regex
    assert s.flags('The original text provided offers insights into the parts.')['meta']
    assert s.flags('The language in this passage is marked by formal diction.')['meta']
    assert s.flags('The following excerpt describes an oration.')['meta']


def test_preamble_flag():
    assert s.flags('As a continuation of this passage: In the halls...')['preamble']
    assert s.flags("Sure — I'll provide one based on that. In the halls...")['preamble']
    assert s.flags("...since you're asking for a continuation, here it is.")['preamble']
    # a clean continuation has no preamble chatter
    assert not s.flags('In the quiet halls the orator carried on his address.')['preamble']


def test_empty_flag():
    assert s.flags('   ')['empty']
    assert s.flags('short')['empty']
    assert not s.flags('this is a sufficiently long clean output line')['empty']


def test_summarize_and_grouping(tmp_path):
    recs = [
        {
            'prompt': 'continue',
            'model': 'm',
            'params': {'mode': 'chat'},
            'text': 'In the quiet halls the orator carried his address onward at length.',
        },
        {'prompt': 'continue', 'model': 'm', 'params': {'mode': 'chat'}, 'text': 'This document discusses the plan in detail.'},
    ]
    p = tmp_path / 'o.jsonl'
    p.write_text('\n'.join(json.dumps(r) for r in recs) + '\n', encoding='utf-8')
    groups = s.score_files([str(p)])
    assert len(groups) == 1
    summ = s.summarize(next(iter(groups.values())))
    assert summ['n'] == 2
    assert summ['good_pct'] == 50.0  # one clean, one meta
    assert summ['rates']['meta'] == 50.0


if __name__ == '__main__':
    import pytest

    raise SystemExit(pytest.main([__file__, '-q']))
