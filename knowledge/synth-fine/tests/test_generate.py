"""Offline unit tests for generate.py (no live LLM server required).

Run:  python -m pytest tests/ -q      (or: python tests/test_generate.py)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate as g


def test_render_is_literal_and_brace_safe():
    tmpl = 'Continue:\n{text}\nEND'
    chunk = 'some data with {curly} braces and 100% signs'
    out = g.render(tmpl, chunk)
    assert chunk in out
    assert out == 'Continue:\nsome data with {curly} braces and 100% signs\nEND'


def test_chunk_respects_max_chunks():
    text = 'word ' * 20000  # far more than a few chunks
    chunks = g.chunk_text(text, chunk_tokens=500, max_chunks=3)
    assert len(chunks) == 3
    assert all(c.strip() for c in chunks)


def test_chunk_empty_returns_nothing():
    assert g.chunk_text('   \n  ', chunk_tokens=100, max_chunks=3) == []


def test_short_text_is_single_chunk():
    chunks = g.chunk_text('A brief passage.', chunk_tokens=3000, max_chunks=3)
    assert len(chunks) == 1
    assert chunks[0] == 'A brief passage.'


def test_iter_seeds_offset_and_limit(tmp_path):
    p = tmp_path / 'seeds.jsonl'
    lines = [json.dumps({'text': f'doc number {i}'}) for i in range(10)]
    lines.insert(3, '')  # blank line should be skipped
    lines.insert(5, '{not valid json')  # malformed should be skipped
    p.write_text('\n'.join(lines), encoding='utf-8')
    got = list(g.iter_seeds(str(p), offset=2, limit=3))
    assert len(got) == 3
    # indices are original file line numbers; texts are well-formed
    assert all(isinstance(idx, int) and txt.startswith('doc number') for idx, txt in got)


def test_build_tasks_cross_product_and_dedup():
    seeds = [(0, 'alpha beta gamma'), (1, 'delta epsilon')]
    templates = {'cont': 'X {text}', 'sum': 'Y {text}'}
    tasks = list(g.build_tasks(seeds, templates, chunk_tokens=3000, max_chunks=3, done=set()))
    # 2 seeds * 1 chunk each * 2 templates
    assert len(tasks) == 4
    done = {('cont', 0, 0)}
    tasks2 = list(g.build_tasks(seeds, templates, chunk_tokens=3000, max_chunks=3, done=done))
    assert len(tasks2) == 3
    assert all(not (t.prompt_name == 'cont' and t.seed_index == 0) for t in tasks2)


def test_load_done_reads_identity(tmp_path):
    p = tmp_path / 'out.jsonl'
    recs = [
        {'prompt': 'cont', 'seed_index': 0, 'chunk_index': 0, 'text': '...'},
        {'prompt': 'sum', 'seed_index': 4, 'chunk_index': 1, 'text': '...'},
    ]
    p.write_text('\n'.join(json.dumps(r) for r in recs) + '\n', encoding='utf-8')
    done = g.load_done(str(p))
    assert ('cont', 0, 0) in done and ('sum', 4, 1) in done
    assert g.load_done(str(tmp_path / 'missing.jsonl')) == set()


def test_load_templates_skips_missing_placeholder(tmp_path):
    good = tmp_path / 'good.txt'
    bad = tmp_path / 'bad.txt'
    good.write_text('do the thing to {text}', encoding='utf-8')
    bad.write_text('no placeholder here', encoding='utf-8')
    tmpls = g.load_templates([str(good), str(bad)])
    assert 'good' in tmpls and 'bad' not in tmpls


if __name__ == '__main__':
    import pytest

    raise SystemExit(pytest.main([__file__, '-q']))
