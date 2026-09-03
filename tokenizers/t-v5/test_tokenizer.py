"""v7 tokenizer tests — ship these with the folder.

Self-contained: every path is relative to this file, so the suite runs
wherever the folder is published (`pytest t-v7/`). The tokenizer-contract
tests skip until tokenizer.json is built; the prepare.py tests run always.

One deliberate charset choice: eight common OCR-era symbols
(§ © « » ½ • ™ ■ — the intersection of the out-of-domain character sets of
two published period tokenizers) are KEPT by the cleaning pipeline instead
of deleted, so the tokenizer may learn them.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

FOLDER = Path(__file__).resolve().parent
TOK = FOLDER / 'tokenizer.json'


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, FOLDER / filename)
    module = importlib.util.module_from_spec(spec)
    # register BEFORE exec so multiprocessing workers can resolve the
    # module's functions by name (prepare.run uses a Pool)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prep = _load('v7_prepare', 'prepare.py')
v7 = _load('v7_gen_tok', 'gen_tok.py')

needs_tokenizer = pytest.mark.skipif(not TOK.is_file(),
                                     reason='tokenizer.json not built yet')
needs_words = pytest.mark.skipif(not v7.WORDS.is_file(),
                                 reason='word dictionary not present')


# --- prepare.py: the deliberate charset choice --------------------------------------
OCR_KEEP = '§©«»½•™■'


def test_ocr_keep_is_the_documented_set():
    assert prep.OCR_KEEP == OCR_KEEP


def test_ocr_keep_chars_survive_every_pass():
    probe = f'Price 1s. 6d. {OCR_KEEP} — “quoted” text.'
    for fn in (prep.flat_clean, prep.record_clean, prep.final_pass):
        out = fn(probe)
        assert all(c in out for c in OCR_KEEP), fn.__name__
    for compose in (prep.clean_dataset_chunk, prep.clean_jsonl_record):
        out = compose(probe)
        assert all(c in out for c in OCR_KEEP)


def test_other_junk_is_still_removed():
    probe = 'ok▼►█ĥ₤ text'
    for fn in (prep.clean_dataset_chunk,
               lambda t: prep.final_pass(prep.record_clean(t))):
        out = fn(probe)
        assert not any(c in out for c in '▼►█₤')


def test_number_explosion_unchanged():
    assert '18 84' in prep.clean_dataset_chunk('the year 1884 came')
    assert '10 0' in prep.clean_dataset_chunk('all 100 men')
    assert ' 42 ' in prep.clean_dataset_chunk('page 42 here')


def test_output_naming():
    out = Path('/tmp/x')
    assert prep.output_path(out, Path('a/book.txt')).name == 'book.txt'
    assert prep.output_path(out, Path('a/data.jsonl')).name == 'data.jsonl.txt'
    assert prep.output_path(out, Path('a/data.jsonl.xz')).name == 'data.jsonl.txt'


def test_prepare_end_to_end(tmp_path):
    """Mixed input directory -> run() -> cleaned shards, resumable."""
    import json as _json
    import lzma as _lzma
    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw / 'book.txt').write_text(
        'Chapter I.\r\nSee page 1884 \u00a7 12 \u2588junk.\n', encoding='utf-8')
    with open(raw / 'news.jsonl', 'w', encoding='utf-8') as f:
        f.write(_json.dumps({'text': 'Price \u00bd\u2122 in 1066!'}) + '\n')
    with _lzma.open(raw / 'extra.jsonl.xz', 'wt', encoding='utf-8') as f:
        f.write(_json.dumps({'text': '\u00ab quoted \u00bb \u2022 350 men'}) + '\n')
    (raw / 'ignored.parquet').write_bytes(b'not supported')

    out = tmp_path / 'clean'
    prep.run(raw, out)
    assert sorted(f.name for f in out.iterdir()) == [
        'book.txt', 'extra.jsonl.txt', 'news.jsonl.txt']
    book = (out / 'book.txt').read_text(encoding='utf-8')
    assert '18 84' in book and '\u00a7' in book and '\u2588' not in book
    news = (out / 'news.jsonl.txt').read_text(encoding='utf-8')
    assert '\u00bd\u2122' in news and '10 66' in news
    extra_out = (out / 'extra.jsonl.txt').read_text(encoding='utf-8')
    assert '\u00ab quoted \u00bb \u2022' in extra_out and '35 0' in extra_out

    # resumable: second run skips everything unless forced
    before = {f.name: f.stat().st_mtime_ns for f in out.iterdir()}
    prep.run(raw, out)
    assert {f.name: f.stat().st_mtime_ns for f in out.iterdir()} == before


# --- the v7 tokenizer contract ------------------------------------------------
def _tok():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOK))


@needs_tokenizer
def test_exact_vocab_size():
    assert _tok().get_vocab_size(with_added_tokens=True) == 32_768


@needs_tokenizer
def test_every_number_is_optimal_modulo_documented_exceptions():
    tok = _tok()
    manifest = json.loads((FOLDER / 'build-manifest.json').read_text())
    exc = {(True, v) for v in manifest['numeric_exceptions_spaced']} \
        | {(False, v) for v in manifest['numeric_exceptions_bare']}
    assert all(2000 <= v <= 2009 for s, v in exc if s)
    for v in range(0, 2100):
        s = str(v)
        want = (len(s) + 1) // 2
        for spaced, text in ((False, s), (True, ' ' + s)):
            n = len(tok.encode(text, add_special_tokens=False).ids)
            allowed = want + 1 if (spaced, v) in exc else want
            assert n == allowed, f'{text!r} -> {n}, want {allowed}'


@needs_tokenizer
def test_no_token_with_three_consecutive_digits():
    spec = json.loads(TOK.read_text())
    bad = [v7.bl_to_text(k) for k in v7.reachable_keys(spec)
           if not v7.digits_ok(v7.bl_to_text(k))]
    assert bad == []


@needs_tokenizer
@needs_words
def test_top2000_coverage_minus_documented_unfixables():
    tok = _tok()
    manifest = json.loads((FOLDER / 'build-manifest.json').read_text())
    unfixable = {' ' + t for t in manifest['unfixable_targets']}
    top = [' ' + w for w in v7.load_top_words()[:2000]]
    enc = tok.encode_batch(top, add_special_tokens=False)
    missing = {t for t, e in zip(top, enc) if len(e.ids) != 1}
    assert missing <= unfixable, f'undocumented gaps: {missing - unfixable}'


@needs_tokenizer
def test_deployment_parity_tokie_and_gigatoken():
    import gigatoken
    import tokie
    hf = _tok()
    probes = v7.STRESS_STRINGS + [' '.join(v7.STRESS_STRINGS) * 5]
    refs = [hf.encode(s, add_special_tokens=False).ids for s in probes]
    for name, t in (('tokie', tokie.Tokenizer.from_json(str(TOK))),
                    ('gigatoken', gigatoken.Tokenizer(str(TOK)))):
        for s, ref in zip(probes, refs):
            e = t.encode(s)
            ids = list(e.ids) if hasattr(e, 'ids') else list(e)
            assert ids == list(ref), f'{name} diverges on {s[:40]!r}'


@needs_tokenizer
def test_roundtrip_byte_exact():
    tok = _tok()
    for s in v7.STRESS_STRINGS:
        assert tok.decode(tok.encode(s).ids) == s
