"""Build the v7 tokenizer: regex-free, universally deployable.

The artifact is a plain ByteLevel BPE (the standard GPT-2 pre-tokenization
scheme, no custom regex), which HuggingFace tokenizers, tokie, and gigatoken
all execute id-exact — the build fails otherwise. All corpus-specific rules
live in the DATA (prepare.py) and in library-neutral vocabulary surgery:

  numbers   prepare.py explodes 3+-digit runs into 2-digit groups, so the
            trainer never sees a 3-digit sequence and no >=3-digit token is
            learnable. Any missing 2-digit chunk merges are APPENDED at the
            END of the merge table — natural merge ranks are never
            reordered: fast BPE engines reproduce HuggingFace only on merge
            tables whose priorities they can replay, and reordering learned
            merges breaks that (measured), while append-only surgery does
            not (verified: full-sample parity). Cost of the natural order,
            measured exhaustively over 0..2099: spaced numbers are optimal
            ceil(digits/2) everywhere except " 2000".." 2009", bare forms
            have a few dozen +1-token exceptions (all recorded in the build
            manifest) — every in-period year in running prose is 2 tokens.
  words     the top-2000 words of the curated words/words-clean.json are
            guaranteed single tokens in their leading-space form. Words the
            GPT-2 scheme cuts into 2+ pre-tokens (contraction-bearing:
            o'clock, i'll, ...) cannot be single tokens and are reported in
            the build manifest, not hidden.
  policy    numeric prune as a backstop: no reachable token may contain 3+
            consecutive digits.

Usage:
    python3 t-v7/prepare.py  # once: DATA/ raw -> DATA-CLEAN/
    python3 t-v7/gen_tok.py  # train (checkpointed) + warrant + verify
"""

import hashlib
import json
import time
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'DATA-CLEAN'  # rebuilt from raw by t-v7/prepare.py
CACHE = HERE / 'cache'
WORDS = HERE.parent / 'words' / 'words-clean.json'
TARGET_VOCAB = 32_768
MIN_FREQUENCY = 1000
TOP_WORDS = 2_000
RECIPE = 'plain-bytelevel-v1'  # cache key component; no custom regex
SPECIAL_TOKENS = ['<|pad|>', '<|unk|>', '<|mask|>', '<|bos|>', '<|eos|>', '<|system|>', '<|user|>', '<|assistant|>'] + [
    f'<|future{i}|>' for i in range(1, 9)
]

# all 100 bare digit-pair chunks must exist; missing ones are APPENDED at the
# end of the merge table (append-only — reordering learned merges breaks
# fast-library parity, see module docstring).
DIGIT_CHUNKS = [f'{v:02d}' for v in range(100)]


def digits_ok(text):
    """The v7 numeric policy: no 3+ consecutive digits in any token."""
    run = 0
    for c in text:
        run = run + 1 if c.isdigit() else 0
        if run >= 3:
            return False
    return True


# --- byte-level key <-> text --------------------------------------------------
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
_missing = [b for b in range(256) if b not in bs]
INV_BYTELEVEL = {chr(c): b for b, c in zip(bs + _missing, bs + [256 + i for i in range(len(_missing))])}


def bl_to_text(key):
    return bytes(INV_BYTELEVEL[ch] for ch in key).decode('utf-8', errors='replace')


# --- reachability machinery ----------------------------------------------------
def reachable_keys(spec):
    have = {k for k in spec['model']['vocab'] if len(k) == 1}
    by_part = {}
    for idx, (a, b) in enumerate(spec['model']['merges']):
        by_part.setdefault(a, []).append(idx)
        by_part.setdefault(b, []).append(idx)
    pending = [i for i, (a, b) in enumerate(spec['model']['merges']) if a in have and b in have]
    while pending:
        idx = pending.pop()
        a, b = spec['model']['merges'][idx]
        merged = a + b
        if merged not in have:
            have.add(merged)
            pending.extend(by_part.get(merged, []))
    return have


def drop_zombies(spec):
    while True:
        have = reachable_keys(spec)
        alive = [[a, b] for a, b in spec['model']['merges'] if a in have and b in have]
        if len(alive) == len(spec['model']['merges']):
            return have
        spec['model']['merges'] = alive


def rebuild_vocab(spec, specials):
    have = reachable_keys(spec)
    old = spec['model']['vocab']
    survivors = sorted((k for k in old if k in have or k in specials), key=lambda k: old[k])
    spec['model']['vocab'] = {k: i for i, k in enumerate(survivors)}
    for entry in spec.get('added_tokens', []):
        entry['id'] = spec['model']['vocab'][entry['content']]
    return len(spec['model']['vocab'])


# --- warrant targets -------------------------------------------------------------
def load_top_words():
    with open(WORDS, encoding='utf-8') as f:
        counts = json.load(f)
    return [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])]


def warrant_targets():
    """Clean top-2000 words (Ġ-form) + spaced 2-digit chunks. Bare chunks are
    handled structurally by canonicalize_digit_merges, not as targets."""
    spaced = [f' {v}' for v in range(100)] + [f' {v:02d}' for v in range(10)]
    return [' ' + w for w in load_top_words()[:TOP_WORDS]] + spaced


# --- training (checkpointed) -------------------------------------------------------
def corpus_files():
    files = sorted(DATA.glob('*.txt'))
    return files


def train_once(files):
    h = hashlib.sha256(RECIPE.encode())
    for f in files:
        h.update(f'{f.name}:{f.stat().st_size}'.encode())
    ckpt = CACHE / f'trained_spec_v{TARGET_VOCAB}_{h.hexdigest()[:12]}.json'
    if ckpt.is_file():
        print(f'reusing cached training pass {ckpt.name}', flush=True)
        return Tokenizer.from_str(ckpt.read_text())
    tok = Tokenizer(BPE(unk_token='<|unk|>'))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = ByteLevelDecoder()
    t0 = time.time()
    tok.train(
        [str(f) for f in files],
        BpeTrainer(
            vocab_size=TARGET_VOCAB, min_frequency=MIN_FREQUENCY, initial_alphabet=ByteLevel.alphabet(), special_tokens=SPECIAL_TOKENS
        ),
    )
    print(f'trained in {time.time() - t0:.0f}s', flush=True)
    CACHE.mkdir(exist_ok=True)
    ckpt.write_text(tok.to_str())
    return tok


# --- warrant ---------------------------------------------------------------------
def missing_targets(tok, targets):
    enc = tok.encode_batch(targets, add_special_tokens=False)
    return [t for t, e in zip(targets, enc) if len(e.ids) != 1]


def append_repairs(spec, tok, targets, protected):
    id2key = {i: k for k, i in tok.get_vocab().items()}
    vocab, merges = spec['model']['vocab'], spec['model']['merges']
    next_id = max(vocab.values()) + 1
    appended = 0
    for t in targets:
        pieces = [id2key[i] for i in tok.encode(t, add_special_tokens=False).ids]
        acc = pieces[0]
        for p in pieces[1:]:
            if [acc, p] not in merges:
                merges.append([acc, p])
                protected.add((acc, p))
                appended += 1
            acc += p
            if acc not in vocab:
                vocab[acc] = next_id
                next_id += 1
    return appended


def ensure_digit_pairs(spec):
    """Append (never reorder) any missing bare digit-pair merges at the end
    of the merge table, so every 2-digit chunk exists. Natural learned ranks
    stay untouched — that is what keeps tokie/gigatoken id-exact."""
    vocab, merges = spec['model']['vocab'], spec['model']['merges']
    have = {a + b for a, b in merges if len(a) == 1 and len(b) == 1 and a.isdigit() and b.isdigit()}
    next_id = max(vocab.values()) + 1
    appended = []
    for key in DIGIT_CHUNKS:
        if key not in have:
            merges.append([key[0], key[1]])
            appended.append(key)
            if key not in vocab:
                vocab[key] = next_id
                next_id += 1
    return appended


def apply_warrant(spec):
    specials = set(SPECIAL_TOKENS)

    # 1. numeric prune: nothing with 3+ consecutive digits may survive
    kept = [m for m in spec['model']['merges'] if digits_ok(bl_to_text(m[0] + m[1]))]
    pruned = len(spec['model']['merges']) - len(kept)
    spec['model']['merges'] = kept
    drop_zombies(spec)
    rebuild_vocab(spec, specials)
    print(f'numeric prune: -{pruned} merges with 3+-digit results', flush=True)

    # 2. chunk completeness: append-only digit-pair repairs (no reordering)
    added = ensure_digit_pairs(spec)
    protected = {(k[0], k[1]) for k in DIGIT_CHUNKS}
    rebuild_vocab(spec, specials)
    print(f'digit pairs: appended {added} missing chunks at end', flush=True)

    # 3. word + spaced-chunk warrant, with unfixables reported
    targets = warrant_targets()
    base_tok = Tokenizer.from_str(json.dumps(spec))
    unfixable = sorted(t for t in missing_targets(base_tok, targets) if len(base_tok.pre_tokenizer.pre_tokenize_str(t)) > 1)
    print(f'unfixable under the GPT-2 scheme ({len(unfixable)}): {[t.strip() for t in unfixable]}', flush=True)
    fixable = [t for t in targets if t not in unfixable]

    for round_no in range(1, 25):
        tok = Tokenizer.from_str(json.dumps(spec))
        todo = [t for t in missing_targets(tok, fixable)]
        if todo:
            n = append_repairs(spec, tok, todo, protected)
            print(f'warrant round {round_no}: repaired {len(todo)} targets (+{n} merges)', flush=True)
        size = rebuild_vocab(spec, specials)
        if size == TARGET_VOCAB and not todo:
            break
        while size > TARGET_VOCAB:
            # trim the LAST unprotected (= least-frequent learned) merge;
            # appended digit pairs and warrant repairs are never touched
            merges = spec['model']['merges']
            idx = max(i for i, m in enumerate(merges) if (m[0], m[1]) not in protected)
            del merges[idx]
            drop_zombies(spec)
            size = rebuild_vocab(spec, specials)
        if size < TARGET_VOCAB:
            extra = [' ' + w for w in load_top_words()[TOP_WORDS : TOP_WORDS + 500]]
            tok = Tokenizer.from_str(json.dumps(spec))
            filler = [t for t in missing_targets(tok, extra) if len(tok.pre_tokenizer.pre_tokenize_str(t)) == 1]
            filler = filler[: TARGET_VOCAB - size]
            print(f'warrant round {round_no}: short by {TARGET_VOCAB - size}, filling with {filler}', flush=True)
            append_repairs(spec, tok, filler, protected)
            rebuild_vocab(spec, specials)
    else:
        raise RuntimeError('warrant did not converge in 25 rounds')
    return spec, unfixable


# --- verify -----------------------------------------------------------------------
STRESS_STRINGS = [
    "In 1884, the cat wasn't sad.\n\nIt cost 12 pounds.",
    "Mr. John went to London in 1899; o'clock it was — don’t!",
    "“Mother’s,” he said. “To-morrow” — well-known; o'er the hills.",
    'αὐτός καὶ ἡ γῆ · naïve façade ſ long s',
    '🙂 emoji 😀 and CJK 漢字 and العربية و עברית',
    '\tTabs\tand  spaces   and\r\nCRLF line endings.',
    'Numbers: 7, 42, 99, 100, 101, 250, 1066, 1104, 1700, 1805, 1884, 12345.',
]


def verify(tok, spec, unfixable):
    assert tok.get_vocab_size(with_added_tokens=True) == TARGET_VOCAB
    for s in STRESS_STRINGS:
        assert tok.decode(tok.encode(s).ids) == s, f'roundtrip failed: {s!r}'

    # numeric policy scan: no reachable token carries 3+ consecutive digits
    bad = [bl_to_text(k) for k in reachable_keys(spec) if not digits_ok(bl_to_text(k))]
    assert not bad, f'3+-digit tokens reachable: {sorted(bad)[:10]}'

    # exhaustive number check 0..2099, bare and spaced. Under natural merge
    # ranks (append-only surgery — the price of fast-library parity) the
    # optimum ceil(digits/2) holds everywhere except a small, documented set:
    # spaced exceptions must lie in 2000..2009 (out-of-period years) and all
    # exceptions cost exactly one extra token.
    spaced_exc, bare_exc = [], []
    for v in range(0, 2100):
        s = str(v)
        want = (len(s) + 1) // 2
        n_b = len(tok.encode(s, add_special_tokens=False).ids)
        n_s = len(tok.encode(' ' + s, add_special_tokens=False).ids)
        if n_b != want:
            assert n_b == want + 1, f'{s!r} -> {n_b}, want {want}'
            bare_exc.append(v)
        if n_s != want:
            assert n_s == want + 1, f'{" " + s!r} -> {n_s}, want {want}'
            spaced_exc.append(v)
    assert all(2000 <= v <= 2009 for v in spaced_exc), spaced_exc
    assert len(bare_exc) <= 80, f'too many bare exceptions: {len(bare_exc)}'

    targets = warrant_targets()
    still = [t for t in missing_targets(tok, targets) if t not in unfixable]
    assert not still, f'fixable warrant targets missing: {still[:10]}'

    top = [' ' + w for w in load_top_words()[:TOP_WORDS]]
    got = sum(1 for e in tok.encode_batch(top, add_special_tokens=False) if len(e.ids) == 1)
    print(
        f'verify OK: exact 2**15, top-{TOP_WORDS} coverage {got}/{TOP_WORDS} '
        f'(unfixable under GPT-2 scheme: {len(unfixable)}), numbers 0..2099 '
        f'optimal except {len(spaced_exc)} spaced (all 200X) / '
        f'{len(bare_exc)} bare at +1 token, roundtrips byte-exact',
        flush=True,
    )
    return spaced_exc, bare_exc


def verify_fast_libraries(json_path):
    """Deployment gate: both fast libraries must reproduce HF ids exactly.

    Parity is proven on real text at scale, not just on stress strings:
    the probe set includes a multi-megabyte slice of every raw source."""
    import tokie
    import gigatoken

    hf = Tokenizer.from_file(json_path)
    probes = STRESS_STRINGS + [' '.join(STRESS_STRINGS) * 20]
    for raw in sorted((HERE.parent / 'DATA').glob('dataset-text*.txt')):
        with open(raw, encoding='utf-8', errors='ignore') as f:
            chunk = f.read(4 << 20)
        probes.append(chunk[: chunk.rfind('\n')])
    refs = [hf.encode(s, add_special_tokens=False).ids for s in probes]
    for name, tok in (('tokie', tokie.Tokenizer.from_json(json_path)), ('gigatoken', gigatoken.Tokenizer(json_path))):
        for s, ref in zip(probes, refs):
            e = tok.encode(s)
            ids = list(e.ids) if hasattr(e, 'ids') else list(e)
            assert ids == list(ref), f'{name} ids diverge on {s[:40]!r}'
    print('deployment check OK: tokie and gigatoken are id-exact', flush=True)


def main():
    files = corpus_files()
    total = sum(f.stat().st_size for f in files) / 1e9
    print(f'building t-v7.FINAL: {len(files)} files, {total:.1f} GB, vocab {TARGET_VOCAB:,}, recipe {RECIPE}', flush=True)

    tok = train_once(files)
    spec = json.loads(tok.to_str())
    n_learned = len(spec['model']['merges'])
    spec, unfixable = apply_warrant(spec)
    tok = Tokenizer.from_str(json.dumps(spec))
    spaced_exc, bare_exc = verify(tok, spec, unfixable)

    out = HERE / 'tokenizer.json'
    tok.save(str(out))
    verify_fast_libraries(str(out))
    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(out), bos_token='<|bos|>', eos_token='<|eos|>', unk_token='<|unk|>', pad_token='<|pad|>'
    )
    fast.save_pretrained(str(HERE))
    (HERE / 'build-manifest.json').write_text(
        json.dumps(
            {
                'experiment': 'regex-free deployable tokenizer: rules in data (DATA-CLEAN2) + merge-rank surgery, no custom splitter',
                'recipe': RECIPE,
                'data': 'DATA-CLEAN2 (prepare.py: charset + 2-digit number explosion)',
                'learned_merges_before_warrant': n_learned,
                'merges_after': len(spec['model']['merges']),
                'top_words': TOP_WORDS,
                'unfixable_targets': [t.strip() for t in unfixable],
                'numeric_policy': 'no token with 3+ consecutive digits; append-only '
                'chunk repairs under natural merge ranks => '
                'ceil(digits/2) tokens for 0..2099 except the '
                'documented +1-token exceptions below',
                'numeric_exceptions_spaced': spaced_exc,
                'numeric_exceptions_bare': bare_exc,
                'target_vocab': TARGET_VOCAB,
            },
            indent=2,
        )
    )

    sample = 'Mr. John went to London in 1884; it cost 100 pounds — 250 men.'
    ids = tok.encode(sample).ids
    assert tok.decode(ids) == sample
    print(f'sample: {[tok.id_to_token(i) for i in ids]}', flush=True)
    print(f'wrote {out} ({tok.get_vocab_size(with_added_tokens=True):,} tokens)')


if __name__ == '__main__':
    main()
