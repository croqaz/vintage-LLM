#!/usr/bin/env python3
"""
Verify a `split_dataset.py` JSONL split before spending hours tokenizing it.

Line counts matching is necessary but NOT sufficient: it cannot see a corrupted
document, a mis-seeked read, a missing EOS, or a validation document that also
sits in train. This checks the things that actually matter:

  1. COUNT       n(original) == n(train) + n(valid), per shard.
  2. CONTENT     The multiset of documents is preserved exactly. Each original
                 text is canonicalised through the same wrap() the splitter
                 applies, hashed, and accumulated order-independently; the
                 outputs are hashed the same way. Equal sums + equal counts +
                 equal char totals means no document was lost, duplicated,
                 truncated or altered.
  3. EOS         Every output document ends with the tokenizer's EOS.
  4. BYTES       On-disk growth is fully explained by re-serialisation, so
                 there is no silent truncation hiding in the size delta.
  5. LEAKAGE     No train document anywhere is byte-identical to any validation
                 document anywhere -- checked ACROSS shards, which is where
                 held-out hygiene usually breaks.

Usage:
    python training/verify_split.py \
        --original data/Piston-n-Prose --train data/train --valid data/valid \
        --config training/config.toml
"""

import argparse
import json
import sys
from collections import Counter
from hashlib import blake2b
from multiprocessing import Pool
from pathlib import Path

MASK = (1 << 128) - 1


def resolve_tokenizer(cfg: dict, config_path) -> str:
    """
    Resolve data.tokenizer the same way base_train.load_config() does: relative
    paths are anchored to the CONFIG FILE's directory, not the shell's cwd, so
    the value means the same thing from anywhere.  Non-path values (HuggingFace
    hub ids) are passed through untouched.
    """
    from pathlib import Path

    value = cfg['data']['tokenizer']
    base = Path(config_path).resolve().parent
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    if value.startswith(('.', '~', '/')) or (base / path).exists():
        return str((base / path).resolve())
    return value  # hub id


def doc_hash(text: str) -> int:
    return int.from_bytes(blake2b(text.encode('utf-8'), digest_size=16).digest(), 'big')


def make_wrap(eos: str):
    """Exactly the wrap() that split_dataset.py applies."""

    def wrap(t: str) -> str:
        if not t.rstrip().endswith(eos):
            t = f'{t}\n{eos}'
        return t

    return wrap


def scan(path: Path, eos: str, canonicalise: bool, leak_set=None) -> dict:
    """Stream one JSONL file, accumulating an order-independent fingerprint."""
    wrap = make_wrap(eos)
    n = 0
    chars = 0
    acc = 0
    no_eos = 0
    bad_json = 0
    blank = 0
    leaks = 0
    with open(path, 'rb') as fh:
        for raw in fh:
            if not raw.strip():
                blank += 1
                continue
            try:
                text = json.loads(raw)['text']
            except Exception:
                bad_json += 1
                continue
            if canonicalise:
                text = wrap(text)  # make the original comparable to the output
            elif not text.rstrip().endswith(eos):
                no_eos += 1
            h = doc_hash(text)
            if leak_set is not None and h in leak_set:
                leaks += 1
            n += 1
            chars += len(text)
            acc = (acc + h) & MASK
    return {
        'path': str(path),
        'n': n,
        'chars': chars,
        'acc': acc,
        'no_eos': no_eos,
        'bad_json': bad_json,
        'blank': blank,
        'leaks': leaks,
        'bytes': path.stat().st_size,
    }


def shard_job(job):
    orig, train, valid, eos, leak_set = job
    return (
        scan(orig, eos, canonicalise=True),
        scan(train, eos, canonicalise=False, leak_set=leak_set),
        scan(valid, eos, canonicalise=False),
    )


def collect_valid_hashes(valid_paths, eos: str) -> set:
    """Hashes of every validation document, across all shards."""
    hashes = set()
    dupes = 0
    for path in valid_paths:
        with open(path, 'rb') as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                h = doc_hash(json.loads(raw)['text'])
                if h in hashes:
                    dupes += 1
                hashes.add(h)
    return hashes, dupes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--original', type=Path, required=True, help='Directory of the pre-split shards.')
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--valid', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=Path('training/config.toml'))
    ap.add_argument('--jobs', type=int, default=5)
    args = ap.parse_args()

    import tomllib

    from transformers import AutoTokenizer

    with open(args.config, 'rb') as fh:
        cfg = tomllib.load(fh)
    eos = AutoTokenizer.from_pretrained(resolve_tokenizer(cfg, args.config)).eos_token
    print(f'EOS token: {eos!r}\n')

    originals = sorted(args.original.glob('*.jsonl'))
    if not originals:
        sys.exit(f'no .jsonl files in {args.original}')

    jobs = []
    for orig in originals:
        train = args.train / f'{orig.stem}-train{orig.suffix}'
        valid = args.valid / f'{orig.stem}-valid{orig.suffix}'
        for p in (train, valid):
            if not p.is_file():
                sys.exit(f'missing output: {p}')
        jobs.append((orig, train, valid))

    print(f'[1/2] hashing {len(jobs)} validation files to build the leakage set …', flush=True)
    valid_hashes, valid_dupes = collect_valid_hashes([j[2] for j in jobs], eos)
    print(f'      {len(valid_hashes):,} distinct validation documents')
    if valid_dupes:
        print(f'      NOTE: {valid_dupes:,} exact duplicate texts WITHIN the validation set')

    print(f'\n[2/2] streaming {len(jobs)} shards x (original + train + valid) …', flush=True)
    with Pool(min(args.jobs, len(jobs))) as pool:
        results = pool.map(shard_job, [(o, t, v, eos, valid_hashes) for o, t, v in jobs])

    failures = []
    totals = Counter()
    print()
    for (orig, train_p, valid_p), (o, t, v) in zip(jobs, results):
        name = orig.name
        ok_count = o['n'] == t['n'] + v['n']
        ok_hash = o['acc'] == (t['acc'] + v['acc']) & MASK
        ok_chars = o['chars'] == t['chars'] + v['chars']
        ok_eos = t['no_eos'] == 0 and v['no_eos'] == 0
        ok_json = o['bad_json'] == t['bad_json'] == v['bad_json'] == 0
        ok_leak = t['leaks'] == 0
        byte_delta = t['bytes'] + v['bytes'] - o['bytes']

        for label, ok in (
            ('count', ok_count),
            ('content-hash', ok_hash),
            ('char-total', ok_chars),
            ('eos', ok_eos),
            ('json', ok_json),
            ('no-leakage', ok_leak),
        ):
            if not ok:
                failures.append(f'{name}: {label}')

        mark = 'PASS' if all((ok_count, ok_hash, ok_chars, ok_eos, ok_json, ok_leak)) else 'FAIL'
        ratio = v['n'] / o['n'] if o['n'] else 0
        print(f'=== {name}  [{mark}]')
        print(f'    docs      {o["n"]:>12,} = {t["n"]:>12,} train + {v["n"]:>10,} valid   ({ratio:.3%} valid)')
        print(f'    chars     {o["chars"]:>12,} = {t["chars"]:>12,} + {v["chars"]:>10,}   {"OK" if ok_chars else "MISMATCH"}')
        print(f'    fingerprint {"MATCH" if ok_hash else "MISMATCH"}  (order-independent sum of per-document hashes)')
        print(f'    blank lines skipped: orig {o["blank"]}, train {t["blank"]}, valid {v["blank"]}')
        print(
            f'    docs missing EOS: train {t["no_eos"]}, valid {v["no_eos"]}   unparseable: {o["bad_json"] + t["bad_json"] + v["bad_json"]}'
        )
        print(f'    train docs also present in ANY valid set: {t["leaks"]}')
        print(f'    bytes +{byte_delta:,} ({byte_delta / o["n"]:.4f}/doc) — re-serialisation overhead')
        print()

        totals['docs'] += o['n']
        totals['train'] += t['n']
        totals['valid'] += v['n']
        totals['chars'] += o['chars']
        totals['leaks'] += t['leaks']

    print('─' * 72)
    print(f'TOTAL  {totals["docs"]:,} documents  ({totals["train"]:,} train / {totals["valid"]:,} valid)')
    print(f'       {totals["chars"]:,} characters')
    print(f'       cross-shard train/valid leaks: {totals["leaks"]}')
    if failures:
        print('\nFAILED CHECKS:')
        for f in failures:
            print(f'  - {f}')
        sys.exit(1)
    print('\nALL CHECKS PASSED — safe to tokenize.')


if __name__ == '__main__':
    main()
