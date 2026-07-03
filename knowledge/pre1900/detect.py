"""Estimate how likely a text is PRE-1900, and keep/drop it against a threshold.

This is the full inference filter in one file. It combines TWO detectors:

  D1  ANACHRONISM VETO (hard):   if the text names a post-1900 concept/brand/
      person/tech (data/banned.txt) or an explicit post-1900 year (1900-2099),
      the score is forced to ~0.03 (dropped). Provided by banned_terms.py.

  D2  LEXICAL MODERNITY (soft):  a Naive-Bayes log-likelihood ratio over word
      frequencies (OLD corpus vs MODERN corpus), calibrated to a probability.
      It learns that modern phrasing/vocabulary => low pre-1900 probability.

Final P(pre-1900) = D2 probability, capped to ~0.03 if D1 fired.

Dependencies: only the Python standard library + banned_terms.py (same folder)
and the data/ folder. No numpy / pyarrow / sklearn needed at inference time.

----------------------------------------------------------------------------
QUICK START
----------------------------------------------------------------------------
    echo "What a beautiful day!"  | python detect.py
    echo "I posted a selfie online in 2016" | python detect.py

    python detect.py mytext.txt
    python detect.py data.jsonl --field text --threshold 0.75 --filter > kept.jsonl

See README.md for every flag and recommended presets.
"""

import argparse
import json
import math
import os
import re
import sys
from typing import Any

from banned_terms import find_anachronisms

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# ---- tuning constants (rarely changed; the CLI flags are the usual knobs) ----
SMOOTH_K = 0.5  # add-k smoothing for word probabilities
GATE_TAU = 200.0  # words with OLD-count >> tau can't be modern evidence
LR_CLIP_LO, LR_CLIP_HI = -8.0, 10.0  # clamp per-word log-ratio
VETO_CAP = 0.03  # score ceiling when an anachronism is found

# D2 logistic features (order must match data/calib.json).
FEATURES = ['sum_lr', 'gated_sum', 'arch_count']

# Archaic markers: their presence nudges the score toward "old".
ARCHAIC = {
    'thou',
    'thee',
    'thy',
    'thine',
    'hath',
    'hast',
    'doth',
    'dost',
    'art',
    'ye',
    'ere',
    'whilst',
    'betwixt',
    'methinks',
    'prithee',
    'forsooth',
    'wherefore',
    'hither',
    'thither',
    'whence',
    'thence',
    'verily',
    'nay',
    'yea',
    'unto',
    'shalt',
    'wilt',
    "'tis",
    "'twas",
    "o'er",
    "ne'er",
}

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*")


def tokenize(text):
    """Lowercase word tokens; keep internal apostrophes ('tis, o'er, don't)."""
    text = text.lower().replace('’', "'").replace('‘', "'")
    return _TOKEN.findall(text)


def _load_freq(name):
    """Load a {word: count} frequency dict; returns (dict, total_tokens)."""
    d = json.load(open(os.path.join(DATA, name), encoding='utf-8'))
    return d, d.pop('__total__')


class Scorer:
    """Loads the dictionaries + calibration once; score() any number of texts."""

    def __init__(self, old_prior=0.0, style_weight=1.0, marker_weight=1.0):
        # default knobs (each can be overridden per call in score())
        self.old_prior = old_prior  # >0 => favor pre-1900 for short texts
        self.style_weight = style_weight  # weight of modern *phrasing* (sum_lr)
        self.marker_weight = marker_weight  # weight of modern *vocabulary* (gated_sum)

        self.old, self.old_tot = _load_freq('old.json')
        self.mod, self.mod_tot = _load_freq('modern.json')
        self.V = len(set(self.old) | set(self.mod))
        self._den_old = self.old_tot + SMOOTH_K * self.V
        self._den_mod = self.mod_tot + SMOOTH_K * self.V

        c = json.load(open(os.path.join(DATA, 'calib.json'), encoding='utf-8'))
        self.w, self.b = c['weights'], c['bias']

        self.cent = {}
        for name in ('17c', '18c', '19c'):
            p = os.path.join(DATA, f'{name}.json')
            if os.path.exists(p):
                self.cent[name] = _load_freq(f'{name}.json')

        self._lr_cache = {}

    def lr(self, w: str) -> float:
        """log P(w|modern) / P(w|old). Positive => the word leans modern."""
        v = self._lr_cache.get(w)
        if v is None:
            p_old = (self.old.get(w, 0) + SMOOTH_K) / self._den_old
            p_mod = (self.mod.get(w, 0) + SMOOTH_K) / self._den_mod
            v = math.log(p_mod / p_old)
            self._lr_cache[w] = v
        return v

    def featurize(self, tokens: list[str]) -> dict[str, float]:
        n = len(tokens)
        if n == 0:
            return dict.fromkeys(FEATURES, 0.0)
        sum_lr = gated = 0.0
        for t in tokens:
            x = self.lr(t)
            sum_lr += min(LR_CLIP_HI, max(LR_CLIP_LO, x))
            if x > 0.0:  # modern evidence, gated so common-in-old words don't count
                gated += x / (1.0 + self.old.get(t, 0) / GATE_TAU)
        arch = sum(1 for t in tokens if t in ARCHAIC)
        return {'sum_lr': sum_lr, 'gated_sum': gated, 'arch_count': float(arch)}

    def _modern_logit(self, feats, style_weight, marker_weight, old_prior) -> float:
        sw = self.style_weight if style_weight is None else style_weight
        mw = self.marker_weight if marker_weight is None else marker_weight
        op = self.old_prior if old_prior is None else old_prior
        scales = {'sum_lr': sw, 'gated_sum': mw}
        total = self.b - op
        for f in FEATURES:
            total += scales.get(f, 1.0) * self.w[f] * feats[f]
        return total

    def score(self, text: str, tokens: list[str], style_weight=None, marker_weight=None, old_prior=None) -> dict[str, Any]:
        """Return a dict: p_pre1900, n_tokens, english_frac, banned_hits, features.

        style_weight / marker_weight / old_prior override the instance defaults
        for this one call (lets you judge, say, prompts and answers differently).
        """
        hits = find_anachronisms(text)  # D1
        feats = self.featurize(tokens)
        logit = self._modern_logit(feats, style_weight, marker_weight, old_prior)
        p = 0.0 if logit > 700 else 1.0 / (1.0 + math.exp(logit))  # D2 (overflow-safe)
        if hits:
            p = min(p, VETO_CAP)  # D1 hard veto
        english_frac = (sum(1 for t in tokens if t in self.old or t in self.mod) / len(tokens)) if tokens else 0.0
        return {
            'p_pre1900': round(p, 4),
            'n_tokens': len(tokens),
            'english_frac': round(english_frac, 3),
            'banned_hits': hits,
            'features': {k: round(v, 4) for k, v in feats.items()},
        }

    def century(self, tokens: list[str]) -> str | None:
        """Best-guess century among 17c/18c/19c (bonus; low confidence)."""
        if not self.cent or not tokens:
            return None
        best, best_ll = None, -1e18
        for c, (freq, tot) in self.cent.items():
            den = tot + SMOOTH_K * self.V
            ll = sum(math.log((freq.get(t, 0) + SMOOTH_K) / den) for t in tokens)
            if ll > best_ll:
                best, best_ll = c, ll
        return best


def iter_texts(path, field):
    """Yield (record_id, text). .jsonl -> one record/line; else whole file."""
    if path.endswith('.jsonl'):
        for i, line in enumerate(open(path, encoding='utf-8', errors='ignore')):
            line = line.strip()
            if line:
                try:
                    yield i, json.loads(line).get(field, '')
                except json.JSONDecodeError:
                    continue
    else:
        yield 0, open(path, encoding='utf-8', errors='ignore').read()


def main():
    ap = argparse.ArgumentParser(description='Pre-1900 text filter.')
    ap.add_argument('files', nargs='*', help='input files (.txt or .jsonl); omit to read STDIN')
    ap.add_argument('--threshold', type=float, default=0.75, help='keep if p_pre1900 >= this (default 0.75)')
    ap.add_argument('--old-prior', type=float, default=0.0, help='bias short/low-evidence texts toward pre-1900 (try 1-3)')
    ap.add_argument('--style-weight', type=float, default=1.0, help='weight of modern *phrasing* (1=full, 0=ignore)')
    ap.add_argument('--marker-weight', type=float, default=1.0, help='weight of modern *vocabulary* (1=full, 0=ignore)')
    ap.add_argument('--min-english', type=float, default=0.0, help='drop docs whose recognized-English fraction < this')
    ap.add_argument('--min-tokens', type=int, default=0, help='drop docs with fewer than this many words')
    ap.add_argument('--field', default='text', help='JSON field to read from .jsonl (default "text")')
    ap.add_argument('--filter', action='store_true', help='output only KEPT records as jsonl')
    ap.add_argument('--century', action='store_true', help='annotate kept items with a century guess')
    ap.add_argument('--json', action='store_true', help='force jsonl output')
    ap.add_argument('--limit', type=int, default=0, help='stop after this many records (0=all)')
    args = ap.parse_args()

    s = Scorer(old_prior=args.old_prior, style_weight=args.style_weight, marker_weight=args.marker_weight)

    def emit(rec, text):
        tokens = tokenize(text)
        r = s.score(text, tokens)
        r['keep'] = r['p_pre1900'] >= args.threshold and r['english_frac'] >= args.min_english and r['n_tokens'] >= args.min_tokens
        if args.century and r['keep']:
            r['century'] = s.century(tokens)
        if args.filter:
            if r['keep']:
                sys.stdout.write(json.dumps({'text': text, **r}, ensure_ascii=False) + '\n')
        elif args.json or (not args.files and not sys.stdin.isatty()):
            sys.stdout.write(json.dumps({'id': rec, **r}, ensure_ascii=False) + '\n')
        else:
            tag = 'KEEP' if r['keep'] else 'DROP'
            hits = f' hits={r["banned_hits"]}' if r['banned_hits'] else ''
            cent = f' [{r.get("century")}]' if r.get('century') else ''
            print(f'{tag} {r["p_pre1900"]:.2f}{cent}{hits}  {text.strip()[:60]!r}')
        return r['keep']

    if not args.files:  # STDIN
        emit(0, sys.stdin.read())
        return
    kept = total = 0
    for path in args.files:
        for rec, text in iter_texts(path, args.field):
            total += 1
            kept += emit(f'{path}:{rec}', text)
            if args.limit > 0 and total >= args.limit:
                break
    if not args.filter:
        sys.stderr.write(f'\n{kept}/{total} kept (threshold {args.threshold})\n')


if __name__ == '__main__':
    main()
