#!/usr/bin/env python3
"""Rank GENERATED synth text (not the source) by quality signals.

Reads one or more output JSONL files produced by generate.py and, for each
group of records sharing the same (prompt, model, mode), reports the rate of
the failure modes we care about in the generated ``text`` field:

  meta   - output opens by describing/analysing the source ("This document...",
           "The original text...", "The language in this passage is...") instead
           of doing the task. This is the "I don't want those!" problem.
  preamble - output acknowledges the task before continuing ("As a continuation
           ...", "I'll provide one...", "since you're asking..."). Model chatter.
  ocr    - output still contains OCR junk glyphs (^ | • ~ ...), i.e. the model
           imitated the dirty source.
  leak   - assistant/refusal leakage ("As an AI", "I cannot", "Sure, here is").
  anach  - contains an obviously post-1850 word (telephone, computer, ...). A
           cheap proxy for the year-1850 constraint, NOT a full check.
  empty  - blank / near-blank output.
  chars  - mean output length in characters (context, not a pass/fail).

Lower is better for every rate; `good%` is the share of outputs with none of
the failure flags. Groups are sorted best-first by good%.

Usage:
    python score_sample.py out/*.jsonl
    python score_sample.py a.jsonl b.jsonl --examples 3   # show sample failures
"""

import argparse
import glob
import json
import re
import sys
from collections import defaultdict

META_RE = re.compile(
    r"^\W*(this|the|here(?:'s| is)|in\s+(?:this|summary))\s+"
    # optional adjective, e.g. "the ORIGINAL text", "the FOLLOWING passage"
    r'(?:(?:original|following|above|preceding|present|foregoing|given|provided)\s+)?'
    r'(document|text|passage|author|speaker|excerpt|writer|piece|paragraph|'
    r'content|article|summary|following|manuscript|language|prose|style|tone)\b',
    re.I,
)
# Task-acknowledgment / preamble the model sometimes emits before (or instead
# of) the actual continuation. Searched only near the start to limit false
# positives on genuine period prose. Distinct from META (which is analysis of
# the source); PREAMBLE is the model talking about its own task.
PREAMBLE_RE = re.compile(
    r"\b(as (?:a|the) continuation|as requested|here(?:'s| is) (?:a|the|my) "
    r"continuation|i(?:'ll| will| shall)?\s+(?:provide|continue|offer)|"
    r"let me (?:continue|provide)|since you(?:'re| are) asking|"
    r"to continue (?:the|this)|i'll provide)\b",
    re.I,
)
LEAK_RE = re.compile(
    r"\b(as an ai|i am an ai|i cannot|i can't|i'm sorry|as a language model|"
    r'sure,?\s+here|here is (?:the|a|your)|certainly[!,]|i apologi[sz]e)\b',
    re.I,
)
OCR_JUNK = set('^|•~¬■□▪●◦♦†‡¤¢∎')
# Small, high-precision anachronism list — a proxy, not authoritative.
ANACHRONISM_RE = re.compile(
    r'\b(computer|internet|website|online|smartphone|telephone|television|'
    r'radio|aeroplane|airplane|automobile|motorcar|nazi|world war|'
    r'electricity|electronic|software|email|video|nuclear|astronaut|'
    r'quantum|penicillin|antibiotic)\b',
    re.I,
)


def flags(text: str) -> dict:
    stripped = text.strip()
    return {
        'empty': len(stripped) < 20,
        'meta': bool(META_RE.match(stripped)),
        'preamble': bool(PREAMBLE_RE.search(stripped[:200])),
        'leak': bool(LEAK_RE.search(stripped)),
        'ocr': any(c in OCR_JUNK for c in text),
        'anach': bool(ANACHRONISM_RE.search(text)),
    }


def group_key(rec: dict) -> tuple:
    params = rec.get('params') or {}
    return (rec.get('prompt', '?'), rec.get('model', '?'), params.get('mode', '?'))


def score_files(paths: list[str]) -> dict:
    groups: dict[tuple, list] = defaultdict(list)
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'text' not in rec:
                    continue
                groups[group_key(rec)].append(rec)
    return groups


FAIL_KEYS = ('meta', 'preamble', 'ocr', 'leak', 'anach', 'empty')


def summarize(recs: list) -> dict:
    n = len(recs)
    all_flags = [flags(r['text']) for r in recs]
    totals = {k: sum(f[k] for f in all_flags) for k in FAIL_KEYS}
    good = sum(1 for f in all_flags if not any(f.values()))
    mean_chars = sum(len(r['text']) for r in recs) / n if n else 0
    return {
        'n': n,
        'good_pct': 100.0 * good / n if n else 0.0,
        'rates': {k: 100.0 * totals[k] / n if n else 0.0 for k in FAIL_KEYS},
        'mean_chars': mean_chars,
    }


def print_report(groups: dict, examples: int) -> None:
    rows = [(k, summarize(v), v) for k, v in groups.items()]
    rows.sort(key=lambda r: (-r[1]['good_pct'], r[0]))

    hdr = f'{"prompt":<22}{"model":<22}{"mode":<11}{"n":>5}{"good%":>7}' + ''.join(f'{k:>9}' for k in FAIL_KEYS) + f'{"chars":>7}'
    print(hdr)
    print('-' * len(hdr))
    for (prompt, model, mode), s, _ in rows:
        line = (
            f'{prompt[:21]:<22}{model[:21]:<22}{mode[:10]:<11}'
            f'{s["n"]:>5}{s["good_pct"]:>7.0f}' + ''.join(f'{s["rates"][k]:>9.0f}' for k in FAIL_KEYS) + f'{s["mean_chars"]:>7.0f}'
        )
        print(line)

    if examples:
        print('\n=== sample flagged outputs ===')
        for (prompt, model, mode), _, recs in rows:
            shown = 0
            for r in recs:
                f = flags(r['text'])
                if any(f.values()) and shown < examples:
                    bad = ','.join(k for k in FAIL_KEYS if f[k])
                    print(f'\n[{prompt} | {model} | {mode}] flags={bad}')
                    print('  ' + repr(r['text'].strip()[:200]))
                    shown += 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('files', nargs='+', help='Output JSONL file(s) or globs.')
    p.add_argument('--examples', type=int, default=0, help='Show N flagged samples per group.')
    args = p.parse_args()

    paths: list[str] = []
    for pat in args.files:
        paths.extend(sorted(glob.glob(pat)) or [pat])
    groups = score_files(paths)
    if not groups:
        sys.exit('No records found.')
    print_report(groups, args.examples)


if __name__ == '__main__':
    main()
