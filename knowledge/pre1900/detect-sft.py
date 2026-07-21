"""General vintage (pre-1900) filter for SFT / instruction datasets.

Replaces the per-dataset processors. It auto-detects the row shape from the
FIRST non-empty line and assumes every other line uses the same shape (fast,
single pass). Three shapes are supported:

  1. Q&A flat fields     {"Question": ..., "Answer": ...}
     (also auto-detects prompt/completion, Input/Anwser, instruction/output,
      question/answer, query/response; or set --question-field/--answer-field)

  2. SFT "messages"      {"messages": [{"role": "user"|"assistant"|"system",
                                        "content": ...}, ...]}

  3. Alpaca "conversation"  {"conversation": [{"from": "human"|"gpt",
                                               "value": ...}, ...]}
     (also accepts "conversations")

Filtering rule (same as before): the human/user side is judged LENIENTLY (modern
phrasing is fine, only post-1900 *concepts* drop it) and the AI answer side is
judged STRICTLY (must sound vintage). A row is KEPT only if every user turn
clears --user-threshold AND every answer turn clears --assistant-threshold.

Answer turns have their <think>...</think> reasoning stripped before scoring
AND in the written output (reasoning traces may contain modern terms; we don't
care and we remove them). Kept rows are written unchanged except for that
stripping, so the output keeps whatever shape the input had.

Input may be JSON Lines (one object per line) OR a single big JSON array
(e.g. [ {...}, {...} ]) -- auto-detected from the first character, or forced
with --json-array / --jsonl. Output is always JSON Lines (one kept row/line).

Examples:
    python detect-sft.py data.jsonl --limit 5000 --dry-run
    python detect-sft.py UltraChat/train_0.jsonl --out kept.jsonl
    python detect-sft.py data.jsonl --question-field prompt --answer-field completion
    python detect-sft.py General-Knowledge/knowledge.json --out kept.jsonl  # JSON array
"""

import argparse
import json
import re
import sys
import time

from detect import Scorer, tokenize

# strip <think>...</think> reasoning (and stray tags) from answer turns
_THINK = re.compile(r'<think>.*?</think>', re.S | re.I)
_STRAY = re.compile(r'</?think>', re.I)

# candidate (question, answer) field-name pairs for the flat Q&A shape
QA_PAIRS = [
    ('Question', 'Answer'),
    ('prompt', 'completion'),
    ('Input', 'Anwser'),
    ('Input', 'Answer'),
    ('instruction', 'output'),
    ('question', 'answer'),
    ('query', 'response'),
]


def strip_think(value):
    return _STRAY.sub(' ', _THINK.sub(' ', value)).strip()


def detect_shape(row, qf, af):
    """Return ('messages'|'conversation'|'qa', question_field, answer_field)."""
    if isinstance(row.get('messages'), list):
        return 'messages', None, None
    if isinstance(row.get('conversation'), list) or isinstance(row.get('conversations'), list):
        return 'conversation', None, None
    if qf and af:
        return 'qa', qf, af
    for q, a in QA_PAIRS:
        if q in row and a in row:
            return 'qa', q, a
    raise SystemExit(f'Could not detect row shape. Keys were: {list(row.keys())}. Use --question-field/--answer-field for a flat Q&A file.')


def get_turns(row, shape, qf, af):
    """Yield (is_answer, text) for each turn. is_answer -> judged strictly."""
    if shape == 'qa':
        yield False, str(row.get(qf, '') or '')
        yield True, str(row.get(af, '') or '')
    elif shape == 'messages':
        for m in row.get('messages', []):
            yield m.get('role') == 'assistant', str(m.get('content', '') or '')
    else:  # conversation
        conv = row.get('conversation') or row.get('conversations') or []
        for t in conv:
            yield t.get('from') in ('gpt', 'assistant'), str(t.get('value', t.get('content', '')) or '')


def clean_row(row, shape, qf, af):
    """Return the row with <think> stripped from every answer turn."""
    if shape == 'qa':
        row[af] = strip_think(str(row.get(af, '') or ''))
    elif shape == 'messages':
        for m in row.get('messages', []):
            if m.get('role') == 'assistant':
                m['content'] = strip_think(str(m.get('content', '') or ''))
    else:
        for t in row.get('conversation') or row.get('conversations') or []:
            if t.get('from') in ('gpt', 'assistant'):
                key = 'value' if 'value' in t else 'content'
                t[key] = strip_think(str(t.get(key, '') or ''))
    return row


def read_rows(path, json_array=None):
    """Yield parsed rows from either JSON Lines or a single JSON array file.

    json_array: True/False forces the mode; None auto-detects from the first
    non-space character ('[' => a JSON array, else JSON Lines).
    """
    with open(path, encoding='utf-8', errors='ignore') as f:
        if json_array is None:
            head = f.read(64).lstrip()
            json_array = head.startswith('[')
            f.seek(0)
        if json_array:
            data = json.load(f)
            if isinstance(data, dict):  # e.g. {"data": [...]} wrappers
                data = next((v for v in data.values() if isinstance(v, list)), [])
            yield from data
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+', help='input jsonl file(s)')
    ap.add_argument('--out', default='vintage_sft.jsonl', help='output jsonl of kept rows')
    ap.add_argument('--question-field', help='override the question field for flat Q&A input')
    ap.add_argument('--answer-field', help='override the answer field for flat Q&A input')
    ap.add_argument(
        '--json-array',
        dest='json_array',
        action='store_true',
        default=None,
        help='force reading input as one big JSON array (auto-detected by default)',
    )
    ap.add_argument('--jsonl', dest='json_array', action='store_false', help='force reading input as JSON Lines (one object per line)')

    # user/question side (lenient)
    ap.add_argument('--user-threshold', type=float, default=0.1)
    ap.add_argument('--user-style', type=float, default=0.0)
    ap.add_argument('--user-marker', type=float, default=0.0)
    # answer side (strict). marker=0.1/thr=0.2 keeps the cleanest answers.
    ap.add_argument('--assistant-threshold', type=float, default=0.2)
    ap.add_argument('--assistant-style', type=float, default=0.0)
    ap.add_argument('--assistant-marker', type=float, default=0.1)
    ap.add_argument('--min-answer-tokens', type=int, default=20)

    ap.add_argument('--limit', type=int, default=0, help='process at most N rows (0 = all)')
    ap.add_argument('--report-every', type=int, default=10_000)
    ap.add_argument('--dry-run', action='store_true', help='do not write output, just print stats + samples')
    args = ap.parse_args()

    s = Scorer()

    def ok_user(text):
        r = s.score(text, tokenize(text), style_weight=args.user_style, marker_weight=args.user_marker)
        return r['p_pre1900'] >= args.user_threshold

    def ok_answer(text):
        r = s.score(text, tokenize(text), style_weight=args.assistant_style, marker_weight=args.assistant_marker)
        return r['p_pre1900'] >= args.assistant_threshold and r['n_tokens'] >= args.min_answer_tokens

    shape = qf = af = None
    out = None if args.dry_run else open(args.out, 'w', encoding='utf-8')
    total = kept = drop_user = drop_answer = 0
    samples = []
    t0 = time.time()
    for path in args.files:
        for row in read_rows(path, args.json_array):
            if not isinstance(row, dict):
                continue
            if shape is None:  # detect once, from the first non-empty row
                shape, qf, af = detect_shape(row, args.question_field, args.answer_field)
                desc = f'{shape}' + (f' ({qf} / {af})' if shape == 'qa' else '')
                print(f'detected shape: {desc}', file=sys.stderr)
            total += 1

            ok = True
            reason = None
            for is_answer, text in get_turns(row, shape, qf, af):
                if is_answer:
                    if not ok_answer(strip_think(text)):
                        ok, reason = False, 'answer'
                        break
                else:
                    if not ok_user(text):
                        ok, reason = False, 'user'
                        break
            if ok:
                kept += 1
                if out:
                    out.write(json.dumps(clean_row(row, shape, qf, af), ensure_ascii=False) + '\n')
                if len(samples) < 5:
                    ans = next((strip_think(t) for a, t in get_turns(row, shape, qf, af) if a), '')
                    samples.append(ans[:150])
            elif reason == 'user':
                drop_user += 1
            else:
                drop_answer += 1

            if total % args.report_every == 0:
                print(f'  {total:,} rows | kept {kept:,} | {total / max(1e-9, time.time() - t0):.0f}/s', file=sys.stderr, flush=True)
            if args.limit and total >= args.limit:
                break
        if args.limit and total >= args.limit:
            break

    if out:
        out.close()
    dt = time.time() - t0
    print(f'\nrows read:          {total:,}  ({dt:.0f}s)', file=sys.stderr)
    print(f'kept (vintage):     {kept:,}  ({100 * kept / max(1, total):.2f}%)', file=sys.stderr)
    print(f'dropped - question: {drop_user:,}  (named a post-1900 concept)', file=sys.stderr)
    print(f'dropped - answer:   {drop_answer:,}  (answer too modern)', file=sys.stderr)
    if not args.dry_run:
        print(f'-> {args.out}', file=sys.stderr)
    print('\nsample kept answers:', file=sys.stderr)
    for sm in samples:
        print(f'   {sm!r}', file=sys.stderr)


if __name__ == '__main__':
    main()
