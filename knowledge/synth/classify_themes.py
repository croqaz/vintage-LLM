#!/usr/bin/env python3
"""
classify_themes.py
==================

Find out *what kinds of Q&A* are in a fine-tuning file so you can diversify the
generator's seeds. Reads the USER turns from one or more chat JSONL files (the
`{"messages":[system,user,assistant], "model":...}` shape that
make_finetune_jsonl.py emits) and classifies each question on TWO axes:

  * DOMAIN  -- what the question is *about*   (science, arts, philosophy, ...)
  * FORM    -- what *kind* of turn it is      (explanation, how-to, chat, ...)

Why two axes? A single topic label hides the "everything is explain/describe"
monotony. The DOMAIN x FORM cross-tab is what reveals e.g. "30% is
explanation about science, ~0% is casual chat or creative writing".

Two modes
---------
  --mode lexical   Pure Python: opening-verb histogram + keyword buckets.
                   Free, instant, no model. A first-glance map; coarse.
  --mode llm       Ask a model to label each question into the taxonomy below.
                   Batched (default 20 questions/call) to save tokens. Runs
                   against the local llama-server (free, default) or OpenRouter.

Outputs
-------
  1. DOMAIN histogram, FORM histogram, and the top DOMAIN x FORM combinations,
     each sorted with counts, % and a small ASCII bar.
  2. A "diversity gap" report: actual share vs. an even baseline, listing the
     most over- and under-represented buckets.
  3. (llm mode) A sidecar `<out>` JSONL = each question + its labels, written
     incrementally so a crash never loses paid classifications. Re-run with
     --mode tally on that sidecar to re-print the report without re-paying.

Usage
-----
  # free first look
  python3 classify_themes.py TypeWriter1.jsonl --mode lexical

  # accurate labels with the local model (free), save labels to themes.jsonl
  python3 classify_themes.py TypeWriter1.jsonl --mode llm --out themes.jsonl

  # accurate labels with OpenRouter (needs OPENROUTER_API_KEY)
  python3 classify_themes.py TypeWriter1.jsonl --mode llm \
      --provider openrouter --model deepseek/deepseek-chat --out themes.jsonl

  # re-print the report from an existing labelled sidecar (no API calls)
  python3 classify_themes.py --mode tally themes.jsonl

Stdlib only. For llm mode you need a running llama-server (local) or an
OPENROUTER_API_KEY (openrouter).
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

# ===========================================================================
# Taxonomy -- EDIT THESE LISTS BY HAND to tune the buckets.
# ===========================================================================
# DOMAIN = what the question is about. Keep labels short and disjoint. The
# model is told to use EXACTLY these labels; anything else collapses to 'other'.
DOMAINS = [
    'arts_poetry_music',  # poems, songs, painting, sculpture, theatre
    'commerce_money',  # trade, prices, work, business
    'food_cooking',  # recipes, preserving, drink, meals
    'geography_places',  # countries, cities, regions, maps, travels
    'health_medicine',  # ailments, remedies, hygiene, the body
    'history',  # past events, figures, civilisations
    'home_practical',  # chores, gardening, repairs, household how-to
    'language_writing',  # grammar, letters, composition, definitions
    'nature_animals',  # weather, plants, animals, husbandry (non-scientific)
    'other',  # genuine catch-all
    'philosophy_ethics',  # right/wrong, meaning, religion-as-thought, logic
    'religion_spirituality',  # faith, scripture, the supernatural
    'science_tech',  # how things work, instruments, engineering, nature-as-science
    'smalltalk_personal',  # opinions about self, casual chat, feelings, hobbies
    'social_etiquette',  # manners, hosting, correspondence conventions
    'travel',  # journeys, places, transport-as-travel
]

# FORM = what kind of turn it is (intent/shape of the answer it wants).
FORMS = [
    'explanation',  # "explain/why/what is ..." -> exposition
    'how_to',  # step-by-step procedure / instructions
    'opinion_advice',  # "should I ...", recommendations, judgement calls
    'creative',  # write a poem/story/song/letter -> generated artefact
    'factual_lookup',  # "list/name/which/who/when" -> short facts
    'casual_chat',  # personal, conversational, rapport ("how are you?")
    'other',
]


# ===========================================================================
# Reading the input
# ===========================================================================
def iter_user_turns(paths):
    """Yield the user-turn text from every record across the given files.
    Skips blank lines and records without a user turn; counts bad JSON."""
    n_bad = 0
    for path in paths:
        try:
            fh = open(path, 'r', encoding='utf-8')
        except OSError as e:
            print(f'! cannot open {path}: {e}', file=sys.stderr)
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                user = None
                for m in rec.get('messages', []):
                    if m.get('role') == 'user':
                        user = (m.get('content') or '').strip()
                        break
                if user:
                    yield user
    if n_bad:
        print(f'[info] skipped {n_bad} unparseable line(s)', file=sys.stderr)


# ===========================================================================
# Reporting helpers (shared by all modes)
# ===========================================================================
def _bar(frac, width=30):
    return '#' * int(round(frac * width))


def histogram(title, counts, total):
    """Print a sorted count/%/bar table for a {label: count} mapping."""
    print(f'\n{title}  (n={total})')
    print('-' * 60)
    for label, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        frac = c / total if total else 0
        print(f'  {label:24s} {c:6d}  {frac * 100:5.1f}%  {_bar(frac)}')


def gap_report(counts, total, axis_name):
    """Compare each bucket's share against an even baseline and list the most
    over- and under-represented. Even baseline = 100% / number-of-buckets."""
    if not total:
        return
    n_buckets = len(counts) or 1
    even = 100.0 / n_buckets
    print(f'\nDiversity gap for {axis_name} (even baseline = {even:.1f}% each)')
    print('-' * 60)
    ranked = sorted(counts.items(), key=lambda kv: kv[1] / total - even / 100)
    print('  most UNDER-represented (add more of these):')
    for label, c in ranked[:5]:
        print(f'    {label:24s} {c / total * 100:5.1f}%  ({c})')
    print('  most OVER-represented (prune these):')
    for label, c in reversed(ranked[-5:]):
        print(f'    {label:24s} {c / total * 100:5.1f}%  ({c})')


def report(domain_counts, form_counts, cross_counts, total):
    """Full report: both axis histograms, top cross-tab combos, gap analysis."""
    histogram('DOMAIN', domain_counts, total)
    histogram('FORM', form_counts, total)
    # Cross-tab: only show the populated combinations, most common first.
    print(f'\nDOMAIN x FORM  (top 20 combinations, n={total})')
    print('-' * 60)
    for (dom, frm), c in sorted(cross_counts.items(), key=lambda kv: -kv[1])[:20]:
        frac = c / total if total else 0
        print(f'  {dom:22s} x {frm:16s} {c:5d}  {frac * 100:4.1f}%')
    gap_report(domain_counts, total, 'DOMAIN')
    gap_report(form_counts, total, 'FORM')


def tally(labelled):
    """Turn a list of (domain, form) into the three count maps + total."""
    dom, frm, cross = {}, {}, {}
    for d, f in labelled:
        dom[d] = dom.get(d, 0) + 1
        frm[f] = frm.get(f, 0) + 1
        cross[(d, f)] = cross.get((d, f), 0) + 1
    return dom, frm, cross, len(labelled)


# ===========================================================================
# Mode 1: lexical (free, no model)
# ===========================================================================
# Keyword buckets approximate DOMAIN by surface words. Crude on purpose: a
# question can hit several buckets (we count it in each it matches, plus an
# 'unmatched' tally). This is only a first-glance map -- use --mode llm for
# real per-question single labels. Edit freely.
LEXICAL_BUCKETS = {
    'arts_poetry_music': ['poem', 'poetry', 'song', 'sing', 'music', 'paint', 'sculpt', 'verse', 'sonnet', 'ballad', 'theatre', 'opera'],
    'commerce_money': ['money', 'price', 'cost', 'trade', 'business', 'wages', 'buy', 'sell', 'market', 'commerce'],
    'food_cooking': [
        'bake',
        'boil',
        'bread',
        'breakfast',
        'brew',
        'cook',
        'dinner',
        'dish',
        'fry',
        'jam',
        'meal',
        'pie',
        'preserv',
        'pudding',
        'recipe',
        'roast',
        'soup',
        'stew',
        'tea',
    ],
    'health_medicine': [
        'ache',
        'cure',
        'disease',
        'doctor',
        'fever',
        'health',
        'hygiene',
        'illness',
        'infection',
        'medicine',
        'pill',
        'remedy',
    ],
    'geography_places': ['geograph', 'country', 'city', 'capital', 'region', 'map', 'river', 'mountain', 'ocean', 'desert'],
    'history': ['history', 'war', 'empire', 'ancient', 'century', 'revolution', 'king', 'queen', 'duke', 'battle', 'civil war'],
    'home_practical': [
        'clean',
        'repair',
        'household',
        'weeded',
        'watered',
        'mend',
        'sew',
        'fire',
        'candle',
        'lamp',
        'kite',
        'blacksmith',
        'chimney',
        'furnace',
        'stove',
        'plough',
    ],
    'language_writing': ['letter', 'grammar', 'synonym', 'sentence', 'quotation', 'spell', 'word ', 'punctuation', 'definition', 'proverb'],
    'nature_animals': ['horse', 'garden', 'plant', 'flower', 'weather', 'bird', 'animal', 'dog', 'cat', 'tree', 'river', 'farm'],
    'philosophy_ethics': ['philosoph', 'ethic', 'moral', 'meaning of life', 'virtue', 'soul', 'wrong to', 'right to', 'justice', 'truth'],
    'religion_spirituality': ['bible', 'god', 'prayer', 'church', 'angel', 'ghost', 'heaven', 'scripture', 'faith', 'tarot'],
    'science_tech': [
        'steam engine',
        'barometer',
        'thermometer',
        'machine',
        'electric',
        'telegraph',
        'pressure',
        'gravity',
        'chemical',
        'experiment',
        'printing press',
        'telescope',
        'microscope',
        'laboratory',
    ],
    'smalltalk_personal': [
        'your favorite',
        'your favourite',
        'how are you',
        'do you like',
        'yourself',
        'your own life',
        'afraid of',
        'feel',
    ],
    'social_etiquette': ['etiquette', 'manners', 'introduce', 'invitation', 'host', 'guest', 'gentleman', 'lady', 'dinner party'],
    'travel': ['travel', 'journey', 'railway', 'steamship', 'excursion', 'voyage', 'visit ', 'continent', 'tour'],
}


def lexical_report(questions):
    """Opening-verb histogram + keyword-bucket histogram. No model calls."""
    total = len(questions)
    # Opening verb / word: a quick read on FORM-ish monotony.
    verbs = {}
    for q in questions:
        toks = q.split()
        if not toks:
            continue
        w = toks[0].lower().strip('",.?!\'')
        verbs[w] = verbs.get(w, 0) + 1
    print(f'\nOpening word  (n={total}; rough proxy for FORM)')
    print('-' * 60)
    for w, c in sorted(verbs.items(), key=lambda kv: -kv[1])[:18]:
        print(f'  {w:24s} {c:6d}  {c / total * 100:5.1f}%  {_bar(c / total)}')

    # Keyword DOMAIN buckets (non-exclusive).
    lows = [q.lower() for q in questions]
    bucket_counts = {}
    matched_any = [False] * total
    for dom, kws in LEXICAL_BUCKETS.items():
        cnt = 0
        for i, q in enumerate(lows):
            if any(k in q for k in kws):
                cnt += 1
                matched_any[i] = True
        bucket_counts[dom] = cnt
    bucket_counts['(unmatched)'] = sum(1 for m in matched_any if not m)
    print(f'\nDOMAIN by keyword  (n={total}; non-exclusive, a question may hit several)')
    print('-' * 60)
    for dom, c in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
        print(f'  {dom:24s} {c:6d}  {c / total * 100:5.1f}%  {_bar(c / total)}')
    print(
        '\n[note] lexical mode is approximate and double-counts. For clean single labels and the DOMAIN x FORM cross-tab, run --mode llm.'
    )


# ===========================================================================
# Mode 2: llm (batched classification)
# ===========================================================================
DEFAULT_BASE_URL = 'http://127.0.0.1:1234'
OPENROUTER_BASE_URL = 'https://openrouter.ai/api'
_AUTH_HEADERS = {}
_DEBUG = False


def _post_json(url, payload, timeout=300):
    data = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if _AUTH_HEADERS:
        headers.update(_AUTH_HEADERS)
    if _DEBUG:
        print(f'\n[debug] POST {url}\n[debug] user msg:\n{payload["messages"][-1]["content"][:1500]}', file=sys.stderr)
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode('utf-8'))
    if _DEBUG:
        print(f'[debug] response:\n{body["choices"][0]["message"]["content"][:1500]}', file=sys.stderr)
    return body


def detect_model(base_url):
    headers = dict(_AUTH_HEADERS)
    req = urllib.request.Request(base_url.rstrip('/') + '/v1/models', headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode('utf-8'))['data'][0]['id']


def chat(base_url, model, system, user, temperature=0.0, max_tokens=1500):
    payload = {
        'model': model,
        'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        'temperature': temperature,
        'max_tokens': max_tokens,
        'stream': False,
    }
    out = _post_json(base_url.rstrip('/') + '/v1/chat/completions', payload)
    return out['choices'][0]['message']['content']


CLASSIFY_SYSTEM = (
    'You are a precise text classifier. You label each question on two axes and '
    'output ONLY JSON. Use EXACTLY these labels.\n'
    'DOMAIN (what it is about): ' + ', '.join(DOMAINS) + '.\n'
    'FORM (what kind of turn): ' + ', '.join(FORMS) + '.\n'
    'For each numbered question reply with one JSON object {"n": <number>, '
    '"domain": "<domain>", "form": "<form>"}. Output a single JSON array of '
    'these objects, in order, and nothing else.'
)


def _coerce(label, allowed, fallback):
    label = (label or '').strip().lower()
    return label if label in allowed else fallback


def classify_batch(base_url, model, batch, args):
    """Classify a list of question strings. Returns list of (domain, form),
    one per input (falls back to ('other','other') on any parse trouble)."""
    numbered = '\n'.join(f'{i + 1}. {q}' for i, q in enumerate(batch))
    user = f'Classify these {len(batch)} questions:\n\n{numbered}'
    try:
        resp = chat(base_url, model, CLASSIFY_SYSTEM, user, temperature=0.0, max_tokens=40 * len(batch) + 200)
    except (urllib.error.URLError, OSError) as e:
        print(f'    ! classify call failed, marking batch as other: {e}', file=sys.stderr)
        return [('other', 'other')] * len(batch)

    # Robust parse: pull the JSON array out of the reply, tolerate stray prose.
    out = [('other', 'other')] * len(batch)
    m = re.search(r'\[.*\]', resp, re.DOTALL)
    items = []
    if m:
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            items = []
    if not items:
        # Fallback: scan object-by-object.
        for om in re.finditer(r'\{[^{}]*\}', resp, re.DOTALL):
            try:
                items.append(json.loads(om.group(0)))
            except json.JSONDecodeError:
                pass
    for obj in items:
        if not isinstance(obj, dict):
            continue
        try:
            idx = int(obj.get('n')) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(batch):
            out[idx] = (
                _coerce(obj.get('domain'), set(DOMAINS), 'other'),
                _coerce(obj.get('form'), set(FORMS), 'other'),
            )
    return out


def setup_provider(args):
    """Resolve provider/model/auth. Returns (base_url, model)."""
    global _AUTH_HEADERS
    if args.provider == 'openrouter':
        key = os.environ.get('OPENROUTER_API_KEY')
        if not key:
            sys.exit('ERROR: OPENROUTER_API_KEY not set (needed for --provider openrouter).')
        _AUTH_HEADERS = {'Authorization': f'Bearer {key}', 'HTTP-Referer': 'classify-themes', 'X-Title': 'classify-themes'}
        base_url = args.base_url if args.base_url != DEFAULT_BASE_URL else OPENROUTER_BASE_URL
        model = args.model or os.environ.get('OPENROUTER_MODEL')
        if not model:
            sys.exit('ERROR: --model is required for OpenRouter (e.g. deepseek/deepseek-chat).')
        return base_url, model
    # local
    base_url = args.base_url
    try:
        model = args.model or detect_model(base_url)
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f'ERROR: could not reach local server at {base_url} ({e}).')
    return base_url, model


def llm_mode(questions, args):
    base_url, model = setup_provider(args)
    print(f'[info] classifying {len(questions)} questions with {model} ({args.provider}), batch={args.batch_size}')

    # Stream labels to the sidecar as we go so paid work survives a crash.
    labelled = []
    out_fh = open(args.out, 'w', encoding='utf-8') if args.out else None
    try:
        for start in range(0, len(questions), args.batch_size):
            batch = questions[start : start + args.batch_size]
            pairs = classify_batch(base_url, model, batch, args)
            for q, (dom, frm) in zip(batch, pairs):
                labelled.append((dom, frm))
                if out_fh:
                    out_fh.write(json.dumps({'question': q, 'domain': dom, 'form': frm}, ensure_ascii=False) + '\n')
            if out_fh:
                out_fh.flush()
            print(f'  ...{len(labelled)}/{len(questions)} classified', file=sys.stderr)
    finally:
        if out_fh:
            out_fh.close()
    return labelled


def tally_mode(paths):
    """Re-read a labelled sidecar (question/domain/form) and re-tally. No API."""
    labelled = []
    for path in paths:
        for line in open(path, 'r', encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'domain' in r and 'form' in r:
                labelled.append((r['domain'], r['form']))
    if not labelled:
        sys.exit('ERROR: no labelled records found (expected {"domain":..,"form":..} lines).')
    return labelled


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description='Classify the themes (domain x form) of Q&A in a fine-tuning JSONL.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('inputs', nargs='+', help='input JSONL file(s)')
    ap.add_argument(
        '--mode',
        choices=['lexical', 'llm', 'tally'],
        default='lexical',
        help='lexical=free keyword pass; llm=model classification; tally=re-report a labelled sidecar',
    )
    ap.add_argument('--provider', choices=['local', 'openrouter'], default='local', help='llm-mode backend')
    ap.add_argument('--base-url', default=DEFAULT_BASE_URL, help='server base URL')
    ap.add_argument('--model', default=None, help='model id (auto-detected for local)')
    ap.add_argument('--batch-size', type=int, default=20, help='questions per llm classification call')
    ap.add_argument('--out', default=None, help='(llm mode) write per-question labels here for auditing / re-tally')
    ap.add_argument('--debug', action='store_true', help='print each llm request/response to stderr')
    args = ap.parse_args()

    global _DEBUG
    _DEBUG = args.debug

    if args.mode == 'tally':
        labelled = tally_mode(args.inputs)
        report(*tally(labelled))
        return

    questions = list(iter_user_turns(args.inputs))
    if not questions:
        sys.exit('ERROR: no user turns found in input.')

    if args.mode == 'lexical':
        lexical_report(questions)
        return

    # llm mode
    labelled = llm_mode(questions, args)
    report(*tally(labelled))
    if args.out:
        print(f'\n[info] per-question labels written to {args.out} (re-report with: python3 classify_themes.py --mode tally {args.out})')


if __name__ == '__main__':
    main()
