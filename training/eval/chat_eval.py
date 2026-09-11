"""Chat-behaviour evaluation: does the model actually hold a conversation?

    python -m eval.chat_eval MODEL [MODEL ...]
    python -m eval.chat_eval ../autoresearch/ft07-winners-141m/full-pad/sft_checkpoints/final \
        --judge ../Llama-141M/final --out chat.json

WHY THIS EXISTS. Every other metric in this package is likelihood on text WE
supply. `chat_target_bpb` renders a conversation and scores the reference
assistant reply token by token; the model never emits anything. So a model can
win a 26-arm hyperparameter sweep and still be unusable, because nothing has
ever checked whether it stops talking, respects turn boundaries, or does what
it was asked.

Five things are measured here, all from text the model actually GENERATES:

  1. STOPPING     does EOS fire inside the budget, and after how many tokens
  2. TURN SAFETY  does the reply leak <|user|>/<|system|>/<|assistant|> and
                  carry on a conversation with itself
  3. INSTRUCTION  mechanical constraint checks (word counts, yes/no, lists)
  4. FORMAT       empty, whitespace-led, unterminated, special-token leakage
  5. VINTAGE      bpb of the GENERATED replies under a judge model, against the
                  same judge's bpb on real held-out prose. Style collapse shows
                  up in generation long before it shows up in likelihood.

READ THE SIGN OF THE JUDGE SCORE, not just its magnitude.
  positive  the replies are HARDER for the vintage judge than real prose:
            the model has drifted out of the pretraining distribution, which
            is the modernisation failure this metric was built to catch.
  negative  the replies are EASIER than real prose, i.e. blander, more
            repetitive, more generic. Measured on 141M, every fine-tune came
            out negative (-0.17 to -0.38) and the untuned base worst of all
            (-0.60, because it degenerates into "THE END OF THE FIRST VOLUME"
            loops). So near zero is good, but a large negative number means
            flat writing, NOT modern writing.

Constraint probes are deliberately NOT factual QA. A 141M model fails factual
recall for reasons that say nothing about instruction-following, and the noise
would swamp the signal. Prompts are written in period register so the question
itself does not push the model out of distribution, which would confound (5).

Greedy by default so runs are comparable. --sample uses a fixed temperature;
the judge score in particular moves with temperature, so do not compare a
sampled run against a greedy one.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

if __package__ in (None, ''):  # allow running the file directly, not just -m
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.helpers import (  # noqa: E402
    DEFAULT_HELDOUT,
    EVAL_DATA,
    free_model,
    load_model_and_tokenizer,
    load_text_items,
    resolve_tokenizer,
)
from eval import metrics as M  # noqa: E402

DEFAULT_PROBES = EVAL_DATA / 'chat_probes.jsonl'
SENT_END = re.compile(r'[.!?]["\')\]]*\s*$')
SENT_SPLIT = re.compile(r'[.!?]+(?:\s|$)')
WORD = re.compile(r"[A-Za-z0-9']+")
# Role markers a chat template can emit. Matched as literal text because a
# merged model's tokenizer may or may not keep them as single ids. Both
# dialects are listed: ours uses bare role names, nanochat uses paired
# start/end markers, and a reply is leaking either way.
OUR_ROLE_MARKERS = ('<|user|>', '<|system|>', '<|assistant|>', '<|bos|>', '<|pad|>', '<|mask|>')
NANOCHAT_ROLE_MARKERS = (
    '<|user_start|>',
    '<|user_end|>',
    '<|assistant_start|>',
    '<|python_start|>',
    '<|output_start|>',
)
ROLE_MARKERS = OUR_ROLE_MARKERS + NANOCHAT_ROLE_MARKERS
# A model that opens a NEW user turn is answering itself, which is the
# damaging case; a stray pad marker is untidy by comparison.
SELF_DIALOGUE_MARKERS = ('<|user|>', '<|system|>', '<|user_start|>')


# ============================================================================
# Checks
# ============================================================================


def _words(t):
    return WORD.findall(t)


def _sentences(t):
    return [s for s in SENT_SPLIT.split(t.strip()) if s.strip()]


def _lines(t):
    return [ln for ln in t.strip().splitlines() if ln.strip()]


def _normalize(t):
    return re.sub(r'[^a-z0-9 ]', '', t.strip().lower()).strip()


def repetition_rate(text, n=6):
    """Share of n-word windows that have already appeared verbatim.

    A looping reply is the failure a reader spots in one second and no
    likelihood metric reports. 0.0 is clean prose; above about 0.3 the reply
    is visibly repeating itself.
    """
    words = [w.lower() for w in _words(text)]
    if len(words) < n * 2:
        return 0.0
    windows = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(windows)) / len(windows)


def _list_items(t):
    """Count enumerated items: comma/semicolon/newline separated, or 1. 2. 3."""
    numbered = re.findall(r'(?m)^\s*(?:\d+[.)]|[-*•])\s+', t)
    if len(numbered) >= 2:
        return len(numbered)
    parts = [p for p in re.split(r'[,;\n]| and ', t) if _words(p)]
    return len(parts)


CHECKS = {
    'max_words': lambda t, v: len(_words(t)) <= v,
    'min_words': lambda t, v: len(_words(t)) >= v,
    'max_sentences': lambda t, v: len(_sentences(t)) <= v,
    'min_sentences': lambda t, v: len(_sentences(t)) >= v,
    'min_list_items': lambda t, v: _list_items(t) >= v,
    'max_list_items': lambda t, v: _list_items(t) <= v,
    'starts_with_any': lambda t, v: any(t.strip().lower().lstrip('"\'').startswith(x) for x in v),
    # Substring. Fine for stems and punctuation ("injur", "?"), WRONG for whole
    # words: "romantic" contains "roma", "thinking" contains "king", "fourteen"
    # contains "four". Use contains_word_any for anything that is a word.
    'contains_any': lambda t, v: any(x in t.lower() for x in v),
    'contains_word_any': lambda t, v: any(re.search(rf'\b{re.escape(x)}\b', t.lower()) for x in v),
    'contains_word_none': lambda t, v: not any(re.search(rf'\b{re.escape(x)}\b', t.lower()) for x in v),
    # Case matters for this one; everything else compares lowercased.
    'contains_cased_any': lambda t, v: any(x in t for x in v),
    # "Do not mention X" is the constraint small models break most often, and
    # it is the one a user notices first.
    'contains_none': lambda t, v: not any(x in t.lower() for x in v),
    'max_lines': lambda t, v: len(_lines(t)) <= v,
    'min_lines': lambda t, v: len(_lines(t)) >= v,
    'every_line_starts_with_digit': lambda t, v: bool(_lines(t)) and all(re.match(r'\s*\d', ln) for ln in _lines(t)) == bool(v),
    'is_upper': lambda t, v: (t.strip() == t.strip().upper() and any(c.isalpha() for c in t)) == bool(v),
    # Exact-answer probes. Compared on letters and digits only, so punctuation
    # and a trailing full stop never decide the result.
    'equals_any': lambda t, v: _normalize(t) in {_normalize(x) for x in v},
}


def run_checks(text, checks):
    return [{'type': c['type'], 'value': c['value'], 'passed': bool(CHECKS[c['type']](text, c['value']))} for c in checks]


# ============================================================================
# Generation
# ============================================================================


def generate_replies(tokenizer, model, probes, max_new_tokens, sample, temperature, top_p, seed, reply_after=None, repeat=1):
    """One reply per probe, chat template applied, EOS NEVER forced.

    min_new_tokens is deliberately absent. The whole point is to find out
    whether the model stops on its own; forcing a floor would hide exactly the
    failure this module exists to catch.
    """
    if tokenizer.chat_template is None:
        raise SystemExit('this tokenizer has no chat_template; chat_eval is meaningless for a base model')
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
    pad_id = model.generation_config.pad_token_id
    if pad_id is None:
        pad_id = getattr(tokenizer, 'pad_token_id', None) or 0

    def one_turn(history, run_seed):
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors='pt', add_special_tokens=False).to(model.device)
        n_in = enc.input_ids.shape[1]
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=pad_id, use_cache=True)
        if sample:
            torch.manual_seed(run_seed)
            if model.device.type == 'cuda':
                torch.cuda.manual_seed_all(run_seed)
            kw.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            g = model.generate(**enc, **kw)
        new_ids = g[0, n_in:].tolist()
        stopped = bool(new_ids) and new_ids[-1] in eos_ids
        # Trim a trailing stop token before decoding so text metrics see the reply.
        body = new_ids[:-1] if stopped else new_ids
        text = tokenizer.decode(body, skip_special_tokens=False)
        # Some models emit a preamble before the reply proper and were trained
        # that way. Violet writes a mood line and then <|assistant|>. Cutting
        # at that marker measures the answer rather than the scaffolding, and
        # stops the marker being counted as a leak.
        if reply_after and reply_after in text:
            text = text.split(reply_after, 1)[1]
        return text, len(new_ids), stopped

    def one_attempt(p, run_seed):
        # A multi-turn probe carries `followup`. The model's OWN first reply
        # becomes the context, which is the only honest way to ask whether it
        # can hold a thread; feeding it a scripted reply would test something
        # easier than what a user does.
        turns = [p['prompt']] + ([p['followup']] if p.get('followup') else [])
        history, replies = [], []
        for turn in turns:
            history.append({'role': 'user', 'content': turn})
            reply, n_new, stopped = one_turn(history, run_seed)
            replies.append({'reply': reply, 'new_tokens': n_new, 'stopped': stopped})
            history.append({'role': 'assistant', 'content': reply})

        last = replies[-1]
        return {
            'id': p['id'],
            'tag': p['tag'],
            'seed': run_seed,
            'prompt': turns[-1],
            'first_prompt': turns[0],
            'reply': last['reply'],
            'turns': len(turns),
            'all_replies': [r['reply'] for r in replies],
            'new_tokens': last['new_tokens'],
            'stopped': all(r['stopped'] for r in replies),
            'truncated': not last['stopped'] and last['new_tokens'] >= max_new_tokens,
            'repetition': repetition_rate(last['reply']),
            'checks': run_checks(last['reply'], p.get('checks', [])),
        }

    out = []
    for p in probes:
        # Seeds are seed, seed+1, ... so a repeat run is reproducible while
        # still being a different draw each time, which is what a person gets.
        attempts = [one_attempt(p, seed + i) for i in range(repeat)]
        if repeat == 1:
            out.append(attempts[0])
            continue
        passed = [bool(a['checks']) and all(c['passed'] for c in a['checks']) for a in attempts]
        # The representative row is the FIRST draw, not the best one. Picking
        # the best would report the model a lucky user meets, which is the
        # illusion this whole mode exists to dispel.
        row = dict(attempts[0])
        row['attempts'] = [
            {'seed': a['seed'], 'reply': a['reply'], 'stopped': a['stopped'], 'passed': ok} for a, ok in zip(attempts, passed, strict=True)
        ]
        row['n_attempts'] = repeat
        row['n_passed'] = sum(passed) if any(a['checks'] for a in attempts) else None
        row['stop_count'] = sum(a['stopped'] for a in attempts)
        out.append(row)
    return out


# ============================================================================
# Scoring
# ============================================================================


def score(samples):
    n = len(samples)
    stopped = [s for s in samples if s['stopped']]
    lens = [s['new_tokens'] for s in stopped]

    def leak(s):
        return [m for m in ROLE_MARKERS if m in s['reply']]

    leaked = [s for s in samples if leak(s)]
    self_dialogue = [s for s in samples if any(m in s['reply'] for m in SELF_DIALOGUE_MARKERS)]

    checked = [s for s in samples if s['checks']]
    all_checks = [c for s in checked for c in s['checks']]

    multiturn = [s for s in samples if s.get('turns', 1) > 1]
    empty = [s for s in samples if not s['reply'].strip()]
    unterminated = [s for s in samples if s['reply'].strip() and not SENT_END.search(s['reply'])]

    by_tag = {}
    for s in checked:
        t = by_tag.setdefault(s['tag'], [0, 0])
        t[0] += sum(c['passed'] for c in s['checks'])
        t[1] += len(s['checks'])

    return {
        'n_probes': n,
        # 1. stopping
        'stop_rate': len(stopped) / n,
        'truncation_rate': sum(s['truncated'] for s in samples) / n,
        'mean_stop_tokens': (sum(lens) / len(lens)) if lens else None,
        'max_stop_tokens': max(lens) if lens else None,
        # 2. turn safety
        'token_leak_rate': len(leaked) / n,
        'self_dialogue_rate': len(self_dialogue) / n,
        # 3. instruction following
        'n_constrained_probes': len(checked),
        'check_pass_rate': (sum(c['passed'] for c in all_checks) / len(all_checks)) if all_checks else None,
        'probes_all_checks_passed': (sum(all(c['passed'] for c in s['checks']) for s in checked) / len(checked)) if checked else None,
        'check_pass_rate_by_tag': {k: (v[0] / v[1]) for k, v in sorted(by_tag.items())},
        # 4. format
        'empty_rate': len(empty) / n,
        'unterminated_rate': len(unterminated) / n,
        'mean_reply_words': sum(len(_words(s['reply'])) for s in samples) / n,
        # 5. looping
        'mean_repetition': sum(s.get('repetition', 0.0) for s in samples) / n,
        'looping_rate': sum(s.get('repetition', 0.0) > 0.30 for s in samples) / n,
        # 6. multi-turn, scored separately because a model can pass every
        # single-turn probe and still lose the thread on the second question.
        'n_multiturn_probes': len(multiturn),
        'multiturn_stop_rate': (sum(s['stopped'] for s in multiturn) / len(multiturn)) if multiturn else None,
        'multiturn_check_pass_rate': (
            (sum(c['passed'] for s in multiturn for c in s['checks']) / sum(len(s['checks']) for s in multiturn))
            if any(s['checks'] for s in multiturn)
            else None
        ),
        'multiturn_looping_rate': (sum(s.get('repetition', 0.0) > 0.30 for s in multiturn) / len(multiturn)) if multiturn else None,
    }


# Weights for chat_capability. Hand-set, not fitted to anything, and the six
# parts are always reported beside the total so the total can be argued with.
#
# The ordering is a claim about what a person notices. A model that never
# stops is unusable whatever else it does, so stopping is weighted most. Then
# following the instruction, because that is what separates a chat model from
# a text continuer. Turn safety is a hard penalty rather than a scored axis:
# a reply that opens a new user turn and answers itself is broken, not weak.
CAPABILITY_WEIGHTS = {
    'stopping': 0.30,
    'instruction': 0.30,
    'multiturn': 0.15,
    'no_looping': 0.10,
    'format': 0.10,
    'brevity_control': 0.05,
}


def reliability(samples):
    """How a model behaves when a person asks the same thing twice.

    A model that answers correctly nine times in ten is a good model that will
    look like a bad one to whoever draws the tenth. Averaging the draws hides
    that; so does reporting only the best. Both numbers belong in the table.

    Returns None unless the run actually repeated, so a greedy run is not
    given a reliability score it did not earn.
    """
    repeated = [s for s in samples if s.get('n_passed') is not None]
    if not repeated:
        return None
    n = len(repeated)
    rates = [s['n_passed'] / s['n_attempts'] for s in repeated]
    always = [s for s in repeated if s['n_passed'] == s['n_attempts']]
    never = [s for s in repeated if s['n_passed'] == 0]
    flaky = [s for s in repeated if 0 < s['n_passed'] < s['n_attempts']]
    attempts = sum(s['n_attempts'] for s in repeated)
    return {
        'n_repeated_probes': n,
        'attempts_per_probe': repeated[0]['n_attempts'],
        # What one random draw gets you: the honest expectation for a person
        # typing a question once.
        'expected_pass_rate': sum(rates) / n,
        # The three bands. always_right is the model you can rely on; flaky is
        # the band where a single unlucky seed misrepresents the model; and
        # never_right is genuine inability, not luck.
        'always_right': len(always) / n,
        'sometimes_right': len(flaky) / n,
        'never_right': len(never) / n,
        # Best case minus expected: how much better the model looks to someone
        # who retries than to someone who does not.
        'retry_premium': (len(always) + len(flaky)) / n - (sum(rates) / n),
        'worst_probe': min(repeated, key=lambda s: s['n_passed'])['id'],
        'stop_rate_over_attempts': sum(s['stop_count'] for s in repeated) / attempts if attempts else None,
    }


def chat_capability(summary, samples):
    """One 0-100 number for "can a person hold a conversation with this".

    Built only from things a reader would also notice: whether it stops,
    whether it does what it was told, whether it survives a second question,
    whether it repeats itself, whether the reply is well formed, and whether
    it can be brief when asked. Vintage fidelity is deliberately absent; the
    rest of the evaluator answers that, and mixing the two produces a number
    that cannot be acted on.

    Missing parts are dropped and the remaining weights renormalised, so a
    probe set without multi-turn items still yields a comparable score. The
    coverage is reported next to it.
    """

    def rate(key, default=None):
        value = summary.get(key)
        return default if value is None else value

    brief = [s for s in samples if any(c['type'] in ('max_words', 'max_sentences', 'max_lines') for c in s['checks'])]
    brief_pass = None
    if brief:
        checks = [c for s in brief for c in s['checks'] if c['type'] in ('max_words', 'max_sentences', 'max_lines')]
        brief_pass = sum(c['passed'] for c in checks) / len(checks)

    parts = {
        'stopping': rate('stop_rate'),
        'instruction': rate('check_pass_rate'),
        'multiturn': rate('multiturn_check_pass_rate'),
        'no_looping': None if rate('looping_rate') is None else 1.0 - rate('looping_rate'),
        'format': None if rate('empty_rate') is None else 1.0 - (rate('empty_rate', 0.0) + rate('unterminated_rate', 0.0)) / 2,
        'brevity_control': brief_pass,
    }
    available = {k: v for k, v in parts.items() if v is not None}
    covered = sum(CAPABILITY_WEIGHTS[k] for k in available)
    if not covered:
        return {'chat_capability': None, 'chat_capability_parts': parts, 'chat_capability_coverage': 0.0}

    score_0_1 = sum(CAPABILITY_WEIGHTS[k] * v for k, v in available.items()) / covered
    # Turn safety is a multiplier, not an addend. Leaking a role marker once
    # in twenty replies should not be averaged away against a good stop rate.
    penalty = 1.0 - min(1.0, rate('self_dialogue_rate', 0.0) + 0.5 * rate('token_leak_rate', 0.0))
    return {
        'chat_capability': round(100 * score_0_1 * penalty, 2),
        'chat_capability_parts': {k: (None if v is None else round(v, 4)) for k, v in parts.items()},
        'chat_capability_turn_safety': round(penalty, 4),
        'chat_capability_coverage': round(covered, 3),
    }


def judge_vintage(judge_dir, tokenizer_override, device, dtype, samples, heldout_texts, max_tokens=320):
    """bpb of GENERATED replies under a judge model, against real prose.

    The judge should be the PRE-FINE-TUNE base model. Then `deviation` reads as
    "how unlike its own pretraining distribution has this model's speech become".
    """
    jdir = Path(judge_dir)
    jtok = resolve_tokenizer(jdir, tokenizer_override)
    jmodel, jtokz = load_model_and_tokenizer(jdir, jtok, device, dtype)
    try:
        ref = M.bits_per_byte_of_texts(jtokz, jmodel, heldout_texts[:100], max_tokens=max_tokens)
        texts = [s['reply'] for s in samples if len(s['reply'].split()) >= 15]
        if not texts:
            return {
                'judge': str(jdir),
                'reference_bpb': ref,
                'generated_bpb': None,
                'deviation_bpb': None,
                'n_scored': 0,
                'note': 'no reply reached 15 words; nothing to judge',
            }
        gen = M.bits_per_byte_of_texts(jtokz, jmodel, texts, max_tokens=max_tokens)
        return {'judge': str(jdir), 'reference_bpb': ref, 'generated_bpb': gen, 'deviation_bpb': gen - ref, 'n_scored': len(texts)}
    finally:
        free_model(jmodel, device)


# ============================================================================
# Report
# ============================================================================


def render_reliability_md(results):
    """The repeat-sampling table: what a person meets, not what one draw shows."""
    L = ['# Reliability under repeated asking', '']
    rel0 = next((r['reliability'] for r in results if r.get('reliability')), None)
    if rel0 is None:
        return ''
    L += [
        f'Every probe asked {rel0["attempts_per_probe"]} times at the run temperature, seeds '
        f'seed..seed+{rel0["attempts_per_probe"] - 1}. A model that is right nine times in ten is a good '
        'model that will look like a bad one to whoever draws the tenth, so the bands matter more than the average.',
        '',
        '| model | expected per try | always right | sometimes right | never right | retry premium |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    ranked = sorted((r for r in results if r.get('reliability')), key=lambda r: -r['reliability']['expected_pass_rate'])
    for r in ranked:
        v = r['reliability']
        L.append(
            f'| {r["label"]} | **{v["expected_pass_rate"]:.0%}** | {v["always_right"]:.0%} | '
            f'{v["sometimes_right"]:.0%} | {v["never_right"]:.0%} | +{v["retry_premium"]:.0%} |'
        )
    L += [
        '',
        'expected per try = share of attempts that passed, i.e. what one question gets you.',
        'always / sometimes / never = share of PROBES the model got right every time, some of',
        'the time, and none of the time. "sometimes" is the band where a single unlucky seed',
        'misrepresents the model. "never" is inability, not luck. retry premium = how much',
        'better the model looks to someone who asks twice than to someone who asks once.',
        '',
    ]
    for r in ranked:
        L += [f'## {r["label"]}', '', '| probe | passed | replies |', '|---|---:|---|']
        for s in r['samples']:
            if s.get('n_passed') is None:
                continue
            seen = []
            for a in s['attempts']:
                text = a['reply'].strip().replace('\n', ' / ').replace('|', '\\|')[:70]
                seen.append(('OK ' if a['passed'] else 'x ') + text)
            L.append(f'| {s["id"]} | {s["n_passed"]}/{s["n_attempts"]} | ' + '<br>'.join(seen) + ' |')
        L.append('')
    return '\n'.join(L)


def render_md(results):
    L = ['# Chat-behaviour evaluation', '']
    L += ['Generated text, not likelihood. Greedy unless a run says otherwise.', '']
    if any(r.get('reliability') for r in results):
        L += [render_reliability_md(results), '']
        return '\n'.join(L)
    L += [
        '| model | capability | stop | checks | multi-turn | loops | leak | self-talk | unterm | judge dev |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in sorted(results, key=lambda x: -(x['summary'].get('chat_capability') or -1)):
        s, j = r['summary'], r.get('vintage') or {}
        dev = j.get('deviation_bpb')

        def pct(value):
            return 'n/a' if value is None else f'{value:.0%}'

        cap = s.get('chat_capability')
        L.append(
            f'| {r["label"]} | **{"n/a" if cap is None else f"{cap:.1f}"}** | {pct(s["stop_rate"])} | '
            f'{pct(s["check_pass_rate"])} | {pct(s.get("multiturn_check_pass_rate"))} | '
            f'{pct(s.get("looping_rate"))} | {pct(s["token_leak_rate"])} | {pct(s["self_dialogue_rate"])} | '
            f'{pct(s["unterminated_rate"])} | {(f"{dev:+.3f}" if dev is not None else "n/a")} |'
        )
    L += [
        '',
        'capability = weighted composite of the columns a reader would notice, 0-100,',
        'stopping 30 / instruction 30 / multi-turn 15 / no looping 10 / format 10 /',
        'brevity 5, multiplied by a turn-safety penalty. It says nothing about vintage',
        'fidelity; the rest of the evaluator answers that.',
        '',
        'stop = EOS fired inside the budget (higher is better). leak = reply contained a',
        'role marker. self-talk = model opened a user or system turn and answered itself.',
        'checks = mechanical constraint pass rate. judge dev = bpb of generated replies minus',
        "the judge's bpb on real held-out prose. Read the sign: POSITIVE means the replies are",
        'harder for the vintage judge than real prose, i.e. drifted modern. NEGATIVE means',
        'easier, i.e. blander and more repetitive. Near zero is good; large negative is flat',
        'writing, not modern writing.',
        '',
    ]
    for r in results:
        L += [f'## {r["label"]}', '']
        parts = r['summary'].get('chat_capability_parts') or {}
        if parts:
            shown = ', '.join(f'{k} {v:.0%}' for k, v in parts.items() if v is not None)
            L += [f'capability parts: {shown} (turn safety x{r["summary"].get("chat_capability_turn_safety", 1):.2f})', '']
        by_tag = r['summary']['check_pass_rate_by_tag']
        if by_tag:
            L += ['constraint pass rate by kind: ' + ', '.join(f'{k} {v:.0%}' for k, v in by_tag.items()), '']
        L += ['| probe | stop | tok | rep | checks | reply |', '|---|---|---:|---:|---|---|']
        for s in r['samples']:
            ok = '' if not s['checks'] else ('pass' if all(c['passed'] for c in s['checks']) else 'FAIL')
            body = s['reply'].strip().replace('\n', ' / ').replace('|', '\\|')
            turns = '' if s.get('turns', 1) == 1 else f' (turn {s["turns"]})'
            L.append(
                f'| {s["id"]}{turns} | {"y" if s["stopped"] else "NO"} | {s["new_tokens"]} | '
                f'{s.get("repetition", 0.0):.2f} | {ok} | {body[:160]} |'
            )
        L.append('')
    return '\n'.join(L)


def main():
    p = argparse.ArgumentParser(prog='python -m eval.chat_eval', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('models', nargs='+', type=Path)
    p.add_argument('--probes', type=Path, default=DEFAULT_PROBES)
    p.add_argument('--judge', type=Path, default=None, help='base model to score generated text against; use the PRE-fine-tune checkpoint')
    p.add_argument('--heldout', type=Path, default=DEFAULT_HELDOUT)
    p.add_argument('--tokenizer', type=Path, default=None)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument(
        '--reply-after',
        default=None,
        help='cut each reply at this marker and score only what follows (Violet: "<|assistant|>")',
    )
    p.add_argument('--sample', action='store_true', help='sample instead of greedy (not comparable across runs)')
    p.add_argument(
        '--repeat',
        type=int,
        default=1,
        help='ask every probe this many times with seeds seed, seed+1, ... Implies --sample. '
        'Measures what a person meets rather than what one lucky draw shows.',
    )
    p.add_argument('--only-tags', default=None, help='comma-separated probe tags to keep, e.g. fact,closed,transform')
    p.add_argument(
        '--exclude-tags',
        default=None,
        help='comma-separated probe tags to drop. Use "fact" for a capability run: knowing the capital '
        'of Spain is a size and corpus question, not an instruction-following one.',
    )
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()

    if not a.probes.exists():
        raise SystemExit(f'probe file not found: {a.probes}')
    probes = [json.loads(l) for l in a.probes.read_text().splitlines() if l.strip()]
    if a.only_tags:
        keep = {t.strip() for t in a.only_tags.split(',')}
        probes = [p for p in probes if p['tag'] in keep]
        if not probes:
            raise SystemExit(f'no probes with tag(s) {sorted(keep)} in {a.probes}')
    if a.exclude_tags:
        drop = {t.strip() for t in a.exclude_tags.split(',')}
        probes = [p for p in probes if p['tag'] not in drop]
        if not probes:
            raise SystemExit(f'every probe in {a.probes} was excluded by {sorted(drop)}')
    if a.repeat < 1:
        raise SystemExit('--repeat must be at least 1')
    if a.repeat > 1 and not a.sample:
        # Repeating a greedy decode gives the same answer every time, which
        # would report perfect reliability for a model that has none.
        a.sample = True
        print(f'--repeat {a.repeat} implies sampling; temperature {a.temperature}, top-p {a.top_p}')

    device = torch.device('cuda' if (a.device == 'auto' and torch.cuda.is_available()) else ('cpu' if a.device == 'auto' else a.device))
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    heldout = [t.text for t in load_text_items(a.heldout, 100)] if a.heldout.exists() else []

    results = []
    for mpath in a.models:
        tok_dir = resolve_tokenizer(mpath, a.tokenizer)
        print(f'loading {mpath} ...')
        model, tokenizer = load_model_and_tokenizer(mpath, tok_dir, device, dtype)
        # Training configs ship use_cache=false; generation wants it on.
        model.config.use_cache = True
        try:
            samples = generate_replies(
                tokenizer, model, probes, a.max_new_tokens, a.sample, a.temperature, a.top_p, a.seed, a.reply_after, a.repeat
            )
        finally:
            free_model(model, device)
        base_summary = score(samples)
        r = {
            'label': mpath.parent.parent.name if mpath.name in ('final', 'final_merged') else mpath.name,
            'checkpoint': str(mpath.resolve()),
            'decoding': 'sample' if a.sample else 'greedy',
            'reply_after': a.reply_after,
            'repeat': a.repeat,
            'n_probes': len(probes),
            'probe_tags': sorted({p['tag'] for p in probes}),
            'temperature': a.temperature if a.sample else None,
            'seed': a.seed,
            'max_new_tokens': a.max_new_tokens,
            'samples': samples,
            # chat_capability is deliberately absent from a repeat run. It would be
            # computed from the first draw of a probe SUBSET, which is neither the
            # capability question nor the reliability one, and having the number
            # sitting in the JSON is an invitation to quote it.
            'summary': {**base_summary, **({} if a.repeat > 1 else chat_capability(base_summary, samples))},
            'reliability': reliability(samples),
        }
        if a.judge and heldout:
            r['vintage'] = judge_vintage(a.judge, a.tokenizer, device, dtype, samples, heldout)
        results.append(r)
        s = r['summary']
        cpr = s['check_pass_rate']
        cpr_s = 'n/a' if cpr is None else f'{cpr:.0%}'
        cap = s.get('chat_capability')
        cap_s = 'n/a' if cap is None else f'{cap:.1f}'
        rel = r.get('reliability')
        if rel:
            print(
                f'  {r["label"]}: expected {rel["expected_pass_rate"]:.0%} per try | '
                f'always {rel["always_right"]:.0%}, sometimes {rel["sometimes_right"]:.0%}, never {rel["never_right"]:.0%}'
            )
        else:
            print(
                f'  {r["label"]}: capability {cap_s} | stop {s["stop_rate"]:.0%}, '
                f'leak {s["token_leak_rate"]:.0%}, checks {cpr_s}, loops {s["looping_rate"]:.0%}'
            )

    out = a.out or Path('chat_eval.json')
    out.write_text(json.dumps({'results': results}, indent=2))
    out.with_suffix('.md').write_text(render_md(results))
    print(f'JSON: {out}\nMD:   {out.with_suffix(".md")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
