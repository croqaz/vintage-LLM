"""Evaluation loop and CORE-metric aggregation.

Every example is scored into a *record* dict (prompt, raw output, prediction,
gold, correctness, ...). Records power both the correctness metric and the
optional debug trace, so what the model actually saw and produced can always be
inspected.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import prompts, scoring

# Answer-token budgets. MC/schema need only a letter, but verbose chat models
# often add a short preamble ("The correct answer is B") before it, so we give
# enough room for the letter to appear rather than truncating to nothing.
MC_MAX_TOKENS = 24
SCHEMA_MAX_TOKENS = 24


def _generate(client, prompt, use_chat, max_tokens, instruction, stop=None, want_logprobs=False):
    if use_chat:
        return client.chat(prompt, max_tokens=max_tokens, temperature=0.0, stop=stop, system=instruction, want_logprobs=want_logprobs)
    return client.complete(prompt, max_tokens=max_tokens, temperature=0.0, stop=stop, want_logprobs=want_logprobs)


def eval_example_generation(client, task, idx, use_chat, capture_logprobs=False):
    """Score one example by prompting the model to produce an answer.
    Returns a record dict including ``correct``."""
    item = task.data[idx]
    t = task.task_type
    rec = {'task': task.label, 'idx': idx, 'task_type': t, 'mode': 'generation'}
    lps = None

    def gen(prompt, mt, instr, stop=None):
        nonlocal lps
        res = _generate(client, prompt, use_chat, mt, instr, stop=stop, want_logprobs=capture_logprobs)
        if capture_logprobs:
            out, lps = res
            return out
        return res

    if t == 'multiple_choice':
        prompt = prompts.render_generation_mc(item, task.data, idx, task.num_fewshot)
        out = gen(prompt, MC_MAX_TOKENS, prompts.MC_INSTRUCTION)
        pred = scoring.parse_letter(out, len(item['choices']))
        if pred is None and not prompts.is_letter_choices(item['choices']):
            # Model answered with the choice text rather than a letter.
            pred = scoring.match_choice_text(out, item['choices'])
        # 'answered' = the model produced something we could map to a choice;
        # False means a refusal/ramble we couldn't parse (counts as wrong).
        rec.update(prompt=prompt, output=out, pred=pred, gold=item['gold'], correct=pred == item['gold'], answered=pred is not None)
    elif t == 'schema':
        prompt = prompts.render_generation_schema(item, task.data, idx, task.num_fewshot, task.continuation_delimiter)
        out = gen(prompt, SCHEMA_MAX_TOKENS, prompts.SCHEMA_INSTRUCTION)
        pred = scoring.parse_letter(out, 2)
        rec.update(prompt=prompt, output=out, pred=pred, gold=item['gold'], correct=pred == item['gold'], answered=pred is not None)
    elif t == 'language_modeling':
        prompt = prompts.render_generation_lm(item, task.data, idx, task.num_fewshot, task.continuation_delimiter)
        mt = scoring.estimate_max_tokens(item['continuation'])
        out = gen(prompt, mt, prompts.LM_INSTRUCTION, stop=['\n\n'])
        correct = scoring.lm_is_correct(out, item['continuation'])
        # For LM tasks a non-empty (post-normalization) output counts as answered.
        rec.update(prompt=prompt, output=out, gold=item['continuation'], correct=correct, answered=bool(scoring.normalize_answer(out)))
    else:
        raise ValueError(f'unknown task type: {t}')
    if capture_logprobs:
        rec['logprobs'] = lps
    return rec


def eval_example_logprob(client, task, idx, style):
    """Score one example faithfully via per-token prompt logprobs.
    Returns a record dict including ``correct``."""
    item = task.data[idx]
    t = task.task_type
    rec = {'task': task.label, 'idx': idx, 'task_type': t, 'mode': 'logprob'}
    if t in ('multiple_choice', 'schema'):
        if t == 'multiple_choice':
            pl, ans, gold = prompts.render_logprob_mc(item, task.data, idx, task.num_fewshot)
        else:
            pl, ans, gold = prompts.render_logprob_schema(item, task.data, idx, task.num_fewshot, task.continuation_delimiter)
        pred, means = scoring.score_mc_logprob(client, pl, ans, style)
        rec.update(prompts=pl, mean_logprobs=means, pred=pred, gold=gold, correct=pred == gold, answered=True)
    elif t == 'language_modeling':
        prompt, ans, cont = prompts.render_logprob_lm(item, task.data, idx, task.num_fewshot, task.continuation_delimiter)
        correct = scoring.lm_greedy_correct(client, prompt, ans, style)
        rec.update(prompt=prompt, gold=cont, correct=correct, answered=True)
    else:
        raise ValueError(f'unknown task type: {t}')
    return rec


def evaluate_task(
    client, task, mode, use_chat, style, concurrency=8, max_per_task=-1, progress=True, on_record=None, capture_logprobs=False
):
    """Evaluate one task. Returns (accuracy, records).

    ``on_record`` (if given) is called with each record as it completes — used to
    stream a debug trace to disk without holding everything in memory.
    """
    n = len(task.data) if max_per_task < 0 else min(max_per_task, len(task.data))

    def run(idx):
        try:
            if mode == 'logprob':
                return eval_example_logprob(client, task, idx, style)
            return eval_example_generation(client, task, idx, use_chat, capture_logprobs=capture_logprobs)
        except Exception as e:
            # A single request failing (transient API error, bad response, ...)
            # must not abort the whole run. Record it as an unanswered error.
            return {
                'task': task.label,
                'idx': idx,
                'task_type': task.task_type,
                'mode': mode,
                'correct': False,
                'answered': False,
                'error': f'{type(e).__name__}: {e}'[:500],
            }

    records = [None] * n
    correct = 0
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(run, i): i for i in range(n)}
        for fut in as_completed(futures):
            idx = futures[fut]
            rec = fut.result()
            records[idx] = rec
            correct += int(bool(rec['correct']))
            done += 1
            if on_record is not None:
                on_record(rec)
            if progress and (done % 25 == 0 or done == n):
                sys.stderr.write(f'\r  {task.label}: {done}/{n}')
                sys.stderr.flush()
    if progress:
        sys.stderr.write('\n')
    return (correct / n if n else 0.0), records


def centered(accuracy, random_baseline_pct):
    b = 0.01 * random_baseline_pct
    return (accuracy - b) / (1.0 - b) if b < 1.0 else accuracy


def core_metric(centered_results):
    if not centered_results:
        return 0.0
    return sum(centered_results.values()) / len(centered_results)
