"""Prompt rendering for both scoring modes.

Two families of renderers:

* ``render_generation_*`` build a natural prompt that asks the model to *produce*
  an answer (a choice letter, or the completion text). This works on any model,
  including chat-only APIs.
* ``render_logprob_*`` reproduce the nanochat/DCLM ICL prompts used by the
  faithful logprob scorer (score each candidate continuation by its per-token
  logprob).

Few-shot example selection mirrors nanochat exactly: for example ``idx`` we seed
``random.Random(1234 + idx)`` and sample ``num_fewshot`` other indices, so runs
are deterministic and comparable across models.
"""

import random

LETTERS = 'ABCDEFGH'

MC_INSTRUCTION = 'Answer the following multiple-choice question. Respond with only the single letter of the correct choice.'
SCHEMA_INSTRUCTION = (
    'Two versions of a sentence are given. Choose the one that is more logical and coherent. Respond with only the single letter A or B.'
)
LM_INSTRUCTION = 'Complete the following exactly. Provide only the missing text, nothing else.'


def sample_fewshot(data, idx, num_fewshot):
    if num_fewshot <= 0:
        return []
    rng = random.Random(1234 + idx)
    available = [i for i in range(len(data)) if i != idx]
    if num_fewshot >= len(available):
        chosen = available
    else:
        chosen = rng.sample(available, num_fewshot)
    return [data[i] for i in chosen]


# ---------------------------------------------------------------------------
# helpers for multiple choice
def is_letter_choices(choices):
    """True when the JSONL already inlined the options and choices are bare letters."""
    return all(isinstance(c, str) and len(c) == 1 and c in LETTERS for c in choices)


def mc_gold_letter(item):
    choices = item['choices']
    if is_letter_choices(choices):
        return choices[item['gold']]
    return LETTERS[item['gold']]


def _render_mc_block(item, include_answer):
    """Render one MC question (optionally with its answer) as a text block."""
    choices = item['choices']
    if is_letter_choices(choices):
        # query already contains the enumerated options and an 'Answer:' cue.
        block = item['query'].rstrip()
        if not block.rstrip().endswith(':'):
            block += '\nAnswer:'
        else:
            block += ' '
    else:
        lines = [item['query'].rstrip()]
        for i, c in enumerate(choices):
            lines.append(f'{LETTERS[i]}. {c}')
        lines.append('Answer:')
        block = '\n'.join(lines)
    if include_answer:
        block = block.rstrip() + ' ' + mc_gold_letter(item)
    return block


def render_generation_mc(item, data, idx, num_fewshot):
    fewshot = sample_fewshot(data, idx, num_fewshot)
    blocks = [_render_mc_block(ex, include_answer=True) for ex in fewshot]
    blocks.append(_render_mc_block(item, include_answer=False))
    return '\n\n'.join(blocks)


# ---------------------------------------------------------------------------
# schema
def _render_schema_block(item, delimiter, include_answer):
    a = item['context_options'][0].rstrip() + delimiter + item['continuation']
    b = item['context_options'][1].rstrip() + delimiter + item['continuation']
    block = f'A. {a}\nB. {b}\nAnswer:'
    if include_answer:
        block = block + ' ' + LETTERS[item['gold']]
    return block


def render_generation_schema(item, data, idx, num_fewshot, delimiter):
    fewshot = sample_fewshot(data, idx, num_fewshot)
    blocks = [_render_schema_block(ex, delimiter, include_answer=True) for ex in fewshot]
    blocks.append(_render_schema_block(item, delimiter, include_answer=False))
    return '\n\n'.join(blocks)


def schema_gold_letter(item):
    return LETTERS[item['gold']]


# ---------------------------------------------------------------------------
# language modeling
def render_generation_lm(item, data, idx, num_fewshot, delimiter):
    """Faithful ICL prompt: few-shot (context+delim+continuation) then the query
    context+delim, left for the model to complete."""
    fewshot = sample_fewshot(data, idx, num_fewshot)
    parts = []
    for ex in fewshot:
        parts.append(ex['context'].strip() + delimiter + ex['continuation'])
    parts.append(item['context'].strip() + delimiter)
    body = '\n\n'.join(parts[:-1])
    if body:
        return body + '\n\n' + parts[-1]
    return parts[-1]


# ---------------------------------------------------------------------------
# logprob (faithful) mode: enumerate candidate full prompts + the answer span text.
def render_logprob_mc(item, data, idx, num_fewshot):
    """Return (prompts, answer_texts, gold). Each prompt = context + one choice;
    answer_texts[i] is the substring whose logprob should be averaged."""
    fewshot = sample_fewshot(data, idx, num_fewshot)
    prefix_blocks = [_render_mc_block(ex, include_answer=True) for ex in fewshot]
    prefix = ('\n\n'.join(prefix_blocks) + '\n\n') if prefix_blocks else ''
    stem = prefix + _render_mc_block(item, include_answer=False).rstrip()
    prompts, answers = [], []
    for i, c in enumerate(item['choices']):
        answer = ' ' + (c if is_letter_choices(item['choices']) else str(c))
        prompts.append(stem + answer)
        answers.append(answer)
    return prompts, answers, item['gold']


def render_logprob_schema(item, data, idx, num_fewshot, delimiter):
    fewshot = sample_fewshot(data, idx, num_fewshot)
    prefix_blocks = []
    for ex in fewshot:
        # few-shot uses the gold context option for schema tasks
        gold_ctx = ex['context_options'][ex['gold']].rstrip()
        prefix_blocks.append(gold_ctx + delimiter + ex['continuation'])
    prefix = ('\n\n'.join(prefix_blocks) + '\n\n') if prefix_blocks else ''
    prompts, answers = [], []
    answer = delimiter + item['continuation']
    for opt in item['context_options']:
        prompts.append(prefix + opt.rstrip() + answer)
        answers.append(answer)
    return prompts, answers, item['gold']


def render_logprob_lm(item, data, idx, num_fewshot, delimiter):
    """Return (prompt, answer_text, continuation). Prompt includes the gold
    continuation; the LM scorer checks greedy-argmax over the answer span."""
    prompt = render_generation_lm(item, data, idx, num_fewshot, delimiter)
    answer = item['continuation']
    return prompt + answer, answer, item['continuation']
