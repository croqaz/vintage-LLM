"""Answer parsing and correctness for both scoring modes."""

import re
import string

from . import prompts


# ---------------------------------------------------------------------------
# generation-mode parsing
def parse_letter(text, num_choices):
    """Extract the chosen option letter from a generation. Returns an index or None.

    We only accept a letter that stands alone as a token (bounded by
    non-letters), so prose like "Class" or "answer-block" never yields a
    spurious 'C'/'B'.
    """
    valid = prompts.LETTERS[:num_choices]
    if not text:
        return None
    # Prefer an explicit "Answer: X" / "(X)" cue with a standalone letter.
    m = re.search(r'answer\s*(?:is)?\s*[:\-]?\s*\(?([A-H])(?![A-Za-z])', text, flags=re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return valid.index(m.group(1).upper())
    # Otherwise the first standalone letter token in range.
    for m in re.finditer(r'(?<![A-Za-z])([A-H])(?![A-Za-z])', text):
        up = m.group(1).upper()
        if up in valid:
            return valid.index(up)
    return None


def match_choice_text(text, choices):
    """Fallback for MC when the model replies with the answer *text* instead of a
    letter. Returns the index of the choice whose normalized text is contained in
    (or contains) the normalized output; prefers the longest unambiguous match.
    Returns None if nothing matches or the match is ambiguous."""
    if not text:
        return None
    out = normalize_answer(text)
    if not out:
        return None
    out_join = ' '.join(out)
    hits = []
    for i, c in enumerate(choices):
        ct = normalize_answer(str(c))
        if not ct:
            continue
        cj = ' '.join(ct)
        if cj in out_join or out_join in cj:
            hits.append((len(ct), i))
    if not hits:
        return None
    hits.sort(reverse=True)
    # Ambiguous if the two longest matches tie on length.
    if len(hits) > 1 and hits[0][0] == hits[1][0]:
        return None
    return hits[0][1]


_ARTICLES = {'a', 'an', 'the'}


def normalize_answer(s):
    """SQuAD-style normalization: lowercase, drop punctuation and articles,
    collapse whitespace."""
    s = s.lower()
    s = ''.join(ch if ch not in string.punctuation else ' ' for ch in s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return tokens


def lm_is_correct(generation, gold):
    """True if the (normalized) gold answer is produced at the start of the
    generation. Tolerant of trailing model chatter."""
    gen_tokens = normalize_answer(generation)
    gold_tokens = normalize_answer(gold)
    if not gold_tokens:
        return len(gen_tokens) == 0
    if len(gen_tokens) < len(gold_tokens):
        return False
    return gen_tokens[: len(gold_tokens)] == gold_tokens


def estimate_max_tokens(gold):
    words = max(1, len(gold.split()))
    return min(200, max(8, words * 3 + 8))


# ---------------------------------------------------------------------------
# logprob-mode scoring
def _span_mean_logprob(token_entries, answer_text):
    """Average per-token logprob over the trailing tokens that make up
    ``answer_text``. Aligns by accumulating decoded-token lengths from the right."""
    target = len(answer_text)
    acc = 0
    span = []
    for entry in reversed(token_entries):
        span.append(entry['logprob'])
        acc += len(entry['token'])
        if acc >= target:
            break
    if not span:
        return float('-inf')
    return sum(span) / len(span)


def score_mc_logprob(client, prompts_list, answers, style):
    """Return (predicted_index, per_choice_mean_logprobs)."""
    means = []
    for prompt, answer in zip(prompts_list, answers):
        entries = client.prompt_logprobs(prompt, style)
        means.append(_span_mean_logprob(entries, answer))
    return means.index(max(means)), means


def lm_greedy_correct(client, prompt, answer_text, style):
    """True if every token of the answer span was the model's argmax (i.e. the
    model would greedily generate the gold continuation)."""
    entries = client.prompt_logprobs(prompt, style)
    target = len(answer_text)
    acc = 0
    span = []
    for entry in reversed(entries):
        span.append(entry)
        acc += len(entry['token'])
        if acc >= target:
            break
    if not span:
        return False
    return all(e['is_greedy'] for e in span)
