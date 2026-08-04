"""Answer parsing and correctness for both scoring modes."""

import re
import string
import sys
import warnings

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
    Returns None if nothing matches or the match is ambiguous.

    A choice that matches the output *exactly* always wins. This matters when one
    option's text nests inside another's ("Present" vs. "Inferred Present"): the
    longest-match rule alone would read a bare "Present" as "Inferred Present".
    """
    if not text:
        return None
    out = normalize_answer(text)
    if not out:
        return None
    out_join = ' '.join(out)
    hits = []
    exact = []
    for i, c in enumerate(choices):
        ct = normalize_answer(str(c))
        if not ct:
            continue
        cj = ' '.join(ct)
        if cj == out_join:
            exact.append(i)
        elif cj in out_join or out_join in cj:
            hits.append((len(ct), i))
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None  # two choices with identical text — genuinely ambiguous
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


# ---------------------------------------------------------------------------
# vintage_qa ROUGE-L scoring
#
# Standard exact-prefix matching fails for vintage examination answers because
# the gold completions are verbose 19th-century prose (4-30 words) while modern
# LLMs produce concise, semantically equivalent answers in contemporary style.
#
# ROUGE-L (Longest Common Subsequence) measures the longest co-occurring token
# sequence, striking a pragmatic balance:
#
# * Short, overlapping answers get high scores (e.g. "Edward II" vs
#   "Edward II. (1307-1327 A. D.)" → ROUGE-L ≈ 0.50).
# * Completely different answers stay low (e.g. "New York" vs
#   "The National Road ..." → ROUGE-L ≈ 0.07).
# * It correlates at 0.82 with embedding cosine similarity on this dataset
#   while being deterministic, dependency-free, and fast.

try:
    from rouge_score import rouge_scorer

    _ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    _HAS_ROUGE = True
except ImportError:
    _ROUGE_SCORER = None
    _HAS_ROUGE = False

_ROUGE_THRESHOLD = 0.30

# Set once the fallback path has shouted, so the banner appears a single time
# per process instead of on every one of the 10,000 vintage_qa items.
_WARNED_ROUGE_FALLBACK = False

# Scoring-mode label recorded in the summary JSON so a degraded run is
# identifiable from its results file alone, long after the console output is gone.
ROUGE_SCORING_MODE = 'rouge-l' if _HAS_ROUGE else 'prefix-fallback-DEGRADED'

ROUGE_MISSING_WARNING = """\
==========================  DEGRADED SCORING  ==========================
 rouge_score is NOT installed, so the vintage_qa task CANNOT be scored
 with ROUGE-L.  It is falling back to exact-prefix matching, which is a
 DIFFERENT AND MUCH STRICTER metric: verbose but correct answers are
 counted as wrong, so vintage_qa scores will be far too low (frontier
 models drop from ~0.20-0.45 to ~0.00) and are NOT comparable to any
 result produced with ROUGE-L available.

 Fix before trusting any vintage_qa number:
     pip install rouge_score

 Already-collected runs do not need re-running: the model outputs are
 stored in the --debug-file and can be re-scored offline.
========================================================================"""


def warn_if_rouge_missing(stream=None):
    """Print the degraded-scoring banner to *stream* when rouge_score is absent.

    Returns True if the warning was emitted.  Callers that touch vintage_qa
    should invoke this so a missing dependency can never pass unnoticed.
    """
    if _HAS_ROUGE:
        return False
    (stream or sys.stderr).write(ROUGE_MISSING_WARNING + '\n')
    (stream or sys.stderr).flush()
    return True


if not _HAS_ROUGE:
    # Loud at import time as a backstop: even a caller that never routes through
    # the CLI cannot silently produce prefix-matched vintage_qa numbers.
    warnings.warn(
        'rouge_score is not installed - vintage_qa will be scored by exact-prefix '
        'match instead of ROUGE-L, producing scores that are far too low and not '
        'comparable to ROUGE-L runs. Install it with: pip install rouge_score',
        RuntimeWarning,
        stacklevel=2,
    )


def rouge_l_correct(generation, gold, threshold=_ROUGE_THRESHOLD):
    """Return True if the generation has ROUGE-L F1 ≥ *threshold* against gold.

    Both texts are normalised before scoring.  The default threshold of 0.30
    was calibrated on Grok's output against the vintage_qa dataset and
    validated against manual inspection.
    """
    if not _HAS_ROUGE:
        # Fall back to exact prefix if rouge_score isn't installed. This is a
        # stricter, non-equivalent metric, so shout once per process rather than
        # letting the substitution pass for a real ROUGE-L score.
        global _WARNED_ROUGE_FALLBACK
        if not _WARNED_ROUGE_FALLBACK:
            _WARNED_ROUGE_FALLBACK = True
            warn_if_rouge_missing()
        return lm_is_correct(generation, gold)
    if not generation or not gold:
        return False
    gen_norm = ' '.join(normalize_answer(generation))
    gold_norm = ' '.join(normalize_answer(gold))
    if not gen_norm or not gold_norm:
        return False
    scores = _ROUGE_SCORER.score(gold_norm, gen_norm)
    return scores['rougeL'].fmeasure >= threshold


def rouge_l_f1(generation, gold):
    """Return the raw ROUGE-L F1 score (for diagnostics/metric reporting)."""
    if not _HAS_ROUGE:
        return 1.0 if lm_is_correct(generation, gold) else 0.0
    if not generation or not gold:
        return 0.0
    gen_norm = ' '.join(normalize_answer(generation))
    gold_norm = ' '.join(normalize_answer(gold))
    if not gen_norm or not gold_norm:
        return 0.0
    return _ROUGE_SCORER.score(gold_norm, gen_norm)['rougeL'].fmeasure
