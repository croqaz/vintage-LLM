"""Metric calculations over models, texts and per-document records.

Flat summaries are built in measurements.py and described in metric_guide.py.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .helpers import LOG2E, ChatItem, TextItem, model_context_limit, prefix_id, stable_hash

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Surface heuristics, not judgments of meaning or usability.
MIN_GENERATION_WORDS = 8
MIN_DISTINCT_2 = 0.55
MAX_LOCAL_REPEAT_RATE = 0.30
MIN_REPEAT_LOOP_WORDS = 12

# A document shorter than this cannot be scored meaningfully; it is counted, not silently dropped.
MIN_SCORED_TOKENS = 8


# ============================================================================
# 1. Fixed-probe sentence scoring (period fidelity)
# ============================================================================


@torch.no_grad()
def token_level_losses(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
    """Target-token log probabilities (nats) and entropy at predicted positions.

    Returns concatenated 1-D tensors over the batch. Standard causal-LM shift:
    position t predicts token t+1; the first token is never scored.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].bool()

    logp_all = F.log_softmax(shift_logits, dim=-1)
    token_logp = logp_all.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(logp_all.exp() * logp_all).sum(-1)
    return token_logp[shift_mask], entropy[shift_mask]


def token_stats(logprobs: torch.Tensor, entropies: torch.Tensor) -> dict:
    """Summarise per-token log-probabilities (nats) into readable stats."""
    probs = logprobs.exp()
    return {
        'mean_token_prob': probs.mean().item(),
        'median_token_prob': probs.median().item(),
        'p10_token_prob': probs.quantile(0.10).item(),
        'min_token_prob': probs.min().item(),
        'token_prob_below_0_01_rate': (probs < 0.01).float().mean().item(),
        'mean_entropy_nats': entropies.mean().item(),
        'ppl': math.exp(-logprobs.mean().item()),
    }


def scored_span_bytes(tokenizer, text: str) -> int:
    """UTF-8 bytes of the part of `text` that token_level_losses actually scores.

    Decode exactly the predicted IDs, including when the tokenizer prepends BOS.
    This is the same denominator convention used by prose and chat scoring.
    """
    ids = tokenizer(text).input_ids
    scored = tokenizer.decode(ids[1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return len(scored.encode('utf-8'))


@torch.no_grad()
def score_probe_sentences(tokenizer, model, sentences: list[str], labels: list[str]) -> dict:
    """Sentence and group likelihood statistics over fixed probe texts."""
    per_sentence = []
    all_logp, all_ent = [], []
    for sent, label in zip(sentences, labels, strict=True):
        enc = tokenizer(sent, return_tensors='pt').to(model.device)
        logp, ent = token_level_losses(model, enc['input_ids'], enc['attention_mask'])
        st = token_stats(logp, ent)
        nbytes = scored_span_bytes(tokenizer, sent)
        st['bpb'] = float(-logp.sum().item() / math.log(2) / nbytes) if nbytes else None
        st['scored_bytes'] = nbytes
        st.update({'label': label, 'sentence': sent, 'tokens_scored': int(logp.numel())})
        per_sentence.append(st)
        all_logp.append(logp)
        all_ent.append(ent)

    all_logp = torch.cat(all_logp)
    all_ent = torch.cat(all_ent)
    overall = token_stats(all_logp, all_ent)
    overall['tokens_scored'] = int(all_logp.numel())
    overall['bpb'] = float(-all_logp.sum().item() * LOG2E / sum(s['scored_bytes'] for s in per_sentence))

    results = {'overall': overall, 'per_sentence': per_sentence}
    for name in set(labels):
        rows = [s for s in per_sentence if s['label'] == name]
        toks = sum(r['tokens_scored'] for r in rows)
        mean_log_loss = sum(math.log(r['ppl']) * r['tokens_scored'] for r in rows) / toks
        nbytes = sum(r['scored_bytes'] for r in rows)
        bits = sum(math.log(r['ppl']) * r['tokens_scored'] / math.log(2) for r in rows)
        results[name] = {
            'sentences_scored': len(rows),
            'ppl': math.exp(mean_log_loss),
            'bpb': bits / nbytes if nbytes else None,
        }
    if 'historical' in results and 'modern' in results:
        results['modern_historical_ppl_ratio'] = results['modern']['ppl'] / max(results['historical']['ppl'], 1e-9)
    return results


# ============================================================================
# 2. Contextual word embeddings (diachronic sense separation)
# ============================================================================


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def find_word_token_positions(tokenizer, context: str, word: str, seq_len: int) -> list[int] | None:
    """Token positions covering `word` inside `context`, via character offsets.

    Robust to BPE merging the word with a leading space (unlike matching
    tokenizer.encode(word) against the sentence's token ids).
    """
    char_start = context.lower().rfind(word.lower())
    if char_start < 0:
        return None
    char_end = char_start + len(word)
    offsets = tokenizer(context, return_offsets_mapping=True)['offset_mapping']
    positions = [i for i, (s, e) in enumerate(offsets[:seq_len]) if e > s and s < char_end and e > char_start]
    return positions or None


def _trunk_module(model):
    """The submodule whose output IS the final hidden state, for models that do
    not implement `output_hidden_states`."""
    for name in ('model', 'transformer', 'gpt_neox', 'backbone', 'decoder'):
        candidate = getattr(model, name, None)
        if isinstance(candidate, torch.nn.Module):
            return candidate
    return None


@torch.no_grad()
def last_hidden_state(model, inputs) -> torch.Tensor:
    """Final-layer hidden states, even for architectures that ignore
    `output_hidden_states`.

    Custom modeling code is free to drop the flag: MODELS/Talkie-1930-13b's
    forward() swallows it via **kwargs and returns
    CausalLMOutputWithPast(loss, logits) with hidden_states=None. A forward hook
    on the trunk captures the tensor regardless, in the SAME forward pass.
    """
    captured = {}

    def hook(_module, _args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured['h'] = getattr(tensor, 'last_hidden_state', tensor)

    trunk = _trunk_module(model)
    handle = trunk.register_forward_hook(hook) if trunk is not None else None
    try:
        out = model(**inputs, output_hidden_states=True)
    finally:
        if handle is not None:
            handle.remove()

    hidden_states = getattr(out, 'hidden_states', None)
    if hidden_states:
        return hidden_states[-1]
    if 'h' in captured and torch.is_tensor(captured['h']):
        return captured['h']
    raise RuntimeError(
        f'{type(model).__name__} returned no hidden_states and no trunk submodule could be hooked; sense separation cannot be computed'
    )


@torch.no_grad()
def extract_word_embeddings(model, tokenizer, words: list[str], contexts: list[str]) -> dict:
    """Last-hidden-layer embedding of each word, averaged over its sub-tokens."""
    embeddings, missing = {}, []
    for word, context in zip(words, contexts, strict=True):
        inputs = tokenizer(context, return_tensors='pt').to(model.device)
        hidden = last_hidden_state(model, inputs)
        positions = find_word_token_positions(tokenizer, context, word, inputs['input_ids'].shape[1])
        if not positions:
            missing.append(word)
            continue
        embeddings[word] = hidden[0, positions].float().mean(dim=0).cpu().numpy()
    return embeddings, missing


@torch.no_grad()
def sense_separation(model, tokenizer, words: list[str], period_contexts: list[str], modern_contexts: list[str]) -> dict:
    """Cosine similarity of the SAME shifted word in a period vs modern sentence.

    Describes contextual vector similarity; no established quality direction.
    """
    hist_emb, missing_hist = extract_word_embeddings(model, tokenizer, words, period_contexts)
    mod_emb, _ = extract_word_embeddings(model, tokenizer, words, modern_contexts)

    present = [w for w in words if w in hist_emb]
    shifts = {w: cosine_similarity(hist_emb[w], mod_emb[w]) for w in present if w in mod_emb}

    pairwise = {}
    all_pairs = []
    for i, w1 in enumerate(present):
        for w2 in present[i + 1 :]:
            s = cosine_similarity(hist_emb[w1], hist_emb[w2])
            pairwise[f'{w1}|{w2}'] = s
            all_pairs.append(s)

    neighbors = {}
    for w in present[:5]:
        sims = sorted(
            ((o, cosine_similarity(hist_emb[w], hist_emb[o])) for o in present if o != w),
            key=lambda kv: -kv[1],
        )[:5]
        neighbors[w] = [(o, round(s, 4)) for o, s in sims]

    arr = np.array(all_pairs) if all_pairs else np.array([0.0])
    return {
        'words': present,
        'missing_words': missing_hist,
        'shift_cosines': shifts,
        'shift_mean_cosine': float(np.mean(list(shifts.values()))) if shifts else None,
        'pairwise_historical': pairwise,
        'historical_pairwise_mean_cosine': float(arr.mean()) if all_pairs else None,
        'historical_pairwise_std_cosine': float(arr.std()) if all_pairs else None,
        'neighbors': neighbors,
    }


# ============================================================================
# 3. Held-out bits-per-byte: prose records and chat records
# ============================================================================


@torch.inference_mode()
def score_prose_records(tokenizer, model, items: list[TextItem], max_tokens: int, progress_every: int = 25) -> tuple[list[dict], int]:
    """Per-document additive bit/byte counts over held-out prose.

    Retaining per-document records lets comparisons use PAIRED bootstrap CIs and
    gives split A/B and early/late diagnostics essentially for free.

    Returns (records, skipped): a document too short to score never becomes a
    record, so it is counted here instead of vanishing from the coverage
    arithmetic.
    """
    records: list[dict] = []
    skipped = 0
    pre = prefix_id(tokenizer, model)
    limit = model_context_limit(model, max_tokens)
    content_budget = limit - (1 if pre is not None else 0)
    for i, item in enumerate(items, 1):
        all_ids = tokenizer(item.text, add_special_tokens=False).input_ids
        ids = all_ids[:content_budget]
        if len(ids) < MIN_SCORED_TOKENS:
            skipped += 1
            continue
        input_ids = ([pre] if pre is not None else []) + ids
        inp = torch.tensor([input_ids], dtype=torch.long, device=model.device)
        logits = model(input_ids=inp, use_cache=False).logits[0, :-1].float()
        labels = inp[0, 1:]
        nll = F.cross_entropy(logits, labels, reduction='none').cpu().numpy().astype(np.float64)
        scored_ids = ids if pre is not None else ids[1:]
        if len(nll) != len(scored_ids):
            raise RuntimeError('internal token/loss alignment error')

        decoded = tokenizer.decode(scored_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        nbytes = len(decoded.encode('utf-8'))
        cut = min(128, len(scored_ids))
        early_text = tokenizer.decode(scored_ids[:cut], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        late_text = tokenizer.decode(scored_ids[cut:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        records.append(
            {
                'id': item.item_id,
                'split': item.split,
                'bits': float(nll.sum() * LOG2E),
                'bytes': nbytes,
                'tokens': len(scored_ids),
                'scored_text_sha256': stable_hash(decoded.encode('utf-8')),
                'truncated': len(all_ids) > len(ids),
                'prefix_token_used': pre is not None,
                'early_bits': float(nll[:cut].sum() * LOG2E),
                'early_bytes': len(early_text.encode('utf-8')),
                'late_bits': float(nll[cut:].sum() * LOG2E),
                'late_bytes': len(late_text.encode('utf-8')),
            }
        )
        if progress_every and i % progress_every == 0:
            print(f' {i}/{len(items)}', end='', flush=True)
    return records, skipped


@torch.inference_mode()
def score_chat_records(tokenizer, model, items: list[ChatItem], max_tokens: int, progress_every: int = 50) -> tuple[list[dict], int]:
    """Score assistant target tokens conditionally on context, via offsets.

    Mirrors score_prose_records when the model exposes neither BOS nor EOS: the
    sequence is scored with no prepended prefix and the very first content token
    simply has no predicting position. Returns (records, skipped).
    """
    records: list[dict] = []
    skipped = 0
    pre = prefix_id(tokenizer, model)
    limit = model_context_limit(model, max_tokens)
    budget = limit - (1 if pre is not None else 0)
    for i, item in enumerate(items, 1):
        full = item.context + item.target
        encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded.input_ids
        offsets = encoded.offset_mapping
        boundary = len(item.context)
        target_positions = [j for j, (start, end) in enumerate(offsets) if end > boundary]
        if not target_positions:
            skipped += 1
            continue
        truncated = len(ids) > budget
        original_target_tokens = len(target_positions)
        if truncated:
            drop = len(ids) - budget
            ids = ids[drop:]
            target_positions = [j - drop for j in target_positions if j >= drop]
        # nll[i] predicts inp[i + 1] and ids[j] sits at inp[j + offset], so ids[j] is
        # scored by nll[j + offset - 1]. With no prefix, ids[0] has no such position.
        offset = 1 if pre is not None else 0
        target_positions = [j for j in target_positions if j + offset >= 1]
        if not target_positions:
            skipped += 1
            continue
        inp = torch.tensor([([pre] if pre is not None else []) + ids], dtype=torch.long, device=model.device)
        logits = model(input_ids=inp, use_cache=False).logits[0, :-1].float()
        labels = inp[0, 1:]
        nll = F.cross_entropy(logits, labels, reduction='none')
        selected = nll[[j + offset - 1 for j in target_positions]].cpu().numpy().astype(np.float64)
        target_ids = [ids[j] for j in target_positions]
        decoded = tokenizer.decode(target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        records.append(
            {
                'id': item.item_id,
                'split': item.split,
                'bits': float(selected.sum() * LOG2E),
                'bytes': len(decoded.encode('utf-8')),
                'tokens': len(target_positions),
                'scored_text_sha256': stable_hash(decoded.encode('utf-8')),
                'input_text_sha256': stable_hash(
                    tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False).encode('utf-8')
                ),
                'truncated': truncated,
                'target_truncated': len(target_positions) < original_target_tokens,
            }
        )
        if progress_every and i % progress_every == 0:
            print(f' c{i}/{len(items)}', end='', flush=True)
    return records, skipped


def is_finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def finite_records(records: list[dict], prefix: str = '') -> list[dict]:
    """The same positive-byte, finite-bit pool for aggregates and bootstrap."""
    return [r for r in records if is_finite(r.get(f'{prefix}bits')) and is_finite(r.get(f'{prefix}bytes')) and r[f'{prefix}bytes'] > 0]


def records_bpb(records: list[dict], split: str | None = None, prefix: str = '') -> float:
    """Sum bits / sum UTF-8 bytes; exclude invalid and empty records."""
    rows = finite_records([r for r in records if split is None or r.get('split') == split], prefix)
    nbytes = sum(r[f'{prefix}bytes'] for r in rows)
    return sum(r[f'{prefix}bits'] for r in rows) / nbytes if nbytes else float('nan')


def count_nonfinite_records(records: list[dict], prefix: str = '') -> int:
    """Records with non-finite or missing bit counts, reported beside BPB."""
    return sum(not is_finite(r.get(f'{prefix}bits')) for r in records)


@torch.no_grad()
def bits_per_byte_of_texts(tokenizer, model, texts: list[str], max_tokens: int = 1024) -> float:
    """Judge likelihood through the same token/byte accounting as prose."""
    items = [TextItem(str(i), text, 'A') for i, text in enumerate(texts)]
    records, _skipped = score_prose_records(tokenizer, model, items, max_tokens, progress_every=0)
    return records_bpb(records)


# ============================================================================
# 4. Forced-choice logic and anachronism traps
# ============================================================================


@torch.no_grad()
def span_bpb(tokenizer, model, prefix_plus_span: str, prefix: str) -> float:
    """Bits/byte the model spends on the part of the text after `prefix`.

    The prefix is cut at its last non-space char so the span owns the leading
    space - these tokenizers glue spaces onto the FOLLOWING word, and cutting
    mid-space makes the two tokenisations non-nested.
    """
    bos = prefix_id(tokenizer, model)
    prefix = prefix.rstrip()
    span = prefix_plus_span[len(prefix) :]
    pre = tokenizer(prefix, add_special_tokens=False).input_ids
    full = tokenizer(prefix + span, add_special_tokens=False).input_ids
    if (not pre and bos is None) or len(full) <= len(pre) or full[: len(pre)] != pre:
        return float('nan')
    inp = torch.tensor([[bos] + full] if bos is not None else [full], device=model.device)
    logits = model(inp).logits[:, :-1].float()
    # nll[i] predicts inp token i+1; the first span token sits one past the
    # prefix (two past it when a BOS is prepended).
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), inp[:, 1:].reshape(-1), reduction='none')
    start = len(pre) if bos is not None else len(pre) - 1
    return nll[start:].sum().item() * LOG2E / len(span.encode('utf-8'))


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; exact, no resampling.

    Used for the small fixed-choice sets, where a normal-approximation interval
    misbehaves near 0 and 1 and a bootstrap over 40 items adds nothing.
    """
    if trials < 1 or not 0 < confidence < 1:
        return float('nan'), float('nan')
    # Two-sided normal quantile via the inverse error function, so scipy is not needed.
    z = math.sqrt(2.0) * _erfinv(confidence)
    phat = successes / trials
    denom = 1.0 + z * z / trials
    center = (phat + z * z / (2 * trials)) / denom
    spread = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def _erfinv(x: float) -> float:
    return float(torch.special.erfinv(torch.tensor(x, dtype=torch.float64)))


@torch.no_grad()
def run_logic(tokenizer, model, items: list[tuple]) -> dict:
    """Forced choice between a coherent and an incoherent continuation.

    Random two-choice baseline = 0.50. Margin = mean bad-minus-good BPB.
    """
    correct, margins, per_cat, records = 0, [], {}, []
    for cat, ctx, good, bad in items:
        gb = span_bpb(tokenizer, model, ctx.rstrip() + good, ctx)
        bb = span_bpb(tokenizer, model, ctx.rstrip() + bad, ctx)
        scored = math.isfinite(gb) and math.isfinite(bb)
        # Retain both alternatives, including skipped pairs, for report inspection.
        records.append(
            {
                'category': cat,
                'context': ctx,
                'good_continuation': good,
                'bad_continuation': bad,
                'good_bpb': gb if math.isfinite(gb) else None,
                'bad_bpb': bb if math.isfinite(bb) else None,
                'margin_bpb': bb - gb if scored else None,
                'correct': gb < bb if scored else None,
            }
        )
        if not scored:
            continue
        ok = gb < bb
        correct += ok
        margins.append(bb - gb)
        per_cat.setdefault(cat, []).append(ok)
    n = len(margins)
    low, high = wilson_interval(correct, n)
    return {
        'accuracy': correct / n if n else float('nan'),
        'accuracy_ci_low': low,
        'accuracy_ci_high': high,
        'accuracy_ci_confidence': 0.95,
        'margin_bpb': sum(margins) / n if n else float('nan'),
        'items_scored': n,
        'items_correct': correct,
        'items_skipped': len(items) - n,
        'categories': {c: {'accuracy': sum(v) / len(v), 'items_scored': len(v)} for c, v in sorted(per_cat.items())},
        'items': records,
    }


@torch.no_grad()
def run_traps(tokenizer, model, pairs: list[tuple]) -> dict:
    """Modern-minus-period phrase BPB on fixed pairs; not a leakage detector."""
    shocks, worst = [], None
    for tstem, tph, cstem, cph in pairs:
        # A malformed pair is skipped like any unscorable item; it must never
        # abort the checkpoint and discard the suites that already ran.
        if tph not in tstem or cph not in cstem:
            continue
        ti = tstem.index(tph)
        ci = cstem.index(cph)
        tb = span_bpb(tokenizer, model, tstem[: ti + len(tph)], tstem[:ti])
        cb = span_bpb(tokenizer, model, cstem[: ci + len(cph)], cstem[:ci])
        if not math.isfinite(tb) or not math.isfinite(cb):
            continue
        s = tb - cb
        shocks.append(s)
        if worst is None or s < worst[1]:
            worst = (tph, s)
    return {
        'mean_delta_bpb': sum(shocks) / len(shocks) if shocks else float('nan'),
        # These sets are tiny; publish the mean's spread beside the mean itself.
        'mean_delta_stderr_bpb': float(np.std(shocks, ddof=1) / math.sqrt(len(shocks))) if len(shocks) > 1 else float('nan'),
        'min_delta_bpb': min(shocks) if shocks else float('nan'),
        'nonpositive_pairs': sum(1 for s in shocks if s <= 0),
        'pairs_scored': len(shocks),
        'pairs_skipped': len(pairs) - len(shocks),
        'min_delta_phrase': worst[0] if worst else None,
    }


# ============================================================================
# 5. Generation text statistics - ONE implementation, used everywhere
# ============================================================================


def longest_loop(words: list[str], max_period: int = 12) -> int:
    """Longest periodic word run with >=2 repeats and period <=max_period; 0 = none."""
    best = 0
    for period in range(1, min(max_period, len(words) // 2) + 1):
        run = 0
        for i in range(len(words) - period):
            if words[i] == words[i + period]:
                run += 1
                if run >= period:
                    best = max(best, run + period)
            else:
                run = 0
    return best


def punct_issues(text: str) -> int:
    """Mechanical punctuation breakage: unbalanced brackets/quotes, doubled
    punctuation, comma after space."""
    issues = abs(text.count('(') - text.count(')')) + text.count('"') % 2
    issues += len(re.findall(r'[,;:]{2,}|\.{4,}|\s,', text))
    return issues


def back_matter_score(text: str) -> dict:
    """Flag back-matter-like formatting; short prose can also trigger this."""
    words = WORD_RE.findall(text)
    n = max(1, len(words))
    chars = max(1, len(text))
    lines = [ln for ln in text.splitlines() if ln.strip()]
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]

    digit_frac = sum(c.isdigit() for c in text) / chars
    initial_frac = len(re.findall(r'\b[A-Z]\.', text)) / n
    caps_frac = sum(1 for w in words if len(w) > 1 and w.isupper()) / n
    short_line_frac = (sum(1 for ln in lines if len(ln.split()) <= 6) / len(lines)) if lines else 0.0
    punct_density = len(re.findall(r'[.,;:\-]', text)) / chars
    mean_sentence_words = float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0

    # Two votes flag a sample for inspection. These heuristics can both miss
    # back matter and flag valid prose; a negative flag does not establish quality.
    votes = (
        (digit_frac > 0.04)
        + (initial_frac > 0.06)
        + (caps_frac > 0.08)
        + (short_line_frac > 0.5)
        + (punct_density > 0.09)
        + (0 < mean_sentence_words < 8)
    )
    return {
        'digit_frac': digit_frac,
        'initial_frac': initial_frac,
        'caps_frac': caps_frac,
        'short_line_frac': short_line_frac,
        'punct_density': punct_density,
        'mean_sentence_words': mean_sentence_words,
        'back_matter_votes': int(votes),
        'is_back_matter': bool(votes >= 2),
    }


def shared_ngram_fraction(texts: list[str], n: int = 4, max_texts: int = 120) -> float:
    """Mean fraction of each text's UNIQUE n-grams appearing in another text.

    Cross-completion overlap: 0.0 = no n-grams are shared, 1.0 = every unique
    n-gram appears in at least one other completion. Words are lowercased and
    extracted by WORD_RE. Each eligible completion has equal weight; repeated
    occurrences within a completion count only once. Texts with fewer than n
    words are excluded; fewer than two eligible texts returns NaN.

    Comparisons require the same prompt
    set, sample count, length budget and decoding settings.

    Cheap by construction - set operations over n-grams, no model, no GPU.
    Capped at max_texts to bound work.
    """
    grams = []
    for text in texts[:max_texts]:
        words = [w.lower() for w in WORD_RE.findall(text)]
        grams.append(set(zip(*(words[i:] for i in range(n)), strict=False)))
    grams = [g for g in grams if g]
    if len(grams) < 2:
        return float('nan')

    # Count how many texts each n-gram appears in, so "appears elsewhere" is a
    # single lookup instead of a pairwise union per text.
    counts: dict = {}
    for g in grams:
        for gram in g:
            counts[gram] = counts.get(gram, 0) + 1
    shared = [sum(1 for gram in g if counts[gram] > 1) / len(g) for g in grams]
    return float(np.mean(shared))


SURFACE_REASONS = ('short_text', 'low_bigram_diversity', 'local_repetition', 'repeat_loop', 'back_matter_like')


def surface_failure_reasons(sample: dict) -> list[str]:
    """Apply surface checks to a continuation's measured statistics."""
    checks = (
        sample['words'] < MIN_GENERATION_WORDS,
        sample['distinct_2'] < MIN_DISTINCT_2,
        sample['echo_rate'] > MAX_LOCAL_REPEAT_RATE,
        sample['longest_loop_words'] >= MIN_REPEAT_LOOP_WORDS,
        sample['is_back_matter'],
    )
    return [reason for reason, triggered in zip(SURFACE_REASONS, checks, strict=True) if triggered]


def text_stats(prompt: str, text: str) -> dict:
    """Surface statistics for one continuation; regex words, not model tokens."""
    words_lower = [w.lower() for w in WORD_RE.findall(text)]
    bigrams = list(zip(words_lower, words_lower[1:], strict=False))
    trigrams = list(zip(words_lower, words_lower[1:], words_lower[2:], strict=False))
    echo = sum(w in words_lower[max(0, i - 4) : i] for i, w in enumerate(words_lower)) / max(1, len(words_lower))
    distinct2 = len(set(bigrams)) / max(1, len(bigrams))

    prompt_words = {w.lower() for w in WORD_RE.findall(prompt)}
    content = [w for w in words_lower if len(w) > 3]
    prompt_overlap = sum(w in prompt_words for w in content) / max(1, len(content))

    loop = longest_loop(words_lower)
    back = back_matter_score(text)
    stats = {
        'words': len(words_lower),
        'distinct_1': (len(set(words_lower)) / max(1, len(words_lower))) if words_lower else 1.0,
        'distinct_2': distinct2,
        'distinct_3': (len(set(trigrams)) / max(1, len(trigrams))) if trigrams else 1.0,
        'echo_rate': echo,
        'longest_loop_words': loop,
        'punct_issues': punct_issues(text),
        'prompt_word_overlap_rate': prompt_overlap,
        'is_back_matter': back['is_back_matter'],
        'back_matter_votes': back['back_matter_votes'],
        'digit_frac': back['digit_frac'],
        'caps_frac': back['caps_frac'],
        'mean_sentence_words': back['mean_sentence_words'],
    }
    reasons = surface_failure_reasons(stats)
    stats.update(
        surface_failure=bool(reasons), surface_failure_reasons=reasons, degenerate=any(reason != 'back_matter_like' for reason in reasons)
    )
    return stats


def summarize_generations(samples: list[dict]) -> dict:
    """Aggregate text_stats rows into checkpoint-level summary numbers."""
    if not samples:
        return {}
    return {
        **{f'{reason}_rate': sum(reason in s['surface_failure_reasons'] for s in samples) / len(samples) for reason in SURFACE_REASONS},
        'n_prompts': len(samples),
        'degenerate_rate': sum(s['degenerate'] for s in samples) / len(samples),
        'mean_distinct_1': float(np.mean([s['distinct_1'] for s in samples])),
        'mean_distinct_2': float(np.mean([s['distinct_2'] for s in samples])),
        'mean_distinct_3': float(np.mean([s['distinct_3'] for s in samples])),
        'mean_echo_rate': float(np.mean([s['echo_rate'] for s in samples])),
        'worst_distinct_2': float(np.min([s['distinct_2'] for s in samples])),
        'worst_echo_rate': float(np.max([s['echo_rate'] for s in samples])),
        'worst_loop_words': max((s['longest_loop_words'] for s in samples), default=0),
        'mean_loop_words': float(np.mean([s['longest_loop_words'] for s in samples])),
        'mean_punct_issues_p100': float(np.mean([100.0 * s['punct_issues'] / max(1, s['words']) for s in samples])),
        'total_punct_issues': sum(s['punct_issues'] for s in samples),
        'mean_prompt_word_overlap_rate': float(np.mean([s['prompt_word_overlap_rate'] for s in samples])),
        'mean_words': float(np.mean([s['words'] for s in samples])),
        # Formatting heuristic, reported separately from repetition and short text.
        'back_matter_rate': float(np.mean([s['is_back_matter'] for s in samples])),
        'mean_digit_frac': float(np.mean([s['digit_frac'] for s in samples])),
        'mean_caps_frac': float(np.mean([s['caps_frac'] for s in samples])),
        'mean_sentence_words': float(np.mean([s['mean_sentence_words'] for s in samples])),
        # Word 4-gram overlap across completions (not a semantic diversity score).
        'shared_4gram_fraction': shared_ngram_fraction([s['continuation'] for s in samples]),
        'surface_failure_rate': float(np.mean([s['surface_failure'] for s in samples])),
    }


@torch.no_grad()
def generate_continuations(
    tokenizer,
    model,
    prompts: list[str],
    modes: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    batch_size: int = 4,
    chat: bool = False,
    on_batch: Callable[[str, list[dict]], None] | None = None,
) -> dict[str, list[dict]]:
    """Generate each prompt once per decoding mode, batched, left-padded.

    modes: any subset of ['greedy', 'sample']. Returns {'greedy': [...],
    'sample': [...]} where each row is {prompt, continuation, ...text_stats}.
    on_batch receives the mode and its cumulative completed rows before each
    progress update. The callback must not modify the rows.
    """
    if not prompts or not modes:
        return {}

    if chat:
        if tokenizer.chat_template is None:
            raise SystemExit('--chat was passed but this tokenizer has no chat_template. Drop --chat for a base model.')
        render_prompts = [
            tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in prompts
        ]
    else:
        render_prompts = prompts

    # A rendered chat template usually emits its own BOS. Letting the tokenizer add
    # a second one silently corrupts every continuation, so only add specials when
    # the rendered text does not already start with the BOS piece.
    bos_piece = getattr(tokenizer, 'bos_token', None)
    add_specials = not (chat and bos_piece and all(text.startswith(bos_piece) for text in render_prompts))

    out: dict[str, list[dict]] = {}
    saved_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    try:
        for mode in modes:
            sampled = mode == 'sample'
            torch.manual_seed(seed)
            if model.device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
            rows: list[dict] = []
            for start in range(0, len(prompts), batch_size):
                batch_render = render_prompts[start : start + batch_size]
                batch_orig = prompts[start : start + batch_size]
                encoded = tokenizer(batch_render, return_tensors='pt', padding=True, add_special_tokens=add_specials).to(model.device)
                input_width = encoded.input_ids.shape[1]
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=sampled,
                    temperature=temperature if sampled else None,
                    top_p=top_p if sampled else None,
                    top_k=top_k if sampled else None,
                    repetition_penalty=1.0,  # deliberately OFF: we are MEASURING loops
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                # Left-padded, so new tokens always start at the padded input width.
                for prompt, sequence in zip(batch_orig, generated, strict=False):
                    continuation = tokenizer.decode(
                        sequence[input_width:], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    ).strip()
                    rows.append({'prompt': prompt, 'continuation': continuation, **text_stats(prompt, continuation)})
                if on_batch is not None:
                    on_batch(mode, rows)
                print(f' g{len(rows)}/{len(prompts)}', end='', flush=True)
            out[mode] = rows
    finally:
        tokenizer.padding_side = saved_padding_side
    return out


# ============================================================================
# 6. Bootstrap confidence intervals and paired comparisons
# ============================================================================


def tokenizer_stats(tokenizer, texts: list[str], max_tokens: int = 1024) -> dict:
    """Bytes/token on token-clipped document prefixes, independent of model scoring.

    This budget excludes special tokens and model-context clipping, so its spans
    can differ from prose_records. It does not measure training throughput.
    """
    total_tokens = total_bytes = 0
    for item in texts:
        # Accepts either raw strings or the TextItem records the harness loads.
        text = item if isinstance(item, str) else getattr(item, 'text', None)
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids[:max_tokens]
        if len(ids) < 8:
            continue
        total_tokens += len(ids)
        total_bytes += len(tokenizer.decode(ids, skip_special_tokens=False).encode('utf-8'))
    try:
        vocab = len(tokenizer)
    except Exception:
        vocab = None
    return {
        'bytes_per_token': (total_bytes / total_tokens) if total_tokens else float('nan'),
        'tokens_scored': total_tokens,
        'bytes_scored': total_bytes,
        'vocab_size': vocab,
    }


def bootstrap_ci(records: list[dict], draws: int, confidence: float, seed: int) -> tuple[float, float]:
    records = finite_records(records)
    if draws < 1 or not 0 < confidence < 1:
        raise ValueError('bootstrap draws must be positive and confidence must be between 0 and 1')
    bits = np.asarray([r['bits'] for r in records], dtype=np.float64)
    nbytes = np.asarray([r['bytes'] for r in records], dtype=np.float64)
    if len(bits) < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(bits), size=(draws, len(bits)))
    values = bits[indices].sum(axis=1) / nbytes[indices].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def paired_bootstrap(candidate: list[dict], leader: list[dict], draws: int, confidence: float, seed: int) -> dict:
    """Paired delta BPB vs a leader, resampling the SAME document IDs."""
    if draws < 1 or not 0 < confidence < 1:
        raise ValueError('bootstrap draws must be positive and confidence must be between 0 and 1')
    candidate, leader = finite_records(candidate), finite_records(leader)
    ca = {r['id']: r for r in candidate}
    le = {r['id']: r for r in leader}
    ids = sorted(ca.keys() & le.keys())
    # Never collapse duplicate documents or compare different scored byte spans.
    compatible = (
        len(ca) == len(candidate)
        and len(le) == len(leader)
        and all(ca[x]['bytes'] == le[x]['bytes'] for x in ids)
        and all(ca[x]['scored_text_sha256'] == le[x]['scored_text_sha256'] for x in ids)
    )
    if len(ids) < 2 or not compatible:
        return dict(delta=float('nan'), low=float('nan'), high=float('nan'), bootstrap_fraction_lower=float('nan'), paired_docs=len(ids))
    cb = np.asarray([ca[x]['bits'] for x in ids])
    cn = np.asarray([ca[x]['bytes'] for x in ids])
    lb = np.asarray([le[x]['bits'] for x in ids])
    ln = np.asarray([le[x]['bytes'] for x in ids])
    delta = cb.sum() / cn.sum() - lb.sum() / ln.sum()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ids), size=(draws, len(ids)))
    values = cb[indices].sum(1) / cn[indices].sum(1) - lb[indices].sum(1) / ln[indices].sum(1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return {
        'delta': float(delta),
        'low': float(low),
        'high': float(high),
        'bootstrap_fraction_lower': float(np.mean(values < 0)),
        'paired_docs': len(ids),
    }


def classify_bpb_interval(low: float | None, high: float | None, margin: float | None) -> str:
    """Classify a candidate-minus-leader BPB interval; positive means worse."""
    if any(v is None or not math.isfinite(v) for v in (low, high, margin)):
        return 'unavailable'
    if margin < 0 or low > high:
        return 'unavailable'
    # Equivalence requires the WHOLE interval inside the practical tolerance.
    if low >= -margin and high <= margin:
        return 'equivalent'
    if low > margin:
        return 'worse'
    if high < -margin:
        return 'better'
    return 'inconclusive'


# ============================================================================
# 7. Experimental composite (fixed transforms, not a validated quality scale)
# ============================================================================

# Hand-set transforms for the composite; not universal across datasets.
BPB_LADDER = [(3.50, 0), (2.00, 20), (1.50, 40), (1.33, 55), (1.19, 70), (1.10, 85), (1.05, 92), (0.95, 100)]
CHAT_LADDER = [(1.60, 0), (1.26, 40), (1.10, 65), (0.93, 85), (0.82, 95), (0.73, 100)]
WEIGHTS = {'bpb': 0.50, 'logic': 0.25, 'chat': 0.15, 'hygiene': 0.10}


def interp(x: float, ladder) -> float:
    """Piecewise-linear map through fixed (value, points) anchors."""
    if not is_finite(x):
        return float('nan')
    if x >= ladder[0][0]:
        return float(ladder[0][1])
    if x <= ladder[-1][0]:
        return float(ladder[-1][1])
    for (x1, y1), (x2, y2) in zip(ladder, ladder[1:]):
        if x2 <= x <= x1:
            return y1 + (y2 - y1) * (x1 - x) / (x1 - x2)
    return float('nan')


def logic_points(acc: float) -> float:
    """0.50 is the two-choice chance floor; 1.00 is the ceiling of the item set.

    The old 0.92 anchor dated from a set whose wrong options were longer than the
    right ones and therefore unwinnable. On the length-matched set a decent model
    already passes 0.88, so anchoring below 1.0 would leave no headroom at all.
    """
    if not is_finite(acc):
        return float('nan')
    return max(0.0, min(100.0, (acc - 0.5) * 200.0))


def hygiene_points(loop_len: float, punct_p100: float) -> float:
    """loop 0 words = 100 pts, 80+ = 0. punct 0/100w = 100 pts, 1+ = 0."""
    lp = max(0.0, 100.0 - loop_len * 1.25)
    pp = max(0.0, 100.0 - punct_p100 * 100.0)
    return 0.7 * lp + 0.3 * pp


def bake_score(parts: dict) -> float:
    total, wsum = 0.0, 0.0
    for k, w in WEIGHTS.items():
        v = parts.get(k)
        if is_finite(v):
            total += w * v
            wsum += w
    return total / wsum if wsum else float('nan')
