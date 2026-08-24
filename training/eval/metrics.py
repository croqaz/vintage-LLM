"""All evaluation metrics for the merged evaluator, in ONE place.

Every function here is pure metric math over a loaded model/tokenizer (or over
plain numbers/texts). The orchestration lives in __main__.py; report rendering
lives in report.py. New metrics (vowel counts, Latin-letter counts, text
entropy, ...) belong in this file - add the computation here, surface it in the
summary dict built by __main__.py, and print it in report.py.

Sections:
  1.  Fixed-probe sentence scoring (period fidelity)
  2.  Contextual word embeddings (diachronic sense separation)
  3.  Span / held-out bits-per-byte (prose records, chat records)
  4.  Forced-choice logic and anachronism traps
  5.  Generation text statistics (one function, used by everything)
  6.  Bootstrap confidence intervals and paired comparisons
  7.  Reference ladders, points, BAKE score and verdicts
"""

from __future__ import annotations

import math
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from .helpers import LOG2E, ChatItem, TextItem, model_context_limit, prefix_id

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Degeneracy thresholds for sampled text at repetition_penalty = 1.0.
ECHO_BAD = 0.30  # fraction of words repeated within the previous 4
DISTINCT2_BAD = 0.60  # fraction of unique word-pairs
# Stricter thresholds for judging a SINGLE probe. The worst of many short
# samples is always somewhat repetitive, so reusing the averaged thresholds
# would flag every checkpoint ever produced.
PROBE_ECHO_BAD = 0.45
PROBE_DISTINCT2_BAD = 0.45


# ============================================================================
# 1. Fixed-probe sentence scoring (period fidelity)
# ============================================================================


@torch.no_grad()
def token_level_losses(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
    """Cross-entropy (nats) and entropy for every real predicted position.

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
        'frac_low_confidence': (probs < 0.01).float().mean().item(),
        'mean_entropy_nats': entropies.mean().item(),
        'perplexity': math.exp(-logprobs.mean().item()),
    }


def scored_span_bytes(tokenizer, text: str) -> int:
    """UTF-8 bytes of the part of `text` that token_level_losses actually scores.

    The first token is never predicted, so it must be excluded, otherwise
    bits-per-byte is silently optimistic on short strings.
    """
    offsets = tokenizer(text, return_offsets_mapping=True)['offset_mapping']
    real = [(s, e) for s, e in offsets if e > s]
    if len(real) < 2:
        return 0
    return len(text[real[1][0] :].encode('utf-8'))


@torch.no_grad()
def score_probe_sentences(tokenizer, model, sentences: list[str], labels: list[str]) -> dict:
    """Period-fidelity metrics on fixed probe sentences (from evaluate.py).

    Per-sentence perplexity/bits-per-byte plus historical-vs-modern group stats.
    """
    per_sentence = []
    all_logp, all_ent = [], []
    for sent, label in zip(sentences, labels, strict=True):
        enc = tokenizer(sent, return_tensors='pt').to(model.device)
        logp, ent = token_level_losses(model, enc['input_ids'], enc['attention_mask'])
        st = token_stats(logp, ent)
        nbytes = scored_span_bytes(tokenizer, sent)
        st['bits_per_byte'] = float(-logp.sum().item() / math.log(2) / nbytes) if nbytes else None
        st['scored_bytes'] = nbytes
        st.update({'label': label, 'sentence': sent, 'num_tokens': int(logp.numel())})
        per_sentence.append(st)
        all_logp.append(logp)
        all_ent.append(ent)

    all_logp = torch.cat(all_logp)
    all_ent = torch.cat(all_ent)
    overall = token_stats(all_logp, all_ent)
    overall['num_tokens'] = int(all_logp.numel())

    results = {'overall': overall, 'per_sentence': per_sentence}
    for name in set(labels):
        rows = [s for s in per_sentence if s['label'] == name]
        toks = sum(r['num_tokens'] for r in rows)
        mean_log_loss = sum(math.log(r['perplexity']) * r['num_tokens'] for r in rows) / toks
        nbytes = sum(r['scored_bytes'] for r in rows)
        bits = sum(math.log(r['perplexity']) * r['num_tokens'] / math.log(2) for r in rows)
        results[name] = {
            'num_sentences': len(rows),
            'perplexity': math.exp(mean_log_loss),
            'bits_per_byte': bits / nbytes if nbytes else None,
        }
    if 'historical' in results and 'modern' in results:
        results['modern_over_historical_ratio'] = results['modern']['perplexity'] / max(results['historical']['perplexity'], 1e-9)
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

    Lower = the model represents the two senses differently (good for a vintage
    model); near 1.0 = it ignores period vs modern usage.
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
        'semantic_shift': shifts,
        'mean_shift_similarity': float(np.mean(list(shifts.values()))) if shifts else None,
        'pairwise_historical': pairwise,
        'pairwise_mean': float(arr.mean()) if all_pairs else None,
        'pairwise_std': float(arr.std()) if all_pairs else None,
        'neighbors': neighbors,
    }


# ============================================================================
# 3. Held-out bits-per-byte: prose records and chat records
# ============================================================================


@torch.inference_mode()
def score_prose_records(tokenizer, model, items: list[TextItem], max_tokens: int, progress_every: int = 25) -> list[dict]:
    """Per-document additive bit/byte counts over held-out prose.

    Retaining per-document records lets comparisons use PAIRED bootstrap CIs and
    gives split A/B and early/late diagnostics essentially for free.
    """
    records: list[dict] = []
    pre = prefix_id(tokenizer, model)
    limit = model_context_limit(model, max_tokens)
    content_budget = limit - (1 if pre is not None else 0)
    for i, item in enumerate(items, 1):
        ids = tokenizer(item.text, add_special_tokens=False).input_ids[:content_budget]
        if len(ids) < 8:
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
                'early_bits': float(nll[:cut].sum() * LOG2E),
                'early_bytes': len(early_text.encode('utf-8')),
                'late_bits': float(nll[cut:].sum() * LOG2E),
                'late_bytes': len(late_text.encode('utf-8')),
            }
        )
        if progress_every and i % progress_every == 0:
            print(f' {i}/{len(items)}', end='', flush=True)
    return records


@torch.inference_mode()
def score_chat_records(tokenizer, model, items: list[ChatItem], max_tokens: int, progress_every: int = 50) -> list[dict]:
    """Score assistant targets conditionally on the context, via offsets.

    This replaces BOTH old chat implementations (evaluate2.run_chat_fit and
    evaluate3.score_chat) with the more rigorous offset-aligned version.
    """
    records: list[dict] = []
    pre = prefix_id(tokenizer, model)
    if pre is None:
        return records
    limit = model_context_limit(model, max_tokens)
    budget = limit - 1
    for i, item in enumerate(items, 1):
        full = item.context + item.target
        encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded.input_ids
        offsets = encoded.offset_mapping
        boundary = len(item.context)
        target_positions = [j for j, (start, end) in enumerate(offsets) if end > boundary]
        if not target_positions:
            continue
        if len(ids) > budget:
            drop = len(ids) - budget
            ids = ids[drop:]
            target_positions = [j - drop for j in target_positions if j >= drop]
        if not target_positions:
            continue
        inp = torch.tensor([[pre] + ids], dtype=torch.long, device=model.device)
        logits = model(input_ids=inp, use_cache=False).logits[0, :-1].float()
        labels = inp[0, 1:]
        nll = F.cross_entropy(logits, labels, reduction='none')
        selected = nll[target_positions].cpu().numpy().astype(np.float64)
        target_ids = [ids[j] for j in target_positions]
        decoded = tokenizer.decode(target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        records.append(
            {
                'id': item.item_id,
                'split': item.split,
                'bits': float(selected.sum() * LOG2E),
                'bytes': len(decoded.encode('utf-8')),
                'tokens': len(target_positions),
            }
        )
        if progress_every and i % progress_every == 0:
            print(f' c{i}/{len(items)}', end='', flush=True)
    return records


def records_bpb(records: list[dict], split: str | None = None, prefix: str = '') -> float:
    """Byte-weighted bits/byte over per-document records.

    NON-FINITE RECORDS ARE EXCLUDED. A single NaN document used to poison the
    whole aggregate: `sum()` returns NaN and the headline prose_bpb silently
    becomes NaN, which reads as "the eval broke" rather than "one of 200
    documents produced a bad forward pass". Observed once on autoresearch2/arch/
    kv_1 (num_key_value_heads=1) where the immediately preceding run of the same
    checkpoint was clean, i.e. transient rather than a property of the model.

    Callers should report `count_nonfinite_records()` alongside this so a dropped
    document is never invisible.
    """
    rows = [r for r in records if split is None or r['split'] == split]
    finite = [r for r in rows if math.isfinite(float(r[f'{prefix}bits']))]
    bits = sum(float(r[f'{prefix}bits']) for r in finite)
    nbytes = sum(int(r[f'{prefix}bytes']) for r in finite)
    return bits / nbytes if nbytes else float('nan')


def count_nonfinite_records(records: list[dict], prefix: str = '') -> int:
    """Documents whose scoring produced NaN/Inf. MUST be reported: a non-zero
    value means the headline number was computed over fewer documents."""
    return sum(1 for r in records if not math.isfinite(float(r[f'{prefix}bits'])))


@torch.no_grad()
def bits_per_byte_of_texts(tokenizer, model, texts: list[str], max_tokens: int = 1024) -> float:
    """Simple scalar held-out loss in bits/byte over raw texts (used by the judge)."""
    bos = prefix_id(tokenizer, model)
    total_bits, total_bytes = 0.0, 0
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False).input_ids[:max_tokens]
        if len(ids) < 8:
            continue
        nbytes = len(tokenizer.decode(ids).encode('utf-8'))
        inp = torch.tensor([[bos] + ids] if bos is not None else [ids], device=model.device)
        logits = model(inp).logits[:, :-1].float()
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), inp[:, 1:].reshape(-1), reduction='sum')
        total_bits += nll.item() * LOG2E
        total_bytes += nbytes
    return total_bits / total_bytes if total_bytes else float('nan')


# ============================================================================
# 4. Forced-choice logic and anachronism traps
# ============================================================================


@torch.no_grad()
def span_bits(tokenizer, model, prefix_plus_span: str, prefix: str) -> float:
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
    if len(full) <= len(pre) or full[: len(pre)] != pre:
        return float('nan')
    inp = torch.tensor([[bos] + full] if bos is not None else [full], device=model.device)
    logits = model(inp).logits[:, :-1].float()
    # nll[i] predicts inp token i+1; the first span token sits one past the
    # prefix (two past it when a BOS is prepended).
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), inp[:, 1:].reshape(-1), reduction='none')
    start = len(pre) if bos is not None else len(pre) - 1
    return nll[start:].sum().item() * LOG2E / len(span.encode('utf-8'))


@torch.no_grad()
def run_logic(tokenizer, model, items: list[tuple]) -> dict:
    """Forced choice between a coherent and an incoherent continuation.

    Chance = 0.50. Margin = mean bits/byte by which coherence is cheaper.
    """
    correct, margins, per_cat = 0, [], {}
    for cat, ctx, good, bad in items:
        gb = span_bits(tokenizer, model, ctx.rstrip() + good, ctx)
        bb = span_bits(tokenizer, model, ctx.rstrip() + bad, ctx)
        if math.isnan(gb) or math.isnan(bb):
            continue
        ok = gb < bb
        correct += ok
        margins.append(bb - gb)
        per_cat.setdefault(cat, []).append(ok)
    n = len(margins)
    return {
        'acc': correct / n if n else float('nan'),
        'margin': sum(margins) / n if n else float('nan'),
        'n': n,
        'per_category': {c: round(sum(v) / len(v), 3) for c, v in sorted(per_cat.items())},
    }


@torch.no_grad()
def run_traps(tokenizer, model, pairs: list[tuple]) -> dict:
    """Anachronism shock: bits on a post-1900 phrase minus a matched period
    phrase in the same sentence shape. Positive = period-bounded."""
    shocks, worst = [], None
    for tstem, tph, cstem, cph in pairs:
        ti = tstem.index(tph)
        ci = cstem.index(cph)
        tb = span_bits(tokenizer, model, tstem[: ti + len(tph)], tstem[:ti])
        cb = span_bits(tokenizer, model, cstem[: ci + len(cph)], cstem[:ci])
        if math.isnan(tb) or math.isnan(cb):
            continue
        s = tb - cb
        shocks.append(s)
        if worst is None or s < worst[1]:
            worst = (tph, s)
    return {
        'mean_shock': sum(shocks) / len(shocks) if shocks else float('nan'),
        'min_shock': min(shocks) if shocks else float('nan'),
        'n_leaked': sum(1 for s in shocks if s <= 0),
        'n': len(shocks),
        'worst_pair': worst[0] if worst else None,
    }


# ============================================================================
# 5. Generation text statistics - ONE implementation, used everywhere
# ============================================================================


def longest_loop(words: list[str], max_period: int = 12) -> int:
    """Longest immediately-repeating block of words; 0 = clean."""
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
    """Detect scanned-book BACK MATTER: indexes, catalogues, tables of contents,
    bibliographies.

    This is a failure mode that distinct-n and loop detection are structurally
    blind to, because such text is lexically DIVERSE - every line is different.
    It was the dominant low-scoring pattern in a real 49k-completion bulk run:

        'Constituent and Legislative Assemblies, 1789-1875. By F. DE PRESSENSE.
         French Revolution.-Mirabeau.- The Constitution of 1791.-Brissot.'

    Signals, all cheap and text-only:
      digit_frac       page numbers and dates
      initial_frac     'F. DE PRESSENSE', 'G. R. PARSONS', 'F.R.S.'
      caps_frac        RUNS OF CAPITALISED NAMES
      short_line_frac  list-shaped rather than prose-shaped
      punct_density    the '.-' / ';' / ',' chains that join index entries
      mean_sentence_words  prose runs long; catalogue entries are stubs
    """
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

    # Vote rather than a hard rule: any SINGLE signal fires on ordinary period
    # prose (dates in narrative, initials in citations), but two together do not.
    #
    # CALIBRATED against 49,413 real completions, using that pipeline's own
    # independent quality score as ground truth:
    #   source: https://huggingface.co/datasets/croqaz/tiny-vintage-completions
    #
    #   threshold   recall on score<50   false positives on score>95 (n=17,085)
    #     >= 2            48.7%                     0.00%
    #     >= 3            15.2%                     0.00%
    #     >= 4             2.1%                     0.00%
    #
    # >= 2 is chosen: it triples recall at zero measured false-positive cost.
    # The detector is PRECISE but not exhaustive -- a 0% back_matter_rate means
    # "none detected", not "none present".
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


def self_bleu(texts: list[str], n: int = 4, max_texts: int = 120) -> float:
    """Mean fraction of each text's n-grams that also appear in ANY other text.

    Mode-collapse detector. distinct-n looks INSIDE one continuation, so 500
    completions that are all subtly the same each score as perfectly diverse.
    This looks ACROSS them: 0.0 = every completion is lexically unique,
    1.0 = they are all reusing the same n-grams.

    Cheap by construction - set operations over n-grams, no model, no GPU.
    Capped at max_texts because the pairwise-union step is O(texts * ngrams).
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


def text_stats(prompt: str, text: str) -> dict:
    """ALL surface text-quality metrics for ONE continuation.

    This is the single source of truth for generation health; the old scripts'
    ngram_stats (evaluate.py), generation_health (evaluate3) and greedy loop /
    punct checks (evaluate2) are all folded in here.
    """
    words_lower = [w.lower() for w in WORD_RE.findall(text)]
    bigrams = list(zip(words_lower, words_lower[1:], strict=False))
    trigrams = list(zip(words_lower, words_lower[1:], words_lower[2:], strict=False))
    echo = sum(w in words_lower[max(0, i - 4) : i] for i, w in enumerate(words_lower)) / max(1, len(words_lower))
    distinct2 = len(set(bigrams)) / max(1, len(bigrams))

    prompt_words = {w.lower() for w in WORD_RE.findall(prompt)}
    content = [w for w in words_lower if len(w) > 3]
    prompt_copy = sum(w in prompt_words for w in content) / max(1, len(content))

    loop = longest_loop(words_lower)
    back = back_matter_score(text)
    # A gate only for unmistakable surface collapse. Diverse nonsense passes by
    # design; held-out BPB and human reading must judge meaning.
    # NOTE: back matter is reported separately, NOT folded into `degenerate` --
    # it is a distinct failure mode (lexically diverse, so the distinct-2/echo/
    # loop gates all pass it) and mixing them would hide which one is happening.
    degenerate = len(words_lower) < 8 or distinct2 < 0.55 or echo > ECHO_BAD or loop >= 12
    return {
        'words': len(words_lower),
        'distinct_1': (len(set(words_lower)) / max(1, len(words_lower))) if words_lower else 1.0,
        'distinct_2': distinct2,
        'distinct_3': (len(set(trigrams)) / max(1, len(trigrams))) if trigrams else 1.0,
        'echo_rate': echo,
        'longest_loop_words': loop,
        'punct_issues': punct_issues(text),
        'prompt_copy_rate': prompt_copy,
        'degenerate': degenerate,
        'is_back_matter': back['is_back_matter'],
        'back_matter_votes': back['back_matter_votes'],
        'digit_frac': back['digit_frac'],
        'caps_frac': back['caps_frac'],
        'mean_sentence_words': back['mean_sentence_words'],
    }


def summarize_generations(samples: list[dict]) -> dict:
    """Aggregate text_stats rows into checkpoint-level summary numbers."""
    if not samples:
        return {}
    return {
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
        'mean_prompt_copy_rate': float(np.mean([s['prompt_copy_rate'] for s in samples])),
        'mean_words': float(np.mean([s['words'] for s in samples])),
        # Back matter: lexically diverse, so invisible to every metric above.
        'back_matter_rate': float(np.mean([s.get('is_back_matter', False) for s in samples])),
        'mean_digit_frac': float(np.mean([s.get('digit_frac', 0.0) for s in samples])),
        'mean_caps_frac': float(np.mean([s.get('caps_frac', 0.0) for s in samples])),
        'mean_sentence_words': float(np.mean([s.get('mean_sentence_words', 0.0) for s in samples])),
        # Mode collapse ACROSS completions (distinct-n only sees within one).
        'self_bleu_4': self_bleu([s.get('continuation', '') for s in samples]),
        # Any completion that is degenerate OR back matter would be dropped by a
        # synth-data filter; this is the closest thing here to a production yield.
        'unusable_rate': float(np.mean([bool(s['degenerate']) or bool(s.get('is_back_matter', False)) for s in samples])),
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
) -> dict[str, list[dict]]:
    """Generate each prompt once per decoding mode, batched, left-padded.

    modes: any subset of ['greedy', 'sample']. Returns {'greedy': [...],
    'sampled': [...]} where each row is {prompt, continuation, ...text_stats}.
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

    out: dict[str, list[dict]] = {}
    old_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    started = time.time()
    for mode in modes:
        sampled = mode == 'sample'
        torch.manual_seed(seed)
        if model.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        rows: list[dict] = []
        for start in range(0, len(prompts), batch_size):
            batch_render = render_prompts[start : start + batch_size]
            batch_orig = prompts[start : start + batch_size]
            encoded = tokenizer(batch_render, return_tensors='pt', padding=True, add_special_tokens=True).to(model.device)
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
            print(f' g{min(start + len(batch_render), len(prompts))}/{len(prompts)}', end='', flush=True)
        out[mode] = rows
    tokenizer.padding_side = old_side
    return out


# ============================================================================
# 6. Bootstrap confidence intervals and paired comparisons
# ============================================================================


def tokenizer_stats(tokenizer, texts: list[str], max_tokens: int = 1024) -> dict:
    """Tokenizer efficiency on the scored corpus -- no model, no GPU.

    bytes_per_token is the compression rate: HIGHER means the same token budget
    carries more text, so a model trained with it sees more data per step. This is
    the number that decides whether a tokenizer swap helps or hurts throughput, and
    it is measured on exactly the documents bits-per-byte is scored on.
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
    bits = np.asarray([r['bits'] for r in records], dtype=np.float64)
    nbytes = np.asarray([r['bytes'] for r in records], dtype=np.float64)
    if not len(bits):
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(bits), size=(draws, len(bits)))
    values = bits[indices].sum(axis=1) / nbytes[indices].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def paired_bootstrap(candidate: list[dict], leader: list[dict], draws: int, confidence: float, seed: int) -> dict:
    """Paired delta BPB vs a leader, resampling the SAME document IDs."""
    ca = {r['id']: r for r in candidate}
    le = {r['id']: r for r in leader}
    ids = sorted(ca.keys() & le.keys())
    if not ids:
        return {'delta': float('nan'), 'low': float('nan'), 'high': float('nan'), 'p_beats': float('nan')}
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
    return {'delta': float(delta), 'low': float(low), 'high': float(high), 'p_beats': float(np.mean(values < 0))}


# ============================================================================
# 7. Reference ladders, points, BAKE score and verdicts
# ============================================================================

# Every anchor was MEASURED with the same code path (same heldout docs, max 1024
# tokens/doc) on real models in this project. Calibrated for tiny (<=1B) models
# on 19th-century English.
BPB_LADDER = [  # (bits/byte on eval_data/heldout.jsonl, bake points)
    (3.50, 0),  # untrained: pure noise
    (2.00, 20),  # word-salad: real words, no sentences
    (1.50, 40),  # broken prose: sentences form, meaning drifts within a line
    (1.33, 55),  # early training (measured: 500M @ 0.7B tokens)
    (1.19, 70),  # solid but visibly undertrained (measured: 500M @ 4.7B tokens)
    (1.10, 85),  # best tiny model measured on this data (341M, Vintage1)
    (1.05, 92),  # a little beyond the best we have seen at this scale
    (0.95, 100),  # estimated ceiling for sub-1B on this corpus
]

# Reference ladder shown in the report (display only -- scoring uses BPB_LADDER
# above). `measured` entries were produced by `python -m eval <path> --force`
# against the SAME held-out file, so they are directly comparable; `synthetic`
# entries are qualitative signposts, not model measurements.
#
# TO ADD A MODEL: run `python -m eval MODELS/<name>` and paste its prose_bpb here.
# TO RE-ANCHOR after changing eval_data/heldout.jsonl: every `measured` row must
# be re-run, or the ladder silently mixes two different held-out sets.
REFERENCE_LADDER = [
    (3.50, 'untrained model (uniform noise)', 'synthetic'),
    (2.00, 'word-salad', 'synthetic'),
    (1.50, 'broken prose', 'synthetic'),
    (1.44380, 'TypeWriter-1913-7B-v1 -- 7.2B, chat-tuned; strong logic, WEAK prose', 'measured'),
    (1.36439, 'Violet-160m -- GPT-NeoX 152M, Victorian-trained (1800-1899)', 'measured'),
    (1.27043, 'TypeWriter-1913-7B-v2 -- 7.2B, chat-tuned; best chat BPB measured (0.707)', 'measured'),
    (1.19189, 'TimeCapsule -- Llama 499M, ~4.7B tokens (undertrained but solid)', 'measured'),
    (1.16932, 'Llama-77M-v1 -- 77M, 16.7B tokens, 80h (saw held-out docs ~1x; NO recall detected)', 'measured'),
    (1.11899, 'vintage-LLM-340m -- Llama 341M, best CLEAN sub-1B on this data', 'measured'),
    (0.95, 'estimated sub-1B ceiling on this corpus', 'synthetic'),
    (0.91380, 'Talkie-1930-13b -- 13.3B, best measured on this data  [int8, UNVERIFIED provenance]', 'measured'),
    # ^ RETIRED and NOT re-evaluable: the model saturates this machine. Its source
    #   JSON holds no results (see MODELS/Talkie-1930-13b/eval-talkie13.3B.json for
    #   why); the numbers behind this rung are recorded in
    #   research/REPORT-reference-models.md.
]

# Llama-77M-v1 trained ~0.64 epochs over a corpus that CONTAINS all 200 held-out
# documents. That is a provenance fact and stays on record. It is NOT, however,
# evidence of memorisation, and the earlier '[CONTAMINATED]' label overstated it.
#
# MEASURED, not argued (research/REPORT-memorization-probe.md): a verbatim
# continuation probe -- feed 256 tokens of each document, greedy-decode 128,
# score exact token overlap -- against a control that saw only 2.4% of the corpus:
#
#            max exact span   docs with span >=16   paired delta vs control
#   llama-77       11 tokens              0 / 200          +0.125 tokens
#   control         7 tokens              0 / 200                  --
#
# llama-77 produced a LONGER span in only 26.5% of documents. Memorisation shows
# up as a heavy right tail; there is none. At 77M parameters (~19 MB of storage at
# a generous 2 bits/param) against a 110 GB corpus seen once, there is no capacity
# to memorise. Its early-vs-late context gap (0.167) is also indistinguishable
# from the control's (0.168).
#
# Treat its 1.16932 as a normal measurement. Keep the provenance note so nobody
# has to re-derive this.
#
# The 1913/1930 models are THIRD-PARTY: we did not build their training sets, and
# our held-out documents are Gutenberg/British-Library derived, so we cannot rule
# out that they saw them. Treat their numbers as aspiration targets, not as clean
# generalisation measurements.
# Provenance flag only: these models trained on a corpus containing the held-out
# documents. See the note above -- for Llama-77M-v1 this was probed and NO recall
# was found. The flag records what we know about the data, not a defect.
TRAINED_ON_HELDOUT_CORPUS = {'Llama-77M-v1'}
CONTAMINATED_REFERENCES = TRAINED_ON_HELDOUT_CORPUS  # legacy alias
UNVERIFIED_PROVENANCE = {'TypeWriter-1913-7B-v1', 'TypeWriter-1913-7B-v2', 'Talkie-1930-13b'}

# Talkie-1930-13b is the ONLY ladder row measured under quantization: 24.7 GiB in
# bf16 does not fit a 15.9 GiB card, so it was scored with `--load-8bit`. On the
# 75M control int8 moved mean NLL by -0.0075 (~0.0025 bpb), i.e. well under the
# 0.0092 gap between adjacent ladder rungs -- but its 0.91380 is not strictly
# comparable to the bf16 rows.
QUANTIZED_REFERENCES = {'Talkie-1930-13b'}

# The two TypeWriter arms show why prose BPB alone is not a model ranking: they
# are the WORST and third-worst rows here on prose, yet post the best chat BPB
# (0.707/0.713 vs 0.981 for vintage-LLM-340m) and the best logic (0.875/0.925 vs
# 0.750). They are chat/instruction-tuned, which trades raw prose likelihood for
# dialogue. Do not read their prose rung as "a 7B is worse than our 341M".

# NOT on the ladder: MODELS/Mr-Chatterbox is a raw nanochat `model.pt` with no
# HF config.json, so `python -m eval` cannot load it. It needs a conversion
# script before it can be anchored.

LOGIC_BANDS = [  # (accuracy ceiling, label)
    (0.55, 'coin-flip - the model does not prefer sense over nonsense'),
    (0.65, 'weak - style without understanding'),
    (0.75, 'normal for a good sub-1B model (Vintage1 scores 0.70-0.78)'),
    (0.85, 'unusually strong for this size'),
    (1.01, '7B territory (TypeWriter scores 0.88-0.93)'),
]

CHAT_LADDER = [  # bits/byte on chat targets -> chat-readiness points
    (1.60, 0),
    (1.26, 40),
    (1.10, 65),
    (0.93, 85),
    (0.82, 95),
    (0.73, 100),
]

# Composite weights (rho vs training step measured on Vintage2's 11 checkpoints).
WEIGHTS = {
    'bpb': 0.50,  # rho -1.000  held-out bits/byte
    'logic': 0.25,  # rho +0.64 (margin)  prefers coherent over incoherent
    'chat': 0.15,  # rho -0.99  fine-tunability for chat
    'hygiene': 0.10,  # rho -0.45/-0.62  greedy looping + punctuation
}

CHINCHILLA_TOKENS_PER_PARAM = 20  # compute-optimal rule of thumb


def interp(x: float, ladder) -> float:
    """Piecewise-linear map through measured (value, points) anchors."""
    if x is None or math.isnan(x):
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
    if acc is None or math.isnan(acc):
        return float('nan')
    return max(0.0, min(100.0, (acc - 0.5) / (0.92 - 0.5) * 100.0))


def hygiene_points(loop_len: float, punct_p100: float) -> float:
    """loop 0 words = 100 pts, 80+ = 0. punct 0/100w = 100 pts, 1+ = 0."""
    lp = max(0.0, 100.0 - loop_len * 1.25)
    pp = max(0.0, 100.0 - punct_p100 * 100.0)
    return 0.7 * lp + 0.3 * pp


def bake_score(parts: dict) -> float:
    total, wsum = 0.0, 0.0
    for k, w in WEIGHTS.items():
        v = parts.get(k)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            total += w * v
            wsum += w
    return total / wsum if wsum else float('nan')


def band_label(acc: float) -> str:
    for hi, label in LOGIC_BANDS:
        if acc < hi:
            return label
    return LOGIC_BANDS[-1][1]


def verdict_text(score: float, lineage: dict) -> tuple[str, str]:
    """(tier, blunt paragraph)."""
    if score is None or math.isnan(score):
        return 'UNSCORED', 'Not enough metrics ran to produce a verdict.'
    if score >= 85:
        tier, body = (
            'BAKED',
            (
                'This is about as good as a sub-1B model gets on this corpus. Further '
                'pretraining will buy very little; if you want a chatbot, fine-tune this '
                'checkpoint. If you want a *better* model, you need more parameters or '
                'better data, not more steps.'
            ),
        )
    elif score >= 70:
        tier, body = (
            'GOLDEN CRUST, SOFT MIDDLE',
            (
                'A solid base model, but measurably short of what this scale can reach. '
                'It writes in period style and mostly holds a sentence together, yet it '
                'still loses the thread of meaning. More clean tokens would still help.'
            ),
        )
    elif score >= 55:
        tier, body = (
            'HALF-BAKED',
            (
                'Clearly undertrained. The style is there but the substance is not: '
                'expect confident nonsense, topic drift and heavy looping under greedy '
                'decoding. Do not fine-tune this for chat yet - keep pretraining.'
            ),
        )
    elif score >= 35:
        tier, body = (
            'DOUGH',
            (
                'Structure is forming - real words, some grammar - but this is not a '
                'usable language model yet. It needs several times more training tokens.'
            ),
        )
    else:
        tier, body = (
            'RAW BATTER',
            (
                'Barely past random guessing. Either training has only just started, or '
                'something is broken (learning rate, data pipeline, tokenizer mismatch).'
            ),
        )
    tpp = lineage.get('tokens_per_param')
    if tpp is not None and lineage.get('tokens_lower_bound'):
        body += (
            f' Compute check: at least ~{tpp:.1f} tokens per parameter, but the '
            f'training was restarted so the true total is unknown (rule of '
            f'thumb for "fully fed": ~{CHINCHILLA_TOKENS_PER_PARAM}).'
        )
    elif tpp is not None:
        if tpp < CHINCHILLA_TOKENS_PER_PARAM * 0.75:
            body += (
                f' Compute check: it has seen ~{tpp:.1f} tokens per parameter; '
                f'the compute-optimal rule of thumb is ~{CHINCHILLA_TOKENS_PER_PARAM}. '
                f'It is undertrained *by construction* - the single cheapest '
                f'improvement is simply more tokens.'
            )
        elif tpp > CHINCHILLA_TOKENS_PER_PARAM * 3:
            body += (
                f' Compute check: ~{tpp:.0f} tokens per parameter is well past '
                f'compute-optimal; if quality has plateaued, more steps are now '
                f'wasted money compared with training a bigger model.'
            )
        else:
            body += (
                f' Compute check: ~{tpp:.1f} tokens per parameter - in the sensible range (rule of thumb: ~{CHINCHILLA_TOKENS_PER_PARAM}).'
            )
    return tier, body
