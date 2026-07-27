#!/usr/bin/env python3
"""
evaluate.py - Checkpoint evaluation harness.

Ports the diachronic-analysis evaluation logic from timecapsule/06_evaluate_model.py
into the main project, unified with the checkpoint-resolution behaviour of
generate.py / vibe_check.py / fine_tune.py:

  * no flags            -> evaluate the LATEST checkpoint in ./checkpoints
  * --checkpoints-dir D -> evaluate the latest checkpoint found in D
  * --checkpoint PATH   -> evaluate exactly this checkpoint directory

The script is organised into pluggable "suites" (see SUITES at the bottom of the
CONSTANTS section). New evaluations (e.g. automatic-essay-grading) should be
added as a new `suite_*` function plus one line in SUITES - nothing else needs
to change.

All metrics are printed with a one-line plain-English explanation, because the
audience is technical but not data-scientists. A full glossary is available
with --explain.

Examples:
  python evaluate.py  # latest checkpoint, all suites
  python evaluate.py --checkpoint checkpoints/checkpoint-xx
  python evaluate.py --suites info perplexity --output results.json
  python evaluate.py --list-suites
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

PROJECT_DIR = Path(__file__).resolve().parent

# ============================================================================
# CONSTANTS - tweak these
# ============================================================================

# --- Checkpoint resolution ---------------------------------------------------
DEFAULT_CHECKPOINTS_DIR = PROJECT_DIR / 'checkpoints'

# --- Historical probe words --------------------------------------------------
# Words with well-documented semantic shifts between ~1800-1875 and today.
# Each word is evaluated inside HISTORICAL_CONTEXTS[i] (period-flavoured usage)
# and MODERN_CONTEXTS[i] (contemporary usage). Comparing the two shows whether
# the model has internalised the historical senses.
HISTORICAL_WORDS = [
    'gay',  # originally "happy, carefree"
    'awful',  # originally "awe-inspiring"
    'nice',  # originally "ignorant, foolish"
    'meat',  # originally "food" in general
    'want',  # originally "lack"
    'python',  # originally a snake
    'commerce',  # trade
    'parliament',  # government
    'science',  # knowledge (broader meaning in 19th century)
    'manufacture',  # literally "made by hand"
]

HISTORICAL_CONTEXTS = [
    'The gay festivities brought joy to all.',
    'The awful majesty of the cathedral inspired reverence.',
    'His nice distinction between the two was quite foolish.',
    'The meat and drink were provided generously.',
    'The people want for bread.',
    'Python is a constrictor snake from Africa.',
    'The commerce between nations flourishes.',
    'Parliament convened to discuss the matter.',
    'He devoted himself to natural science.',
    'The manufacture of cotton was done by hand.',
]

MODERN_CONTEXTS = [
    'He came out as gay last year.',
    'The awful weather ruined our picnic.',
    'She is a really nice person.',
    "I don't eat meat, I'm vegan.",
    'Most people want the new iPhone.',
    'Python is a popular programming language.',
    'Mobile commerce keeps growing.',
    'European Union parliament voted on the bill.',
    'She studies computer science.',
    'Robots manufacture cars.',
]

# --- Generation probes -------------------------------------------------------
# Short open-ended prompts continued by the model to sanity-check fluency.
GENERATION_PROMPTS = [
    # generic fluency
    'The history of the world is',
    'What is God? God is',
    # era-specific registers (dates, industry, letters, novels, monarchy)
    'In the year of our Lord eighteen hundred and',
    'The manufacture of cotton',
    'The steam engine',
    'The telegraph',
    'My dearest sister,',
    'Greetings, my friend',
    'Her Majesty the Queen',
    'Chapter I.',
]

# --- Perplexity probe --------------------------------------------------------
# The perplexity suite scores these sentences (historical + modern contexts by
# default). Lower perplexity on the historical set relative to the modern set
# indicates the model has absorbed period language.
PERPLEXITY_SENTENCES = HISTORICAL_CONTEXTS + MODERN_CONTEXTS
PERPLEXITY_LABELS = ['historical'] * len(HISTORICAL_CONTEXTS) + ['modern'] * len(MODERN_CONTEXTS)

# --- Suite behaviour knobs ---------------------------------------------------
TOP_K_NEIGHBORS = 5  # neighbours shown per word in the embedding suite
SIMILARITY_MATRIX_WORDS = 5  # how many words to show in the similarity matrix
GEN_MAX_NEW_TOKENS = 60
GEN_TEMPERATURE = 0.8
GEN_TOP_P = 0.9
GEN_TOP_K = 25
GEN_REPETITION_PENALTY = 1.1
DEFAULT_SEED = 1337

# ============================================================================
# ARGUMENTS / DEVICE / CHECKPOINT RESOLUTION
# ============================================================================


def select_device(name: str) -> torch.device:
    if name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(name)


def select_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != 'auto':
        return getattr(torch, name)
    if device.type == 'cuda':
        return torch.bfloat16
    return torch.float32


def resolve_checkpoint(checkpoint: Path | None, checkpoints_dir: Path) -> Path:
    """Pick --checkpoint if given, else the numerically-latest checkpoint."""
    if checkpoint is not None:
        if not (checkpoint / 'config.json').exists():
            raise FileNotFoundError(f'No config.json in {checkpoint} - not a valid HF checkpoint directory.')
        return checkpoint
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f'Checkpoints directory not found: {checkpoints_dir}')
    latest = get_last_checkpoint(str(checkpoints_dir))
    if latest is None:
        raise FileNotFoundError(f'No checkpoint found in {checkpoints_dir}')
    return Path(latest)


def load_model_and_tokenizer(checkpoint_dir: Path, device: torch.device, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, dtype=dtype)
    model.config.use_cache = True
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # enables correct batched generation
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, help='Specific checkpoint directory to evaluate.')
    p.add_argument(
        '--compare',
        action='store_true',
        help='Evaluate EVERY checkpoint in --checkpoints-dir, print a comparison table '
        'and a plain-English verdict (is the model baked enough?). '
        'Per-checkpoint JSONs are cached in --output (default: eval_results/compare/) '
        'so interrupted runs resume where they left off.',
    )
    p.add_argument('--force', action='store_true', help='With --compare: recompute even if a cached JSON exists.')
    p.add_argument(
        '--checkpoints-dir',
        type=Path,
        default=DEFAULT_CHECKPOINTS_DIR,
        help='Directory containing checkpoints (latest is used when --checkpoint is absent).',
    )
    p.add_argument('--suites', nargs='+', default=None, metavar='NAME', help='Suites to run (default: all). See --list-suites.')
    p.add_argument('--list-suites', action='store_true', help='List available suites and exit.')
    p.add_argument('--explain', action='store_true', help='Print a glossary of every metric at the end of the report.')
    p.add_argument(
        '--output', type=Path, default=None, help='Also write the full numeric results to this JSON file (for checkpoint comparison).'
    )
    p.add_argument('--tokens', type=int, default=GEN_MAX_NEW_TOKENS, help='New tokens per generation probe.')
    p.add_argument('--temperature', type=float, default=GEN_TEMPERATURE)
    p.add_argument('--top-p', type=float, default=GEN_TOP_P)
    p.add_argument('--top-k', type=int, default=GEN_TOP_K)
    p.add_argument('--repetition-penalty', type=float, default=GEN_REPETITION_PENALTY)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda', 'mps'))
    p.add_argument('--dtype', default='auto', choices=('auto', 'float32', 'float16', 'bfloat16'))
    p.add_argument('--chat', action='store_true', help='Apply the chat template to generation prompts.')
    p.add_argument('--show-special-tokens', action='store_true')
    return p.parse_args()


# ============================================================================
# SHARED STATISTICS HELPERS
# ============================================================================


def token_stats_from_logprobs(logprobs: torch.Tensor, entropies: torch.Tensor) -> dict:
    """Summarise per-token log-probabilities (nats) into human-readable stats."""
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


def per_token_loss(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """
    Cross-entropy and entropy for every real (non-pad, non-first) position.
    Returns concatenated 1-D tensors over the whole batch.
    """
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Predict token t+1 from position t (standard causal-LM shift).
    shift_logits = logits[:, :-1].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].bool()

    logp_all = F.log_softmax(shift_logits, dim=-1)
    token_logp = logp_all.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(logp_all.exp() * logp_all).sum(-1)

    return token_logp[shift_mask], entropy[shift_mask]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ngram_stats(text: str) -> dict:
    """Distinct-n and repetition metrics over whitespace tokens of ONE text."""
    words = text.split()
    out = {'num_words': len(words)}
    for n in (1, 2):
        grams = list(zip(*(words[i:] for i in range(n)))) if len(words) >= n else []
        out[f'distinct_{n}'] = (len(set(grams)) / len(grams)) if grams else 1.0
    # Fraction of words identical to the word 1..4 positions earlier (echo detector).
    repeats = sum(1 for i, w in enumerate(words) if i > 0 and w in words[max(0, i - 4) : i])
    out['echo_rate'] = repeats / max(1, len(words))
    return out


def fmt(v, nd=4):
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def banner(title: str):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


# ============================================================================
# SUITES
# Each suite: fn(model, tokenizer, device, args) -> dict of JSON-safe results.
# To add a new evaluation: write a suite_* function, register it in SUITES.
# ============================================================================


def suite_info(model, tokenizer, device, args) -> dict:
    """Static facts about the checkpoint: architecture, size, tokenizer."""
    banner('SUITE: MODEL INFO')
    cfg = model.config
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = model.get_input_embeddings().weight
    weight_files = list(Path(args._checkpoint_dir).glob('*.safetensors'))
    disk_mb = sum(f.stat().st_size for f in weight_files) / 1e6

    results = {
        'checkpoint': str(args._checkpoint_dir),
        'model_type': cfg.model_type,
        'architecture': type(model).__name__,
        'parameters': params,
        'parameters_millions': round(params / 1e6, 2),
        'hidden_size': cfg.hidden_size,
        'num_layers': cfg.num_hidden_layers,
        'num_attention_heads': cfg.num_attention_heads,
        'num_key_value_heads': getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads),
        'vocab_size_config': cfg.vocab_size,
        'vocab_size_tokenizer': len(tokenizer),
        'max_position_embeddings': cfg.max_position_embeddings,
        'tie_word_embeddings': getattr(cfg, 'tie_word_embeddings', None),
        'disk_size_mb': round(disk_mb, 1),
        'device': str(device),
        'dtype': str(next(model.parameters()).dtype),
        'has_chat_template': tokenizer.chat_template is not None,
        # Embedding-matrix health: mean norm + anisotropy (avg pairwise cos of
        # a sample of rows). High anisotropy (~1) = embeddings collapsed into a
        # narrow cone, common in undertrained models.
        'embedding_mean_norm': None,
        'embedding_anisotropy': None,
    }

    with torch.no_grad():
        results['embedding_mean_norm'] = float(emb.float().norm(dim=-1).mean())
        sample = emb[torch.randperm(emb.shape[0])[:512]].float()
        sample = sample / sample.norm(dim=-1, keepdim=True)
        sims = sample @ sample.T
        off_diag = sims[~torch.eye(len(sample), dtype=bool, device=sample.device)]
        results['embedding_anisotropy'] = float(off_diag.mean())

    for k in ('checkpoint', 'model_type', 'architecture', 'device', 'dtype'):
        print(f'  {k:24}: {results[k]}')
    print(f'  {"parameters":24}: {params:,} ({results["parameters_millions"]}M)')
    print(f'  {"disk size":24}: {results["disk_size_mb"]:.0f} MB')
    print(
        f'  {"layers / hidden / heads":24}: {cfg.num_hidden_layers} / {cfg.hidden_size} / '
        f'{cfg.num_attention_heads} (KV heads: {results["num_key_value_heads"]})'
    )
    print(f'  {"vocab (cfg / tokenizer)":24}: {cfg.vocab_size} / {len(tokenizer)}')
    print(f'  {"context length":24}: {cfg.max_position_embeddings}')
    print(f'  {"chat template":24}: {results["has_chat_template"]}')
    print(
        f'  {"embedding mean norm":24}: {fmt(results["embedding_mean_norm"])}'
        '   <- typical row length of the embedding matrix; very small or huge values signal trouble'
    )
    print(
        f'  {"embedding anisotropy":24}: {fmt(results["embedding_anisotropy"])}'
        '   <- avg cosine between random embeddings; near 1.0 means tokens are barely distinguishable'
    )
    if results['vocab_size_tokenizer'] > results['vocab_size_config']:
        print('  WARNING: tokenizer vocab is LARGER than config vocab_size - checkpoint/tokenizer mismatch?')
    elif results['vocab_size_config'] != results['vocab_size_tokenizer']:
        print(f'  (note: config vocab padded by {results["vocab_size_config"] - results["vocab_size_tokenizer"]} unused rows - harmless)')
    return results


def suite_perplexity(model, tokenizer, device, args) -> dict:
    """
    Perplexity of fixed probe sentences (historical vs modern contexts).
    No external dataset needed, and results are directly comparable across
    checkpoints because the sentences are constants in this file.
    """
    banner('SUITE: PERPLEXITY ON FIXED PROBE SENTENCES')
    print(
        f'  Scoring {len(PERPLEXITY_SENTENCES)} sentences '
        f'({PERPLEXITY_LABELS.count("historical")} historical, {PERPLEXITY_LABELS.count("modern")} modern)'
    )

    per_sentence = []
    all_logp, all_ent = [], []
    for sent, label in zip(PERPLEXITY_SENTENCES, PERPLEXITY_LABELS):
        enc = tokenizer(sent, return_tensors='pt').to(device)
        logp, ent = per_token_loss(model, enc['input_ids'], enc['attention_mask'])
        st = token_stats_from_logprobs(logp, ent)
        st.update({'label': label, 'sentence': sent, 'num_tokens': int(logp.numel())})
        per_sentence.append(st)
        all_logp.append(logp)
        all_ent.append(ent)

    all_logp = torch.cat(all_logp)
    all_ent = torch.cat(all_ent)
    overall = token_stats_from_logprobs(all_logp, all_ent)
    overall['num_tokens'] = int(all_logp.numel())

    results = {'overall': overall, 'per_sentence': per_sentence}
    for name in ('historical', 'modern'):
        rows = [s for s in per_sentence if s['label'] == name]
        toks = sum(r['num_tokens'] for r in rows)
        # Exact token-weighted group perplexity: exp of the mean log-loss,
        # recovered from each sentence's perplexity (log ppl = mean log-loss).
        mean_log_loss = sum(math.log(r['perplexity']) * r['num_tokens'] for r in rows) / toks
        results[name] = {
            'num_sentences': len(rows),
            'perplexity': math.exp(mean_log_loss),
            'mean_sentence_perplexity': float(np.mean([r['perplexity'] for r in rows])),
        }

    print(f'\n  OVERALL ({overall["num_tokens"]} tokens)')
    print(
        f'    perplexity            : {fmt(overall["perplexity"], 2)}'
        "   <- 'average branching factor'; lower = less surprised = better fit to this text"
    )
    print(f'    mean token prob       : {fmt(overall["mean_token_prob"])}   <- average confidence on the actual next word')
    print(f'    median token prob     : {fmt(overall["median_token_prob"])}   <- typical confidence (robust to a few very bad tokens)')
    print(
        f'    worst-token prob (min): {overall["min_token_prob"]:.1e}'
        "   <- the single most surprising word; tiny values = the model really didn't expect it"
    )
    print(f"    10th-percentile prob  : {overall['p10_token_prob']:.1e}   <- confidence on the model's worst 10% of guesses")
    print(
        f'    low-confidence tokens : {fmt(overall["frac_low_confidence"] * 100, 2)}%'
        '   <- share of tokens predicted with <1% probability (red flags)'
    )
    print(
        f'    mean entropy          : {fmt(overall["mean_entropy_nats"], 3)} nats'
        '   <- how spread out the predictions are; ~0 = very certain, higher = hedging'
    )

    print('\n  GROUP COMPARISON')
    print(f'    {"group":<12}{"sentences":>10}{"perplexity":>14}')
    for name in ('historical', 'modern'):
        g = results[name]
        print(f'    {name:<12}{g["num_sentences"]:>10}{g["perplexity"]:>14.2f}')
    ratio = results['modern']['perplexity'] / max(results['historical']['perplexity'], 1e-9)
    results['modern_over_historical_ratio'] = ratio
    print(
        f'    modern/historical ratio: {ratio:.2f}'
        '   <- >1 means period text is easier for the model than modern text'
        ' (desired for a vintage model with a ~1900 knowledge cutoff)'
    )

    print('\n  PER-SENTENCE DETAIL (sorted worst-first)')
    for st in sorted(per_sentence, key=lambda r: -r['perplexity']):
        print(f'    [{st["label"]:<10}] ppl={st["perplexity"]:>8.2f}  minp={st["min_token_prob"]:.1e}  {st["sentence"][:52]}')
    return results


def find_word_token_positions(tokenizer, context: str, word: str, seq_len: int) -> list[int] | None:
    """
    Token positions covering `word` inside `context`, found via character offsets.
    Robust to BPE merging the word with a leading space (unlike matching
    tokenizer.encode(word) against the sentence's token ids).
    """
    char_start = context.lower().find(word.lower())
    if char_start < 0:
        return None
    char_end = char_start + len(word)
    offsets = tokenizer(context, return_offsets_mapping=True)['offset_mapping']
    positions = [
        i
        for i, (s, e) in enumerate(offsets[:seq_len])
        if e > s and s < char_end and e > char_start  # real token overlapping the word's chars
    ]
    return positions or None


def extract_word_embeddings(model, tokenizer, device, words, contexts) -> dict:
    """Last-hidden-layer embedding of each word, averaged over its sub-tokens."""
    embeddings, missing = {}, []
    for word, context in zip(words, contexts):
        inputs = tokenizer(context, return_tensors='pt').to(device)
        with torch.no_grad():
            hidden = model(**inputs, output_hidden_states=True).hidden_states[-1]
        positions = find_word_token_positions(tokenizer, context, word, inputs['input_ids'].shape[1])
        if not positions:
            missing.append(word)
            continue
        embeddings[word] = hidden[0, positions].float().mean(dim=0).cpu().numpy()
    if missing:
        print(f'  (skipped {len(missing)} words not locatable in their context: {", ".join(missing)})')
    return embeddings


def suite_embeddings(model, tokenizer, device, args) -> dict:
    """Diachronic embedding analysis (ported from timecapsule 06_evaluate_model.py)."""
    banner('SUITE: DIACHRONIC WORD-EMBEDDING ANALYSIS')
    print(f'  Extracting embeddings for {len(HISTORICAL_WORDS)} words in historical AND modern contexts...')

    hist_emb = extract_word_embeddings(model, tokenizer, device, HISTORICAL_WORDS, HISTORICAL_CONTEXTS)
    mod_emb = extract_word_embeddings(model, tokenizer, device, HISTORICAL_WORDS, MODERN_CONTEXTS)

    words = [w for w in HISTORICAL_WORDS if w in hist_emb]
    results = {'words': words, 'pairwise_historical': {}, 'semantic_shift': {}, 'neighbors': {}}

    # --- 1. How much does each word's meaning move between contexts? ---------
    print('\n  CONTEXT SENSITIVITY (cosine similarity of the SAME word, historical vs modern context)')
    print("    low value = the word's representation changes a lot with context (semantic shift)")
    shifts = {}
    for w in words:
        if w in mod_emb:
            shifts[w] = cosine_similarity(hist_emb[w], mod_emb[w])
    frozen_prefix = []
    for w, s in sorted(shifts.items(), key=lambda kv: kv[1]):
        print(f'    {w:<14}: {s:+.3f}')
        if s >= 0.999:
            frozen_prefix.append(w)
    if frozen_prefix:
        print(
            f'    (note: {", ".join(frozen_prefix)} score exactly 1.0 because the words sit after an'
            ' identical sentence prefix in both contexts - a causal LM cannot see words to its'
            ' right, so the representation is literally the same. Edit the context constants'
            ' to vary the prefixes if you want these words measured.)'
        )
    results['semantic_shift'] = shifts
    if shifts:
        results['mean_shift_similarity'] = float(np.mean(list(shifts.values())))
        print(
            f'    {"MEAN":<14}: {results["mean_shift_similarity"]:+.3f}'
            '   <- near 1.0 = model ignores historical vs modern usage; lower = senses differ'
        )

    # --- 2. Similarity matrix within the historical embeddings ---------------
    show = words[:SIMILARITY_MATRIX_WORDS]
    print(f'\n  PAIRWISE COSINE SIMILARITY (historical contexts, first {len(show)} words)')
    print('    1.0 = identical meaning, 0 = unrelated, negative = opposites')
    header = '    ' + ' ' * 13 + ''.join(f'{w[:10]:>11}' for w in show)
    print(header)
    all_pairs = []
    for i, w1 in enumerate(words):
        if w1 in show:
            row = f'    {w1[:12]:<13}'
            for j, w2 in enumerate(show):
                row += f'{cosine_similarity(hist_emb[w1], hist_emb[w2]):>11.3f}' if j >= i else ' ' * 11
            print(row)
        for j in range(i + 1, len(words)):
            s = cosine_similarity(hist_emb[words[i]], hist_emb[words[j]])
            results['pairwise_historical'][f'{words[i]}|{words[j]}'] = s
            all_pairs.append(s)
    if all_pairs:
        arr = np.array(all_pairs)
        results['pairwise_mean'] = float(arr.mean())
        results['pairwise_std'] = float(arr.std())
        print(
            f'    all {len(all_pairs)} pairs: mean={arr.mean():+.3f} std={arr.std():.3f}'
            '   <- high mean = embeddings collapsed together (bad); healthy spread is good'
        )

    # --- 3. Nearest neighbours -----------------------------------------------
    print('\n  NEAREST NEIGHBOURS inside the probe-word set (historical contexts)')
    for w in show:
        sims = sorted(((o, cosine_similarity(hist_emb[w], hist_emb[o])) for o in words if o != w), key=lambda kv: -kv[1])[:TOP_K_NEIGHBORS]
        results['neighbors'][w] = [(o, round(s, 4)) for o, s in sims]
        joined = ', '.join(f'{o} ({s:+.2f})' for o, s in sims)
        print(f'    {w:<14}: {joined}')
    return results


def suite_generation(model, tokenizer, device, args) -> dict:
    """Continue fixed prompts and measure fluency, confidence and repetition."""
    banner('SUITE: GENERATION PROBES')
    torch.manual_seed(args.seed)
    prompts = []
    for p in GENERATION_PROMPTS:
        if args.chat:
            prompts.append(tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True))
        else:
            prompts.append(p)

    inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
    do_sample = args.temperature > 0
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            top_k=args.top_k if do_sample else None,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
    elapsed = time.perf_counter() - t0
    # NOTE: prompts are LEFT-padded, so generated tokens always start at the
    # padded input width - slicing by per-row prompt length would wrongly
    # include tail prompt tokens for the shorter (padded) rows.
    input_width = inputs['input_ids'].shape[1]

    generations = []
    total_new = 0
    stop_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    for i, prompt in enumerate(GENERATION_PROMPTS):
        seq = out.sequences[i]
        gen_ids = list(seq[input_width:])
        # Rows that finished early (EOS) are padded with pad tokens up to the
        # batch's max length; exclude them from text and stats.
        gen_ids = [t for t in gen_ids if t.item() not in stop_ids]
        total_new += len(gen_ids)
        text = tokenizer.decode(gen_ids, skip_special_tokens=not args.show_special_tokens, clean_up_tokenization_spaces=False)
        # Confidence of the sampling distribution at each generated step.
        probs, ents = [], []
        for step, scores in enumerate(out.scores):
            if step >= len(gen_ids):
                break
            lp = F.log_softmax(scores[i].float(), dim=-1)
            probs.append(lp[gen_ids[step]].exp().item())
            ents.append(float(-(lp.exp() * lp).sum()))
        ng = ngram_stats(text)
        rec = {
            'prompt': prompt,
            'continuation': text.strip(),
            'new_tokens': len(gen_ids),
            'mean_token_prob': float(np.mean(probs)) if probs else None,
            'min_token_prob': float(np.min(probs)) if probs else None,
            'mean_entropy_nats': float(np.mean(ents)) if ents else None,
            **ng,
        }
        generations.append(rec)
        print(f'\n  --- Probe {i + 1} ---')
        print(f'  prompt      : {prompt}')
        print(f'  continuation: {rec["continuation"][:400]}')
        print(
            f'  stats: {rec["new_tokens"]} tokens | mean prob {fmt(rec["mean_token_prob"], 3)}'
            f' | min prob {rec["min_token_prob"]:.1e}'
            f' | distinct-1 {fmt(rec["distinct_1"], 2)} | distinct-2 {fmt(rec["distinct_2"], 2)}'
            f' | echo {fmt(rec["echo_rate"], 2)}'
        )
        print(
            '         (mean/min prob = confidence in sampled words; distinct-n = vocabulary variety,'
            ' near 0 = repetitive; echo = fraction of words repeated from the previous 4)'
        )

    results = {
        'generations': generations,
        'total_new_tokens': total_new,
        'elapsed_seconds': round(elapsed, 2),
        'tokens_per_second': round(total_new / max(elapsed, 1e-9), 1),
        'mean_distinct_1': float(np.mean([g['distinct_1'] for g in generations])),
        'mean_distinct_2': float(np.mean([g['distinct_2'] for g in generations])),
        'mean_echo_rate': float(np.mean([g['echo_rate'] for g in generations])),
        'mean_token_prob': float(np.mean([g['mean_token_prob'] for g in generations if g['mean_token_prob'] is not None])),
    }
    print(f'\n  AGGREGATE over {len(generations)} probes: {total_new} tokens in {elapsed:.1f}s ({results["tokens_per_second"]} tok/s)')
    print(
        f'    mean distinct-1 {fmt(results["mean_distinct_1"], 3)}'
        f' | mean distinct-2 {fmt(results["mean_distinct_2"], 3)}'
        f' | mean echo {fmt(results["mean_echo_rate"], 3)}'
        f' | mean prob {fmt(results["mean_token_prob"], 3)}'
    )
    if results['mean_echo_rate'] > 0.4 or results['mean_distinct_2'] < 0.5:
        print('    WARNING: high repetition - typical of an undertrained checkpoint or too-low temperature.')
    return results


# ============================================================================
# SUITE REGISTRY - add future evaluations (e.g. automatic-essay-grading) here
# ============================================================================

SUITES = {
    'info': suite_info,
    'perplexity': suite_perplexity,
    'embeddings': suite_embeddings,
    'generation': suite_generation,
}

GLOSSARY = """
GLOSSARY (plain English)
------------------------
perplexity            e^(average loss). Roughly "on average, the model hesitated between
                      N equally-likely next words". Lower is better. Comparable across
                      checkpoints ONLY on the same fixed sentences (that's why they're constants).
token probability     Softmax probability the model assigned to the word that actually came
                      next. Mean = average confidence, median = typical confidence,
                      min/10th-percentile = how bad the worst guesses get.
low-confidence share  Fraction of tokens predicted with <1% probability. Spikes here
                      usually mean OOV words, noise, or genuine surprise.
entropy (nats)        Spread of the next-word distribution. ~0 = dead certain; high =
                      many plausible continuations. Not good or bad by itself - read with
                      token probability.
anisotropy            Average cosine similarity between random embedding rows. Near 1.0
                      means all token vectors point the same way (undertrained/collapsed).
context sensitivity   Cosine similarity of the same word in a historical vs a modern
                      sentence. Lower = the model represents the two senses differently.
distinct-1 / -2       Fraction of unique words / word-pairs in generated text.
                      Low values = the model is looping.
echo rate             Fraction of generated words that already appeared within the
                      previous 4 words. High values = stuttering/repetition.
"""

# ============================================================================
# COMPARE MODE - evaluate every checkpoint, then explain what it all means
# ============================================================================


def list_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """All valid checkpoints, sorted by training step (checkpoint-NNNNN)."""

    def step_of(p: Path) -> float:
        try:
            return int(p.name.rsplit('-', 1)[1])
        except (IndexError, ValueError):
            return math.inf  # e.g. 'final' sorts last

    ckpts = [p for p in checkpoints_dir.iterdir() if p.is_dir() and (p / 'config.json').exists()]
    return sorted(ckpts, key=step_of)


def evaluate_checkpoint(checkpoint_dir: Path, device, dtype, args, selected) -> dict:
    """Run the selected suites on one checkpoint and free the model afterwards."""
    args._checkpoint_dir = checkpoint_dir
    banner('CHECKPOINT EVALUATION')
    print(f'  checkpoint : {checkpoint_dir}')
    print(f'  device     : {device}   dtype: {dtype}   seed: {args.seed}')
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir, device, dtype)
    results = {'checkpoint': str(checkpoint_dir), 'suites': {}}
    for name in selected:
        results['suites'][name] = SUITES[name](model, tokenizer, device, args)
    del model, tokenizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return results


def checkpoint_provenance(ckpt_dir: Path) -> str:
    """Human-readable training lineage, sniffed from files inside the checkpoint."""
    import tomllib

    def flatten(d, out):
        for k, v in d.items():
            if isinstance(v, dict):
                flatten(v, out)
            else:
                out[k] = v
        return out

    ctx = None
    cfg_json = ckpt_dir / 'config.json'
    if cfg_json.exists():
        with open(cfg_json) as f:
            ctx = json.load(f).get('max_position_embeddings')
    ctx_note = f', ctx {ctx}' if ctx else ''

    ft = ckpt_dir / 'fine_tune_config.toml'
    if ft.exists():
        with open(ft, 'rb') as f:
            flat = flatten(tomllib.load(f), {})
        base = Path(str(flat.get('base_model', '?'))).name
        lr = flat.get('learning_rate', '?')
        return f'SFT from {base} @ lr={lr}{ctx_note}'
    if any((ckpt_dir / n).exists() for n in ('training_config.toml', 'trainer_state.json', 'trainer_state2.json')):
        return f'base pretraining{ctx_note}'
    return f'(unknown recipe{ctx_note})'


def summarize_row(results: dict) -> dict:
    """Flatten one checkpoint's results into the key comparison metrics."""
    s = results['suites']
    ppl = s.get('perplexity', {})
    emb = s.get('embeddings', {})
    gen = s.get('generation', {})
    info = s.get('info', {})
    name = Path(results['checkpoint']).name
    try:
        step = int(name.rsplit('-', 1)[1])
    except (IndexError, ValueError):
        step = None
    return {
        'name': name,
        'step': step,
        'hist_ppl': ppl.get('historical', {}).get('perplexity'),
        'mod_ppl': ppl.get('modern', {}).get('perplexity'),
        'ratio': ppl.get('modern_over_historical_ratio'),
        'shift': emb.get('mean_shift_similarity'),
        'aniso': info.get('embedding_anisotropy'),
        'gen_prob': gen.get('mean_token_prob'),
        'dist2': gen.get('mean_distinct_2'),
        'echo': gen.get('mean_echo_rate'),
    }


def _cell(v, nd=1, pct=False):
    if v is None:
        return '-'.rjust(9)
    return f'{v * 100:8.1f}%' if pct else f'{v:9.{nd}f}'


def print_compare_report(rows: list[dict]):
    """The table + the 'so what?' - written for technical non-data-scientists."""
    banner('CHECKPOINT COMPARISON - all numbers computed on the SAME fixed probes')
    print("  Lower step = less training. Read each row as 'how vintage-fluent is the model at this point'.")
    print()
    print(
        f'  {"checkpoint":<16}{"HIST ppl":>9}{"MODERN ppl":>11}{"ratio":>7}{"shift":>7}{"gen prob":>9}{"dist-2":>8}{"echo":>7}{"aniso":>7}'
    )
    for r in rows:
        print(
            f'  {r["name"]:<16}{_cell(r["hist_ppl"])}{r["mod_ppl"]:11.1f}'
            f'{r["ratio"]:7.2f}{r["shift"]:7.3f}{r["gen_prob"]:9.3f}'
            f'{r["dist2"]:8.3f}{r["echo"]:7.3f}{r["aniso"]:7.3f}'
        )
    print('\n  Training lineage (sniffed from files inside each checkpoint):')
    for r in rows:
        if r.get('provenance'):
            print(f'    {r["name"]:<16}: {r["provenance"]}')
    print("""
  Column cheat-sheet:
    HIST ppl   perplexity on 1800s-style sentences. Lower = more fluent in period English.
    MODERN ppl perplexity on present-day sentences. For a vintage model HIGH is fine/desired.
    ratio      MODERN/HIST. >1 = the model prefers period text (that's the whole point).
    shift      how differently the same word (gay, awful, python...) is represented in
               historical vs modern sentences. Lower = stronger sense separation.
    gen prob   avg confidence in the words it wrote. Trending up = more sure-footed.
    dist-2     phrase variety of generated text. Near 1.0 = no looping. Below 0.5 = broken.
    echo       stutter rate (word repeated within 4 words). Near 0 = good. Above 0.4 = looping.
    aniso      embedding collapse detector. Near 0 = healthy. Above 0.7 = degenerate.
""")

    rows = [r for r in rows if r['hist_ppl'] is not None]
    if len(rows) < 2:
        return

    def healthy(r):
        return (
            (r['echo'] is None or r['echo'] <= 0.4)
            and (r['dist2'] is None or r['dist2'] >= 0.5)
            and (r['aniso'] is None or r['aniso'] <= 0.7)
        )

    usable = [r for r in rows if healthy(r)]
    best = min(usable, key=lambda r: r['hist_ppl']) if usable else None
    first, last = rows[0], rows[-1]

    # Per-step relative improvements of historical perplexity.
    rel_changes = [(a['hist_ppl'] - b['hist_ppl']) / a['hist_ppl'] for a, b in zip(rows, rows[1:])]
    regressions = [(rows[i]['name'], rows[i + 1]['name'], c) for i, c in enumerate(rel_changes) if c < -0.05]

    # Anomaly: checkpoints AFTER the best one got >10% worse. If so, judge the
    # 'baked?' question from the healthy trend leading UP TO the best one,
    # not from the anomalous tail.
    best_idx = rows.index(best) if best else len(rows) - 1
    anomaly = best is not None and last['hist_ppl'] > best['hist_ppl'] * 1.10
    trend_rows = rows[: best_idx + 1] if anomaly else rows
    trend_changes = [(a['hist_ppl'] - b['hist_ppl']) / a['hist_ppl'] for a, b in zip(trend_rows, trend_rows[1:])]
    recent = trend_changes[-2:] if len(trend_changes) >= 2 else trend_changes
    still_improving = (sum(recent) / max(1, len(recent))) > 0.02  # avg >2% gain lately

    banner('VERDICT 1 - IS THE MODEL BAKED ENOUGH?')
    anchor = trend_rows[-1]
    total_gain = (first['hist_ppl'] - anchor['hist_ppl']) / first['hist_ppl']
    print(
        f'  Historical perplexity went from {first["hist_ppl"]:.0f} (step {first["step"]}) '
        f'to {anchor["hist_ppl"]:.0f} (step {anchor["step"]}): {total_gain:.0%} improvement.'
    )
    if anomaly:
        print(f'  NOTE: checkpoints after {anchor["name"]} got WORSE on BOTH text types')
        print(f'  (latest {last["name"]}: {last["hist_ppl"]:.0f} vs best {anchor["hist_ppl"]:.0f}).')
        print('  That is not normal training drift - the lineage above shows those were')
        print('  aggressive fine-tunes / context surgery, so they are excluded from the')
        print('  baked-ness judgement below.')
    if still_improving:
        print('  -> The curve was STILL improving (>2% gains) when it reached the best checkpoint.')
        print('     The base model is NOT fully baked: more pretraining steps should still help.')
    else:
        print('  -> The curve had FLATTENED (<2% gains) by the time it reached the best checkpoint.')
        print('     The model is about as baked as this data/recipe will make it. More of the')
        print('     same training would give diminishing returns - to go further, add data')
        print('     or change the recipe, not just the step count.')
    if regressions:
        print('  WARNING - perplexity went UP (got worse) at these jumps:')
        for a, b, c in regressions:
            print(f'     {a} -> {b}: {c:+.0%}. Possible LR instability or a noisy save point.')

    banner('VERDICT 2 - ARE THESE NUMBERS GOOD OR BAD?')
    print("  Don't compare the raw perplexities to big-model benchmarks: they are measured on")
    print('  20 deliberately tricky probe sentences, so only the TREND and the RATIO matter.')
    print(f'  - ratio {last["ratio"]:.2f} at the latest checkpoint: the model finds period text')
    print(f'    {last["ratio"]:.1f}x easier than modern text. For a ~1900 knowledge-cutoff model,')
    print('    >1 is the goal; bigger = more thoroughly vintage.')
    if last['dist2'] is not None and last['echo'] is not None:
        if last['dist2'] >= 0.9 and last['echo'] <= 0.1:
            print(f'  - dist-2 {last["dist2"]:.2f} / echo {last["echo"]:.2f}: generation is clean,')
            print('    no looping or stuttering.')
        else:
            print(f'  - dist-2 {last["dist2"]:.2f} / echo {last["echo"]:.2f}: generation quality is')
            print('    questionable - inspect the continuations before trusting this checkpoint.')
    if last['aniso'] is not None:
        if last['aniso'] <= 0.3:
            print(f'  - aniso {last["aniso"]:.3f}: embeddings are well spread out (healthy).')
        else:
            print(f'  - aniso {last["aniso"]:.3f}: embeddings look collapsed - a sign of')
            print('    undertraining. This checkpoint needs more steps, not fewer.')

    banner('VERDICT 3 - WHICH CHECKPOINT SHOULD I USE?')
    if best is None:
        print('  No checkpoint passed the basic health checks - inspect them individually.')
    elif best['name'] == last['name']:
        print(f'  Use the LATEST checkpoint ({best["name"]}): lowest historical perplexity')
        print(f'  ({best["hist_ppl"]:.0f}) among healthy checkpoints, and it is the most trained.')
    else:
        print(f'  Best historical perplexity: {best["name"]} ({best["hist_ppl"]:.0f}),')
        print(f'  while the latest ({last["name"]}) sits at {last["hist_ppl"]:.0f}.')
        print('  Later checkpoints did NOT improve on period text - consider the earlier one,')
        print('  or check the regression warnings above.')


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    args = parse_args()

    if args.list_suites:
        print('Available suites:')
        for name, fn in SUITES.items():
            print(f'  {name:<12} {fn.__doc__.strip().splitlines()[0]}')
        return

    selected = args.suites or list(SUITES)
    unknown = [s for s in selected if s not in SUITES]
    if unknown:
        print(f'Unknown suite(s): {unknown}. Available: {list(SUITES)}', file=sys.stderr)
        sys.exit(2)

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    torch.manual_seed(args.seed)

    if args.compare:
        outdir = args.output or (PROJECT_DIR / 'eval_results' / 'compare')
        outdir.mkdir(parents=True, exist_ok=True)
        ckpts = list_checkpoints(args.checkpoints_dir)
        if not ckpts:
            print(f'No checkpoints found in {args.checkpoints_dir}', file=sys.stderr)
            sys.exit(1)
        print(f'Comparing {len(ckpts)} checkpoints from {args.checkpoints_dir} (cache: {outdir}, use --force to recompute)')
        rows = []
        for ck in ckpts:
            cache = outdir / f'{ck.name}.json'
            if cache.exists() and not args.force:
                print(f'  [cached] {ck.name}')
                with open(cache) as f:
                    results = json.load(f)
            else:
                results = evaluate_checkpoint(ck, device, dtype, args, selected)
                with open(cache, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f'  [saved] {cache}')
            row = summarize_row(results)
            row['provenance'] = checkpoint_provenance(ck)
            rows.append(row)
        print_compare_report(rows)
        if args.explain:
            print(GLOSSARY)
        return

    checkpoint_dir = resolve_checkpoint(args.checkpoint, args.checkpoints_dir)
    results = evaluate_checkpoint(checkpoint_dir, device, dtype, args, selected)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nFull numeric results written to {args.output} (keep these JSONs to compare checkpoints side by side).')

    if args.explain:
        print(GLOSSARY)

    banner('EVALUATION COMPLETE')


if __name__ == '__main__':
    main()
