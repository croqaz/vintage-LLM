#!/usr/bin/env python3
"""
evaluate.py - Checkpoint inspection harness: architecture, lineage, period fidelity.

SCOPE (read this first)
----------------------
This script answers "WHAT is this checkpoint, and is it still VINTAGE?".
It does NOT answer "is the model baked enough / which checkpoint is best".

Those questions need real held-out text; they are answered by evaluate2.py, which
scores hundreds of unseen period documents in bits-per-byte.
What this script is good for:

  * INFO       architecture, parameter count, disk size, tokenizer/config sanity.
  * LINEAGE    training recipe sniffed from files inside each checkpoint - base
               pretraining vs SFT, the base it was fine-tuned from, its LR, and
               context-length surgery. This is the one thing evaluate2.py cannot
               see, and it explains most "why did this checkpoint change" puzzles.
  * PERIOD     is period text easier for the model than modern text, and does it
               represent shifted words (gay, awful, nice...) differently in a
               period vs a modern sentence. Orthogonal to training progress
               (measured correlation with training step: ~+0.2), so read it as a
               style/era gauge, never as a quality gauge.
  * HYGIENE    looping/stuttering of sampled text with NO repetition penalty, so
               the degeneracy detectors can actually see degeneracy.

Checkpoint resolution matches generate.py / vibe_check.py / fine_tune.py:

  * no flags            -> the LATEST checkpoint in ./checkpoints
  * --checkpoints-dir D -> the latest checkpoint found in D
  * --checkpoint PATH   -> exactly this checkpoint directory
  * --compare           -> every checkpoint in --checkpoints-dir, as a table

Examples:
  python evaluate.py
  python evaluate.py --checkpoint checkpoints/checkpoint-xx
  python evaluate.py --compare --checkpoints-dir Vintage1
  python evaluate.py --suites info perplexity --output results.json
  python evaluate.py --list-suites
"""

import argparse
import contextlib
import io
import json
import math
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

PROJECT_DIR = Path(__file__).parent if Path(__file__).parent != Path('.') else Path.cwd()

# ============================================================================
# CONSTANTS - tweak these
# ============================================================================

# --- Checkpoint resolution ---------------------------------------------------
DEFAULT_CHECKPOINTS_DIR = PROJECT_DIR / 'checkpoints'
DEFAULT_RESULTS_DIR = PROJECT_DIR / 'eval_results'

# --- Historical probe words --------------------------------------------------
# Words with well-documented semantic shifts between ~1800-1900 and today.
# Each word is evaluated inside HISTORICAL_CONTEXTS[i] (period sense) and
# MODERN_CONTEXTS[i] (present-day sense).
#
# IMPORTANT CONSTRAINT on these sentences: this is a CAUSAL language model, so a
# token's representation contains only the text to its LEFT. Every probe word is
# therefore placed LAST, with all the disambiguating words before it - otherwise
# the "period sense" and "modern sense" representations are computed from an
# identical prefix and come out bit-identical (cosine exactly 1.0), which is what
# the earlier version of this file measured for several words.
HISTORICAL_WORDS = [
    'gay',  # originally "merry, carefree"
    'awful',  # originally "awe-inspiring"
    'nice',  # originally "over-fussy, hair-splitting"
    'meat',  # originally "food" in general
    'want',  # originally "lack, destitution"
    'python',  # originally only a snake
    'commerce',  # trade between nations
    'parliament',  # the British institution
    'science',  # systematic knowledge, esp. natural philosophy
    'manufacture',  # literally "making, by hand"
]

HISTORICAL_CONTEXTS = [
    'The ballroom was filled with dancing and laughter, and every heart was gay',
    'The mountain rose above the valley in a silence solemn and awful',
    'He drew a distinction so fine and over-scrupulous that his critics called it nice',
    'The Lord provideth for all his creatures, giving them drink and meat',
    'The labouring poor of this parish are reduced to great want',
    'The keeper fed the great serpent which the naturalists call a python',
    'The merchants of the port have grown rich upon their foreign commerce',
    'Her Majesty was pleased to summon the Lords and Commons to Parliament',
    'He gave up his fortune to the patient study of natural science',
    'The weavers at their looms are employed in the woollen manufacture',
]

MODERN_CONTEXTS = [
    'After years of hiding it from everyone at work, he told his parents he is gay',
    'Three hours stuck in traffic in the pouring rain made the commute awful',
    'She gave up her whole weekend to help me move apartments, which was really nice',
    'I switched to a plant-based diet last year and I no longer eat meat',
    'Millions of shoppers queue outside the store because they want the new smartphone',
    'The engineering team rewrote the whole backend microservice in Python',
    'Small retail startups now run almost all of their business through online commerce',
    'Members of the European Union parliament voted on the digital privacy bill',
    'She is finishing a graduate degree in machine learning and computer science',
    'Robots on the automated assembly line handle the entire car manufacture',
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
# The perplexity suite scores these sentences (historical + modern contexts).
# The MODERN/HISTORICAL ratio is the meaningful output; the absolute numbers are
# not comparable to anything outside this file.
PERPLEXITY_SENTENCES = HISTORICAL_CONTEXTS + MODERN_CONTEXTS
PERPLEXITY_LABELS = ['historical'] * len(HISTORICAL_CONTEXTS) + ['modern'] * len(MODERN_CONTEXTS)

# --- Suite behaviour knobs ---------------------------------------------------
TOP_K_NEIGHBORS = 5  # neighbours shown per word in the embedding suite
SIMILARITY_MATRIX_WORDS = 5  # how many words to show in the similarity matrix
GEN_MAX_NEW_TOKENS = 60
GEN_TEMPERATURE = 0.8
GEN_TOP_P = 0.9
GEN_TOP_K = 25
# Default 1.0 = OFF, deliberately. A repetition penalty suppresses exactly the
# looping that the echo / distinct-n detectors below exist to find; with the old
# default of 1.1 a checkpoint that loops for 57 straight words under greedy
# decoding still scored "generation is clean".
GEN_REPETITION_PENALTY = 1.0
DEFAULT_SEED = 1337

# Degeneracy thresholds for sampled text at repetition_penalty = 1.0.
ECHO_BAD = 0.30  # fraction of words repeated within the previous 4
DISTINCT2_BAD = 0.60  # fraction of unique word-pairs
# Stricter thresholds for judging a SINGLE probe. The worst of 10 short samples is
# always somewhat repetitive, so reusing the averaged thresholds above flags every
# checkpoint ever produced and carries no information.
PROBE_ECHO_BAD = 0.45
PROBE_DISTINCT2_BAD = 0.45

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


def resolve_tokenizer(checkpoint_dir: Path, checkpoints_dir: Path, cli_tokenizer: Path | None) -> Path:
    """Find a tokenizer: --tokenizer flag, checkpoint dir, a tokenizers/ subdir, parent."""
    if cli_tokenizer is not None:
        return cli_tokenizer
    candidates = (
        checkpoint_dir,
        checkpoints_dir / 'tokenizers',
        checkpoints_dir,
        checkpoints_dir.parent / 'tokenizers',
        checkpoints_dir.parent,
    )
    for candidate in candidates:
        if (candidate / 'tokenizer.json').exists():
            return candidate
    raise FileNotFoundError(
        f'No tokenizer.json found in the checkpoint, {checkpoints_dir}, a tokenizers/ subdirectory, or the parent.\n'
        'Pass --tokenizer PATH to point to a HuggingFace tokenizer directory.'
    )


def load_model_and_tokenizer(checkpoint_dir: Path, tokenizer_dir: Path, device: torch.device, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, dtype=dtype)
    model.config.use_cache = True
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # enables correct batched generation
    return model, tokenizer


def free_model(model, device: torch.device) -> None:
    """Actually release the weights.

    `del model` only drops the local name - the caller and any suite closure may
    still hold references, leaving several GB resident and OOM-ing the next load.
    Moving the parameters to the 'meta' device frees the real storage regardless.
    """
    with contextlib.suppress(Exception):  # best effort; freeing must never break a run
        model.to('meta')
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# ============================================================================
# SHARED STATISTICS HELPERS
# ============================================================================


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


def scored_span_bytes(tokenizer, text: str) -> int:
    """
    UTF-8 bytes of the part of `text` that per_token_loss actually scores.

    The first token is never predicted, so it must be excluded, otherwise
    bits-per-byte is silently optimistic on short strings.
    """
    offsets = tokenizer(text, return_offsets_mapping=True)['offset_mapping']
    real = [(s, e) for s, e in offsets if e > s]
    if len(real) < 2:
        return 0
    return len(text[real[1][0] :].encode('utf-8'))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ngram_stats(text: str) -> dict:
    """Distinct-n and repetition metrics over whitespace tokens of ONE text."""
    words = text.split()
    out = {'num_words': len(words)}
    for n in (1, 2):
        grams = list(zip(*(words[i:] for i in range(n)), strict=False)) if len(words) >= n else []
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
# TRAINING LINEAGE
# The most useful thing in this file: what recipe produced this checkpoint.
# Sniffed from files the trainer leaves inside the checkpoint directory.
# ============================================================================


def checkpoint_provenance(ckpt_dir: Path) -> dict:
    """Structured training lineage for one checkpoint."""

    def flatten(d, out):
        for k, v in d.items():
            if isinstance(v, dict):
                flatten(v, out)
            else:
                out[k] = v
        return out

    info = {
        'kind': 'unknown',
        'base_model': None,
        'learning_rate': None,
        'context': None,
        'global_step': None,
        'epoch': None,
    }

    cfg_json = ckpt_dir / 'config.json'
    if cfg_json.exists():
        with open(cfg_json) as f:
            info['context'] = json.load(f).get('max_position_embeddings')

    state = ckpt_dir / 'trainer_state.json'
    if state.exists():
        try:
            with open(state) as f:
                st = json.load(f)
            info['global_step'] = st.get('global_step')
            info['epoch'] = st.get('epoch')
        except (json.JSONDecodeError, OSError):
            pass

    ft = ckpt_dir / 'fine_tune_config.toml'
    if ft.exists():
        try:
            with open(ft, 'rb') as f:
                flat = flatten(tomllib.load(f), {})
            info['kind'] = 'sft'
            info['base_model'] = Path(str(flat.get('base_model', '?'))).name
            info['learning_rate'] = flat.get('learning_rate')
        except (tomllib.TOMLDecodeError, OSError):
            info['kind'] = 'sft'
    elif any((ckpt_dir / n).exists() for n in ('training_config.toml', 'trainer_state.json', 'trainer_state2.json')):
        info['kind'] = 'base'

    return info


def provenance_line(info: dict) -> str:
    """One-line human-readable form of checkpoint_provenance()."""
    ctx = f', ctx {info["context"]}' if info.get('context') else ''
    step = ''
    if info.get('global_step') is not None:
        step = f', trainer step {info["global_step"]}'
        if info.get('epoch') is not None:
            step += f' (epoch {info["epoch"]:.2f})'
    if info['kind'] == 'sft':
        lr = info.get('learning_rate')
        lr_s = f' @ lr={lr}' if lr is not None else ''
        return f'SFT from {info.get("base_model") or "?"}{lr_s}{ctx}{step}'
    if info['kind'] == 'base':
        return f'base pretraining{ctx}{step}'
    return f'(unknown recipe{ctx}{step})'


# ============================================================================
# SUITES
# Each suite: fn(model, tokenizer, device, args) -> dict of JSON-safe results.
# To add a new evaluation: write a suite_* function, register it in SUITES.
# ============================================================================


def suite_info(model, tokenizer, device, args) -> dict:
    """Static facts about the checkpoint: architecture, size, tokenizer, lineage."""
    banner('SUITE: MODEL INFO AND TRAINING LINEAGE')
    cfg = model.config
    params = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight
    ckpt_dir = Path(args._checkpoint_dir)
    weight_files = list(ckpt_dir.glob('*.safetensors'))
    disk_mb = sum(f.stat().st_size for f in weight_files) / 1e6
    prov = checkpoint_provenance(ckpt_dir)

    results = {
        'checkpoint': str(ckpt_dir),
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
        'provenance': prov,
        'provenance_line': provenance_line(prov),
        # Mean embedding row norm, and mean pairwise cosine of a fixed random
        # sample of rows. See the printout below for what these do and do NOT mean.
        'embedding_mean_norm': None,
        'embedding_mean_cosine': None,
    }

    with torch.no_grad():
        results['embedding_mean_norm'] = float(emb.float().norm(dim=-1).mean())
        gen = torch.Generator(device='cpu').manual_seed(args.seed)  # same rows every run
        pick = torch.randperm(emb.shape[0], generator=gen)[:512].to(emb.device)
        sample = emb[pick].float()
        sample = sample / sample.norm(dim=-1, keepdim=True)
        sims = sample @ sample.T
        off_diag = sims[~torch.eye(len(sample), dtype=bool, device=sample.device)]
        results['embedding_mean_cosine'] = float(off_diag.mean())

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
    print(f'  {"training lineage":24}: {results["provenance_line"]}')
    print(
        f'  {"embedding mean norm":24}: {fmt(results["embedding_mean_norm"])}'
        '   <- typical row length of the embedding matrix; very small or huge values signal trouble'
    )
    print(
        f'  {"embedding mean cosine":24}: {fmt(results["embedding_mean_cosine"])}'
        '   <- avg cosine between random embedding rows. Descriptive only: it starts near 0.0'
    )
    print(f'  {"":24}  because RANDOM init is near-orthogonal, and it GROWS as the model trains.')
    print(f'  {"":24}  Near 0 therefore does NOT mean "healthy" - it can mean "barely trained".')
    if results['vocab_size_tokenizer'] > results['vocab_size_config']:
        print('  WARNING: tokenizer vocab is LARGER than config vocab_size - checkpoint/tokenizer mismatch?')
    elif results['vocab_size_config'] != results['vocab_size_tokenizer']:
        print(f'  (note: config vocab padded by {results["vocab_size_config"] - results["vocab_size_tokenizer"]} unused rows - harmless)')
    return results


def suite_perplexity(model, tokenizer, device, args) -> dict:
    """Period-vs-modern fidelity: perplexity and bits-per-byte on fixed probe sentences."""
    banner('SUITE: PERIOD FIDELITY ON FIXED PROBE SENTENCES')
    print(
        f'  Scoring {len(PERPLEXITY_SENTENCES)} sentences '
        f'({PERPLEXITY_LABELS.count("historical")} historical, {PERPLEXITY_LABELS.count("modern")} modern)'
    )
    print('  The MODERN/HISTORICAL ratio is the output that means something. The absolute')
    print('  numbers come from ~20 sentences: too noisy to rank checkpoints (use evaluate2.py).')

    per_sentence = []
    all_logp, all_ent = [], []
    for sent, label in zip(PERPLEXITY_SENTENCES, PERPLEXITY_LABELS, strict=True):
        enc = tokenizer(sent, return_tensors='pt').to(device)
        logp, ent = per_token_loss(model, enc['input_ids'], enc['attention_mask'])
        st = token_stats_from_logprobs(logp, ent)
        nbytes = scored_span_bytes(tokenizer, sent)
        # bits-per-byte: total surprise in bits divided by the UTF-8 bytes it covers.
        # Unlike perplexity this is comparable across DIFFERENT tokenizers.
        st['bits_per_byte'] = float(-logp.sum().item() / math.log(2) / nbytes) if nbytes else None
        st['scored_bytes'] = nbytes
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
        nbytes = sum(r['scored_bytes'] for r in rows)
        bits = sum(math.log(r['perplexity']) * r['num_tokens'] / math.log(2) for r in rows)
        results[name] = {
            'num_sentences': len(rows),
            'perplexity': math.exp(mean_log_loss),
            'bits_per_byte': bits / nbytes if nbytes else None,
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
    print(f'    {"group":<12}{"sentences":>10}{"perplexity":>14}{"bits/byte":>12}')
    for name in ('historical', 'modern'):
        g = results[name]
        bpb = f'{g["bits_per_byte"]:>12.3f}' if g['bits_per_byte'] is not None else f'{"-":>12}'
        print(f'    {name:<12}{g["num_sentences"]:>10}{g["perplexity"]:>14.2f}{bpb}')
    ratio = results['modern']['perplexity'] / max(results['historical']['perplexity'], 1e-9)
    results['modern_over_historical_ratio'] = ratio
    print(
        f'    modern/historical ratio: {ratio:.2f}'
        '   <- >1 means period text is easier for the model than modern text'
        ' (desired for a vintage model with a ~1900 knowledge cutoff)'
    )
    print('    (bits/byte is the tokenizer-independent version - the only column here you may')
    print('     compare between models that use DIFFERENT tokenizers.)')

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
    char_start = context.lower().rfind(word.lower())
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
    for word, context in zip(words, contexts, strict=True):
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
    """Diachronic sense separation: same word in a period vs a modern sentence."""
    banner('SUITE: DIACHRONIC WORD-SENSE SEPARATION')
    print(f'  Extracting contextual embeddings for {len(HISTORICAL_WORDS)} shifted words in period AND modern sentences.')
    print('  Every probe word sits LAST in its sentence, so the causal model has actually read')
    print('  the disambiguating context before it represents the word.')

    hist_emb = extract_word_embeddings(model, tokenizer, device, HISTORICAL_WORDS, HISTORICAL_CONTEXTS)
    mod_emb = extract_word_embeddings(model, tokenizer, device, HISTORICAL_WORDS, MODERN_CONTEXTS)

    words = [w for w in HISTORICAL_WORDS if w in hist_emb]
    results = {
        'words': words,
        'pairwise_historical': {},
        'semantic_shift': {},
        'neighbors': {},
    }

    # --- 1. How much does each word's representation move between contexts? ---
    print('\n  SENSE SEPARATION (cosine similarity of the SAME word, period vs modern sentence)')
    print('    lower value = the model represents the two senses differently (good sign for a')
    print('    period model); ~1.0 = it treats them as the same word regardless of context')
    shifts = {}
    for w in words:
        if w in mod_emb:
            shifts[w] = cosine_similarity(hist_emb[w], mod_emb[w])
    for w, s in sorted(shifts.items(), key=lambda kv: kv[1]):
        flag = '   <- suspiciously identical: check the sentence prefixes differ' if s >= 0.999 else ''
        print(f'    {w:<14}: {s:+.3f}{flag}')
    results['semantic_shift'] = shifts
    if shifts:
        results['mean_shift_similarity'] = float(np.mean(list(shifts.values())))
        print(
            f'    {"MEAN":<14}: {results["mean_shift_similarity"]:+.3f}'
            '   <- near 1.0 = model ignores period vs modern usage; lower = senses differ'
        )

    # --- 2. Similarity matrix within the historical embeddings ---------------
    show = words[:SIMILARITY_MATRIX_WORDS]
    print(f'\n  PAIRWISE COSINE SIMILARITY (period sentences, first {len(show)} words)')
    print('    these are CONTEXTUAL vectors, so they reflect the sentences as much as the words;')
    print('    read the spread, not individual pairs. High mean = representations collapsed.')
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
        print(f'    all {len(all_pairs)} pairs: mean={arr.mean():+.3f} std={arr.std():.3f}   <- healthy spread is good')

    # --- 3. Nearest neighbours -----------------------------------------------
    print('\n  NEAREST NEIGHBOURS inside the probe-word set (period sentences)')
    for w in show:
        sims = sorted(
            ((o, cosine_similarity(hist_emb[w], hist_emb[o])) for o in words if o != w),
            key=lambda kv: -kv[1],
        )[:TOP_K_NEIGHBORS]
        results['neighbors'][w] = [(o, round(s, 4)) for o, s in sims]
        joined = ', '.join(f'{o} ({s:+.2f})' for o, s in sims)
        print(f'    {w:<14}: {joined}')
    return results


def suite_generation(model, tokenizer, device, args) -> dict:
    """Continue fixed prompts and measure looping/stuttering (no repetition penalty)."""
    banner('SUITE: GENERATION PROBES')
    if args.repetition_penalty > 1.0:
        print(f'  WARNING: --repetition-penalty {args.repetition_penalty} is ON. It masks the looping')
        print('  that distinct-n / echo below are meant to detect. Use 1.0 for an honest reading.')
    torch.manual_seed(args.seed)
    prompts = []
    for p in GENERATION_PROMPTS:
        if args.chat:
            if tokenizer.chat_template is None:
                raise SystemExit('--chat was passed but this tokenizer has no chat_template. Drop --chat for a base model.')
            prompts.append(
                tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
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
        seq = out[i]
        gen_ids = list(seq[input_width:])
        # Rows that finished early (EOS) are padded with pad tokens up to the
        # batch's max length; exclude them from text and stats.
        gen_ids = [t for t in gen_ids if t.item() not in stop_ids]
        total_new += len(gen_ids)
        text = tokenizer.decode(
            gen_ids,
            skip_special_tokens=not args.show_special_tokens,
            clean_up_tokenization_spaces=False,
        )
        # NOTE: no "confidence in the sampled tokens" metric here on purpose.
        # generate(output_scores=True) returns logits AFTER temperature, top-k and
        # repetition-penalty processing, so it is not the model's belief; and
        # scoring a model on text it chose itself rewards degenerate loops, which
        # are trivially predictable. Fluency is judged by repetition below and by
        # held-out loss in evaluate2.py.
        rec = {
            'prompt': prompt,
            'continuation': text.strip(),
            'new_tokens': len(gen_ids),
            **ngram_stats(text),
        }
        generations.append(rec)
        print(f'\n  --- Probe {i + 1} ---')
        print(f'  prompt      : {prompt}')
        print(f'  continuation: {rec["continuation"][:400]}')
        print(
            f'  stats: {rec["new_tokens"]} tokens'
            f' | distinct-1 {fmt(rec["distinct_1"], 2)} | distinct-2 {fmt(rec["distinct_2"], 2)}'
            f' | echo {fmt(rec["echo_rate"], 2)}'
        )
        print('         (distinct-n = vocabulary/phrase variety, near 0 = looping; echo = fraction of words repeated from the previous 4)')

    results = {
        'generations': generations,
        'total_new_tokens': total_new,
        'elapsed_seconds': round(elapsed, 2),
        'tokens_per_second': round(total_new / max(elapsed, 1e-9), 1),
        'repetition_penalty': args.repetition_penalty,
        'temperature': args.temperature,
        'mean_distinct_1': float(np.mean([g['distinct_1'] for g in generations])),
        'mean_distinct_2': float(np.mean([g['distinct_2'] for g in generations])),
        'mean_echo_rate': float(np.mean([g['echo_rate'] for g in generations])),
        'worst_distinct_2': float(np.min([g['distinct_2'] for g in generations])),
        'worst_echo_rate': float(np.max([g['echo_rate'] for g in generations])),
    }
    print(f'\n  AGGREGATE over {len(generations)} probes: {total_new} tokens in {elapsed:.1f}s ({results["tokens_per_second"]} tok/s)')
    print(
        f'    mean distinct-1 {fmt(results["mean_distinct_1"], 3)}'
        f' | mean distinct-2 {fmt(results["mean_distinct_2"], 3)}'
        f' | mean echo {fmt(results["mean_echo_rate"], 3)}'
        f' | worst probe: distinct-2 {fmt(results["worst_distinct_2"], 3)} / echo {fmt(results["worst_echo_rate"], 3)}'
    )
    if results['mean_echo_rate'] > ECHO_BAD or results['mean_distinct_2'] < DISTINCT2_BAD:
        print('    WARNING: high repetition across probes - typical of an undertrained checkpoint.')
    elif results['worst_echo_rate'] > ECHO_BAD or results['worst_distinct_2'] < DISTINCT2_BAD:
        print('    NOTE: at least one probe looped badly even though the average looks fine - read the probes above.')
    print('    (sampling hides loops that greedy decoding exposes; evaluate2.py measures the greedy loop length.)')
    return results


# ============================================================================
# SUITE REGISTRY - add future evaluations here
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
                      checkpoints ONLY on the same fixed sentences (that's why they're
                      constants) and ONLY with the same tokenizer.
bits per byte         Same surprise, divided by UTF-8 bytes instead of tokens. The only
                      loss number here that is valid across DIFFERENT tokenizers.
                      evaluate2.py uses this on real held-out documents.
token probability     Softmax probability the model assigned to the word that actually came
                      next. Mean = average confidence, median = typical confidence,
                      min/10th-percentile = how bad the worst guesses get.
low-confidence share  Fraction of tokens predicted with <1% probability. Spikes here
                      usually mean OOV words, noise, or genuine surprise.
entropy (nats)        Spread of the next-word distribution. ~0 = dead certain; high =
                      many plausible continuations. Not good or bad by itself - read with
                      token probability.
modern/historical     Ratio of the two group perplexities. >1 = period text is easier for
                      the model than modern text, which is the point of a vintage model.
                      It does NOT track training progress - a barely-trained checkpoint
                      can already score >1.
embedding mean cosine Average cosine between random embedding rows. DESCRIPTIVE ONLY: it
                      starts near 0 at random init and GROWS with training, so a low
                      value is not a clean bill of health.
sense separation      Cosine similarity of the same word in a period vs a modern sentence.
                      Lower = the model represents the two senses differently.
distinct-1 / -2       Fraction of unique words / word-pairs in generated text.
                      Low values = the model is looping. Only meaningful with
                      repetition_penalty = 1.0.
echo rate             Fraction of generated words that already appeared within the
                      previous 4 words. High values = stuttering/repetition.
training lineage      Recipe sniffed from files inside the checkpoint: base pretraining vs
                      SFT, which model it was fine-tuned from, its LR, context length,
                      and the trainer's own global_step/epoch.
"""

# ============================================================================
# COMPARE MODE - inspect every checkpoint, then explain what it all means
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


def gen_params_of(args) -> dict:
    """Generation settings that affect cached results (used to invalidate the cache)."""
    return {
        'tokens': args.tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'repetition_penalty': args.repetition_penalty,
        'seed': args.seed,
        'chat': args.chat,
    }


def evaluate_checkpoint(checkpoint_dir: Path, tokenizer_dir: Path, device, dtype, args, selected) -> dict:
    """Run the selected suites on one checkpoint and free the model afterwards."""
    args._checkpoint_dir = checkpoint_dir
    banner('CHECKPOINT INSPECTION')
    print(f'  checkpoint : {checkpoint_dir}')
    print(f'  tokenizer  : {tokenizer_dir}')
    print(f'  device     : {device}   dtype: {dtype}   seed: {args.seed}')
    model, tokenizer = load_model_and_tokenizer(checkpoint_dir, tokenizer_dir, device, dtype)
    results = {
        'checkpoint': str(checkpoint_dir),
        'suites': {},
        'suites_run': list(selected),
        'gen_params': gen_params_of(args),
        'provenance': checkpoint_provenance(checkpoint_dir),
    }
    for name in selected:
        results['suites'][name] = SUITES[name](model, tokenizer, device, args)
    free_model(model, device)
    return results


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
    prov = results.get('provenance') or info.get('provenance') or {}
    return {
        'name': name,
        'step': step,
        'hist_ppl': ppl.get('historical', {}).get('perplexity'),
        'mod_ppl': ppl.get('modern', {}).get('perplexity'),
        'hist_bpb': ppl.get('historical', {}).get('bits_per_byte'),
        'ratio': ppl.get('modern_over_historical_ratio'),
        'shift': emb.get('mean_shift_similarity'),
        'emb_cos': info.get('embedding_mean_cosine'),
        'dist2': gen.get('mean_distinct_2'),
        'echo': gen.get('mean_echo_rate'),
        'worst_dist2': gen.get('worst_distinct_2'),
        'worst_echo': gen.get('worst_echo_rate'),
        'provenance': prov,
        'provenance_line': provenance_line(prov) if prov else None,
    }


def _cell(v, width=9, nd=1):
    """Fixed-width numeric cell that tolerates missing values (partial --suites runs)."""
    if v is None:
        return '-'.rjust(width)
    return f'{v:>{width}.{nd}f}'


def print_compare_report(rows: list[dict]):
    """The table + the 'so what?' - written for technical non-data-scientists."""
    banner('CHECKPOINT COMPARISON - all numbers computed on the SAME fixed probes')
    print('  Read this as "what is each checkpoint, and is it still vintage".')
    print('  It deliberately does NOT rank the checkpoints by quality: ~20 probe sentences')
    print('  cannot separate adjacent checkpoints. Run evaluate2.py for that verdict.')
    print()
    print(f'  {"checkpoint":<16}{"HIST ppl":>9}{"MODERN ppl":>11}{"ratio":>7}{"sense":>7}{"dist-2":>8}{"echo":>7}{"emb cos":>9}  recipe')
    for r in rows:
        print(
            f'  {r["name"]:<16}{_cell(r["hist_ppl"])}{_cell(r["mod_ppl"], 11)}'
            f'{_cell(r["ratio"], 7, 2)}{_cell(r["shift"], 7, 3)}{_cell(r["dist2"], 8, 3)}'
            f'{_cell(r["echo"], 7, 3)}{_cell(r["emb_cos"], 9, 3)}  {r.get("provenance_line") or ""}'
        )
    print("""
  Column cheat-sheet:
    HIST ppl   perplexity on 1800s-style sentences. Noisy - do NOT rank checkpoints with it.
    MODERN ppl perplexity on present-day sentences. For a vintage model HIGH is fine/desired.
    ratio      MODERN/HIST. >1 = the model prefers period text (that's the whole point).
    sense      how differently the same word (gay, awful, python...) is represented in a
               period vs a modern sentence. Lower = stronger sense separation.
    dist-2     phrase variety of sampled text. Near 1.0 = no looping. Below 0.6 = broken.
    echo       stutter rate (word repeated within 4 words). Near 0 = good. Above 0.3 = looping.
    emb cos    avg cosine between random embedding rows. Descriptive: grows with training.
    recipe     training lineage sniffed from files inside the checkpoint.
""")

    scored = [r for r in rows if r['ratio'] is not None]
    if not scored:
        print('  (no period-fidelity numbers in these results - run the perplexity suite for the verdicts)')
        return
    last = scored[-1]

    banner('VERDICT 1 - IS IT STILL VINTAGE?')
    print(f'  Latest checkpoint ({last["name"]}) finds period text {last["ratio"]:.1f}x easier than')
    print('  modern text (MODERN/HIST perplexity ratio).')
    if last['ratio'] < 1.0:
        print('  -> BELOW 1.0: modern text is EASIER for this model than period text. That points at')
        print('     modern data leaking into the training mix. Investigate the corpus.')
    elif last['ratio'] < 1.5:
        print('  -> Barely above 1.0: the period preference is weak. Either the corpus is mixed or')
        print('     the model is too early in training to have specialised.')
    else:
        print('  -> Comfortably period-biased, as intended for a ~1900 knowledge cutoff.')
    ratios = [r['ratio'] for r in scored]
    if len(ratios) >= 3:
        drift = ratios[-1] - ratios[0]
        direction = 'strengthened' if drift > 0.2 else ('weakened' if drift < -0.2 else 'held roughly steady')
        print(f'  Across the run the period preference {direction} ({ratios[0]:.1f}x -> {ratios[-1]:.1f}x).')
    shifts = [r['shift'] for r in scored if r['shift'] is not None]
    if shifts:
        print(f'  Word-sense separation at the latest checkpoint: {scored[-1]["shift"]:.3f} mean cosine')
        print('  between the period and modern reading of the same word (lower = more separated).')

    banner('VERDICT 2 - IS THE TEXT COMING OUT CLEAN?')
    clean = [r for r in scored if r['dist2'] is not None and r['echo'] is not None]
    if not clean:
        print('  (generation suite not run)')
    else:
        loopers = [r for r in clean if r['echo'] > ECHO_BAD or r['dist2'] < DISTINCT2_BAD]
        # Single-probe thresholds are deliberately much stricter than the averaged
        # ones: on 10 probes of 60 tokens the worst probe is always somewhat
        # repetitive, so a loose threshold here flags every checkpoint and says
        # nothing. Only genuine degeneracy should surface.
        per_probe = [
            r
            for r in clean
            if r not in loopers and ((r['worst_echo'] or 0) > PROBE_ECHO_BAD or (r['worst_dist2'] or 1) < PROBE_DISTINCT2_BAD)
        ]
        if loopers:
            print('  These checkpoints loop or stutter on average - do not ship them:')
            for r in loopers:
                print(f'    {r["name"]:<16} distinct-2 {r["dist2"]:.3f}  echo {r["echo"]:.3f}')
        else:
            print(f'  No checkpoint loops on average (all distinct-2 >= {DISTINCT2_BAD}, echo <= {ECHO_BAD}).')
        if per_probe:
            print(f'  {len(per_probe)} checkpoint(s) had one badly degenerate probe (distinct-2 < {PROBE_DISTINCT2_BAD}')
            print(f'  or echo > {PROBE_ECHO_BAD} on a single prompt) - read those continuations before trusting them:')
            for r in sorted(per_probe, key=lambda r: r['worst_dist2'] or 1)[:4]:
                print(f'    {r["name"]:<16} worst probe: distinct-2 {r["worst_dist2"]:.3f}  echo {r["worst_echo"]:.3f}')
        best_clean = max(clean, key=lambda r: r['dist2'])
        worst_clean = min(clean, key=lambda r: r['dist2'])
        if best_clean['name'] != worst_clean['name']:
            print(f'  Range across the run: distinct-2 {worst_clean["dist2"]:.3f} ({worst_clean["name"]})')
            print(f'  to {best_clean["dist2"]:.3f} ({best_clean["name"]}).')
        print('  Remember these are SAMPLED at temperature > 0, which hides loops. Greedy decoding is')
        print('  the harsher test and evaluate2.py reports the greedy loop length.')

    banner('VERDICT 3 - WHAT CHANGED BETWEEN THESE CHECKPOINTS?')
    print('  Only the points where the recipe CHANGED are listed (identical recipes are collapsed):')
    prev = None
    events = 0
    for r in rows:
        prov = r.get('provenance') or {}
        # Recipe identity deliberately excludes global_step/epoch: those differ on
        # every checkpoint, so including them would print one line per checkpoint
        # and hide the actual transitions.
        recipe = (prov.get('kind'), prov.get('base_model'), prov.get('learning_rate'), prov.get('context'))
        if prev is None or recipe != prev['recipe']:
            marker = ''
            if prev is not None and prov.get('context') and prev['prov'].get('context') and prov['context'] != prev['prov']['context']:
                marker = f'   <- context length changed {prev["prov"]["context"]} -> {prov["context"]}'
                events += 1
            if prov.get('kind') == 'sft' and (prev is None or prev['prov'].get('kind') != 'sft'):
                marker += '   <- switched from pretraining to fine-tuning'
                events += 1
            elif prov.get('kind') == 'base' and prev is not None and prev['prov'].get('kind') == 'sft':
                marker += '   <- back to base pretraining after a fine-tune'
                events += 1
            print(f'    {r["name"]:<16}: {r.get("provenance_line") or "(unknown)"}{marker}')
        prev = {'recipe': recipe, 'prov': prov}

    # A trainer step far below the directory number means training was RESTARTED:
    # the trainer_state only covers the latest segment, so any "tokens seen"
    # estimate derived from it (evaluate2.py does this) is a LOWER BOUND.
    mismatched = []
    for r in rows:
        prov = r.get('provenance') or {}
        gs, step = prov.get('global_step'), r.get('step')
        if gs is not None and step is not None and gs < step * 0.9:
            mismatched.append((r['name'], step, gs))
    if mismatched:
        print()
        print("  RESTART DETECTED: these directory numbers do not match the trainer's own step count,")
        print('  which means training resumed in a fresh run rather than continuing one counter:')
        for name, step, gs in mismatched[:6]:
            print(f'    {name:<16} directory says {step}, trainer_state says {gs}')
        print('  Consequence: any tokens-seen figure computed from trainer_state (evaluate2.py does')
        print('  this) only covers the LAST segment and is a lower bound on the real total.')
    sft_rows = [r for r in rows if (r.get('provenance') or {}).get('kind') == 'sft']
    if sft_rows:
        names = ', '.join(r['name'] for r in sft_rows)
        print()
        print(f'  {len(sft_rows)} checkpoint(s) are fine-tunes, not pretraining: {names}.')
        print('  Fine-tuned checkpoints usually score WORSE on general period text and BETTER on')
        print('  their target task. Do not compare them head-to-head with base checkpoints, and')
        print('  fine-tune your own downstream model from a BASE checkpoint unless you want this')
        print("  SFT's behaviour baked in.")
        low_lr = [
            r
            for r in sft_rows
            if isinstance((r.get('provenance') or {}).get('learning_rate'), (int, float)) and r['provenance']['learning_rate'] <= 1e-6
        ]
        if low_lr:
            names = ', '.join(r['name'] for r in low_lr)
            print(f'  NOTE: {names} ran SFT at lr <= 1e-6, which barely moves the weights - expect it to')
            print('  behave almost identically to the checkpoint it started from.')
    if not events and not sft_rows:
        print('  Uniform recipe across the whole run: no fine-tuning or context surgery detected.')

    banner('FOR "IS IT BAKED ENOUGH / WHICH CHECKPOINT IS BEST"')
    print('  Use evaluate2.py. It scores hundreds of real held-out period documents in')
    print('  bits-per-byte, checks tokens-seen against parameter count, and gives a')
    print('  calibrated verdict. Nothing in THIS file has the statistical power to')
    print('  rank two adjacent checkpoints, and it would be dishonest to pretend otherwise.')


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# ============================================================================
# MAIN
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, help='Specific checkpoint directory to evaluate.')
    p.add_argument(
        '--compare',
        action='store_true',
        help='Inspect EVERY checkpoint in --checkpoints-dir and print a comparison table plus the '
        'training lineage. Per-checkpoint JSONs are cached under --results-dir/compare-<folder>/ '
        'so interrupted runs resume where they left off.',
    )
    p.add_argument(
        '--force',
        action='store_true',
        help='With --compare: recompute even if a cached JSON exists.',
    )
    p.add_argument(
        '--checkpoints-dir',
        type=Path,
        default=DEFAULT_CHECKPOINTS_DIR,
        help='Directory containing checkpoints (latest is used when --checkpoint is absent).',
    )
    p.add_argument(
        '--suites',
        nargs='+',
        default=None,
        metavar='NAME',
        help='Suites to run (default: all). See --list-suites.',
    )
    p.add_argument('--list-suites', action='store_true', help='List available suites and exit.')
    p.add_argument(
        '--explain',
        action='store_true',
        help='Print a glossary of every metric at the end of the report.',
    )
    p.add_argument(
        '--results-dir',
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help='Where reports and cached per-checkpoint JSONs are written (default: eval_results/).',
    )
    p.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Write the full numeric results to this JSON file instead of the default name.',
    )
    p.add_argument(
        '--tokens',
        type=int,
        default=GEN_MAX_NEW_TOKENS,
        help='New tokens per generation probe.',
    )
    p.add_argument('--temperature', type=float, default=GEN_TEMPERATURE)
    p.add_argument('--top-p', type=float, default=GEN_TOP_P)
    p.add_argument('--top-k', type=int, default=GEN_TOP_K)
    p.add_argument(
        '--repetition-penalty',
        type=float,
        default=GEN_REPETITION_PENALTY,
        help='Leave at 1.0 (off). Anything higher hides the looping this script is trying to detect.',
    )
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda', 'mps'))
    p.add_argument('--dtype', default='auto', choices=('auto', 'float32', 'float16', 'bfloat16'))
    p.add_argument(
        '--tokenizer',
        type=Path,
        default=None,
        help='Tokenizer directory (when checkpoint dirs lack tokenizer files). Auto-detected from --checkpoints-dir/parent when omitted.',
    )
    p.add_argument(
        '--chat',
        action='store_true',
        help='Apply the chat template to generation prompts (needs a template in the tokenizer).',
    )
    p.add_argument('--show-special-tokens', action='store_true')
    return p.parse_args()


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
        folder = args.checkpoints_dir.name or 'checkpoints'
        outdir = args.results_dir / f'compare-{folder}'
        outdir.mkdir(parents=True, exist_ok=True)
        ckpts = list_checkpoints(args.checkpoints_dir)
        if not ckpts:
            print(f'No checkpoints found in {args.checkpoints_dir}', file=sys.stderr)
            sys.exit(1)
        print(f'Inspecting {len(ckpts)} checkpoints from {args.checkpoints_dir} (cache: {outdir}, use --force to recompute)')
        # Resolve the tokenizer once from the first checkpoint (all share the same one).
        tokenizer_dir = resolve_tokenizer(ckpts[0], args.checkpoints_dir, args.tokenizer)
        print(f'  tokenizer  : {tokenizer_dir}')
        rows = []
        for ck in ckpts:
            cache = outdir / f'{ck.name}.json'
            results = None
            if cache.exists() and not args.force:
                with open(cache) as f:
                    cached = json.load(f)
                # Stale-cache guard: reuse only if the same suites ran with the
                # same generation settings, otherwise the table mixes runs.
                same_suites = set(cached.get('suites', {})) >= set(selected)
                same_params = cached.get('gen_params') == gen_params_of(args)
                if same_suites and same_params:
                    print(f'  [cached] {ck.name}')
                    results = cached
                else:
                    print(f'  [stale cache, recomputing] {ck.name}')
            if results is None:
                results = evaluate_checkpoint(ck, tokenizer_dir, device, dtype, args, selected)
                write_json(cache, results)
                print(f'  [saved] {cache}')
            rows.append(summarize_row(results))
        # Capture the report so the same text the user just read is also on disk;
        # a table this wide is unreadable once it has scrolled past.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_compare_report(rows)
            if args.explain:
                print(GLOSSARY)
        report = buf.getvalue()
        print(report, end='')
        summary = args.output or (args.results_dir / f'evaluate-compare-{folder}.json')
        write_json(summary, rows)
        report_path = summary.with_suffix('.txt')
        write_text(report_path, report)
        print(f'\nComparison summary written to {summary}')
        print(f'Readable report written to  {report_path}')
        print(f'Per-checkpoint detail in {outdir}/')
        return

    checkpoint_dir = resolve_checkpoint(args.checkpoint, args.checkpoints_dir)
    tokenizer_dir = resolve_tokenizer(checkpoint_dir, args.checkpoints_dir, args.tokenizer)
    results = evaluate_checkpoint(checkpoint_dir, tokenizer_dir, device, dtype, args, selected)

    out = args.output or (args.results_dir / f'evaluate-{checkpoint_dir.name}.json')
    write_json(out, results)
    print(f'\nFull numeric results written to {out}')

    if args.explain:
        print(GLOSSARY)

    banner('INSPECTION COMPLETE')
    print('  This covered WHAT the checkpoint is, its lineage, and whether it is still vintage.')
    print('  For "is it baked enough", run evaluate2.py on the same path.')


if __name__ == '__main__':
    main()
