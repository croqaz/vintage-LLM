"""Merged evaluator CLI - run everything, load each model ONCE.

Usage examples:

  python -m eval                                  # latest checkpoint in ./checkpoints
  python -m eval path/to/checkpoint-22944         # one checkpoint
  python -m eval autoresearch autoresearch2       # every final/ export in those trees
  python -m eval Vintage1 --gen-mode sample       # cheaper generation pass
  python -m eval --render-report old-results.json # re-render Markdown only

For every checkpoint this computes, in ONE model load:
  * architecture / size / tokenizer info and training lineage
  * period fidelity on fixed historical-vs-modern probe sentences
  * diachronic word-sense separation (contextual embeddings)
  * held-out prose BPB with per-document records (A/B splits, early/late)
  * conditional chat-target BPB
  * forced-choice logic accuracy and anachronism-trap shocks
  * ONE merged prompt battery generated once per decoding mode (greedy +
    sampled), with ALL text-quality metrics computed on those same texts

Outputs:
  * <out>.json  - machine-readable payload; results[i].summary is a FLAT dict of
    canonically-named headline numbers meant for agents to grep across runs;
    per-document records allow bootstrap comparisons without reloading models.
  * <out>.md    - human-readable report rendered by report.py.

Re-runs reuse cached per-checkpoint results when the fingerprint (weights,
tokenizer) and the settings hash are unchanged, so pointing this at
autoresearch/ + autoresearch2/ repeatedly is cheap.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

from . import metrics as M
from . import report as R
from .helpers import (
    DEFAULT_RESULTS_DIR,
    EVAL_DATA,
    atomic_json,
    banner,
    checkpoint_lineage,
    die,
    file_hash,
    fmt,
    free_model,
    interp_curve,
    load_chat_items,
    load_model_and_tokenizer,
    load_text_items,
    model_fingerprint,
    model_slug,
    peek_model_identity,
    provenance_line,
    resolve_targets,
    resolve_tokenizer,
    select_device,
    select_dtype,
    training_curve,
)
from .metric_guide import HOW_TO_READ, PRIMARY_METRIC, describe, undocumented
from .prompts import (
    HISTORICAL_CONTEXTS,
    HISTORICAL_WORDS,
    LOGIC_ITEMS,
    MODERN_CONTEXTS,
    PROBE_LABELS,
    PROBE_SENTENCES,
    SEED_SETS,
    TRAP_PAIRS,
)

SCHEMA_VERSION = 1


# ============================================================================
# Settings identity for the cache
# ============================================================================


def settings_fingerprint(args, heldout_hash: str, chat_hash: str | None) -> dict:
    """Identity of everything except model weights that affects the numbers."""
    return {
        'schema': SCHEMA_VERSION,
        'heldout_sha256': heldout_hash,
        'chat_sha256': chat_hash,
        'docs': args.docs,
        'chat_docs': args.chat_docs,
        'max_tokens': args.max_tokens,
        'chat_max_tokens': args.chat_max_tokens,
        'generation_modes': args.generation_modes,
        'generation_tokens': args.gen_tokens,
        'generation_batch_size': args.generation_batch_size,
        'seed_set': args.seed_set,
        'temp_sweep': args.temp_sweep,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'seed': args.seed,
        'dtype': args.dtype,
        # int8 changes the numbers, so a quantized result must never be served
        # from (or written into) the same cache slot as a bf16 one.
        'load_8bit': args.load_8bit,
        'logic_items': len(LOGIC_ITEMS),
        'trap_pairs': len(TRAP_PAIRS),
        'n_prompts': len(SEED_SETS[args.seed_set]()),
    }


# ============================================================================
# Per-checkpoint evaluation - the single model load happens here
# ============================================================================


def _temp_label(temp: float) -> str:
    """Stable key fragment for a temperature.

    f'{1.0:g}' renders '1', so `sweep_t1_*` would break anyone grepping
    `sweep_t1.0_*` while 0.8 and 1.2 keep their decimal. Always keep one.
    """
    label = f'{temp:g}'
    return label if '.' in label else f'{label}.0'


def evaluate_checkpoint(checkpoint: Path, tok_dir: Path, device, dtype, args, prompts, data, snapshot=None) -> dict:
    started = time.time()

    def save(stage: str) -> None:
        """Persist everything computed SO FAR for this checkpoint.

        A model evaluation is many expensive suites in one load. Talkie-1930-13b
        completed prose, chat, logic and traps over 44 minutes and then OOMed in
        the generation suite, and all of it was lost because results are only
        recorded once a checkpoint finishes. Snapshotting after each EXPENSIVE
        suite (not the cheap ones - this must not hammer the disk) makes that
        work recoverable.
        """
        if snapshot is not None:
            try:
                snapshot(stage, result)
            except Exception as exc:  # a safety net must never break the run
                print(f'  (snapshot after {stage} failed: {exc})')

    print(f'  loading model ({tok_dir}) ...', flush=True)
    model, tokenizer = load_model_and_tokenizer(checkpoint, tok_dir, device, dtype, load_8bit=args.load_8bit)
    n_params = sum(p.numel() for p in model.parameters())
    # HF measures total_flos against non-embedding params, so the tokens-seen estimate
    # must divide by the SAME count (see checkpoint_lineage). num_parameters() is the
    # exact function Trainer.floating_point_ops calls.
    try:
        n_params_no_embed = int(model.num_parameters(exclude_embeddings=True))
    except Exception:
        n_params_no_embed = None

    result: dict = {
        'checkpoint': str(checkpoint),
        # Experiment identity for cross-run comparison: relative to the project
        # root when possible (e.g. autoresearch/optimizer/lion_8bit/final).
        'label': str(checkpoint.relative_to(Path.cwd())) if checkpoint.is_relative_to(Path.cwd()) else str(checkpoint),
        'name': checkpoint.name,
        'tokenizer': str(tok_dir),
        'params': n_params,
        'params_no_embed': n_params_no_embed,
        'params_millions': round(n_params / 1e6, 2),
    }
    try:
        # ---- info -----------------------------------------------------------
        banner('SUITE: MODEL INFO AND TRAINING LINEAGE')
        weight_files = list(checkpoint.glob('*.safetensors'))
        disk_mb = sum(f.stat().st_size for f in weight_files) / 1e6
        lineage = checkpoint_lineage(checkpoint, n_params, n_params_no_embed)
        cfg = model.config
        emb = model.get_input_embeddings().weight
        with torch.no_grad():
            emb_norm = float(emb.float().norm(dim=-1).mean())
            gen = torch.Generator(device='cpu').manual_seed(args.seed)
            pick = torch.randperm(emb.shape[0], generator=gen)[:512].to(emb.device)
            sample = emb[pick].float()
            sample = sample / sample.norm(dim=-1, keepdim=True)
            sims = sample @ sample.T
            off_diag = sims[~torch.eye(len(sample), dtype=bool, device=sample.device)]
            emb_cos = float(off_diag.mean())
        result['embedding_stats'] = {'mean_norm': emb_norm, 'mean_cosine': emb_cos}
        print(f'  embedding mean norm {emb_norm:.3f} | mean cosine {emb_cos:.3f}')
        result['info'] = {
            'model_type': cfg.model_type,
            'architecture': type(model).__name__,
            'hidden_size': cfg.hidden_size,
            'num_layers': cfg.num_hidden_layers,
            'num_attention_heads': cfg.num_attention_heads,
            'num_key_value_heads': getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads),
            'vocab_size_config': cfg.vocab_size,
            'vocab_size_tokenizer': len(tokenizer),
            'max_position_embeddings': cfg.max_position_embeddings,
            'tie_word_embeddings': getattr(cfg, 'tie_word_embeddings', None),
            'disk_size_mb': round(disk_mb, 1),
            'has_chat_template': tokenizer.chat_template is not None,
        }
        result['lineage'] = {k: v for k, v in lineage.items() if k != '_state_file'}
        result['lineage']['line'] = provenance_line(lineage)
        print(f'  params {n_params:,} | disk {disk_mb:.0f} MB | lineage: {result["lineage"]["line"]}')

        # ---- period fidelity on fixed probes ---------------------------------
        if not args.skip_probes:
            banner('SUITE: PERIOD FIDELITY ON FIXED PROBE SENTENCES')
            result['period_probes'] = M.score_probe_sentences(tokenizer, model, prompts['probe_sentences'], prompts['probe_labels'])
            pp = result['period_probes']
            print(
                f'  hist ppl {pp["historical"]["perplexity"]:.2f} | modern ppl '
                f'{pp["modern"]["perplexity"]:.2f} | ratio {pp.get("modern_over_historical_ratio", float("nan")):.2f}'
            )

        # ---- word-sense separation -------------------------------------------
        if not args.skip_embeddings:
            banner('SUITE: DIACHRONIC WORD-SENSE SEPARATION')
            # This suite needs internals (hidden states) that some custom
            # architectures simply do not expose. It is a descriptive diagnostic,
            # not a ranking input, so a failure here must NOT lose the prose BPB
            # that the same expensive model load just produced.
            try:
                result['embeddings'] = M.sense_separation(
                    model,
                    tokenizer,
                    prompts['historical_words'],
                    prompts['historical_contexts'],
                    prompts['modern_contexts'],
                )
                if result['embeddings'].get('missing_words'):
                    print(f'  (skipped words not locatable in context: {", ".join(result["embeddings"]["missing_words"])})')
                print(f'  mean shift similarity: {fmt(result["embeddings"].get("mean_shift_similarity"), 3)}')
            except Exception as exc:
                result['embeddings'] = {'unavailable': f'{type(exc).__name__}: {exc}'}
                print(f'  UNAVAILABLE for this architecture: {type(exc).__name__}: {exc}')

        # ---- held-out prose --------------------------------------------------
        if data['heldout']:
            banner('SUITE: HELD-OUT PROSE (per-document records)')
            result['prose_records'] = M.score_prose_records(tokenizer, model, data['heldout'], args.max_tokens)
            print()
            save('prose')

        # ---- chat targets ----------------------------------------------------
        if data['chat']:
            if not tokenizer.is_fast:
                print('  (chat skipped: requires a fast tokenizer)')
            else:
                banner('SUITE: CONDITIONAL CHAT-TARGET LOSS')
                result['chat_records'] = M.score_chat_records(tokenizer, model, data['chat'], args.chat_max_tokens)
                print()
                save('chat')

        # ---- tokenizer efficiency + optimisation health ----------------------
        # Neither needs the GPU. Both were previously only obtainable by hand-writing
        # a throwaway script against the tokenizer / trainer_state.json.
        if data['heldout']:
            result['tokenizer_stats'] = M.tokenizer_stats(tokenizer, data['heldout'], args.max_tokens)
        result['training_curve'] = training_curve(checkpoint)

        # ---- logic + traps ---------------------------------------------------
        banner('SUITE: LOGIC FORCED CHOICE AND ANACHRONISM TRAPS')
        result['logic'] = M.run_logic(tokenizer, model, prompts['logic_items'])
        result['traps'] = M.run_traps(tokenizer, model, prompts['trap_pairs'])
        print(
            f'  logic acc {result["logic"]["acc"]:.3f} (margin {result["logic"]["margin"]:+.3f}) | '
            f'trap shock {result["traps"]["mean_shock"]:+.3f} bits/byte'
        )
        save('logic_traps')

        # ---- THE one generation pass over the MERGED prompt list --------------
        if args.generation_modes:
            banner('SUITE: GENERATION OVER MERGED PROMPT BATTERY')
            print(f'  {len(prompts["generation"])} unique prompts x modes {args.generation_modes}, {args.gen_tokens} new tokens each')
            modes_out = M.generate_continuations(
                tokenizer,
                model,
                prompts['generation'],
                modes=args.generation_modes,
                max_new_tokens=args.gen_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
                batch_size=args.generation_batch_size,
                chat=args.chat,
            )
            gen_block = {'temperature': args.temperature, 'top_p': args.top_p, 'top_k': args.top_k}
            for mode, prefix in (('greedy', 'greedy'), ('sample', 'sampled')):
                rows = modes_out.get(mode, [])
                gen_block[f'{prefix}_samples'] = rows
                gen_block[f'{prefix}_summary'] = M.summarize_generations(rows)
            result['generation'] = gen_block
            save('generation')

            # ---- optional temperature sweep ---------------------------------
            # Degeneracy at high temperature is a TAIL property of the
            # distribution; BPB measures the head and is nearly blind to it. A
            # model can compress beautifully and still fall apart at t=1.2,
            # which is the regime a bulk synth-data run actually uses.
            # Off by default: each extra temperature costs one more full
            # sampled pass.
            if args.temp_sweep and 'sample' in args.generation_modes:
                sweep = {}
                for temp in args.temp_sweep:
                    label = _temp_label(temp)
                    if abs(temp - args.temperature) < 1e-9 and gen_block.get('sampled_summary'):
                        sweep[label] = gen_block['sampled_summary']
                        continue
                    print(f'  temperature sweep: t={temp:g}', flush=True)
                    swept = M.generate_continuations(
                        tokenizer,
                        model,
                        prompts['generation'],
                        modes=['sample'],
                        max_new_tokens=args.gen_tokens,
                        temperature=temp,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        seed=args.seed,
                        batch_size=args.generation_batch_size,
                        chat=args.chat,
                    )
                    sweep[label] = M.summarize_generations(swept.get('sample', []))
                    gen_block['temperature_sweep'] = sweep
                    result['generation'] = gen_block
                    save(f'sweep_t{label}')
                gen_block['temperature_sweep'] = sweep
            result['generation'] = gen_block
    finally:
        free_model(model, device)
        print(f'  model freed ({time.time() - started:.0f}s elapsed)', flush=True)

    # ---- derived numbers (pure math over what we just collected) ------------
    finalize_result(result, args)
    # Put the headline numbers at the TOP of the object. A human or an agent
    # opening a 200 KB result should hit `summary` on line 3, not after 4,000
    # lines of per-document records.
    lead = ('label', 'checkpoint', 'summary', 'verdict', 'bake_score', 'points')
    return {**{k: result[k] for k in lead if k in result}, **result}


def finalize_result(result: dict, args) -> None:
    """Build summary + bake score from the raw measurements. No model needed."""
    s: dict = {}

    prose = result.get('prose_records') or []
    if prose:
        s['prose_bpb'] = M.records_bpb(prose)
        # Never let a dropped document be invisible: a non-zero count means the
        # headline bpb was computed over fewer than n_prose_docs documents.
        s['prose_nonfinite_docs'] = M.count_nonfinite_records(prose)
        if s['prose_nonfinite_docs']:
            print(
                f'  WARNING: {s["prose_nonfinite_docs"]} document(s) scored NaN/Inf and were '
                f'EXCLUDED from prose_bpb (of {len(prose)} total)'
            )
        s['prose_bpb_split_a'] = M.records_bpb(prose, 'A')
        s['prose_bpb_split_b'] = M.records_bpb(prose, 'B')
        s['prose_bpb_early'] = M.records_bpb(prose, prefix='early_')
        s['prose_bpb_late'] = M.records_bpb(prose, prefix='late_')

    chat = result.get('chat_records') or []
    if chat:
        s['chat_bpb'] = M.records_bpb(chat)

    logic = result.get('logic') or {}
    if logic:
        s['logic_acc'] = logic.get('acc')
        s['logic_margin'] = logic.get('margin')

    traps = result.get('traps') or {}
    if traps:
        s.update(
            trap_mean_shock=traps.get('mean_shock'),
            trap_min_shock=traps.get('min_shock'),
            trap_n_leaked=traps.get('n_leaked'),
            trap_n_pairs=traps.get('n'),
        )

    probes = result.get('period_probes') or {}
    if probes:
        s['probe_historical_ppl'] = probes.get('historical', {}).get('perplexity')
        s['probe_modern_ppl'] = probes.get('modern', {}).get('perplexity')
        s['probe_modern_over_historical_ratio'] = probes.get('modern_over_historical_ratio')
        # Model-health signals from the legacy evaluate.py token-stat block: they were
        # computed but only reachable deep in period_probes.overall. Surfaced flat so
        # they can be grepped across runs like every other headline number.
        _ov = probes.get('overall') or {}
        s['probe_frac_low_confidence'] = _ov.get('frac_low_confidence')
        s['probe_mean_entropy_nats'] = _ov.get('mean_entropy_nats')
        s['probe_mean_token_prob'] = _ov.get('mean_token_prob')

    emb = result.get('embeddings') or {}
    if emb:
        s['sense_shift_mean_cosine'] = emb.get('mean_shift_similarity')

    emb_stats = result.get('embedding_stats') or {}
    s['embedding_mean_norm'] = emb_stats.get('mean_norm')
    s['embedding_mean_cosine'] = emb_stats.get('mean_cosine')

    lineage = result.get('lineage') or {}
    s['tokens_seen_estimate'] = lineage.get('tokens_seen')
    s['tokens_per_param'] = lineage.get('tokens_per_param')

    tstats = result.get('tokenizer_stats') or {}
    s['tokenizer_bytes_per_token'] = tstats.get('bytes_per_token')
    s['tokenizer_vocab_size'] = tstats.get('vocab_size')

    curve = result.get('training_curve') or {}
    s['final_eval_loss'] = curve.get('final_eval_loss')
    s['final_eval_ppl'] = curve.get('final_eval_ppl')
    s['final_eval_step'] = curve.get('final_eval_step')
    s['final_train_loss'] = curve.get('final_train_loss')
    s['grad_norm_max'] = curve.get('grad_norm_max')
    s['grad_norm_mean'] = curve.get('grad_norm_mean')
    s['grad_norm_min'] = curve.get('grad_norm_min')
    s['grad_norm_nonfinite'] = curve.get('grad_norm_nonfinite')

    gen = result.get('generation') or {}
    greedy_sum, sampled_sum = gen.get('greedy_summary') or {}, gen.get('sampled_summary') or {}
    if greedy_sum:
        s['greedy_mean_loop_words'] = greedy_sum.get('mean_loop_words')
        s['greedy_worst_loop_words'] = greedy_sum.get('worst_loop_words')
    if sampled_sum:
        s['sampled_mean_distinct_1'] = sampled_sum.get('mean_distinct_1')
        s['sampled_mean_distinct_2'] = sampled_sum.get('mean_distinct_2')
        s['sampled_mean_echo_rate'] = sampled_sum.get('mean_echo_rate')
        s['sampled_temperature'] = (result.get('generation') or {}).get('temperature')
        s['sampled_mean_loop_words'] = sampled_sum.get('mean_loop_words')
        s['sampled_degenerate_rate'] = sampled_sum.get('degenerate_rate')
        s['sampled_prompt_copy_rate'] = sampled_sum.get('mean_prompt_copy_rate')
        s['sampled_punct_issues_p100'] = sampled_sum.get('mean_punct_issues_p100')
        # Back matter (indexes/catalogues/TOC): lexically diverse, so distinct-n,
        # echo and loop detection are ALL blind to it. It was the dominant
        # low-scoring pattern in a real bulk run.
        s['sampled_back_matter_rate'] = sampled_sum.get('back_matter_rate')
        s['sampled_mean_sentence_words'] = sampled_sum.get('mean_sentence_words')
        # Mode collapse ACROSS completions; distinct-n only sees within one.
        s['sampled_self_bleu_4'] = sampled_sum.get('self_bleu_4')
        # degenerate OR back matter = what a synth-data filter would drop.
        s['sampled_unusable_rate'] = sampled_sum.get('unusable_rate')
        s['greedy_self_bleu_4'] = greedy_sum.get('self_bleu_4') if greedy_sum else None
        s['greedy_back_matter_rate'] = greedy_sum.get('back_matter_rate') if greedy_sum else None

    sweep = (result.get('generation') or {}).get('temperature_sweep') or {}
    for temp, row in sweep.items():
        s[f'sweep_t{temp}_unusable_rate'] = row.get('unusable_rate')
        s[f'sweep_t{temp}_degenerate_rate'] = row.get('degenerate_rate')
        s[f'sweep_t{temp}_back_matter_rate'] = row.get('back_matter_rate')
        s[f'sweep_t{temp}_self_bleu_4'] = row.get('self_bleu_4')
        s[f'sweep_t{temp}_mean_loop_words'] = row.get('mean_loop_words')
    result['summary'] = s

    # ---- BAKE score: computed exactly ONCE from these same numbers ----------
    points = {}
    if s.get('prose_bpb') is not None:
        points['bpb'] = M.interp(s['prose_bpb'], M.BPB_LADDER)
    if s.get('logic_acc') is not None:
        points['logic'] = M.logic_points(s['logic_acc'])
    if s.get('chat_bpb') is not None:
        points['chat'] = M.interp(s['chat_bpb'], M.CHAT_LADDER)
    if greedy_sum and sampled_sum:
        loop = greedy_sum.get('mean_loop_words', 0.0)
        punct = sampled_sum.get('mean_punct_issues_p100', 0.0)
        if loop is not None and punct is not None:
            points['hygiene'] = M.hygiene_points(loop, punct)
    result['points'] = points
    # HONESTY GUARD: bake_score() renormalises over whichever components are
    # present, so a model evaluated with --gen-mode none silently scores as if
    # hygiene did not exist. That is exactly how Talkie-1930-13b posted 97.6
    # from {bpb, logic, chat} while missing the one axis it fails. Record what
    # was missing so no ranking can quietly compare a partial score to a full
    # one.
    missing = [k for k in M.WEIGHTS if k not in points]
    result['bake_components_missing'] = missing
    result['bake_is_partial'] = bool(missing)
    s['bake_components_missing'] = ','.join(missing) if missing else None
    s['bake_is_partial'] = bool(missing)
    s['bake_weight_covered'] = round(sum(w for k, w in M.WEIGHTS.items() if k in points), 3)
    if missing:
        print(
            f'  WARNING: bake score is PARTIAL — missing {", ".join(missing)} '
            f'({100 * (1 - s["bake_weight_covered"]):.0f}% of the weight); not comparable to a full score'
        )
    score = M.bake_score(points)
    result['bake_score'] = score
    tier, text = M.verdict_text(score, lineage)
    result['verdict'] = {'tier': tier, 'text': text}
    s['bake_score'] = score
    s['verdict_tier'] = tier
    for k, v in points.items():
        s[f'points_{k}'] = v


# ============================================================================
# Rankings (paired bootstrap on the retained per-document records)
# ============================================================================


def rank_by_prose(results: list[dict], args) -> list[dict]:
    eligible = [r for r in results if r.get('prose_records') and math.isfinite(r['summary'].get('prose_bpb', float('nan')))]
    if not eligible:
        return []
    eligible.sort(key=lambda r: r['summary']['prose_bpb'])
    leader = eligible[0]
    epsilon = leader['summary']['prose_bpb'] * args.equivalence
    rows = []
    for rank, r in enumerate(eligible, 1):
        ci_lo, ci_hi = M.bootstrap_ci(r['prose_records'], args.bootstrap, args.confidence, args.seed)
        pair = M.paired_bootstrap(r['prose_records'], leader['prose_records'], args.bootstrap, args.confidence, args.seed + rank)
        rows.append(
            {
                'rank': rank,
                'label': r['label'],
                'checkpoint': r['checkpoint'],
                'bpb': r['summary']['prose_bpb'],
                'ci_low': ci_lo,
                'ci_high': ci_hi,
                'delta_vs_leader': pair['delta'],
                'delta_ci_low': pair['low'],
                'delta_ci_high': pair['high'],
                'p_beats_leader': pair['p_beats'],
                'equivalent_to_leader': pair['low'] <= epsilon,
                'chat_bpb': r['summary'].get('chat_bpb'),
                'bake_score': r['bake_score'],
            }
        )
        r['ci_low'], r['ci_high'] = ci_lo, ci_hi  # surfaced in the detail section too
    return rows


# ============================================================================
# Optional big-model judge (loaded once, after evaluated models are freed)
# ============================================================================


def run_judge(args, results, device, dtype, heldout_texts) -> None:
    from .helpers import resolve_tokenizer as rt

    judge_dir = Path(args.judge)
    jtok_dir = rt(judge_dir, args.tokenizer)
    print(f'loading judge {judge_dir} ...')
    judge_model, judge_tok = load_model_and_tokenizer(judge_dir, jtok_dir, device, dtype)
    try:
        real_bpb = M.bits_per_byte_of_texts(judge_tok, judge_model, heldout_texts[:100], max_tokens=320)
        for r in results:
            texts = [s['continuation'] for s in (r.get('generation') or {}).get('sampled_samples', [])]
            texts = [t for t in texts if len(t.split()) >= 20]
            if not texts:
                continue
            gen_bpb = M.bits_per_byte_of_texts(judge_tok, judge_model, texts, max_tokens=320)
            dev = abs(gen_bpb - real_bpb)
            r.setdefault('summary', {}).update(judge_bpb_real=real_bpb, judge_bpb_gen=gen_bpb, judge_deviation=dev)
            print(f'  {r["label"]}: judge deviation {dev:.3f} bits/byte')
    finally:
        free_model(judge_model, device)


# ============================================================================
# CLI
# ============================================================================


# ============================================================================
# Cross-run collection: compare models evaluated in SEPARATE invocations
# ============================================================================


def collect_results(roots: list[Path]) -> list[dict]:
    """Load every per-model eval-*.json under `roots` into one result list.

    Per-document prose records are stored in each JSON, so a full PAIRED bootstrap
    across separately-evaluated models needs no model loads and no GPU at all.
    """
    seen: dict[str, dict] = {}
    for root in roots:
        root = root.resolve()
        files = sorted(root.rglob('eval-*.json')) if root.is_dir() else [root]
        for f in files:
            try:
                payload = json.loads(f.read_text())
            except Exception as exc:
                print(f'  skipping {f}: {exc}')
                continue
            for r in payload.get('results', []):
                # Key on the PROJECT-RELATIVE label, not the absolute checkpoint path:
                # JSONs produced on another machine or in a container carry unrelated
                # absolute paths (e.g. /root/PWD/...) and would escape de-duplication.
                key = r.get('label') or r.get('checkpoint')
                if not key:
                    continue
                prev = seen.get(key)
                if prev is None:
                    seen[key] = r
                    continue

                # Same model from two JSONs (a folder-level file and a per-model file).
                # Prefer the one whose checkpoint path exists here, then the richer record.
                def _score(rec):
                    ck = rec.get('checkpoint') or ''
                    return (Path(ck).exists(), len(rec.get('summary') or {}))

                if _score(r) > _score(prev):
                    seen[key] = r
    return list(seen.values())


def compare_curves(results: list[dict], grid_points: int = 10) -> dict:
    """Eval loss for every run interpolated onto ONE common step grid.

    `eval_steps` is in MINUTES, so runs of different speed evaluate at different
    step numbers and their raw curves are not directly comparable. The grid spans
    the largest step every run actually reached.
    """
    curves = {r['label']: (r.get('training_curve') or {}).get('eval_curve') or [] for r in results}
    curves = {k: v for k, v in curves.items() if len(v) >= 2}
    if len(curves) < 2:
        return {}
    common_max = min(v[-1][0] for v in curves.values())
    common_min = max(v[0][0] for v in curves.values())
    if common_max <= common_min:
        return {}
    step = (common_max - common_min) / (grid_points - 1)
    grid = [round(common_min + i * step) for i in range(grid_points)]
    return {
        'grid': grid,
        'series': {k: [interp_curve(v, g) for g in grid] for k, v in curves.items()},
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='python -m eval',
        description='Merged evaluation for tiny vintage LLMs: info+lineage, period fidelity, '
        'held-out BPB, bake score, logic/traps, chat readiness and generation hygiene - '
        'one model load, one prompt battery.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('targets', nargs='*', type=Path, help='Checkpoint dir, folder of checkpoints, or experiment tree.')
    p.add_argument('--tokenizer', type=Path, default=None, help='Force one tokenizer for every checkpoint.')
    p.add_argument('--heldout', type=Path, default=EVAL_DATA / 'heldout-Sprocket-n-Say.jsonl')
    p.add_argument('--chat-data', type=Path, default=EVAL_DATA / 'chat_sample.jsonl')
    p.add_argument('--docs', type=int, default=200, help='Held-out docs to score.')
    p.add_argument('--chat-docs', type=int, default=200)
    p.add_argument('--no-chat', action='store_true', help='Skip the conditional chat-target diagnostic.')
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--chat-max-tokens', type=int, default=768)
    p.add_argument(
        '--gen-mode',
        choices=('both', 'greedy', 'sample', 'none'),
        default='both',
        help="Decoding modes over the merged prompt battery ('both' = greedy for loop detection + sampled for quality).",
    )
    p.add_argument(
        '--gen-tokens',
        type=int,
        default=256,
        help='New tokens per continuation. Default raised from 120 to 256: a real 49k-completion '
        'bulk run had median 240 tokens (p90 727), and loops often only develop past 120.',
    )
    p.add_argument('--generation-batch-size', type=int, default=4)
    p.add_argument(
        '--seed-set',
        choices=('curated', 'cold', 'both'),
        default='both',
        help="Prompt battery. 'curated' = 46 topical stems; 'cold' = 26 bare function-word openers "
        'sampled from a real 49k bulk run (much harder, no subject handed to the model); '
        "'both' = the default 72.",
    )
    p.add_argument(
        '--temp-sweep',
        type=str,
        default=None,
        metavar='T1,T2,...',
        help='Also summarise sampled generation at these temperatures, e.g. 0.8,1.0,1.2. '
        'Each extra temperature costs one more full sampled pass. Off by default.',
    )
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--top-k', type=int, default=25)
    p.add_argument('--chat', action='store_true', help='Apply a chat template to generation prompts.')
    p.add_argument('--skip-probes', action='store_true', help='Skip fixed-probe period-fidelity suite.')
    p.add_argument('--skip-embeddings', action='store_true', help='Skip word-sense separation suite.')
    p.add_argument('--judge', type=str, default=None, help='Optional big period model dir for the coherence judge.')
    p.add_argument('--bootstrap', type=int, default=10_000)
    p.add_argument('--confidence', type=float, default=0.95)
    p.add_argument('--equivalence', type=float, default=0.001, help='Relative BPB difference treated as practically equivalent.')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    p.add_argument('--dtype', choices=('auto', 'float32', 'float16', 'bfloat16'), default='auto')
    p.add_argument(
        '--load-8bit',
        action='store_true',
        help='Load weights with bitsandbytes int8. Halves resident size so a 13B fits a 16GB card, '
        'at a small measurement cost (-0.0075 nll on the 75M control). Recorded in the results JSON.',
    )
    p.add_argument('--threads', type=int, default=None, help='Set PyTorch CPU worker threads.')
    p.add_argument('--include-checkpoints', action='store_true', help='Include checkpoint-N dirs during recursive discovery.')
    p.add_argument(
        '--out',
        '--output',
        '-o',
        dest='out',
        type=Path,
        default=None,
        help='JSON output path (Markdown written beside it). Default: eval-<arch><size>.json in the folder that was pointed at.',
    )
    p.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument(
        '--slim',
        action='store_true',
        help='Drop per-document records and generated texts from the main JSON (~85%% smaller). '
        'The -summary.json sidecar is written either way. NOTE: a slim file cannot be used for '
        'paired-bootstrap --collect or a full --render-report.',
    )
    p.add_argument(
        '--audit-guide',
        action='store_true',
        help='Check that every summary key in an existing results JSON has a metric_guide entry, '
        'then exit. Takes the JSON path as the positional argument.',
    )
    p.add_argument('--force', action='store_true', help='Ignore matching cached checkpoint results.')
    p.add_argument('--render-report', type=Path, default=None, help='Only re-render Markdown from an existing results JSON.')
    p.add_argument(
        '--collect',
        nargs='+',
        type=Path,
        default=None,
        help='Merge existing eval-*.json under these paths into ONE ranked comparison '
        '(paired bootstrap across separately-run models). No models are loaded.',
    )
    return p.parse_args(argv)


def audit_guide(path: Path) -> int:
    """Fail loudly if any summary key in `path` lacks a metric_guide entry."""
    payload = json.loads(path.read_text())
    keys: set[str] = set()
    for r in payload.get('results', []):
        keys.update((r.get('summary') or {}).keys())
    missing = undocumented(keys)
    print(f'{path}: {len(keys)} summary keys, {len(keys) - len(missing)} documented')
    if missing:
        print('UNDOCUMENTED (add these to eval/metric_guide.py):')
        for k in missing:
            print(f'  {k}')
        return 1
    print('OK - every summary key is documented.')
    return 0


def main(argv=None) -> None:
    args = parse_args(argv)

    # ---- report-only mode ----------------------------------------------------
    if args.render_report:
        src = Path(args.render_report)
        payload = json.loads(src.read_text())
        # Refresh the self-description in place. This costs NOTHING (no model
        # load, no GPU) and is the only way to bring a result forward onto the
        # current schema when re-running the model is impractical -- e.g.
        # MODELS/Talkie-1930-13b, whose 13B int8 weights cannot be re-evaluated
        # on this machine safely.
        keys: set[str] = set()
        for r in payload.get('results', []):
            keys.update((r.get('summary') or {}).keys())
        payload['primary_metric'] = PRIMARY_METRIC
        payload['how_to_read'] = HOW_TO_READ
        payload['metric_guide'] = describe(sorted(keys))
        missing = undocumented(keys)
        if missing:
            payload['metric_guide_undocumented'] = missing
            print(f'  WARNING: {len(missing)} summary keys have no metric_guide entry: {", ".join(missing)}')
        atomic_json(src, payload)
        slim = {
            'primary_metric': payload['primary_metric'],
            'how_to_read': payload['how_to_read'],
            'models': [
                {'label': r.get('label'), 'params_millions': r.get('params_millions'), **(r.get('summary') or {})}
                for r in payload.get('results', [])
            ],
            'metric_guide': payload['metric_guide'],
        }
        atomic_json(src.with_name(src.stem + '-summary.json'), slim)
        out_md = args.out or src.with_suffix('.md')
        out_md.write_text(R.render_report(payload), encoding='utf-8')
        print(f'report: {out_md}')
        print(f'summary: {src.with_name(src.stem + "-summary.json")}')
        return

    # ---- collect-only mode: compare models evaluated in separate runs ---------
    if args.collect:
        results = collect_results(args.collect)
        if not results:
            die(f'no eval-*.json found under {", ".join(map(str, args.collect))}')
        print(f'collected {len(results)} model results (no models loaded)')
        payload = {
            'schema_version': SCHEMA_VERSION,
            'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'target_label': 'collected: ' + ', '.join(str(p) for p in args.collect),
            'settings': {'collected_from': [str(p) for p in args.collect], 'bootstrap': args.bootstrap},
            'environment': {'python': platform.python_version(), 'torch': torch.__version__},
            'notes': [
                'Collected from existing per-model JSONs. Metrics were computed in their '
                'original runs; only the rankings and curve comparison are new.',
                'Paired bootstrap uses the retained per-document records, so it is exact even though the models were evaluated separately.',
            ],
            'bootstrap_confidence': args.confidence,
            # SELF-DESCRIPTION. This collected file is the one a cold agent opens
            # FIRST - it is the cross-model ranking - so it must carry the same
            # guide as a single-model result, not less.
            'primary_metric': PRIMARY_METRIC,
            'how_to_read': HOW_TO_READ,
            'metric_guide': {},
            'results': results,
            'skipped': [],
            'failures': [],
            'rankings': {
                'prose': rank_by_prose(results, args),
                'bake': sorted(
                    (
                        {'label': r['label'], 'bake_score': r.get('bake_score'), 'tier': (r.get('verdict') or {}).get('tier')}
                        for r in results
                        if r.get('bake_score') is not None
                    ),
                    key=lambda x: -x['bake_score'],
                ),
            },
            'curve_comparison': compare_curves(results),
        }
        for i, row in enumerate(payload['rankings']['bake'], 1):
            row['rank'] = i

        seen_keys: set[str] = set()
        for r in results:
            seen_keys.update((r.get('summary') or {}).keys())
        payload['metric_guide'] = describe(sorted(seen_keys))
        missing = undocumented(seen_keys)
        if missing:
            payload['metric_guide_undocumented'] = missing
            print(f'  WARNING: {len(missing)} summary keys have no metric_guide entry: {", ".join(missing)}')

        out_json = (args.out or args.results_dir / 'eval-collected.json').resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(out_json, payload)
        atomic_json(
            out_json.with_name(out_json.stem + '-summary.json'),
            {
                'primary_metric': payload['primary_metric'],
                'how_to_read': payload['how_to_read'],
                'models': [
                    {'label': r.get('label'), 'params_millions': r.get('params_millions'), **(r.get('summary') or {})} for r in results
                ],
                'metric_guide': payload['metric_guide'],
            },
        )
        out_md = out_json.with_suffix('.md')
        out_md.write_text(R.render_report(payload), encoding='utf-8')
        print(f'JSON (for agents): {out_json}')
        print(f'Markdown (humans): {out_md}')
        return

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.no_chat or args.chat_docs == 0:
        args.chat_docs = 0

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    args.generation_modes = {'both': ['greedy', 'sample'], 'greedy': ['greedy'], 'sample': ['sample'], 'none': []}[args.gen_mode]
    args.temp_sweep = [float(t) for t in args.temp_sweep.split(',') if t.strip()] if args.temp_sweep else []
    if args.audit_guide:
        if not args.targets:
            die('--audit-guide needs a results JSON path as the positional argument')
        raise SystemExit(audit_guide(Path(args.targets[0])))

    # ---- data (loaded once for the whole run) ---------------------------------
    from .prompts import (
        LOGIC_ITEMS,
        SEED_SETS,
        TRAP_PAIRS,
    )

    prompts = {
        'generation': SEED_SETS[args.seed_set](),
        'logic_items': LOGIC_ITEMS,
        'trap_pairs': TRAP_PAIRS,
        'probe_sentences': PROBE_SENTENCES,
        'probe_labels': PROBE_LABELS,
        'historical_words': HISTORICAL_WORDS,
        'historical_contexts': HISTORICAL_CONTEXTS,
        'modern_contexts': MODERN_CONTEXTS,
    }

    extra_notes = []
    heldout_path = args.heldout.resolve()
    chat_path = args.chat_data.resolve()
    if heldout_path.exists():
        heldout = load_text_items(heldout_path, args.docs)
        if heldout_path != (EVAL_DATA / 'heldout-Sprocket-n-Say.jsonl').resolve():
            extra_notes.append('Custom held-out set: ladder placement is approximate; overlapping training data flatters BPB.')
    else:
        heldout = []
        extra_notes.append('NO HELD-OUT DATA FOUND - the strongest metric was skipped. Pass --heldout FILE.jsonl.')
    chats = [] if (args.no_chat or args.chat_docs == 0) or not chat_path.exists() else load_chat_items(chat_path, args.chat_docs)
    data = {'heldout': heldout, 'chat': chats}

    # ---- checkpoints ----------------------------------------------------------
    targets = args.targets or [Path.cwd() / 'checkpoints']
    resolved = resolve_targets(targets, args.include_checkpoints)
    if not resolved.checkpoints:
        die(f'no HF checkpoints found under {", ".join(map(str, targets))}')
    for item in resolved.skipped:
        print(f'  skipping {item["path"]}: {item["reason"]}')

    # A target that was skipped (missing path, unloadable format) must not name the run
    # or decide where output lands -- otherwise results get filed under a model that was
    # never evaluated. Fall back to the raw targets only if every one was skipped.
    _skipped_paths = {item['path'] for item in resolved.skipped}
    eval_targets = [t for t in targets if str(t.resolve()) not in _skipped_paths] or targets
    target_label = eval_targets[0].name if len(resolved.checkpoints) > 1 else resolved.checkpoints[0].name

    # ---- output location: next to what was pointed at ------------------------
    # Default: the folder that was pointed to (llama-77/final/ -> results in
    # llama-77/final/eval-<arch><size>.json, e.g. eval-llama77M.json).
    # Override with --output/--out. Falls back to --results-dir if unwritable.
    if args.out:
        out_json = Path(args.out)
    else:
        base = eval_targets[0].resolve()
        out_dir = base if base.is_dir() else base.parent
        if len(resolved.checkpoints) == 1:
            model_type, approx_params = peek_model_identity(resolved.checkpoints[0])
            slug = model_slug(model_type, approx_params) if approx_params else resolved.checkpoints[0].name
            out_json = out_dir / f'eval-{slug}.json'
        else:
            out_json = out_dir / f'eval-{out_dir.name}.json'
    import os

    if not args.out and not os.access(out_json.parent, os.W_OK):
        fallback = args.results_dir / out_json.name
        print(f'warning: {out_json.parent} is not writable, falling back to {fallback}')
        out_json = fallback
    out_json = out_json.resolve()
    out_md = out_json.with_suffix('.md')

    # ---- resumable cache -------------------------------------------------------
    settings = settings_fingerprint(args, file_hash(heldout_path), file_hash(chat_path) if chats else None)
    payload: dict = {
        'schema_version': SCHEMA_VERSION,
        'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'target_label': str(eval_targets[0]) if len(resolved.checkpoints) > 1 else str(resolved.checkpoints[0]),
        'settings': settings,
        'environment': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'transformers': __import__('transformers').__version__,
            'device': str(device),
            'dtype': str(dtype).removeprefix('torch.'),
        },
        'notes': extra_notes,
        'bootstrap_confidence': args.confidence,
        # ---- SELF-DESCRIPTION -------------------------------------------
        # This file must be readable by an agent on another machine with no
        # access to this repo's history. `how_to_read` states the three rules
        # that prevent wrong conclusions; `metric_guide` gives every summary
        # key a unit, a direction and its comparability caveats.
        'primary_metric': PRIMARY_METRIC,
        'how_to_read': HOW_TO_READ,
        'metric_guide': {},
        'results': [],
        'skipped': resolved.skipped,
        'failures': [],
    }
    # SNAPSHOT BEFORE OVERWRITING. `payload['results']` starts EMPTY and is
    # written incrementally, so a --force run that dies partway (e.g. an OOM in
    # the generation suite) leaves an empty file where good results used to be.
    # That destroyed a 13B evaluation that could not be re-run. Never again:
    # keep the previous file beside the new one until the run succeeds.
    if out_json.exists():
        try:
            prev = out_json.with_name(out_json.stem + '.prev.json')
            prev.write_bytes(out_json.read_bytes())
        except Exception as exc:  # insurance must never block the run
            print(f'  (could not snapshot previous results: {exc})')

    cached: dict[str, dict] = {}
    if out_json.exists() and not args.force:
        try:
            old = json.loads(out_json.read_text())
            if old.get('settings') == settings:
                cached = {r['fingerprint']: r for r in old.get('results', []) if r.get('fingerprint')}
        except Exception as exc:
            print(f'warning: could not read cache {out_json}: {exc}', file=sys.stderr)

    print(
        f'eval: {len(resolved.checkpoints)} checkpoint(s), {len(heldout)} prose docs, {len(chats)} chat docs, '
        f'{len(prompts["generation"])} generation prompts on {device}/{dtype}'
    )
    print(f'output: {out_json}')

    for index, checkpoint in enumerate(resolved.checkpoints, 1):
        banner(f'CHECKPOINT {index}/{len(resolved.checkpoints)}: {checkpoint.name}')
        try:
            tok_dir = resolve_tokenizer(checkpoint, args.tokenizer)
            fingerprint = model_fingerprint(checkpoint, tok_dir)
            if fingerprint in cached:
                result = cached[fingerprint]
                # Lineage files are tiny and may have changed (resumed runs);
                # refresh them even when expensive scores come from cache.
                result['lineage'] = {
                    k: v
                    for k, v in checkpoint_lineage(checkpoint, result.get('params'), result.get('params_no_embed')).items()
                    if k != '_state_file'
                }
                result['lineage']['line'] = provenance_line(result['lineage'])
                print('  [cached]')
            else:
                partial_path = out_json.with_name(out_json.stem + '.partial.json')

            def _snapshot(stage: str, partial: dict, _p=partial_path, _c=checkpoint) -> None:
                # ONE small file per model, overwritten in place - not a growing
                # pile. If a later suite dies, everything up to `stage` survives.
                atomic_json(
                    _p,
                    {
                        'incomplete': True,
                        'stage_completed': stage,
                        'checkpoint': str(_c),
                        'note': 'Partial snapshot written mid-evaluation. Superseded by the '
                        'main eval-*.json once the checkpoint finishes; kept so an '
                        'expensive run that dies late is not lost.',
                        'result': partial,
                    },
                )

            result = evaluate_checkpoint(checkpoint, tok_dir, device, dtype, args, prompts, data, snapshot=_snapshot)
            result['fingerprint'] = fingerprint
            result['n_prose_docs'] = len(heldout)
            result['n_chat_docs'] = len(chats)
            payload['results'].append(result)
            atomic_json(out_json, payload)  # incremental: safe to interrupt
            s = result.get('summary', {})
            print(
                f'  => bake {fmt(result.get("bake_score"), 0)}/100 | bpb {fmt(s.get("prose_bpb"), 4)}'
                f' | logic {fmt(s.get("logic_acc"), 3)} | tier {result["verdict"]["tier"]}'
            )
        except KeyboardInterrupt:
            atomic_json(out_json, payload)
            raise
        except Exception as exc:  # keep going: one bad checkpoint must not kill a 50-model sweep
            import traceback

            traceback.print_exc()
            payload['failures'].append({'path': str(checkpoint), 'reason': f'{type(exc).__name__}: {exc}'})
            atomic_json(out_json, payload)

    if not payload['results']:
        die('every checkpoint failed')

    # ---- optional judge --------------------------------------------------------
    if args.judge:
        banner('SUITE: BIG-MODEL COHERENCE JUDGE')
        run_judge(args, payload['results'], device, dtype, [t.text for t in heldout])
        atomic_json(out_json, payload)

    # ---- rankings ----------------------------------------------------------------
    payload['rankings'] = {'prose': rank_by_prose(payload['results'], args)}
    scored = [r for r in payload['results'] if isinstance(r.get('bake_score'), float) and not math.isnan(r['bake_score'])]
    payload['rankings']['bake'] = [
        {'rank': i, 'label': r['label'], 'bake_score': r['bake_score'], 'tier': r['verdict']['tier']}
        for i, r in enumerate(sorted(scored, key=lambda r: -r['bake_score']), 1)
    ]
    if not payload['results']:
        # Every checkpoint failed. Writing this would replace a good previous
        # result with an empty one - exactly the failure that lost Talkie's 13B
        # evaluation. Leave whatever is on disk alone.
        prev = out_json.with_name(out_json.stem + '.prev.json')
        if prev.exists():
            out_json.write_bytes(prev.read_bytes())
            print(f'  RESTORED previous results from {prev} (this run produced none)')
        die('every checkpoint failed - previous results left intact')

    atomic_json(out_json, payload)

    # ---- describe exactly the keys this run produced -------------------------
    seen_keys: set[str] = set()
    for r in payload['results']:
        seen_keys.update((r.get('summary') or {}).keys())
    payload['metric_guide'] = describe(sorted(seen_keys))
    missing = undocumented(seen_keys)
    if missing:
        # A metric nobody can interpret is worse than no metric. Loud, not fatal.
        payload['metric_guide_undocumented'] = missing
        print(f'  WARNING: {len(missing)} summary keys have no metric_guide entry: {", ".join(missing)}')
    atomic_json(out_json, payload)

    # ---- small sidecars for humans and greps ---------------------------------
    # The full JSON is ~85% per-document records and generated texts, which exist
    # so --collect can run paired bootstrap and --render-report can re-render
    # without reloading models. Nobody should have to read that to get a number.
    # NUMBERS FIRST, dictionary after: metric_guide is ~600 lines and the data is
    # ~60, so putting the guide first would bury exactly what you opened this for.
    slim = {
        'primary_metric': payload['primary_metric'],
        'how_to_read': payload['how_to_read'],
        'models': [
            {'label': r.get('label'), 'params_millions': r.get('params_millions'), **(r.get('summary') or {})} for r in payload['results']
        ],
        'metric_guide': payload['metric_guide'],
    }
    atomic_json(out_json.with_name(out_json.stem + '-summary.json'), slim)

    # One line per model, append-only: greppable and diffable across runs.
    try:
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_RESULTS_DIR / 'summaries.jsonl', 'a', encoding='utf-8') as fh:
            for row in slim['models']:
                fh.write(json.dumps({'generated': payload.get('created'), **row}) + '\n')
    except Exception as exc:  # a convenience index must never fail a run
        print(f'  (could not append to summaries.jsonl: {exc})')

    # ---- Markdown report ------------------------------------------------------------
    report_text = R.render_report(payload)
    out_md.write_text(report_text, encoding='utf-8')
    notes = '\n'.join(f'*{n}*' for n in extra_notes)
    if notes:
        with open(out_md, 'a', encoding='utf-8') as f:
            f.write('\n' + notes + '\n')

    if args.slim:
        # Destroys paired-bootstrap and re-render ability for this file; the
        # numbers are all still in the -summary.json sidecar.
        for r in payload['results']:
            for key in ('prose_records', 'chat_records', 'period_probes', 'training_curve'):
                r.pop(key, None)
            gen = r.get('generation') or {}
            for key in ('greedy_samples', 'sampled_samples'):
                gen.pop(key, None)
        payload['slim'] = True
        payload['notes'] = [
            *payload.get('notes', []),
            'SLIM: per-document records and generated texts were dropped. '
            '--collect cannot paired-bootstrap this file and --render-report '
            'cannot fully re-render it. Re-run without --slim to restore.',
        ]
        atomic_json(out_json, payload)

    print(f'\nJSON (for agents): {out_json}')
    print(f'Summary (small):   {out_json.with_name(out_json.stem + "-summary.json")}')
    print(f'Markdown (humans): {out_md}')


if __name__ == '__main__':
    main()
