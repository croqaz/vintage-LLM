"""Checkpoint evaluation CLI: python -m eval [targets ...].

Loads each model once. Writes measurements to a full JSON and a Markdown report.
Collection resamples saved document records; report rendering does not score models.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

from . import SCHEMA_VERSION
from . import metrics as M
from . import report as R
from .comparison import chat_comparison_key, composite_comparison_key, identity, prose_comparison_key
from .helpers import (
    DEFAULT_HELDOUT,
    DEFAULT_RESULTS_DIR,
    EVAL_DATA,
    LADDER_CALIBRATION_HELDOUT,
    atomic_json,
    banner,
    checkpoint_lineage,
    die,
    file_hash,
    fmt,
    free_model,
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
    weight_bytes,
)
from .measurements import summarize_result
from .nanochat_models import is_nanochat_checkpoint
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

# ============================================================================
# Settings identity for the cache
# ============================================================================


# Modules whose contents can change a measured number. Rendering, documentation and
# the standalone side scripts are deliberately excluded: hashing them would invalidate
# every cached result AND split comparison groups on a cosmetic edit, because this
# digest is part of comparison.prose_comparison_key.
SCORING_MODULES = ('helpers.py', 'measurements.py', 'metrics.py', 'prompts.py')


def settings_fingerprint(args, heldout_hash: str, chat_hash: str | None) -> dict:
    """Identity of everything except model weights that affects the numbers."""
    return {
        'schema': SCHEMA_VERSION,
        'scoring_code_sha256': identity({name: file_hash(Path(__file__).parent / name) for name in SCORING_MODULES}),
        'prompts_sha256': identity(
            {
                'generation': SEED_SETS[args.seed_set](),
                'logic': LOGIC_ITEMS,
                'traps': TRAP_PAIRS,
                'probes': PROBE_SENTENCES,
                'probe_labels': PROBE_LABELS,
                'words': HISTORICAL_WORDS,
                'historical': HISTORICAL_CONTEXTS,
                'modern': MODERN_CONTEXTS,
            }
        ),
        'chat_template_enabled': args.chat,
        'skip_probes': args.skip_probes,
        'skip_embeddings': args.skip_embeddings,
        'device': str(select_device(args.device)),
        'resolved_dtype': str(select_dtype(args.dtype, select_device(args.device))),
        'torch': torch.__version__,
        'transformers': __import__('transformers').__version__,
        'threads': torch.get_num_threads(),
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
    """Temperature key with at least one decimal digit (for example, sweep_t1.0)."""
    label = f'{temp:g}'
    return label if '.' in label else f'{label}.0'


def evaluate_checkpoint(checkpoint: Path, tok_dir: Path, device, dtype, args, prompts, data, snapshot=None) -> dict:
    started = time.time()

    def save(stage: str) -> None:
        """Persist completed suites and generation batches for interrupted evaluations."""
        if snapshot is not None:
            try:
                snapshot(stage, result)
            except Exception as exc:  # a safety net must never break the run
                print(f'  (snapshot after {stage} failed: {exc})')

    print(f'  loading model ({tok_dir}) ...', flush=True)
    model, tokenizer = load_model_and_tokenizer(checkpoint, tok_dir, device, dtype, load_8bit=args.load_8bit)
    # nanochat attends over the whole row with no padding mask, so a left-padded
    # batch would score pad tokens as content. One row at a time is the only
    # correct setting; the wrapper raises rather than let it slide.
    gen_batch_size = args.generation_batch_size
    if is_nanochat_checkpoint(checkpoint):
        if gen_batch_size != 1:
            print(f'  nanochat: generation batch {gen_batch_size} -> 1 (no padding-mask support)')
        gen_batch_size = 1
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
        disk_mb = weight_bytes(checkpoint) / 1e6
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

        # ---- fixed-sentence likelihood on fixed probes ---------------------------------
        if not args.skip_probes:
            banner('SUITE: FIXED-SENTENCE LIKELIHOOD ON FIXED PROBE SENTENCES')
            result['period_probes'] = M.score_probe_sentences(tokenizer, model, prompts['probe_sentences'], prompts['probe_labels'])
            pp = result['period_probes']
            print(
                f'  hist ppl {pp["historical"]["ppl"]:.2f} | modern ppl '
                f'{pp["modern"]["ppl"]:.2f} | ratio {pp.get("modern_historical_ppl_ratio", float("nan")):.2f}'
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
                print(f'  mean shift similarity: {fmt(result["embeddings"].get("shift_mean_cosine"), 3)}')
            except Exception as exc:
                result['embeddings'] = {'unavailable': f'{type(exc).__name__}: {exc}'}
                print(f'  UNAVAILABLE for this architecture: {type(exc).__name__}: {exc}')

        # ---- held-out prose --------------------------------------------------
        if data['heldout']:
            banner('SUITE: HELD-OUT PROSE (per-document records)')
            result['prose_records'], result['prose_docs_skipped'] = M.score_prose_records(
                tokenizer, model, data['heldout'], args.max_tokens
            )
            print()
            save('prose')

        # ---- chat targets ----------------------------------------------------
        if data['chat']:
            if not tokenizer.is_fast:
                print('  (chat skipped: requires a fast tokenizer)')
            else:
                banner('SUITE: CONDITIONAL CHAT-TARGET LOSS')
                result['chat_records'], result['chat_target_docs_skipped'] = M.score_chat_records(
                    tokenizer, model, data['chat'], args.chat_max_tokens
                )
                print()
                save('chat')

        # ---- tokenizer and training-log diagnostics ----------------------
        if data['heldout']:
            result['tokenizer_stats'] = M.tokenizer_stats(tokenizer, data['heldout'], args.max_tokens)
        result['training_curve'] = training_curve(checkpoint)

        # ---- logic + traps ---------------------------------------------------
        banner('SUITE: LOGIC FORCED CHOICE AND ANACHRONISM TRAPS')
        result['logic'] = M.run_logic(tokenizer, model, prompts['logic_items'])
        result['traps'] = M.run_traps(tokenizer, model, prompts['trap_pairs'])
        print(
            f'  logic acc {result["logic"]["accuracy"]:.3f} (margin {result["logic"]["margin_bpb"]:+.3f} bits/byte) | '
            f'trap mean delta {result["traps"]["mean_delta_bpb"]:+.3f} bits/byte'
        )
        save('logic_traps')

        # ---- THE one generation pass over the MERGED prompt list --------------
        if args.generation_modes:
            banner('SUITE: GENERATION OVER MERGED PROMPT BATTERY')
            print(f'  {len(prompts["generation"])} unique prompts x modes {args.generation_modes}, {args.gen_tokens} new tokens each')
            gen_block = {'temperature': args.temperature, 'top_p': args.top_p, 'top_k': args.top_k}
            gen_block['chat_template_sha256'] = identity(tokenizer.chat_template) if args.chat and tokenizer.chat_template else None
            result['generation'] = gen_block

            def save_generation_batch(mode: str, rows: list[dict]) -> None:
                prefix = 'sampled' if mode == 'sample' else 'greedy'
                gen_block[f'{prefix}_samples'] = rows
                gen_block[f'{prefix}_summary'] = M.summarize_generations(rows)
                save(f'generation_{prefix}_g{len(rows)}/{len(prompts["generation"])}')

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
                batch_size=gen_batch_size,
                chat=args.chat,
                on_batch=save_generation_batch,
            )
            for mode, prefix in (('greedy', 'greedy'), ('sample', 'sampled')):
                rows = modes_out.get(mode, [])
                gen_block[f'{prefix}_samples'] = rows
                gen_block[f'{prefix}_summary'] = M.summarize_generations(rows)
            save('generation')

            # ---- optional temperature sweep ---------------------------------
            # Each additional temperature adds a sampled pass over the same prompts.
            if args.temp_sweep and 'sample' in args.generation_modes:
                sweep = {}
                gen_block['temperature_sweep'] = sweep
                for temp in args.temp_sweep:
                    label = _temp_label(temp)
                    if abs(temp - args.temperature) < 1e-9 and gen_block.get('sampled_summary'):
                        sweep[label] = gen_block['sampled_summary']
                        continue
                    print(f'  temperature sweep: t={temp:g}', flush=True)

                    def save_sweep_batch(mode: str, rows: list[dict], _label=label, _temp=temp) -> None:
                        sweep[_label] = M.summarize_generations(rows)
                        # Keep the active sweep's text in the snapshot; completed
                        # sweeps retain aggregate measurements in the full report.
                        gen_block['temperature_sweep_in_progress'] = {'temperature': _temp, 'samples': rows}
                        save(f'sweep_t{_label}_g{len(rows)}/{len(prompts["generation"])}')

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
                        batch_size=gen_batch_size,
                        chat=args.chat,
                        on_batch=save_sweep_batch,
                    )
                    sweep[label] = M.summarize_generations(swept.get('sample', []))
                    gen_block.pop('temperature_sweep_in_progress', None)
                    save(f'sweep_t{label}')
    finally:
        free_model(model, device)
        print(f'  model freed ({time.time() - started:.0f}s elapsed)', flush=True)

    # ---- derived numbers (pure math over what we just collected) ------------
    summarize_result(result)
    # Put the headline numbers at the TOP of the object. A human or an agent
    # opening a 200 KB result should hit `summary` on line 3, not after 4,000
    # lines of per-document records.
    lead = ('label', 'checkpoint', 'summary', 'bake_score', 'points')
    return {**{k: result[k] for k in lead if k in result}, **result}


# ============================================================================
# Rankings (paired bootstrap on the retained per-document records)
# ============================================================================


def _rank_records(results: list[dict], args, *, block: str, key_fn, prefix: str) -> list[dict]:
    """Rank one per-document BPB family within matching protocol/coverage groups.

    Shared by prose and chat targets: both retain per-document records, so both
    get a marginal bootstrap CI and a PAIRED interval against their group leader.
    """
    if args.bootstrap < 1 or not 0 < args.confidence < 1 or not M.is_finite(args.equivalence) or args.equivalence < 0:
        raise ValueError('positive bootstrap draws, 0 < confidence < 1 and nonnegative equivalence required')
    comparison_key = prefix + '_comparison'
    bpb_key = prefix + '_bpb'
    groups = {}
    for result in results:
        if block in result:
            result.pop(comparison_key, None)
            summarize_result(result)
        records = M.finite_records(result.get(block) or [])
        if not records:
            continue
        if len({r['id'] for r in records}) != len(records):
            # Conservative, but never silent: an unrankable result says why.
            result[comparison_key] = {
                'comparison_to_leader': 'unavailable',
                'comparison_unavailable_reason': 'duplicate document IDs cannot be paired unambiguously',
            }
            summarize_result(result)
            print(f'  note: {result["label"]} is not ranked on {prefix}: duplicate document IDs')
            continue
        groups.setdefault(key_fn(result), []).append(result)
    rows = []
    for group, eligible in sorted(groups.items()):
        eligible.sort(key=lambda r: r['summary'][bpb_key])
        leader = eligible[0]
        epsilon = leader['summary'][bpb_key] * args.equivalence
        for rank, r in enumerate(eligible, 1):
            ci_lo, ci_hi = M.bootstrap_ci(r[block], args.bootstrap, args.confidence, args.seed)
            # One common set of resample indices for every candidate in the group, so
            # the deltas are mutually consistent and the order cannot flip on the seed.
            pair = M.paired_bootstrap(r[block], leader[block], args.bootstrap, args.confidence, args.seed)
            is_leader = r is leader
            comparison = 'leader' if is_leader else M.classify_bpb_interval(pair['low'], pair['high'], epsilon)
            # A model compared with itself has no fraction to report; null, not zero.
            fraction = None if is_leader else pair['bootstrap_fraction_lower']
            rows.append(
                {
                    'rank': rank,
                    'metric': bpb_key,
                    'comparison_group': group,
                    'leader_label': leader['label'],
                    'label': r['label'],
                    'checkpoint': r['checkpoint'],
                    'bpb': r['summary'][bpb_key],
                    'ci_low': ci_lo,
                    'ci_high': ci_hi,
                    'delta_vs_leader': pair['delta'],
                    'delta_ci_low': pair['low'],
                    'delta_ci_high': pair['high'],
                    'bootstrap_fraction_lower_than_leader': fraction,
                    'paired_docs': pair['paired_docs'],
                    'equivalence_margin_bpb': epsilon,
                    'comparison_to_leader': comparison,
                    'equivalent_to_leader': comparison in ('leader', 'equivalent'),
                    'chat_target_bpb': r['summary'].get('chat_target_bpb'),
                    'bake_score': r.get('bake_score'),
                }
            )
            r[comparison_key] = {
                'bpb_ci_low': ci_lo,
                'bpb_ci_high': ci_hi,
                'bpb_ci_confidence': args.confidence,
                'comparison_group': group,
                'comparison_leader': leader['label'],
                'comparison_to_leader': comparison,
                'delta_vs_leader_bpb': pair['delta'],
                'delta_ci_low_bpb': pair['low'],
                'delta_ci_high_bpb': pair['high'],
                'equivalence_margin_bpb': epsilon,
                'paired_docs': pair['paired_docs'],
                'bootstrap_fraction_lower_than_leader': fraction,
            }
            summarize_result(r)
    return rows


def rank_by_prose(results: list[dict], args) -> list[dict]:
    return _rank_records(results, args, block='prose_records', key_fn=prose_comparison_key, prefix='prose')


def rank_by_chat(results: list[dict], args) -> list[dict]:
    """Chat targets carry per-document records too, so they get the same treatment."""
    return _rank_records(results, args, block='chat_records', key_fn=chat_comparison_key, prefix='chat_target')


def rank_by_composite(results: list[dict]) -> list[dict]:
    """Only complete scores, ranked within the same recorded evaluation protocol."""
    groups = {}
    for r in results:
        s = r.get('summary') or {}
        if (
            s.get('bake_status') == 'complete'
            and M.is_finite(r.get('bake_score'))
            and r.get('evaluation_settings')
            and r.get('prose_records')
            and r.get('chat_records')
        ):
            groups.setdefault(composite_comparison_key(r), []).append(r)
    return [
        {'rank': rank, 'label': r['label'], 'bake_score': r['bake_score'], 'comparison_group': group}
        for group, members in sorted(groups.items())
        for rank, r in enumerate(sorted(members, key=lambda r: -r['bake_score']), 1)
    ]


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
            r['judge'] = {
                'checkpoint': str(judge_dir.resolve()),
                'tokenizer': str(jtok_dir),
                'fingerprint': model_fingerprint(judge_dir, jtok_dir),
                'max_tokens': 320,
            }
            texts = [s['continuation'] for s in (r.get('generation') or {}).get('sampled_samples', [])]
            texts = [t for t in texts if len(t.split()) >= 20]
            if not texts:
                continue
            gen_bpb = M.bits_per_byte_of_texts(judge_tok, judge_model, texts, max_tokens=320)
            dev = abs(gen_bpb - real_bpb)
            r['judge']['scores'] = dict(reference_bpb=real_bpb, generated_bpb=gen_bpb, absolute_bpb_difference=dev)
            summarize_result(r)
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
    seen: dict[tuple, dict] = {}
    for root in roots:
        root = root.resolve()
        files = sorted(root.rglob('eval-*.json')) if root.is_dir() else [root]
        files = [f for f in files if not f.stem.endswith(('.partial', '.prev'))]
        for f in files:
            try:
                payload = read_results(f)
            except Exception as exc:
                print(f'  skipping {f}: {exc}')
                continue
            for r in payload.get('results', []):
                r.setdefault('source_file', str(f))
                # Key on the PROJECT-RELATIVE label, not the absolute checkpoint path:
                # JSONs produced on another machine or in a container carry unrelated
                # absolute paths (e.g. /root/PWD/...) and would escape de-duplication.
                model_label = r.get('model_label') or r.get('label') or r.get('checkpoint')
                key = (model_label, r.get('fingerprint'), identity([r['evaluation_settings'], r['evaluation_environment']]))
                if not model_label:
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
    results = list(seen.values())
    labels = [r.get('model_label') or r['label'] for r in results]
    for r, label in zip(results, labels):
        r['model_label'] = label
        if labels.count(label) > 1:
            r['label'] = (
                label
                + ' [eval '
                + identity([r.get('fingerprint'), r.get('evaluation_settings'), r.get('evaluation_environment')])[:8]
                + ']'
            )
    return results


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog='python -m eval',
        description='Merged evaluation for tiny vintage LLMs: info+lineage, fixed-sentence likelihood, '
        'held-out BPB, experimental composite, logic/traps, chat-target loss and surface statistics - '
        'one model load, one prompt battery.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('targets', nargs='*', type=Path, help='Checkpoint dir, folder of checkpoints, or experiment tree.')
    p.add_argument('--tokenizer', type=Path, default=None, help='Force one tokenizer for every checkpoint.')
    p.add_argument('--heldout', type=Path, default=DEFAULT_HELDOUT)
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
        help='Decoding modes over the prompt battery.',
    )
    p.add_argument(
        '--gen-tokens',
        type=int,
        default=256,
        help='Maximum new tokens per continuation.',
    )
    p.add_argument('--generation-batch-size', type=int, default=4)
    p.add_argument(
        '--seed-set',
        choices=('curated', 'cold', 'both'),
        default='both',
        help="Prompt battery: 'curated' topical stems, 'cold' function-word openers, or both.",
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
    p.add_argument('--skip-probes', action='store_true', help='Skip fixed-probe fixed-sentence suite.')
    p.add_argument('--skip-embeddings', action='store_true', help='Skip word-sense separation suite.')
    p.add_argument('--judge', type=str, default=None, help='Optional reference model directory for reference/generated-text BPB.')
    p.add_argument('--bootstrap', type=int, default=10_000)
    p.add_argument('--confidence', type=float, default=0.95)
    p.add_argument(
        '--equivalence',
        type=float,
        default=0.001,
        help='Relative BPB tolerance: the entire paired CI must lie within +/- this fraction of leader BPB.',
    )
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    p.add_argument('--dtype', choices=('auto', 'float32', 'float16', 'bfloat16'), default='auto')
    p.add_argument(
        '--load-8bit',
        action='store_true',
        help='Load weights with bitsandbytes int8; recorded as a separate scoring precision.',
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
        'A slim file cannot be used for paired-bootstrap --collect or a full --render-report.',
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
    args = p.parse_args(argv)
    if args.bootstrap < 1 or not 0 < args.confidence < 1 or not M.is_finite(args.equivalence) or args.equivalence < 0:
        p.error('use positive --bootstrap, 0 < --confidence < 1 and nonnegative --equivalence')
    if args.docs < 0 or args.chat_docs < 0 or args.max_tokens < 8 or args.chat_max_tokens < 8:
        p.error('document counts must be nonnegative and context budgets at least 8 tokens')
    if args.gen_tokens < 1 or args.generation_batch_size < 1:
        p.error('generation length and batch size must be positive')
    return args


def read_results(path: Path) -> dict:
    """Read a result payload using the evaluator's schema."""
    payload = json.loads(path.read_text())
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'{path}: expected evaluation schema {SCHEMA_VERSION}')
    return payload


def audit_guide(path: Path) -> int:
    """Fail loudly if any summary key in `path` lacks a metric_guide entry."""
    payload = read_results(path)
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
        payload = read_results(src)
        out_md = args.out or src.with_suffix('.md')
        out_md.write_text(R.render_report(payload), encoding='utf-8')
        print(f'report: {out_md}')
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
                'recorded protocols; collection recomputes the rankings.',
                'Paired bootstrap resamples matching scored documents within protocol groups; it estimates document-sampling uncertainty, not training-seed uncertainty.',
            ],
            'bootstrap_confidence': args.confidence,
            'primary_metric': PRIMARY_METRIC,
            'how_to_read': HOW_TO_READ,
            'metric_guide': {},
            'results': results,
            'skipped': [],
            'failures': [],
            'rankings': {
                'prose': rank_by_prose(results, args),
                'chat_target': rank_by_chat(results, args),
                'bake': rank_by_composite(results),
            },
        }

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
        heldout = load_text_items(heldout_path, args.docs) if args.docs else []
        if heldout_path != LADDER_CALIBRATION_HELDOUT.resolve():
            extra_notes.append(
                'Composite anchors were calibrated on heldout-Sprocket-n-Say.jsonl and are fixed; bake_score is not calibrated for this held-out set. bpb itself is unaffected.'
            )
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
    partial_path = out_json.with_name(out_json.stem + '.partial.json')
    prev_path = out_json.with_name(out_json.stem + '.prev.json')

    # ---- resumable cache -------------------------------------------------------
    settings = settings_fingerprint(
        args, file_hash(heldout_path) if heldout_path.exists() else None, file_hash(chat_path) if chats else None
    )
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
        'primary_metric': PRIMARY_METRIC,
        'how_to_read': HOW_TO_READ,
        'metric_guide': {},
        'results': [],
        'skipped': resolved.skipped,
        'failures': [],
    }
    # Retain a recoverable copy before writing incremental results.
    if out_json.exists():
        try:
            prev_path.write_bytes(out_json.read_bytes())
        except Exception as exc:  # insurance must never block the run
            print(f'  (could not snapshot previous results: {exc})')

    cached: dict[str, dict] = {}
    if out_json.exists() and not args.force:
        try:
            saved = read_results(out_json)
            if saved['settings'] == settings:
                cached = {r['fingerprint']: r for r in saved['results']}
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

            def _snapshot(stage: str, partial: dict, _p=partial_path, _c=checkpoint) -> None:
                # Retain the latest completed suite or batch until all outputs are saved.
                atomic_json(
                    _p,
                    {
                        'incomplete': True,
                        'stage_completed': stage,
                        'checkpoint': str(_c),
                        'note': 'In-progress measurements; removed after successful evaluation and report writing.',
                        'result': partial,
                    },
                )

            # Both protocol and checkpoint identity must match for a cache hit.
            if fingerprint in cached:
                result = cached[fingerprint]
                # Lineage files are tiny and may have changed (resumed runs);
                # refresh them even when the expensive scores come from cache.
                result['lineage'] = {
                    k: v
                    for k, v in checkpoint_lineage(checkpoint, result.get('params'), result.get('params_no_embed')).items()
                    if k != '_state_file'
                }
                result['lineage']['line'] = provenance_line(result['lineage'])
                print('  [cached]')
            else:
                result = evaluate_checkpoint(checkpoint, tok_dir, device, dtype, args, prompts, data, snapshot=_snapshot)
            result['fingerprint'] = fingerprint
            # The optional judge runs after the evaluated models are freed.
            # Never carry a previous judge pass through the model-score cache.
            result.pop('judge', None)
            result['evaluation_settings'] = settings
            result['evaluation_environment'] = payload['environment']
            result.update(prose_docs_requested=len(heldout), chat_target_docs_requested=len(chats))
            summarize_result(result)
            payload['results'].append(result)
            atomic_json(out_json, payload)  # incremental: safe to interrupt
            s = result.get('summary', {})
            covered = s.get('bake_weight_covered')
            coverage = f'{100 * covered:.0f}% weight' if M.is_finite(covered) else 'no weight'
            print(
                f'  => bake {fmt(result.get("bake_score"), 0)}/100 ({s.get("bake_status")}, {coverage})'
                f' | bpb {fmt(s.get("prose_bpb"), 4)} | logic accuracy {fmt(s.get("logic_accuracy"), 3)}'
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
        if prev_path.exists():
            out_json.write_bytes(prev_path.read_bytes())
            print(f'  restored previous results from {prev_path}; failed run produced no complete results')
        die('every checkpoint failed')

    # ---- optional judge --------------------------------------------------------
    if args.judge:
        banner('SUITE: REFERENCE-MODEL TEXT LIKELIHOOD')
        run_judge(args, payload['results'], device, dtype, [t.text for t in heldout])
        atomic_json(out_json, payload)

    # ---- rankings ----------------------------------------------------------------
    payload['rankings'] = {'prose': rank_by_prose(payload['results'], args)}
    payload['rankings']['chat_target'] = rank_by_chat(payload['results'], args)
    payload['rankings']['bake'] = rank_by_composite(payload['results'])

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

    # ---- Markdown report ------------------------------------------------------------
    report_text = R.render_report(payload)
    out_md.write_text(report_text, encoding='utf-8')
    notes = '\n'.join(f'*{n}*' for n in extra_notes)
    if notes:
        with open(out_md, 'a', encoding='utf-8') as f:
            f.write('\n' + notes + '\n')

    if args.slim:
        # Keep aggregate measurements in results[].summary; dropping raw records
        # prevents paired bootstrap and full report re-rendering.
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

    if not payload['failures']:
        partial_path.unlink(missing_ok=True)
        prev_path.unlink(missing_ok=True)

    print(f'\nJSON (for agents): {out_json}')
    print(f'Markdown (humans): {out_md}')


if __name__ == '__main__':
    main()
