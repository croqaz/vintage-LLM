# eval/ - merged evaluation harness

One entry point that loads each model **once** and computes every metric the
three legacy scripts (`evaluate.py`, `evaluate2.py`, `evaluate3.py`) used to
compute separately:

```bash
python -m eval                                   # latest checkpoint in ./checkpoints
python -m eval path/to/checkpoint-22944          # one checkpoint
python -m eval autoresearch autoresearch2        # every final/ export in those trees
python -m eval Vintage1 --gen-mode sample        # cheaper generation pass
python -m eval --render-report results.json      # re-render Markdown only
```

## Files

| file | role |
|---|---|
| `prompts.py` | THE single merged prompt list (eval1 + eval2 + eval3 prompts) and all probe/logic/trap constants |
| `helpers.py` | device/dtype, checkpoint & tokenizer discovery, model load/free, training-lineage sniffing, data loading |
| `metrics.py` | ALL metric math. New metrics go here. |
| `report.py` | JSON -> Markdown rendering. Tweak independently of metric code. |
| `__main__.py` | CLI orchestration, caching, rankings |

## What runs per checkpoint (one model load)

1. **Model info + training lineage** - architecture, params, disk size,
   base-vs-SFT recipe, LR, context length, tokens-seen estimate (from
   `total_flos`), wall-clock budget bucket.
2. **Period fidelity on fixed probe sentences** - historical vs modern
   perplexity/BPB and the modern/historical ratio.
3. **Diachronic word-sense separation** - cosine of the same shifted word in a
   period vs a modern sentence.
4. **Held-out prose BPB** - per-document records with A/B splits and early/late
   context diagnostics (the eval3 code path; also feeds ranking).
5. **Conditional chat-target BPB** (offset-aligned; replaces both old chat
   implementations).
6. **Logic forced choice + anachronism traps** (from evaluate2).
7. **Generation over the merged prompt battery** - generated once per
   decoding mode (`--gen-mode both|greedy|sample|none`, default `both`);
   distinct-1/2/3, echo rate, longest loop, punctuation issues per 100 words,
   prompt-copy rate, degenerate flag - computed on exactly those texts.

The **bake score** is then computed once from these same numbers: held-out BPB
(0.50) + logic accuracy (0.25) + chat readiness (0.15) + hygiene (0.10).

## Outputs

By default the results are written **next to what you point the eval at**:
pointing at `llama-77/final/` writes `llama-77/final/eval-llama77M.json` +
`.md` (the file name is a model slug from architecture + parameter count,
matching `base_train.make_model_id()`); pointing at a folder of checkpoints
writes one combined `eval-<folder>.json` in that folder. Override with
`--output PATH` (falls back to `--results-dir` if the target dir is not
writable).

- `<out>.json` - machine-readable payload. `results[i].summary` is a FLAT dict
  with stable key names for agents to grep across runs:
  `prose_bpb`, `chat_bpb`, `logic_acc`, `trap_mean_shock`,
  `probe_modern_over_historical_ratio`, `sense_shift_mean_cosine`,
  `sampled_mean_distinct_2`, `greedy_mean_loop_words`, `bake_score`,
  `verdict_tier`, ... Per-document bit/byte records are retained so paired
  bootstrap comparisons never need to reload models.
- `<out>.md` - human-readable report rendered by `report.py`.
- Rankings: `rankings.prose` (paired bootstrap vs leader, equivalence ties)
  and `rankings.bake`.

Re-runs reuse cached per-checkpoint results when the model fingerprint and
settings hash are unchanged (`--force` recomputes). Interrupted sweeps resume
where they stopped because the JSON is written incrementally.

## Cross-run comparison (`--collect`)

Models evaluated in SEPARATE invocations can be compared without reloading anything,
because each JSON retains its per-document bit/byte records:

```bash
python -m eval --collect autoresearch autoresearch2 -o eval_results/all.json
```

Produces one ranked JSON + Markdown with:

- `rankings.prose` - **paired** bootstrap vs the leader. The paired CI is ~30x tighter
  than the marginal CIs because the same documents are resampled for both models;
  marginal CIs overlap for models that are in fact clearly separated.
- `rankings.bake`
- `curve_comparison` - every run's eval loss interpolated onto ONE common step grid.

> **`curve_comparison` compares eval loss, which is computed on each run's OWN
> validation split.** Runs that trained on different datasets are therefore NOT
> comparable on this table, only on `prose_bpb` (same held-out file for everyone).
> `eval_steps` is also in MINUTES, which is why interpolation is mandatory even for
> runs on identical data.

## Numbers that used to need a throwaway script

| in `summary` | replaces |
|---|---|
| `tokenizer_bytes_per_token`, `tokenizer_vocab_size` | ad-hoc AutoTokenizer compression measurement. HIGHER bytes/token = more text per token budget. Check this before blaming a tokenizer for a quality gap. |
| `final_eval_loss`, `final_eval_ppl`, `final_eval_step`, `final_train_loss` | hand-parsing `trainer_state.json` |
| `grad_norm_max` / `_mean` / `_min` / `_nonfinite` | ditto - `grad_norm_nonfinite > 0` is the instability signal |
| `probe_frac_low_confidence`, `probe_mean_entropy_nats`, `probe_mean_token_prob` | were buried in `period_probes.overall` |
| `result.training_curve.eval_curve` | the full curve, retained for step-matched comparison |
| `lineage.learning_rate`, `optim`, `lr_scheduler`, `warmup_steps`/`stable_steps`/`decay_steps`, `max_steps`, `tokens_per_step`, `seed` | grepping `training_config.toml` by hand. Base runs used to report none of this -- only SFT configs were parsed. |

### Lineage self-checks (`lineage.warnings`, rendered as **Check:** lines)

- **wall clock vs `max_train_minutes`** - the budget is PER PROCESS and restarts on
  resume, so an interrupted run legitimately exceeds it. The report now shows the
  per-segment breakdown (`181 min = 61.4 + 120.1 over 2 segments (resumed)`) instead
  of a bare total that looks like a bug.
- **tokens seen, two ways** - `total_flos / (6 * non-embedding params)` versus
  `global_step * tokens_per_step`. Disagreement >2% means the trainer counters and the
  FLOP counter cover different spans (usually a restart without `--resume`).
- **trained sequence length vs `config.json` `max_position_embeddings`** - the latter is
  the architecture window, not what the run actually trained on.

`trainer_state.json` / `train.log` are looked up in the checkpoint directory, and one
level up ONLY when the target is a `final/` or `checkpoint-*/` export -- pointing the
eval at a run directory used to read the sibling run's files from the parent tree.

## The reference ladder

`REFERENCE_LADDER` in `metrics.py` drives the "Where it sits" figure. Rows tagged
`measured` are real models scored through THIS eval on the SAME held-out file, so the
comparison is exact; `synthetic` rows are qualitative signposts and `historical` rows
are past measurements whose checkpoint is no longer on disk. A model that is itself on
the ladder has its own row suppressed (detected by exact bits/byte match).

To add a model: `python -m eval MODELS/<name> --force`, then paste its `prose_bpb`.
If `eval_data/heldout.jsonl` ever changes, EVERY `measured` row must be re-run or the
ladder silently mixes two different held-out sets.

`REFERENCE_LADDER` (display) is separate from `BPB_LADDER` (bake-score anchors) on
purpose: editing the display ladder must never silently move everyone's bake score.

## Adding a new metric (vowel counts, Latin-letter counts, entropy, ...)

1. Compute it in `metrics.py` (a pure function; text stats belong inside or
   next to `text_stats()`).
2. Surface the headline number in `finalize_result()` in `__main__.py`
   (one stable snake_case key in `summary`).
3. Print it in `report.py`.

Legacy scripts are kept at the repo root for reference but are superseded by
this package.
