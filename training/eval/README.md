# eval/ - Evaluation harness

One entry point that loads each model **once** and computes every metric:

```bash
python -m eval                                   # latest checkpoint in ./checkpoints
python -m eval path/to/checkpoint-12345          # one checkpoint
python -m eval autoresearch autoresearch2        # every final/ export in those trees
python -m eval Vintage1 --gen-mode sample        # cheaper generation pass
python -m eval --render-report results.json      # re-render Markdown only
```

## Files

| file | role |
|---|---|
| `prompts.py` | Big prompt list and all probe/logic/trap constants |
| `helpers.py` | device/dtype, checkpoint & tokenizer discovery, model load/free, training-lineage sniffing, data loading |
| `metrics.py` | ALL metric math. New metrics go here. |
| `report.py`  | JSON -> Markdown rendering. Tweak independently of metric code. |
| `__main__.py` | CLI orchestration, caching, rankings |

## What runs per checkpoint

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
6. **Logic forced choice + anachronism traps**.
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

## WARNING: the two held-out sets are NOT interchangeable

There are two prose held-out files, one per corpus. **Each is contaminated relative to
its own corpus** — pick the one that does NOT match the model's training data.

| file | clean for | contaminated for |
|---|---|---|
| `heldout-Sprocket-n-Say.jsonl` | nothing measured clean | Sprocket-n-Say (20/20 probed, see REPORT-tokenizer-tv3-vs-tv4.md) **and** Piston-n-Prose (152/153) |
| `heldout-Piston-n-Prose.jsonl` | Piston-n-Prose models (0/200) | untested against Sprocket-n-Say |

So: score a **Piston-n-Prose** model on `heldout-Piston-n-Prose.jsonl`. A
Sprocket-n-Say model has no verified-clean prose set — `heldout-Sprocket-n-Say.jsonl`
was measured 20/20 inside that corpus, so its historical `prose_bpb` numbers carry a
memorization component. They remain mutually comparable (every arm was inflated the
same way) but are not absolute.

`eval_data/chat_sample.jsonl` was checked against Piston-n-Prose and is **clean
(0/199)**, so `chat_bpb` is valid there.

**Never compare a score on one prose file against a score on the other.** They are
different text of different difficulty — the same model scores ~1.28 bits/byte on
`heldout-Sprocket-n-Say` and ~1.10 on `heldout-Piston-n-Prose`. That gap is the file,
not the model.

`BPB_LADDER` and every `measured` row in `REFERENCE_LADDER` are anchored on
`heldout-Sprocket-n-Say.jsonl`, which is why it remains the CLI default. Adding a
Piston-trained model to that ladder would mix two eval sets.

Caveat on `heldout-Piston-n-Prose.jsonl`: it is drawn from the Piston-n-Prose
validation shards, so it is in-distribution for models trained on that corpus and
out-of-distribution for older ones. It is clean, not neutral.

Full account: `research/REPORT-piston-v5-first-1h-run.md`.

## The reference ladder

`REFERENCE_LADDER` in `metrics.py` drives the "Where it sits" figure. Rows tagged
`measured` are real models scored through THIS eval on the SAME held-out file, so the
comparison is exact; `synthetic` rows are qualitative signposts and `historical` rows
are past measurements whose checkpoint is no longer on disk. A model that is itself on
the ladder has its own row suppressed (detected by exact bits/byte match).

To add a model: `python -m eval MODELS/<name> --force`, then paste its `prose_bpb`.
If `eval_data/heldout-Sprocket-n-Say.jsonl` ever changes, EVERY `measured` row must be re-run or the
ladder silently mixes two different held-out sets.

`REFERENCE_LADDER` (display) is separate from `BPB_LADDER` (bake-score anchors) on
purpose: editing the display ladder must never silently move everyone's bake score.

## Generation-regime metrics (added 2026-08-23)

BPB measures the HEAD of the distribution. Bulk generation lives in the TAIL, and
a model can compress beautifully while producing unusable text at t=1.2. These
close that gap:

| in `summary` | what it catches |
|---|---|
| `sampled_back_matter_rate` | index / catalogue / table-of-contents output. Lexically DIVERSE, so `distinct_2`, `echo_rate` and loop detection are ALL blind to it. Calibrated on 49,413 real completions ([tiny-vintage-completions](https://huggingface.co/datasets/croqaz/tiny-vintage-completions)): 48.7% recall on the worst tail, **0.00% false positives** on 17,085 clean ones. Precise, not exhaustive -- 0% means "none detected". |
| `sampled_self_bleu_4` | mode collapse ACROSS completions. `distinct_n` looks inside ONE completion, so 500 near-identical completions each score as perfectly diverse. This looks between them. |
| `sampled_unusable_rate` | degenerate OR back matter -- the closest thing here to the keep-rate of a synth-data filter. |
| `sweep_t<T>_*` | the same numbers per temperature (`--temp-sweep 0.8,1.0,1.2`). |
| `bake_is_partial`, `bake_components_missing`, `bake_weight_covered` | `bake_score()` renormalises over PRESENT components, so `--gen-mode none` silently produces a score that ignores hygiene. These make that visible. |

### Flags

```bash
python -m eval MODEL --temp-sweep 0.8,1.0,1.2   # tail behaviour; +1 sampled pass per temp
python -m eval MODEL --seed-set cold            # 26 bare function-word openers
python -m eval MODEL --seed-set curated         # the original 46 topical stems
python -m eval MODEL --gen-tokens 512           # longer than the 256 default
python -m eval MODEL --load-8bit                # int8: fits a 13B on a 16GB card
```

Defaults changed: `--gen-tokens` 120 -> **256** and `--seed-set` -> **both** (72
prompts). Both were chosen from a real 49,413-completion bulk run
([tiny-vintage-completions](https://huggingface.co/datasets/croqaz/tiny-vintage-completions)) whose generations had median 240 tokens
(p90 727) and 36,753 distinct two-word openers, mostly bare function words. The old 46 curated topical stems handed the model a
subject, which is an easier test than production use.

Cost on a 75M model: full battery with a 3-point temperature sweep = **79s**.
Without the sweep it is well under a minute.

## Reading the JSON cold (self-description)

Every results JSON describes itself, so an agent on another machine with no
access to this repo's history can answer "which number do I quote, and which way
is good?" without reading the source:

| top-level key | what it gives you |
|---|---|
| `primary_metric` | `"prose_bpb"` -- the one number to rank on |
| `how_to_read` | the three rules that prevent WRONG conclusions |
| `metric_guide` | per summary key: `desc`, `unit`, `direction` (`lower_is_better` / `higher_is_better` / `neutral`), `comparable_across`, and a `caveat` where a naive reading misleads |

The three rules, because they matter more than anything else here:

1. **Rank on `prose_bpb`** (lower is better). Byte-normalised, fixed 1024-token
   window -- the only headline valid across tokenizers, context lengths and sizes.
2. **Never rank on `final_eval_loss`** across models trained on different data.
   It uses each run's OWN validation split. Its guide entry says so.
3. **Check `bake_is_partial` before comparing `bake_score`.** A partial score was
   renormalised over only the measured components and is inflated.

`python -m eval --audit-guide results.json` fails if any summary key lacks a
guide entry. A metric nobody can interpret is worse than no metric, so adding one
without documenting it is a bug, not a warning.

## File sizes

A full result JSON is ~230 KB / ~6,600 lines, of which about 85% is
`generation.*_samples` and `*_records`. That bulk is not decoration: `--collect`
needs the per-document records for PAIRED bootstrap, and `--render-report` needs
the generated texts. Two sidecars exist so you never have to open it:

* `eval-<slug>-summary.json` -- flat summary per model, ~2 KB. Written always.
* `eval_results/summaries.jsonl` -- one line per model per run, append-only;
  greppable and diffable over time.
* `--slim` drops the bulk from the main JSON (~85% smaller) when you know you
  will not need paired bootstrap or a full re-render.

## Adding a new metric (vowel counts, Latin-letter counts, entropy, ...)

1. Compute it in `metrics.py` (a pure function; text stats belong inside or
   next to `text_stats()`).
2. Surface the headline number in `finalize_result()` in `__main__.py`
   (one stable snake_case key in `summary`).
3. Print it in `report.py`.
