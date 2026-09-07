# Evaluator

Loads each model once; scores prose, chat targets, fixed-choice/sentence probes,
embedding diagnostics and generated continuations.

```bash
python -m eval path/to/final --out /tmp/check.json
python -m eval --collect autoresearch llama-75 -o /tmp/comparison.json  # no model loads
python -m eval --render-report /tmp/check.json                       # writes Markdown only
python -m eval /tmp/check.json --audit-guide
python3 -m unittest discover -s eval/tests -v
```

Default output is `eval-<architecture><size>.json` beside the requested target,
plus a matching Markdown report. All aggregate measurements live in
`results[].summary`; per-document records and generated texts stay in detail
blocks. The embedded `metric_guide` describes every summary key.
Markdown titles show model type, parameter count, depth, width, attention/KV heads,
context limit, vocabulary size and embedding tying; the checkpoint path stays in the body.

During evaluation, `.partial.json` retains in-progress measurements and `.prev.json`
backs up an existing result. Every generation `gN/total` update saves completed
continuations and their statistics first, including the active temperature sweep.
Both files are removed after successful completion; failures and interruptions retain them.

## Names and units

| Key or family | What it measures |
|---|---|
| `prose_bpb`, `chat_target_bpb` | Total negative log-likelihood in **bits / scored UTF-8 bytes**, not token perplexity. Chat scores target tokens given context. |
| `prose_uniform_token_bpb` | log2(model vocabulary size) × scored tokens / scored bytes. Equal token probabilities on the prose scoring pool, not an evaluated untrained network. |
| `prose_docs_scored`, `prose_docs_excluded`, `prose_scored_bytes` | Actual aggregate coverage; analogous `chat_target_*` keys. `*_docs_requested` counts loaded inputs; inputs skipped before recording are not in `*_docs_excluded`. |
| `prose_truncated_docs`, `chat_input_truncated_docs`, `chat_target_truncated_docs` | Records clipped to a context budget, including whether chat targets were clipped. |
| `prose_bpb_ci_low/high/confidence`, `prose_delta_*_bpb` | Document-bootstrap uncertainty and candidate-minus-group-leader BPB differences. |
| `logic_accuracy`, `logic_margin_bpb`, `logic_items_scored` | Good continuation has lower BPB than bad (ties incorrect); mean bad-minus-good BPB; denominator. Category results use `logic_category_*`. |
| `trap_mean_delta_bpb`, `trap_min_delta_bpb`, `trap_nonpositive_pairs` | Modern-minus-period phrase BPB and count at or below zero; **not leakage evidence**. |
| `probe_{historical,modern,overall}_{bpb,ppl}` | Fixed-sentence BPB and token perplexity. Other token statistics use `probe_overall_*`. |
| `{greedy,sampled,sweep_t1.2}_*` | Identical suffixes for each generation mode/temperature: `surface_failure_rate`, `mean_distinct_1/2/3`, `mean_echo_rate`, `mean_loop_words`, `shared_4gram_fraction`, etc. |
| `*_mean_prompt_word_overlap_rate` | Fraction of generated words longer than 3 letters also in the prompt; overlap, not proof of copying. |
| `trainer_eval_loss_nats_per_token`, `trainer_train_loss_nats_per_token` | Logged trainer losses; `trainer_eval_ppl` is exp(eval loss). These are not this evaluator's held-out BPB. |
| `embedding_*`, `sense_shift_*_cosine`, `grad_norm_*` | Descriptive vector/training-log statistics, without an established good/bad direction. |
| `bake_score`, `bake_status`, `bake_weight_covered` | Experimental composite and its coverage; not a model-readiness verdict. |

Fractions are 0–1 in JSON; Markdown may display percentages. Missing/non-finite
numbers serialize as `null`, never as zero. Per-document totals, all generation
aggregates, comparison intervals and probe BPBs are also in the flat summaries.

## Surface flags

`surface_failure_rate` is the union of short-text/repetition and back-matter-like
formatting flags. Each saved sample has `surface_failure_reasons`; per-reason
rates are exposed too (they overlap, so do not add them).

- `short_text`: fewer than 8 regex English words.
- `low_bigram_diversity`: distinct-2 below 0.55.
- `local_repetition`: over 30% of words repeat within the previous 4 words.
- `repeat_loop`: a periodic run of at least 12 words.
- `back_matter_like`: digit/capitalization/short-sentence/newline heuristics.

Valid short text can fail; diverse nonsense can pass. `shared_4gram_fraction`
measures unique word 4-grams shared across completions, not coherence.
Generation comparisons need matching prompts, sample count, length budget,
template, seed and decoding settings.

## Comparisons

Prose is byte-weighted over finite, positive-byte records. Bootstrap uses that
same pool. Comparisons require matching document coverage, scored spans and
scoring/window/precision protocol; `--collect` retains per-result settings and
keeps differing protocols in separate groups. Records hash the actual scored
text. Different tokenizers may score different prefixes at the same token limit, so byte
normalization alone does not make their results interchangeable.

`comparison_to_leader` is `equivalent` only if the **whole** paired interval is
inside ±`equivalence_margin_bpb`; wholly above/below is `worse`/`better`.
Otherwise it is `inconclusive`, or `unavailable` when evidence is missing.
The lowest observed BPB within each group is `leader`.
Defaults: 95% confidence; margin 0.1% of leader BPB
(`--confidence .95 --equivalence .001`). At least two matching documents are
needed for a bootstrap interval. These intervals estimate document-sampling
uncertainty, not training-seed uncertainty or a multiple-comparison correction.
`bootstrap_fraction_lower_than_leader` is a replicate fraction, **not** a
posterior probability or p-value.

Trainer-loss curves are recorded per result. Matching optimizer steps cannot
verify matching validation data, tokenizer or loss reduction. Early/late prose BPB compares different token
positions, not a controlled long-context test. Chat-target loss does not predict
fine-tuning readiness; fixed phrase/probe preferences do not establish a cutoff.

The composite uses fixed transforms and weights:
prose .50, logic .25, chat-target .15, generation .10.
`bake_status` is `complete`, `partial`, or `unscored`.
Partial scores renormalize measured weights and are excluded from full-score
rankings; even complete scores are ranked only within matching recorded protocols.

## Running and extending

Summaries are derived from measurements. Rendering reads them without changing
the JSON; collection recomputes comparisons from saved document records.

Cache identity includes evaluation code/prompt contents, suite/template flags,
resolved runtime settings, tokenizer configuration and weight file size/mtime.
Use `--force` if weights were changed while preserving those file stats.
`--slim` removes records needed for later paired comparisons.

Implementation: `metrics.py` (math), `measurements.py` (flat summaries),
`comparison.py` (protocol identities), `metric_guide.py` (units),
`report.py` (Markdown), `__main__.py` (CLI).
