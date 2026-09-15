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

## Sibling modules

Two evaluations live outside `python -m eval` because they answer questions the
main harness is not shaped for.

```bash
python -m eval.probe_memorization SUSPECT CONTROL     # did it memorise, or learn the domain?
python -m eval.chat_eval MODEL --judge BASE_MODEL     # does it hold a conversation?
```

`chat_eval` exists because everything in the main harness is likelihood on text
WE supply. `chat_target_bpb` renders a conversation and scores the reference
assistant reply token by token, so the model never emits anything: it measures
how surprised the model is by a good reply, not what it would actually say. A
model can therefore win a hyperparameter sweep on every metric here and still
be unusable.

`chat_eval` scores generated text on five axes: whether EOS fires inside the
budget and after how many tokens, whether the reply leaks role markers or
opens a turn and answers itself, mechanical constraint checks (word and
sentence counts, yes/no, list length), format hygiene (empty, unterminated),
and a vintage score. That last one reuses the judge idea: point `--judge` at
the PRE-fine-tune checkpoint and it reports the bpb of the model's own replies
minus the judge's bpb on real held-out prose. Style collapse shows up in
generation long before it shows up in likelihood.

Read the sign, not just the magnitude. **Positive** means the replies are
harder for the vintage judge than real prose, so the model has drifted out of
its pretraining distribution: the modernisation failure. **Negative** means
easier, so blander and more repetitive. On Llama-141M every fine-tune scored
negative (-0.17 to -0.38) and the untuned base scored worst (-0.60, because it
loops). Near zero is good; a large negative is flat writing, not modern
writing.

Probes live in `eval_data/chat_probes.jsonl`, tagged `length`, `closed`,
`list`, `transform`, `instruct`, `multiturn`, `open` and `fact`. A multi-turn
probe carries a `followup`, and the model's OWN first reply becomes the
context; feeding it a scripted reply would test something easier than what a
person does. Greedy by default so runs compare.

Probes get corrected when a flaw is found, so every run records the probe
file's path and sha256. `fact_capital_italy` accepted only Rome until
2026-09-15, which marked a period-correct Turin as wrong: the Kingdom of Italy
moved its capital from Turin (1861) to Florence (1865) to Rome (1871), all
inside the period our corpus covers. It now accepts all three. Numbers
published before that date used the strict version; compare `probes_sha256`
before comparing scores.

`--only-tags` and `--exclude-tags` select a subset. `fact` is separated from
the rest on purpose: knowing the capital of Spain is a size and corpus
question, not an instruction-following one, so a capability run passes
`--exclude-tags fact`.

Constraint checks are literal and easy to get wrong. `contains_any` is
SUBSTRING matching, which is right for stems and punctuation (`injur`, `?`)
and wrong for words: "romantic" contains "roma", "thinking" contains "king",
"fourteen" contains "four". Use `contains_word_any` for anything that is a
word, and `contains_cased_any` when case carries the answer (`THUNDER`, the
chemical symbol `Au`). Every trap in that list scored a wrong answer as
correct before it was caught, so each one has a test in
`tests/test_chat_checks.py`.

### chat_capability

One 0-100 composite over the axes a reader notices, weighted stopping 30,
instruction 30, multi-turn 15, no looping 10, format 10, brevity on demand 5.
Turn safety is a MULTIPLIER rather than an axis: a reply that opens a new user
turn and answers itself is broken, not weak, and averaging that against a good
stop rate would hide it. Missing axes renormalise the remaining weights and
the coverage is reported. Vintage fidelity is deliberately absent, because the
rest of the evaluator answers that and mixing the two gives a number nobody
can act on.

### Reliability: the same question, ten times

```bash
python -m eval.chat_eval MODEL --only-tags fact,closed,transform \
  --repeat 10 --temperature 0.8 --seed 1337
```

Greedy decoding reports the one answer a model gives. A person gets a random
one. A model that answers correctly nine times in ten is a good model that
will look like a bad one to whoever draws the tenth, and a single greedy
sample cannot tell that case apart from a model that is simply wrong.

`--repeat N` asks every probe N times with seeds `seed .. seed+N-1`, so the
run is reproducible while each draw differs. It forces sampling: repeating a
greedy decode would report perfect reliability for a model that has none. The
representative row in the report is the FIRST draw, never the best one.

Results split into three bands over probes, and the bands matter more than the
average: **always right** is the model you can rely on, **never right** is
genuine inability, and **sometimes right** is the band where one unlucky seed
misrepresents the model. `retry_premium` is how much better the model looks to
someone who asks twice than to someone who asks once.

### Repetition penalty

`--repetition-penalty` defaults to **1.0**, meaning off, in both the main
evaluator and `chat_eval`. Looping is a property of the model, and a penalty
hides it: at 1.1 the `looping_rate` column collapses toward zero for
everything and stops telling you anything.

But 1.1 is the llama.cpp and ollama default, so it is the condition a person
deploying the model is actually in. Neither number answers the question
alone. Run both and read the gap:

- 8% looping at 1.0 falling to 1% at 1.1 is a model that was fine already.
- 45% falling to 3% is a model that is only usable because of the sampler.
  That is a training result, not a sampler setting, and the 1.1-only view
  hides it completely.

For small models there is a cost on the other side. A penalty applies to every
repeated token, including the ones period prose legitimately repeats, so
expect some loss of register alongside the smoother output. Whether that trade
is worth making should be a decision, not an inherited default.

### Models that need help

`--reply-after MARKER` cuts each reply at a marker and scores only what
follows. Violet emits a mood line and then `<|assistant|>` before the answer
proper, which is its trained format, not leakage.

Violet also ships no chat template at all; the one in its model directory is
transcribed from its own README. Without it the model is being asked in a
format it never saw, which is not a measurement of anything.

## nanochat checkpoints

`nanochat_models.py` loads Karpathy-style nanochat checkpoints through the same
code path as any transformers model. Point `python -m eval` at the directory
and it works; there is no separate command and no conversion step.

```bash
python -m eval MODELS/GPT1900-instruct          # tokenizer/ is found automatically
python -m eval MODELS/SomeModel --tokenizer DIR # when the repo ships it elsewhere
NANOCHAT_REPO=/path/to/nanochat python -m eval MODELS/SomeModel
```

A nanochat checkpoint is a `model_<step>.pt` state dict beside a
`meta_<step>.json`, with the vocabulary in a pickled tiktoken `Encoding` at
`tokenizer/tokenizer.pkl`. There is no `config.json`, no `tokenizer.json` and
no HF architecture, so discovery, tokenizer resolution, fingerprinting, size
reporting and lineage all branch on `is_nanochat_checkpoint()`.

The weights are not converted. The architecture has drifted well away from
Llama: value embeddings on alternating layers, a learned smear of the previous
token's embedding, a mid-depth backout subtraction, per-layer residual and x0
scalars, parameter-free QK norm, relu-squared MLPs, tanh logit softcapping and
a sliding-window pattern. Writing that as a Llama config would change the
numbers. Instead the real nanochat `GPT` runs inside a `PreTrainedModel`
wrapper that only translates the calling convention, and you need a nanochat
source tree for it (clone into `MODELS/nanochat` or set `NANOCHAT_REPO`).

Two limits are real and the code raises rather than hides them.

**No attention mask.** nanochat attends causally over the whole row and has no
padding-mask path, so a left-padded batch would score pad tokens as content.
Scoring is one document at a time anyway; generation drops to batch size 1
automatically and says so.

**No KV cache.** nanochat's cache belongs to its own Engine and to
FlashAttention-3. The wrapper recomputes the prefix at every decode step, which
makes generation the slow part of a nanochat run.

Forks move things. The value-embedding gate width is a source constant that
GPT-1900 trained at 32 and the current tree hard-codes at 12, so the loader
resizes the gate to match the checkpoint. Parameters that postdate a
checkpoint (smear, backout) are set to their identity values, which reproduces
the older architecture exactly; anything missing that cannot be neutralised is
an error, not a warning.

Tokenizers vary more than the weights do. nanochat trains a fresh BPE per run,
so two 32,768-entry vocabularies built from different corpora disagree on
nearly every id; a substitute produces confident nonsense rather than an error,
and the loader checks vocabulary size against the model before scoring
anything. Some forks replace the tokenizer outright: Mr. Chatterbox ships a
HuggingFace `tokenizers` BPE instead of a tiktoken pickle, and renames every
role token (`<|bos|>` becomes `<|endoftext|>`, `<|user_start|>` becomes
`<human>`, the user turn has no closing marker). `load_nanochat_tokenizer`
accepts either format and `pick_markers` reads the conversation dialect off
the vocabulary. A wrong dialect does not raise; it feeds the model a prompt it
was never trained on and quietly lowers the score, which is why this is
detected rather than assumed.

## The leaderboard

Every report opens with `### Prose BPB leaderboard`, because the first thing
anyone wants is a rough sense of where a checkpoint sits. It is one table, and
it only works if it stays one table.

It used to be two. The reference points were hard coded in `report.py` from a
2026-08-23 run scored on `heldout-Sprocket-n-Say.jsonl`; when the default
held-out moved to `heldout-Piston-n-Prose.jsonl` the renderer noticed the
mismatch and split them into a comparable half and an incomparable half.
Correct, and impossible to read.

Now the rows are measured rather than recorded. `leaderboard.py` owns
`eval_data/leaderboard.json`, each entry carries the held-out hash it was
scored against, and the report shows only the entries matching the current run
and says how many it hid. A row that cannot be ranked is counted in the
caption, never printed beside one that can.

```bash
python -m eval --update-leaderboard MODELS Llama-141M/final-anneal-3h
```

That reads existing `eval-*.json` files, loads no models and touches no GPU.
Measurements are overwritten every refresh; the curated fields `display`,
`kind`, `origin` and `note` are written by hand and survive.

The table carries `modern/hist`, the modern-probe BPB over the historical one.
Above 1 means the model finds period text easier than modern text, which is
what a vintage model should do. At or below 1 it is more at home in modern
English whatever its prose BPB says. That column is the reason it is worth
having: MonadGPT has the best prose BPB in the current table at 0.83172 and a
ratio of 0.82, which is its OpenHermes-Mistral base showing through. Prose BPB
alone would have ranked it first.

The retired pre-v1 figures are in `eval_data/leaderboard-legacy.json`. They
were produced by a different evaluator on a different corpus and are kept for
provenance only. Re-run those models to put them back on the live table.

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

## Embedding geometry

The evaluator records how much of its embedding space a model actually uses.
`embedding_mean_cosine` alone is misleading: nearly every transformer carries a
large shared offset vector that every token includes, and subtracting it sends
the cosine to zero, so a high raw value mostly measures that benign offset.
`embedding_mean_cosine_centered` is reported beside it for exactly that reason.

`embedding_effective_dims` is the participation ratio of the centred covariance
spectrum, meaning the number of independent directions carrying per-token
signal. A stock Llama at initialisation measures about 750 of 768. Well below
that means the vocabulary has been squeezed into a narrow subspace and tokens
are forced to resemble one another.

`embedding_residual_norm` is the mean length after removing the shared offset.
Compare it against the initialisation scale, `0.02 * sqrt(width)` for a stock
Llama, to see whether training grew the per-token part or let it shrink.

Measured on our own models, same tokenizer and corpus, different recipes:
Llama-75M ends at 467 of 768, Llama-141M at 30. Both start at 750.

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
