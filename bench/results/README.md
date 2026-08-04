# Benchmark Results — Baseline / Calibration

This directory contains evaluation results for several LLMs run against the
Vintage CORE benchmark suite. Some models were evaluated at
`max_per_task=20` (a quick calibration pass), others at `max_per_task=500`
(a deeper evaluation). All use generation mode.

The purpose of this report is to establish a **baseline / calibration** — not
to rank models. Results are presented in alphabetical order by model name.

Coverage is complete: all models have a number for all tasks.
The one caveat to keep in mind when reading the tables: the OpenRouter
models are scored on **20 items per task** to limit API spend, so their numbers
carry roughly ±11pp of sampling noise at 50% and single-item granularity (one
item = 0.05). The three local models use 500 items per task. Do not read small
gaps between an OpenRouter row and a local row as real.

---

## Benchmark Tasks

| Task | Type | Description |
|------|------|-------------|
| `agi_eval_lsat_ar` | multiple_choice | LSAT Analytical Reasoning (AGI Eval) |
| `arc_challenge` | multiple_choice | Grade-school science (hard set) |
| `arc_easy` | multiple_choice | Grade-school science (easy set) |
| `basic_math` | language_modeling | Simple arithmetic |
| `bigbench_operators` | language_modeling | Symbolic operator manipulation |
| `bigbench_qa_wikidata` | language_modeling | QA over Wikidata facts |
| `bigbench_repeat_copy_logic` | language_modeling | Repetition / copy logic |
| `boolq` | multiple_choice | Boolean questions from Wikipedia |
| `commonsense_qa` | multiple_choice | Commonsense reasoning |
| `copa` | multiple_choice | Causal reasoning (plausible alternatives) |
| `coqa` | language_modeling | Conversational QA |
| `hellaswag` | multiple_choice | Commonsense inference (few-shot) |
| `hellaswag_zeroshot` | multiple_choice | Commonsense inference (zero-shot) |
| `jeopardy` | language_modeling | Jeopardy-style trivia |
| `lambada_openai` | language_modeling | Next-word prediction from books |
| `openbook_qa` | multiple_choice | Open-book science QA |
| `piqa` | multiple_choice | Physical commonsense reasoning |
| `squad` | language_modeling | SQuAD reading comprehension |
| `winograd` | schema | Winograd pronoun resolution |
| `winogrande` | schema | Winograd schema challenge |
| `vintage_qa` | language_modeling | Pre-1900 oral exam questions, scored via ROUGE-L (10,000 items) |
| `hist_llm` | multiple_choice | Expert-level global history — presence/absence of a characteristic for a polity and time frame (Seshat HiST-LLM, 7,455 items) |

---

## Model Results

### 1. `Falcon-H1-1.5B` (local)

- **Scoring mode:** generation | **Endpoint:** chat
- **max_per_task:** 500 | **Time:** 3013.8s (2889.2s + 124.5s for `hist_llm`)
- **Avg raw score:** 0.624
- **Avg centered score:** 0.527

| Task | Score | Centered | Unparsed |
|------|:-----:|:--------:|:--------:|
| agi_eval_lsat_ar | 0.343 | 0.179 | 0.00 |
| arc_challenge | 0.808 | 0.744 | 0.00 |
| arc_easy | 0.900 | 0.867 | 0.00 |
| basic_math | 0.933 | 0.933 | 0.00 |
| bigbench_operators | 0.667 | 0.667 | 0.00 |
| bigbench_qa_wikidata | 0.580 | 0.580 | 0.00 |
| bigbench_repeat_copy_logic | 0.250 | 0.250 | 0.00 |
| boolq | 0.838 | 0.562 | 0.00 |
| commonsense_qa | 0.794 | 0.743 | 0.00 |
| copa | 0.940 | 0.880 | 0.00 |
| coqa | 0.480 | 0.480 | 0.00 |
| hellaswag | 0.630 | 0.507 | 0.00 |
| hellaswag_zeroshot | 0.590 | 0.453 | 0.00 |
| jeopardy | 0.404 | 0.404 | 0.00 |
| lambada_openai | 0.530 | 0.530 | 0.00 |
| openbook_qa | 0.768 | 0.691 | 0.00 |
| piqa | 0.756 | 0.512 | 0.00 |
| squad | 0.736 | 0.736 | 0.00 |
| winograd | 0.707 | 0.414 | 0.00 |
| winogrande | 0.596 | 0.192 | 0.00 |
| **vintage_qa** | **0.074** | **0.074** | **0.00** |
| **hist_llm** | **0.402** | **0.203** | **0.00** |

---

### 2. `google/gemma-4-26b-a4b-it` (OpenRouter)

- **Scoring mode:** generation | **Endpoint:** chat
- **max_per_task:** 20 | **Time:** 91.6s (67.0s + 24.7s for the two added tasks)
- **Avg raw score:** 0.773
- **Avg centered score:** 0.710

| Task | Score | Centered | Unparsed |
|------|:-----:|:--------:|:--------:|
| agi_eval_lsat_ar | 0.40 | 0.250 | 0.00 |
| arc_challenge | 1.00 | 1.000 | 0.00 |
| arc_easy | 0.95 | 0.933 | 0.00 |
| basic_math | 1.00 | 1.000 | 0.00 |
| bigbench_operators | 0.90 | 0.900 | 0.00 |
| bigbench_qa_wikidata | 0.95 | 0.950 | 0.00 |
| bigbench_repeat_copy_logic | 0.80 | 0.800 | 0.00 |
| boolq | 0.85 | 0.595 | 0.00 |
| commonsense_qa | 0.80 | 0.750 | 0.00 |
| copa | 0.80 | 0.600 | 0.10 |
| coqa | 0.65 | 0.650 | 0.00 |
| hellaswag | 0.70 | 0.600 | 0.00 |
| hellaswag_zeroshot | 0.45 | 0.267 | 0.20 |
| jeopardy | 0.85 | 0.850 | 0.00 |
| lambada_openai | 0.50 | 0.500 | 0.00 |
| openbook_qa | 1.00 | 1.000 | 0.00 |
| piqa | 0.95 | 0.900 | 0.00 |
| squad | 0.85 | 0.850 | 0.00 |
| winograd | 1.00 | 1.000 | 0.00 |
| winogrande | 0.80 | 0.600 | 0.00 |
| **vintage_qa** | **0.30** | **0.300** | **0.00** |
| **hist_llm** | **0.50** | **0.333** | **0.00** |

---

### 3. `mistralai/mistral-nemo` (OpenRouter)

- **Scoring mode:** generation | **Endpoint:** chat
- **max_per_task:** 20 | **Time:** 708.4s (699.7s + 8.8s for the two added tasks)
- **Avg raw score:** 0.705
- **Avg centered score:** 0.625

| Task | Score | Centered | Unparsed |
|------|:-----:|:--------:|:--------:|
| agi_eval_lsat_ar | 0.30 | 0.125 | 0.00 |
| arc_challenge | 0.85 | 0.800 | 0.00 |
| arc_easy | 0.85 | 0.800 | 0.00 |
| basic_math | 0.95 | 0.950 | 0.00 |
| bigbench_operators | 0.75 | 0.750 | 0.00 |
| bigbench_qa_wikidata | 0.85 | 0.850 | 0.00 |
| bigbench_repeat_copy_logic | 0.45 | 0.450 | 0.00 |
| boolq | 0.80 | 0.459 | 0.00 |
| commonsense_qa | 0.70 | 0.625 | 0.00 |
| copa | 0.90 | 0.800 | 0.00 |
| coqa | 0.70 | 0.700 | 0.00 |
| hellaswag | 0.65 | 0.533 | 0.00 |
| hellaswag_zeroshot | 0.80 | 0.733 | 0.00 |
| jeopardy | 0.80 | 0.800 | 0.00 |
| lambada_openai | 0.50 | 0.500 | 0.00 |
| openbook_qa | 0.70 | 0.600 | 0.00 |
| piqa | 0.95 | 0.900 | 0.00 |
| squad | 0.80 | 0.800 | 0.00 |
| winograd | 0.80 | 0.600 | 0.00 |
| winogrande | 0.75 | 0.500 | 0.00 |
| **vintage_qa** | **0.20** | **0.200** | **0.00** |
| **hist_llm** | **0.45** | **0.267** | **0.00** |

---

### 4. `qwen/qwen3.7-flash` (OpenRouter)

- **Scoring mode:** generation | **Endpoint:** chat
- **max_per_task:** 20 | **Time:** 43.2s
- **Avg raw score:** 0.791
- **Avg centered score:** 0.736
- **Required `--extra-body '{"reasoning":{"enabled":false}}'`** — see the note below.

| Task | Score | Centered | Unparsed |
|------|:-----:|:--------:|:--------:|
| agi_eval_lsat_ar | 0.30 | 0.125 | 0.00 |
| arc_challenge | 1.00 | 1.000 | 0.00 |
| arc_easy | 0.95 | 0.933 | 0.00 |
| basic_math | 1.00 | 1.000 | 0.00 |
| bigbench_operators | 0.95 | 0.950 | 0.00 |
| bigbench_qa_wikidata | 0.85 | 0.850 | 0.00 |
| bigbench_repeat_copy_logic | 0.65 | 0.650 | 0.00 |
| boolq | 0.90 | 0.730 | 0.00 |
| commonsense_qa | 0.90 | 0.875 | 0.00 |
| copa | 0.95 | 0.900 | 0.00 |
| coqa | 0.70 | 0.700 | 0.00 |
| hellaswag | 0.85 | 0.800 | 0.00 |
| hellaswag_zeroshot | 0.80 | 0.733 | 0.00 |
| jeopardy | 0.80 | 0.800 | 0.00 |
| lambada_openai | 0.75 | 0.750 | 0.00 |
| openbook_qa | 0.95 | 0.933 | 0.00 |
| piqa | 0.85 | 0.700 | 0.00 |
| squad | 0.90 | 0.900 | 0.00 |
| winograd | 0.90 | 0.800 | 0.00 |
| winogrande | 0.85 | 0.700 | 0.00 |
| **vintage_qa** | **0.30** | **0.300** | **0.00** |
| **hist_llm** | **0.30** | **0.067** | **0.00** |

**This model cannot be benchmarked with default settings.** It is a
mandatory-reasoning model: it returns its tokens in `message.reasoning` and
leaves `message.content` as `null`. Because the harness sizes `max_tokens` from
the length of the gold answer (often under 20 tokens), reasoning consumes the
whole budget, every response comes back with `finish_reason: "length"` and empty
content, and the API reports success — so the first attempt scored **0.000 with
100% unanswered on all 22 tasks and no errors at all**. Disabling reasoning fixes
it completely (0% unanswered everywhere):

```bash
python run_benchmark.py --base-url https://openrouter.ai/api/v1 \
  --model qwen/qwen3.7-flash --api-key "$OPENROUTER_API_KEY" \
  --max-per-task 20 --extra-body '{"reasoning":{"enabled":false}}' \
  --debug-file results/openrouter_qwen37flash_debug.jsonl \
  --output results/openrouter_qwen37flash_summary.json
```

Note that `{"enable_thinking":false}` and `{"chat_template_kwargs":
{"enable_thinking":false}}` do **not** work for this model on OpenRouter; only
`{"reasoning":{"enabled":false}}` (or `{"reasoning":{"effort":"none"}}`) does.
The scores below are therefore **non-reasoning** scores and understate what the
model can do with its reasoning budget intact — most visibly on `hist_llm`
(0.300, barely above the 0.25 baseline) and `agi_eval_lsat_ar` (0.300), the two
tasks that most reward deliberation.

---

### 5. `SmolLM2-1.7B` (local)

- **Scoring mode:** generation | **Endpoint:** chat
- **max_per_task:** 500 | **Time:** 1055.3s (1019.0s + 36.3s for `hist_llm`)
- **Avg raw score:** 0.448
- **Avg centered score:** 0.290

| Task | Score | Centered | Unparsed |
|------|:-----:|:--------:|:--------:|
| agi_eval_lsat_ar | 0.270 | 0.087 | 0.00 |
| arc_challenge | 0.590 | 0.453 | 0.00 |
| arc_easy | 0.826 | 0.768 | 0.00 |
| basic_math | 0.083 | 0.083 | 0.00 |
| bigbench_operators | 0.076 | 0.076 | 0.00 |
| bigbench_qa_wikidata | 0.658 | 0.658 | 0.00 |
| bigbench_repeat_copy_logic | 0.125 | 0.125 | 0.00 |
| boolq | 0.644 | 0.038 | 0.00 |
| commonsense_qa | 0.660 | 0.575 | 0.00 |
| copa | 0.710 | 0.420 | 0.18 |
| coqa | 0.374 | 0.374 | 0.00 |
| hellaswag | 0.412 | 0.216 | 0.00 |
| hellaswag_zeroshot | 0.446 | 0.261 | 0.00 |
| jeopardy | 0.498 | 0.498 | 0.00 |
| lambada_openai | 0.582 | 0.582 | 0.00 |
| openbook_qa | 0.488 | 0.317 | 0.00 |
| piqa | 0.640 | 0.280 | 0.00 |
| squad | 0.372 | 0.372 | 0.00 |
| winograd | 0.495 | −0.011 | 0.00 |
| winogrande | 0.518 | 0.036 | 0.00 |
| **vintage_qa** | **0.096** | **0.096** | **0.00** |
| **hist_llm** | **0.300** | **0.067** | **0.00** |

---

### 6. `SmolLM2-360M` (local)

- **Scoring mode:** generation | **Endpoint:** chat
- **max_per_task:** 500 | **Time:** 718.3s (695.1s + 23.3s for `hist_llm`)
- **Avg raw score:** 0.235
- **Avg centered score:** 0.014

| Task | Score | Centered | Unparsed |
|------|:-----:|:--------:|:--------:|
| agi_eval_lsat_ar | 0.278 | 0.098 | 0.00 |
| arc_challenge | 0.254 | 0.005 | 0.00 |
| arc_easy | 0.228 | −0.029 | 0.00 |
| basic_math | 0.028 | 0.028 | 0.00 |
| bigbench_operators | 0.000 | 0.000 | 0.00 |
| bigbench_qa_wikidata | 0.006 | 0.006 | 0.00 |
| bigbench_repeat_copy_logic | 0.000 | 0.000 | 0.00 |
| boolq | 0.384 | −0.665 | 0.00 |
| commonsense_qa | 0.230 | 0.037 | 0.00 |
| copa | 0.570 | 0.140 | 0.00 |
| coqa | 0.174 | 0.174 | 0.00 |
| hellaswag | 0.254 | 0.005 | 0.00 |
| hellaswag_zeroshot | 0.268 | 0.024 | 0.00 |
| jeopardy | 0.016 | 0.016 | 0.00 |
| lambada_openai | 0.228 | 0.228 | 0.00 |
| openbook_qa | 0.212 | −0.051 | 0.00 |
| piqa | 0.498 | −0.004 | 0.00 |
| squad | 0.248 | 0.248 | 0.00 |
| winograd | 0.502 | 0.004 | 0.00 |
| winogrande | 0.488 | −0.024 | 0.00 |
| **vintage_qa** | **0.052** | **0.052** | **0.00** |
| **hist_llm** | **0.254** | **0.005** | **0.00** |

---

### 7. `x-ai/grok-4.5` (OpenRouter)

- **Scoring mode:** generation | **Endpoint:** chat
- **max_per_task:** 20 | **Time:** 734.1s (663.9s + 70.2s for the two added tasks)
- **Avg raw score:** 0.884
- **Avg centered score:** 0.857
- **vintage_qa:** 0.450 — best `vintage_qa` result of any model tested
- **hist_llm:** 0.600 — best `hist_llm` result of any model tested

| Task | Score | Centered | Unparsed |
|------|:-----:|:--------:|:--------:|
| agi_eval_lsat_ar | 1.00 | 1.000 | 0.00 |
| arc_challenge | 1.00 | 1.000 | 0.00 |
| arc_easy | 0.95 | 0.933 | 0.00 |
| basic_math | 1.00 | 1.000 | 0.00 |
| bigbench_operators | 1.00 | 1.000 | 0.00 |
| bigbench_qa_wikidata | 0.95 | 0.950 | 0.00 |
| bigbench_repeat_copy_logic | 0.95 | 0.950 | 0.00 |
| boolq | 0.90 | 0.730 | 0.00 |
| commonsense_qa | 0.85 | 0.813 | 0.00 |
| copa | 1.00 | 1.000 | 0.00 |
| coqa | 0.70 | 0.700 | 0.00 |
| hellaswag | 0.90 | 0.867 | 0.00 |
| hellaswag_zeroshot | 0.85 | 0.800 | 0.00 |
| jeopardy | 0.90 | 0.900 | 0.00 |
| lambada_openai | 0.75 | 0.750 | 0.00 |
| openbook_qa | 1.00 | 1.000 | 0.00 |
| piqa | 0.95 | 0.900 | 0.00 |
| squad | 0.85 | 0.850 | 0.00 |
| winograd | 1.00 | 1.000 | 0.00 |
| winogrande | 0.90 | 0.800 | 0.00 |
| **vintage_qa** | **0.45** | **0.450** | **0.00** |
| **hist_llm** | **0.60** | **0.467** | **0.00** |

---

## Summary Table

| Model | Samples/Task | Avg Raw | Avg Centered | vintage_qa | hist_llm | Time |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| `google/gemma-4-26b-a4b-it` | 20 | 0.773 | 0.710 | 0.300 | 0.500 | 92s |
| `mistralai/mistral-nemo` | 20 | 0.705 | 0.625 | 0.200 | 0.450 | 708s |
| `qwen/qwen3.7-flash` † | 20 | 0.791 | 0.736 | 0.300 | 0.300 | 43s |
| `x-ai/grok-4.5` | 20 | 0.884 | 0.857 | 0.450 | 0.600 | 734s |
| `Falcon-H1-1.5B` (local) | 500 | 0.624 | 0.527 | 0.074 | 0.402 | 3014s |
| `SmolLM2-1.7B` (local) | 500 | 0.448 | 0.290 | 0.096 | 0.300 | 1055s |
| `SmolLM2-360M` (local) | 500 | 0.235 | 0.014 | 0.052 | 0.254 | 718s |

Every **Avg** column is the mean over the same 22 tasks for every row, so the
columns are internally consistent — but see the sampling caveat at the top before
comparing a 20-item OpenRouter row against a 500-item local row. `Avg Centered`
is identical to the `core_metric` field stored in each summary JSON.

† `qwen/qwen3.7-flash` was run with reasoning disabled because it cannot produce
parseable output otherwise; its row understates the model. See its section above.

---

## Key Observations

- **Scoring mode** is `generation` (free-form text generation) for all models.
- **Centered scores** adjust for chance-level guessing in multiple-choice tasks
  (raw score minus chance baseline, normalised).
- **`vintage_qa`** is scored via ROUGE-L F1 with a threshold of 0.30 instead of
  exact prefix match, because gold answers are verbose 19th-century prose while
  models produce modern equivalents. The earlier prediction that larger models
  would reach 25-30% is now confirmed and slightly exceeded: grok-4.5 scores
  0.450, gemma 0.300 and mistral-nemo 0.200, against 5-10% for the three small
  local models.
- **`vintage_qa` requires the `rouge_score` package, and silently mis-scored
  without it.** `vintage_core.scoring.rouge_l_correct()` used to fall back to
  exact-prefix matching when the import failed, with no warning. The first
  OpenRouter runs of this task were made in an environment missing the package
  and reported 0.000 for all four models — verbose-but-correct answers were all
  counted wrong. The stored model outputs were unaffected, so the task was
  re-scored offline from the `--debug-file` records rather than re-run, moving
  grok 0.000 → 0.450, gemma 0.000 → 0.300 and mistral 0.000 → 0.200. The three
  local models were unaffected (0 flips over 1,500 records — they had been run
  with the package present). The fallback is now loud: an import-time
  `RuntimeWarning`, a banner before any tokens are spent, and a
  `vintage_qa_scoring` field in each summary JSON that reads
  `prefix-fallback-DEGRADED` when the metric is not real ROUGE-L. `rouge_score`
  is now pinned in `requirements.txt`.
- **`hist_llm`** is scored as ordinary four-way multiple choice (accuracy plus a
  centered score against a 25% chance baseline), so it needs no special metric.
  All three local runs came back with `NO-ANS% = 0.00` and no request errors,
  which is the expected shape: a model that replies with the option wording
  ("Inferred Present") instead of the letter is mapped back to a choice rather
  than counted unanswered. `SmolLM2-360M` exercised that path heavily — 81 of its
  500 replies were full option strings like `D: Absent` — and still recorded zero
  unanswered, so the mapping is doing its job.
- **`hist_llm` answer-class bias matters more than the headline number here.**
  Falcon-H1-1.5B scores 0.402 (centered 0.203) and spreads its answers over all
  four classes (A 178 / D 151 / B 124 / C 47 out of 500), so it is doing real
  work above chance. SmolLM2-1.7B scores 0.300 (centered 0.067) but answered `D`
  ("Absent") on 459 of 500 items — that 0.300 is essentially the frequency of the
  `D` class in this sample (147/500 = 0.294), not history knowledge. Read the
  1.7B number as chance-level with a degenerate-answer bias. `SmolLM2-360M` lands
  at 0.254 (centered 0.005) — indistinguishable from the 25% baseline — and never
  emitted `A` at all (B 279 / D 202 / C 19 / A 0), so it is also answering from
  format habit rather than content. Only Falcon-H1-1.5B is meaningfully above
  chance on this task.
- **`hist_llm` is the hardest task in the suite for every model tested.** The best
  score is grok-4.5 at 0.600; no other model clears 0.500, and the frontier models
  sit closer to the 1.5B local model than they do on any knowledge task. It is
  expert-level Seshat data, so this is the expected shape rather than a harness
  problem — but it does mean `hist_llm` has the most headroom of any task here.
- These results serve as a **calibration baseline** for the benchmark suite.
  When evaluating new models, compare against these numbers to gauge relative
  performance.
