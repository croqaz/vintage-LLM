# Benchmark Results — Baseline / Calibration

This directory contains evaluation results for several LLMs run against the
Vintage CORE benchmark suite (21 tasks). Some models were evaluated at
`max_per_task=20` (a quick calibration pass), others at `max_per_task=500`
(a deeper evaluation). All use generation mode.

The purpose of this report is to establish a **baseline / calibration** — not
to rank models. Results are presented in alphabetical order by model name.

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

---

## Model Results

### 1. `Falcon-H1-1.5B` (local)

- **Scoring mode:** generation | **Endpoint:** chat
- **max_per_task:** 500 | **Time:** 2889.2s
- **Avg raw score:** 0.635
- **Avg centered score:** 0.543

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

---

### 2. `google/gemma-4-26b-a4b-it` (OpenRouter)

- **Scoring mode:** generation | **Endpoint:** chat
- **max_per_task:** 20 | **Time:** 67.0s
- **Avg raw score:** 0.810
- **Avg centered score:** 0.750
- **vintage_qa:** N/A (not yet evaluated)

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

---

### 3. `mistralai/mistral-nemo` (OpenRouter)

- **Scoring mode:** generation | **Endpoint:** chat
- **max_per_task:** 20 | **Time:** 699.7s
- **Avg raw score:** 0.743
- **Avg centered score:** 0.664
- **vintage_qa:** N/A (not yet evaluated)

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

---

### 4. `qwen/qwen3-vl-30b-a3b-thinking` (OpenRouter)

- **Scoring mode:** generation | **Endpoint:** chat
- **max_per_task:** 20 | **Time:** 1000.6s
- **Avg raw score:** 0.798
- **Avg centered score:** 0.728
- **vintage_qa:** N/A (not yet evaluated)

| Task | Score | Centered | Unparsed |
|------|:-----:|:--------:|:--------:|
| agi_eval_lsat_ar | 0.15 | −0.063 | 0.85 |
| arc_challenge | 0.85 | 0.800 | 0.00 |
| arc_easy | 0.80 | 0.733 | 0.00 |
| basic_math | 1.00 | 1.000 | 0.00 |
| bigbench_operators | 1.00 | 1.000 | 0.00 |
| bigbench_qa_wikidata | 0.95 | 0.950 | 0.05 |
| bigbench_repeat_copy_logic | 0.90 | 0.900 | 0.05 |
| boolq | 0.90 | 0.730 | 0.00 |
| commonsense_qa | 0.80 | 0.750 | 0.00 |
| copa | 0.90 | 0.800 | 0.10 |
| coqa | 0.65 | 0.650 | 0.00 |
| hellaswag | 0.80 | 0.733 | 0.00 |
| hellaswag_zeroshot | 0.40 | 0.200 | 0.50 |
| jeopardy | 0.85 | 0.850 | 0.00 |
| lambada_openai | 0.70 | 0.700 | 0.00 |
| openbook_qa | 0.95 | 0.933 | 0.05 |
| piqa | 0.90 | 0.800 | 0.05 |
| squad | 0.80 | 0.800 | 0.05 |
| winograd | 0.90 | 0.800 | 0.05 |
| winogrande | 0.75 | 0.500 | 0.10 |

---

### 5. `SmolLM2-1.7B` (local)

- **Scoring mode:** generation | **Endpoint:** chat
- **max_per_task:** 500 | **Time:** 1019.0s
- **Avg raw score:** 0.455
- **Avg centered score:** 0.300
- **vintage_qa:** 0.096

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

---

### 6. `x-ai/grok-4.5` (OpenRouter)

- **Scoring mode:** generation | **Endpoint:** chat
- **max_per_task:** 20 | **Time:** 663.9s
- **Avg raw score:** 0.920
- **Avg centered score:** 0.897
- **vintage_qa:** N/A (not yet evaluated)

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

---

## Summary Table

| Model | Samples/Task | Avg Raw (20 std) | Avg Centered (20 std) | vintage_qa | Time |
|-------|:---:|:---:|:---:|:---:|:---:|
| `google/gemma-4-26b-a4b-it` | 20 | 0.810 | 0.750 | N/A | 67s |
| `mistralai/mistral-nemo` | 20 | 0.743 | 0.664 | N/A | 700s |
| `qwen/qwen3-vl-30b-a3b-thinking` | 20 | 0.798 | 0.728 | N/A | 1001s |
| `x-ai/grok-4.5` | 20 | 0.920 | 0.897 | N/A | 664s |
| `Falcon-H1-1.5B` (local) | 500 | 0.635 | 0.543 | 0.074 | 2889s |
| `SmolLM2-1.7B` (local) | 500 | 0.455 | 0.300 | 0.096 | 1019s |

---

## Key Observations

- **Scoring mode** is `generation` (free-form text generation) for all models.
- **Centered scores** adjust for chance-level guessing in multiple-choice tasks
  (raw score minus chance baseline, normalised).
- **`vintage_qa`** is scored via ROUGE-L F1 with a threshold of 0.30 instead of
  exact prefix match, because gold answers are verbose 19th-century prose while
  models produce modern equivalents. Small models score 7-10%; larger models
  are expected to reach 25-30% (to be verified).
- The OpenRouter models (20 samples each) were evaluated under the old 22-task
  configuration and have not yet been re-evaluated with `vintage_qa`. Their
  `vintage_qa` scores are listed as N/A.
- `hellaswag_zeroshot` and `agi_eval_lsat_ar` show elevated unparsed rates for
  some models, suggesting these tasks are harder to format correctly.
- These results serve as a **calibration baseline** for the benchmark suite.
  When evaluating new models, compare against these numbers to gauge relative
  performance.
