# Vintage CORE

A portable, self-contained benchmark that evaluates any language model served
behind an OpenAI-compatible API — **or a local HuggingFace checkpoint** — on
**Vintage CORE** — the 22-task
[DCLM CORE](https://arxiv.org/abs/2406.11794) evaluation suite, with every item
rewritten into pre-1900 English prose. (The suite ships 22 task configurations:
19 from upstream CORE — `bigbench_language_identification` is omitted — plus
`basic_math` (elementary arithmetic over operands 1–10) and two `vintage_exam`
tasks built from pre-1900 schoolbook examination corpora.)

The benchmark ships with all of its data. Clone or copy this folder anywhere,
point it at an API endpoint (or a local checkpoint directory), and run — no
datasets to download, no GPU required, and the only Python dependencies are
`requests` and `PyYAML` (plus `torch` + `transformers` for local checkpoints).

```bash
pip install -r requirements.txt

python run_benchmark.py \
  --base-url http://localhost:1234/v1 \
  --model my-model

# or a local HF checkpoint, no server needed:
pip install torch transformers   # or: pip install .[local]
python run_benchmark.py --local-path checkpoints/checkpoint-xx
```

## What it measures

The suite has 22 task configurations over 21 datasets (HellaSwag runs both
zero-shot and ten-shot), grouped into six categories:

| Category | Tasks |
| --- | --- |
| World knowledge | `arc_easy`, `arc_challenge`, `jeopardy`, `bigbench_qa_wikidata` |
| Commonsense reasoning | `copa`, `piqa`, `openbook_qa`, `commonsense_qa` |
| Language understanding | `hellaswag`, `hellaswag_zeroshot`, `winograd`, `winogrande`, `lambada_openai` |
| Reading comprehension | `boolq`, `squad`, `coqa` |
| Symbolic problem solving | `agi_eval_lsat_ar`, `bigbench_operators`, `bigbench_repeat_copy_logic`, `basic_math` |
| Vintage examination | `vintage_exam` (short answers), `vintage_exam_recall` (full-answer verbatim recall) |

Each task reports **accuracy** and a **centered** score that subtracts the
task's random-chance baseline: `centered = (acc − b) / (1 − b)`. The headline
**CORE metric** is the mean of the centered scores across all tasks. Centered
scores can be **negative** — a model that scores below chance (e.g. from a
strong position bias) legitimately lands below zero; zero means exactly chance.

The results table also shows **`NO-ANS%`** — the share of items where the model
produced no parseable answer (a refusal or off-format reply, counted as wrong).
A high value is a warning that the score reflects the model's *format-following*
rather than its knowledge (small instruct models often choke on many-shot
prompts crammed into one chat turn). It is reported per task in the summary JSON
as `unparsed_rate`. For a format-agnostic read on such models, use faithful
`--scoring logprob` on a backend that supports prompt log-probs.

## Scoring modes

Tasks come in three shapes — multiple-choice, Winograd-style *schema*, and
*language-modeling* completion. There are two ways to score them:

- **`logprob` (faithful).** Reproduces the original CORE method: score each
  candidate continuation by its per-token log-probability and pick the best;
  for language-modeling tasks, check whether the gold continuation is the
  greedy argmax. This requires a backend that returns **prompt log-probs**
  (vLLM's `prompt_logprobs`, or the OpenAI-legacy `echo` + `logprobs` on
  `/v1/completions`). Numbers are directly comparable to nanochat/DCLM CORE.
- **`generation` (universal).** Prompts the model to *produce* an answer — a
  choice letter, or the completion text — then parses/normalizes it. Works on
  **any** chat or completion API, including those that don't expose log-probs
  (OpenAI chat, Anthropic-via-proxy, most hosted endpoints). Scores are not
  identical to the logprob method but track model quality closely.

By default (`--scoring auto`) the tool probes the backend and uses `logprob`
when available, otherwise `generation`.

## Local checkpoints (no server)

Pass `--local-path DIR` (a directory with `config.json`, `model.safetensors`,
tokenizer files, ...) to evaluate a local HuggingFace checkpoint in-process —
no API server, no GPU required:

```bash
# Fast subset while iterating on a fine-tune
python run_benchmark.py --local-path checkpoints/checkpoint-xx \
  --tasks arc_easy,boolq,copa,winograd,squad --max-per-task 50 \
  --output results/ckptxx.json
```

Local runs support every scoring mode. With the default `--scoring auto` they
use the **faithful logprob** method (prompt log-probs come straight from the
model's logits, equivalent to vLLM's `prompt_logprobs`), and `--api auto`
resolves to the checkpoint's own chat template for generation mode. Notes:

- Requires `torch` + `transformers` (`pip install .[local]`). Runs on CPU;
  pass `--device cuda` if a GPU is available (auto-detected by default).
- `--max-context` (default 4096) left-truncates over-long prompts.
- The faithful MC scorer benefits from an internal KV prefix cache, so the
  per-candidate forward passes after the first are nearly free.
- `use_cache` is forced on even if the checkpoint config disables it.

The programmatic API mirrors this: `evaluate(local_path="checkpoints/checkpoint-xx", ...)`.

## Endpoints: chat vs. completion

- **Instruct / chat models** (the common case for hosted APIs): the default
  `--api chat` path sends each question as a single user turn. This is right for
  virtually all API-served models.
- **Base models** (e.g. a raw checkpoint on llama-server or vLLM): pass
  `--api completions` for a faithful in-context-learning prompt. For base
  models this is both more accurate and required for `--scoring logprob`.

## Usage

```bash
# Local llama-server (instruct model)
python run_benchmark.py --base-url http://localhost:1234/v1 --model Falcon-H1-0.5B

# OpenAI
OPENAI_API_KEY=sk-... python run_benchmark.py \
  --base-url https://api.openai.com/v1 --model gpt-4o-mini

# A base model on vLLM, faithful logprob scoring, full run, save JSON
python run_benchmark.py --base-url http://host:8000/v1 --model meta-llama/Llama-3.1-8B \
  --api completions --scoring logprob --output results/llama.json

# Fast subset while iterating
python run_benchmark.py --base-url http://localhost:1234/v1 --model m \
  --tasks arc_easy,boolq,squad --max-per-task 50
```

### Quick smoke test on OpenRouter

Pick a cheap instruct model and run a small subset. Your OpenRouter key can go in
`--api-key` or the `OPENAI_API_KEY` env var:

```bash
python run_benchmark.py \
  --base-url https://openrouter.ai/api/v1 \
  --model meta-llama/llama-3.2-3b-instruct \
  --api-key "$OPENROUTER_API_KEY" \
  --tasks arc_easy,boolq,squad,winograd,piqa \
  --max-per-task 20 \
  --debug-file results/openrouter_debug.jsonl \
  --output results/openrouter_summary.json
```

At ~100 requests this costs a fraction of a cent on the cheapest models. Note:
OpenRouter's chat API exposes log-probs only for *generated* tokens, not the
*prompt* log-probs the faithful scorer needs, so runs there use `generation`
mode. For faithful `logprob` scoring use a backend that returns prompt log-probs
(vLLM `prompt_logprobs`, or an OpenAI-legacy `echo` completion endpoint).

### Debugging: capture every model output

`--debug-file PATH` streams one JSON record per example as the run proceeds —
the exact prompt sent, the raw model output, the parsed prediction, the gold
answer, and whether it was correct. This is the fastest way to check that
prompting/parsing behaves for a given model:

```bash
python run_benchmark.py --base-url ... --model ... \
  --tasks arc_easy --max-per-task 10 --debug-file debug.jsonl

# eyeball the ones it got wrong
python -c "import json; [print(r['output'][:60], '->', r['pred'], 'gold', r['gold']) \
  for r in map(json.loads, open('debug.jsonl')) if not r['correct']]"
```

Record fields: `task`, `idx`, `task_type`, `mode`, `prompt`, `output` (generation
mode), `pred`, `gold`, `correct`, `answered`; logprob mode adds `prompts` and
`mean_logprobs` per choice. A request that fails (transport error, or a gateway
that returns HTTP 200 with an error body and no `choices`) does **not** abort the
run — it is recorded with `answered: false` and an `error` field, and reported in
the summary's `error_rate` / `error_samples` and a warning under the table.

### Seeing log-probs

There are two distinct things called "logprobs":

- **Generated-token logprobs** — the model's confidence in the tokens it *wrote*
  (e.g. the answer letter). Most chat APIs including OpenRouter expose these.
  Add `--show-logprobs` and each debug record gains a `logprobs` list of
  `{token, logprob, top: [...]}`. Handy for seeing whether "B" barely beat "A".
- **Prompt log-probs** — the per-token likelihood the model assigns to text you
  *supply* (the candidate continuations). This is what the faithful `--scoring
  logprob` mode needs. OpenRouter's chat API does **not** provide it; use a
  vLLM (`prompt_logprobs`) or OpenAI-legacy `echo` completions endpoint.

```bash
python run_benchmark.py --base-url https://openrouter.ai/api/v1 \
  --model mistralai/mistral-nemo --api-key "$OPENROUTER_API_KEY" \
  --tasks arc_easy --max-per-task 10 --show-logprobs --debug-file dbg.jsonl
```

## Using it from your own experiment code

Import `evaluate` to score a model programmatically and compare experiment
variants. It returns a plain dict — full suite or any subset:

```python
from vintage_core import evaluate

def score(model_name, base_url, **kw):
    return evaluate(
        base_url=base_url,
        model=model_name,
        tasks=None,           # None = all 22; or e.g. ["arc_easy", "boolq"]
        max_per_task=-1,      # -1 = all; use e.g. 200 for a fast signal
        scoring="auto",       # "generation" | "logprob" | "auto"
        api="auto",           # "chat" | "completions" | "auto"
        concurrency=16,
        debug_file=None,      # set a path to dump per-example records
        return_records=False, # True to also get records in memory
        **kw,
    )

baseline = score("run-A", "http://localhost:8000/v1")
variant  = score("run-B", "http://localhost:8001/v1")
print(f"CORE: A={baseline['core_metric']:.4f}  B={variant['core_metric']:.4f}  "
      f"Δ={variant['core_metric'] - baseline['core_metric']:+.4f}")

# per-task deltas
for t in baseline["results"]:
    a, b = baseline["results"][t], variant["results"][t]
    print(f"  {t:<34} {a:.3f} -> {b:.3f}  ({b-a:+.3f})")
```

The return dict contains `core_metric`, `results` (accuracy per task),
`centered_results`, `scoring_mode`, `endpoint`, and `num_tasks`. Few-shot
selection is deterministic, so two variants are compared on identical prompts.
For a quick iteration signal, keep `max_per_task` small (say 100–200); for a
final number, use `-1` (all examples).

### Key options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--base-url` | *(required for API)* | API root, e.g. `.../v1` |
| `--model` | *(required for API)* | Model name sent in requests |
| `--local-path` | — | Local HF checkpoint dir; replaces `--base-url`/`--model` |
| `--device` | auto | Torch device for `--local-path` (`cuda` / `cpu`) |
| `--max-context` | `4096` | Max prompt tokens for `--local-path` (left-truncated) |
| `--api-key` | `$OPENAI_API_KEY` | Bearer token (omit for local servers) |
| `--scoring` | `auto` | `auto` \| `generation` \| `logprob` |
| `--api` | `auto` | `auto` \| `chat` \| `completions` |
| `--tasks` | all | Comma-separated task labels |
| `--max-per-task` | `-1` (all) | Cap examples per task |
| `--concurrency` | `8` | In-flight requests |
| `--output` | — | Write full results JSON |
| `--extra-body` | — | JSON merged into each request (e.g. `'{"top_p":1}'`) |

Few-shot example selection is deterministic (seeded per example), so repeated
runs of the same model are directly comparable.

## Known limitations

- **Reasoning / "thinking" models are not yet supported in generation mode.**
  Models that emit a long internal chain-of-thought before answering (e.g.
  MiniMax M2) can exhaust the small answer-token budget while still "thinking"
  and return **empty content**, scoring ~0 on everything. This surfaces
  unmistakably as **`NO-ANS% = 100%`** with `error_rate = 0` — treat any run
  with a high `NO-ANS%` as **invalid, not a real capability score**. Thinking
  models whose final answer fits the budget (e.g. Qwen3 thinking variants) work
  fine. Faithful `logprob` mode side-steps this entirely (no generation).
- Generation-mode scores are **not identical** to the faithful logprob CORE
  metric; compare models scored the same way.

## Verifying the install

```bash
python tests/test_selfcheck.py     # data integrity, parsing, rendering — no server needed
```

## Data provenance & license

The code in this repository is released under the **MIT License** (see
`LICENSE`), used with the permission of the original authors.

The evaluation data under `data/` is a derivative of the DCLM CORE evaluation
bundle distributed with [nanochat](https://github.com/karpathy/nanochat), with
every task rewritten into a pre-1900 register. It is provided for benchmark use;
see `data/RESTYLE_REPORT.md` for per-task restyle coverage. The MIT license
covers the code and does not relicense upstream dataset content, which remains
under its original terms.
