# ꧁⎝ 𓆩༺ The Patina Project ༻𓆪 ⎠꧂

The goal of this project is to take a modern LLM and age/ oldify/ victorify 🎩 it.
We want to steam-tune the LLM until it behaves like a Victorian person the past.

All the datasets we use have a hard cut at year 1900.

**Target period: 1875-1899.** Matches the data - every system prompt in `vintage-ft-v1` says
*"the year 1899"*, `vintage_qa` is framed as *"the year is 1890"*, the oral-exam rows say
*1885*, and `banned_terms.py` draws its boundary at 1900.

**What counts as an anachronism is already written down - don't reinvent it.**
`knowledge/README.md` holds the curated lists: `## Allowed knowledge (before year 1900)`
and `## Banned knowledge (after the year 1900)`, with dates and reasoning per item
(electrical telegraph 1840s, soda siphon 1829, …; World Wars, Great Depression, Cold War,
viruses-as-particles 1931, … all banned). The machine-readable side is
`knowledge/pre1900/banned_terms.py` (~800 terms). Any prompt set, filter or eval in this
project must derive its allowed/banned notion from **those two files**, never from a list
improvised in a config or a docstring.

---

# Environment

Measured, not assumed (python scripts/gpu_info.py):

```
AMD Radeon RX 9070 XT  ·  15.92 GB VRAM  ·  ROCm 7.2
BF16 supported ✅ (Ampere+ cc>=8.0)
BF16 supported ✅ (torch)
TF32 supported ✅

FP32 :  15.99 TFLOPS  (8.60 ms/iter)
FP16 : 120.28 TFLOPS  (1.14 ms/iter)
BF16 : 128.45 TFLOPS  (1.07 ms/iter)
INT8 :  74.12 TFLOPS  (1.85 ms/iter)

Python 3.14.0
torch 2.13.0+rocm7.2   transformers 5.10.2   trl 1.4.0   peft 0.19.1   accelerate 1.13.0
fla (flash-linear-attention) 0.5.1  ✅ installed
flash_attn                          ❌ can't install on this machine
```

---

## Attention: what actually works

**Verified by running a forward *and* backward pass on `SmolLM2-135M--Instruct` in bf16 on
this GPU:**

| `attn_implementation` | verdict after real training runs |
|---|---|
| **`"eager"`** | ✅ **USE THIS.** Completed 7,034/7,034 steps clean. ~+15% slower than sdpa. |
| `"sdpa"` | 🚨 **UNSAFE - NaNs non-deterministically.** Died at steps 400 / 1700 / 1900 / 3800 across both precisions, both optimizers and two learning rates. |
| `"flex_attention"` | 🚨 **UNUSABLE - `grad_norm = nan` on step 1.** Passes an isolated forward+backward, fails instantly in the training loop. |
| `"flash_attention_2"` | ❌ `flash_attn` not installed, never installs |

⚠️ **An isolated forward+backward pass is NOT evidence of trainability.** Both `sdpa` and
`flex_attention` pass that test on this GPU and both are broken in practice. Only a multi-thousand
step run counts - **800-step probes produced two false "fixes"**, and one sdpa run looked perfectly
healthy for 3,700 steps before exploding.

**Proof it is the kernel and not the data:** the `eager` run logged loss 1.8226 / 1.8129 at steps
2900 / 3000, matching a failed `sdpa` run's 1.8227 / 1.8129 to four decimals - identical data
order, identical trajectory, one finished and one exploded at 3800.

**Verdict on `flash-linear-attention[rocm]`: it is NOT a replacement for `flash_attn` here.**
Checked directly against the installed `fla` 0.5.1:

- `fla` is a Triton kernel library for **linear-attention architectures** - its 40-odd op
  modules are `gla`, `retention`, `rwkv4/6/7`, `delta_rule`, `mamba`-family, `lightning_attn`,
  `based`, `titans`, etc. SmolLM2 is a standard **softmax-attention** Llama-architecture model.
  These kernels implement a *different attention mathematics*; they are not a faster
  implementation of the one SmolLM2 uses.
- It *does* ship `fla.ops.attn.parallel_attn`, a Triton softmax-attention kernel whose
  signature includes `cu_seqlens: torch.LongTensor` "consistent with the FlashAttention API".
  So a bridge is technically conceivable.
- But `transformers` 5.10.2 registers only
  `['sdpa', 'flash_attention_2', 'flash_attention_3', 'flash_attention_4', 'flex_attention', 'paged|*']`
  in `ALL_ATTENTION_FUNCTIONS`. Passing `attn_implementation = "fla"` will simply be rejected.
  Using `parallel_attn` means registering a custom attention function and monkeypatching the
  model - **model-surgery work, not a config flag**, and out of scope for `fine_tune.py`.

**🚨 RETRACTED: flex_attention is not the path to packing.** An earlier version of this file
called it "experimental, worth testing - the real path to packing" on the strength of one
successful forward+backward pass. In an actual training run it produces `grad_norm = nan` on
**step 1** (`diag/flexattn.bf16/`). **Packing is therefore dead on this machine** - it needs a
document-masking attention implementation, and the only two candidates are broken (flex) or
missing (flash_attn).

**All arms use `attn_implementation = "eager"` + `packing = false`.**

An alternative worth investigating later: the TRL docs mention `padding_free = True` with
`attn_implementation = "kernels-community/flash-attn2"`, which fetches a prebuilt kernel instead
of compiling `flash_attn` locally. Untested here, and given that two attention kernels already
NaN on this card, it should be treated as low-probability.

## 📋 Status - see training/patina/EXPERIMENT_LOG.md for the full record

| | |
|---|---|
| LoRA pipeline | ✅ **works**, after 5 fixes to `fine_tune.py` |
| arm B, 135M-Instruct, 1 epoch | ✅ **completed** - 7,034 steps, 25.4 min, eval ppl 6.19 → 5.80 |
| style verdict | ❌ **FAIL** - register improved (+0.04 `p_pre1900`), but archaic-fantasy drift **+44%** and **0/5 trap prompts refused** (base also 0/5) |
| 🚨 biggest gotcha | **`attn_implementation` must be `"eager"`.** `sdpa` NaNs non-deterministically; `flex_attention` NaNs on step 1 |
| 🔑 biggest insight | **Style transfer is easy; period-knowledge discipline is not.** v1 has no examples of the assistant *declining* post-1900 topics, so it cannot teach that. This is a **data** problem, not a hyperparameter problem |
| remaining | CORE benchmark (Step 4), arm A, 360M, base models, LoRA variants |

---

---

# Method: LoRA only

**No full fine-tuning.** Every experiment uses `method = "lora"` or a LoRA variant.

✅ **VERIFIED WORKING** - `exp/SmolLM2-135M-Instruct/smoke/…` ran 50 LoRA steps end to end:
trained, saved an adapter, merged it, and the merged model loads and generates. Measured:
**~5.2–5.5 it/s** at effective batch 4 / seqlen 1024, **peak 4.3 GB of 17.1 GB**, LoRA
trainable **4,884,480 / 139,399,488 = 3.50%**, loss 2.93 → 2.47, eval ppl 10.92 → 10.21,
grad_norm 0.26–0.34 with no spikes. Huge headroom: 360M, higher `r`, longer `seqlen` and
bigger batches are all affordable.

Three edits to `fine_tune.py` were required to get there (all additive):

1. **`max_steps` was hardcoded to `-1`** - a short smoke test was impossible to configure.
   Now `max_steps=int(train_cfg.get('max_steps', -1))`. **This was the real blocker.**
2. **`lr_scheduler_kwargs` was commented out.** Now passed through *and guarded*:
   `min_lr_rate` is only accepted by `lr_scheduler_type = "cosine_with_min_lr"` - with
   `linear`/`cosine` it raises `TypeError`. **That is why it was commented out.** The script now
   drops the key with a warning instead of crashing. `warmup_ratio` also added.
3. **LoRA variants were unreachable.** `build_peft_config` now passes `use_dora`,
   `use_rslora`, `init_lora_weights` and `modules_to_save` when present - so DoRA / rsLoRA /
   PiSSA are configurable. All four confirmed to exist as `LoraConfig` fields in peft 0.19.1.

Confirmed behaviours to respect:

- 🚨 **`final/` holds the adapter only** - `adapter_config.json` + `adapter_model.safetensors`
  (19 MB), no `config.json`, no `model.safetensors`. `--local-path` on it **fails**.
  **Always benchmark `final_merged/`** and keep `merge_after_training = true`.
- ✅ **The merge is real** - diffing merged vs base weights: **210 of 273 tensors changed**,
  max |Δ| ≈ 0.004 across attention and MLP projections. Not a no-op copy.
- **Keep `gradient_checkpointing = false`.** With LoRA and checkpointing, frozen base inputs
  have no `requires_grad` and training dies unless `enable_input_require_grads()` is called -
  which this script does not do.
- **LoRA wants ~10× the full-FT learning rate.** `2e-4`, not `3e-5`.
- **`load_best_model_at_end = true` + PEFT + `torch_compile`** is a known rough edge; the code
  unwraps `_orig_mod` before `merge_and_unload()`, but `torch_compile = false` removes the risk.
- ⚠️ **`report_to = ["tensorboard"]` fails - tensorboard is NOT installed.** All patina configs
  use `report_to = []`.
- ⚠️ **transformers 5.x gotcha:** `apply_chat_template(..., return_tensors='pt')` returns a
  **dict**, not a tensor, so `model.generate(ids, ...)` dies on `.shape`. Render with
  `tokenize=False`, then tokenize. `style_eval.py` does this.

Still untested: `torch_compile`, `packing`, `flex_attention`, `gradient_checkpointing`,
DoRA/rsLoRA/PiSSA, multi-GPU, resume-from-checkpoint.

---

# Models

Our distant goal is to release the best model, which I'm guessing will be the larger one.
For quick testing, we will iterate on the smaller models first. All four live locally under
`training/patina/`.

| model | trainable today? | note |
|---|---|---|
| `SmolLM2-135M` | ⚠️ needs `chat_template` | see below |
| `SmolLM2-135M--Instruct` | ✅ | ChatML template present |
| `SmolLM2-360M` | ⚠️ needs `chat_template` | see below |
| `SmolLM2-360M--Instruct` | ✅ | ChatML template present |

**The base-model blocker.** `fine_tune.py:150` raises
`ValueError('Tokenizer is missing chat_template!')`, and neither base model has one. The
docstring above that line promises an Alpaca-style fallback; the code does not implement it.

**The fix is cheap.** All four models share `vocab_size 49152`, `tie_word_embeddings: True`,
and - crucially - `<|im_start|>` and `<|im_end|>` are **already in the base tokenizers'
`added_tokens_decoder`**. So enabling a base model means copying the Instruct `chat_template`
into a **new** `tokenizer_config.json` inside the experiment folder. **No embedding resize, no
new tokens, no `lm_head` surprises.** The only genuine confound is that the base never
*trained* on those tokens - which is what the fine-tune is for.

Deferred to Step 5 so the first result isn't confounded.

---

# Datasets

## `vintage-ft-v1/` - Experiment 1's data

**87,251 rows / ~27.8M tokens**, all rows valid `{"messages": [...]}`, system prompts uniformly
*"the year 1899"* - so **no system-prompt tweaking is needed here**; that item is closed.
Token counts are `chars/4` estimates.

| file | rows | ~tokens | arm A | arm B |
|---|---:|---:|:-:|:-:|
| `ChatGPT.jsonl` | 642 | 319,257 | ✅ | ✅ |
| `Claude.jsonl` | 501 | 260,075 | ✅ | ✅ |
| `DarkOdity.jsonl` | 955 | 581,349 | ✅ | ✅ |
| `DeepSeek.jsonl` | 2,264 | 837,066 | ✅ | ✅ |
| `GemAI.jsonl` | 1,233 | 600,218 | ✅ | ✅ |
| `gpt-OSS.jsonl` | 987 | 682,950 | ✅ | ✅ |
| `GPT1900.jsonl` | 1,078 | 823,180 | ✅ | ✅ |
| `Maverick.jsonl` | 536 | 235,210 | ✅ | ✅ |
| `Mistral.jsonl` | 2,626 | 1,263,348 | ✅ | ✅ |
| `MonadGPT.jsonl` | 2,451 | 956,048 | ✅ | ✅ |
| `MythoMax-13B.jsonl` | 98 | 42,491 | ✅ | ✅ |
| `Qwen.jsonl` | 2,208 | 912,679 | ✅ | ✅ |
| `Talkie-13B.jsonl` | 11,182 | 2,022,841 | ✅ | ✅ |
| `Vox-12B.jsonl` | 2,094 | 1,422,650 | ✅ | ✅ |
| **`TypeWriter-7B.jsonl`** | **58,396** | **16,846,693** | ✅ | ❌ |
| **arm A total** | **87,251** | **~27.8M** | | |
| **arm B total** | **28,855** | **~11.0M** | | |

### How v1 was built (this is why it scores so high)

Every file was generated with an explicit period system prompt -

> *"You are a distinguished gentleman living in the year 1899. You speak in cultivated, clear,
> late-nineteenth-century English: warm, witty, precise, and courteous. You know nothing of
> events, inventions, books, or persons later than 1900. You always write in English and never
> use foreign words or phrases."*

— then responses were **aggressively dropped if they matched any of the ~800 banned terms**,
then filtered **twice more** with different quality heuristics. Rows that failed were deleted,
not kept.

**Consequence for interpretation:** a 98–100% `detect-sft.py` pass rate on these files is
**expected by construction**, not evidence of anything. They are a filtered survivor set. Do
not read a high `p_pre1900` on v1 as a quality signal - it is a selection artefact. (The one
historical exception: `Talkie-13B` briefly leaked modern rows because a later generation batch
wasn't re-filtered. Fixed; the unfiltered predecessor has been deleted.)

### Why arm A / arm B

`TypeWriter-7B.jsonl` is **67% of rows and 61% of tokens** from a single 7B source. The ablation
is *not* a quality accusation - measured on a **random** 1,200-row sample it looks fine
(98.17% pass `detect-sft`, median assistant answer 637 chars, p90 1,427). The point is that one
source **dominates** the corpus, so arm A measures "does more data win" and arm B measures
"does a smaller, more diverse corpus win". Without both, any v1 verdict is really a verdict on
TypeWriter-7B.

✅ **`vintage_qa` is uncontaminated by v1.** Normalized cross-match of every v1 user and
assistant turn against `bench/data/eval_data/vintage/vintage_qa.jsonl`:

```
bench distinct questions: 8,920
  matched in v1: 6  (0.07%, all in TypeWriter-7B.jsonl)
  answer matches: NONE
```

So Experiment 1 may report the **full 21-task `core_metric` including `vintage_qa`**, directly
comparable to `bench/results/local_*`. No exclusions, no footnote.

## `vintage-ft-v2/` - deferred; one gem, two landmines

| file | rows | ~tokens | shape | status |
|---|---:|---:|---|---|
| `violet_sft_dataset.jsonl` | 49,981 | ~12.75M | `{text, category, metadata}` | 🟡 **best content in v2**, needs conversion |
| `shard1_messages.jsonl` | 109,107 | ~32.1M | `messages` ✅ | 🔴 archangels + 100% `vintage_qa` overlap |
| `shard2_messages.jsonl` | 54,409 | ~48.9M | `messages` ✅ | 🔴 53,458 rows have no system prompt |
| `shard3_messages.jsonl` | 6,408 | ~1.0M | `messages` ✅ | 🟡 no system prompt |

The three shards are **format-valid**: 169,924 / 169,924 rows parse as `{"messages": [...]}` ✅.
(The original `training-mess.jsonl` had 54,409 JSON-*array* lines and 6,408 flat
`{question, answer}` rows; both are now normalized.) `violet_sft_dataset.jsonl` is not.

### `violet_sft_dataset.jsonl` - the most on-target data in the project

Content categories are exactly what this project wants:

```
advice_personal 2,962 · etiquette_rules 2,606 · victorian_society 2,369
propriety_questions 2,369 · social_dilemmas 2,369 · opinion_requests 2,369
letter_writing 2,369 · victorian_technology 2,132 · history_events 2,132
definitions_concepts 2,132 · moral_questions 2,132 · descriptive_writing 2,132
decision_advice 2,014 · lists_practical 1,777 · paraphrasing 1,777
```

Three things must be fixed before it can be used:

1. **It is not `messages`.** Each row is `{text, category, metadata}` where `text` uses a
   custom template with **literal strings, not tokens**:
   `<|user|>\n…\n<|violet_mood|>\nConsiderate\n<|assistant|>\n…`. Feed this to SmolLM2 and
   `<|violet_mood|>` tokenizes as garbage punctuation.
2. **🚨 It will break the loader if mixed in.** `fine_tune.py:96-110` keeps whichever of
   `messages` / `text` a file has - this file has `text`, the others have `messages` - and then
   `concatenate_datasets` across mismatched features **raises**. It cannot simply be appended
   to `dataset_path`.
3. **🚨 Line 49,981 is a bare integer (`2052`), not an object.** One malformed final line.

**Conversion is straightforward and the metadata makes it clean:** `metadata` already carries
`question`, `mood` and `multi_turn`, so a converter can emit proper `messages` and fold `mood`
into the system prompt - *"You are a distinguished gentleman living in the year 1899 … Your
present humour is **Considerate**."* This is what "needs system prompt tweaks" means for this
file: it has **no system prompt at all**, and one must be synthesised per row from `mood`.
Note the user turns are deliberately modern English (*"…really starting to bug me"*) - that is
fine and matches how `detect-sft.py` judges user turns leniently.

### The shards

Dominant personas in `shard1`: **40,637 rows with an Archangel system prompt** (Michael ~25,000,
"Grabrail" ~12,300+), 25,000 **Archimedes of Syracuse** (d. 212 BC, 5 prompt variants), ~2,900
**Isaac Newton** (1600s), ~1,600 **Leonardo da Vinci** (Renaissance), and 36,443 rows with no
system prompt. Only 2,042 rows carry an *"oral examination"* system prompt in the 1885/1890
framing. Across all three shards **126,309 rows (74%) have no system prompt.**

🚨 **`vintage_qa` contamination in v2 is total.** **8,919 of 8,920** bench questions (100.0%)
appear in `shard1`; 7,273 shard1 rows are verbatim bench items (question *and* answer).
Consequences whenever v2 is used:

- `vintage_qa` is **1 of the 21 tasks inside `core_metric`**. Training on v2 and then quoting
  `core_metric` against `bench/results/local_*` (whose models never saw the answer key) is not
  a valid comparison. Pull `vintage_qa` out and report it separately as *"recall (train-set)"*.
- Holding out *random* shard1 rows does **not** decontaminate the benchmark - there is no
  headroom inside the bench set. Decontaminating means excluding those 7,273 rows specifically.
- The clean eval already exists: **87,277 distinct shard1 questions are *not* in the
  benchmark.** Carve a held-out slice from those and you get a real generalization number
  alongside the recall number.

v2 earns a GPU run only after `triage_v2.py` (below) reports how much of it is period-appropriate.
`violet_sft_dataset.jsonl` is the likely first winner from that triage.

---

# Available tools

Work folder:

- `training/patina/` - where we save the experiments and the logs; the only folder we are
  allowed to edit.

Read-only:

- `training/patina/vintage-ft-v1` & `vintage-ft-v2` - the fine-tuning datasets
- `knowledge/README.md` - the curated allowed/banned knowledge lists (**the source of truth
  for anachronisms**)
- `training/fine_tune.py` - the FT script; a new TOML config per experiment, kept in that
  experiment's folder
- `knowledge/pre1900/detect-sft.py` (and `detect.py`, `banned_terms.py`) - estimate how likely
  a text is PRE-1900, against a threshold
- `bench/run_benchmark.py` - the benchmark; we don't run the full suite, `--max-per-task 50`
  is enough
- `bench/results/local*.json` - previous runs, used as calibration/baseline
- `training/evaluate.py` - checkpoint diagnostics

## `training/fine_tune.py`

Thin wrapper around HuggingFace TRL:

```bash
python training/fine_tune.py --cfg training/patina/<exp-folder>/config.toml
```

- `--cfg` defaults to `training/fine_tune_config.toml` - always pass it explicitly
- **Relative paths inside the TOML resolve against that TOML's parent directory**
  (`fine_tune.py:296-310`). A config in `training/patina/<exp>/` must therefore use
  `../SmolLM2-360M--Instruct` and `../vintage-ft-v1/*.jsonl`
- `bf16` / `fp16` are honoured (`fine_tune.py:406-407`); `bf16` defaults to `BF16_SUPPORTED`,
  which is `True` on this card
- The loader keeps only the `messages` / `text` column and requires one of them
  (`fine_tune.py:96-110`) - see the `violet` warning above
- `load_best_model_at_end = true` requires `save_steps == eval_steps`
- **With LoRA, benchmark `<final_model_dir>_merged`, not `<final_model_dir>`** - see the Method
  section

## `bench/run_benchmark.py`

21-task CORE capability suite:

```bash
python bench/run_benchmark.py \
  --local-path  training/patina/<exp-folder>/final_merged \
  --max-per-task 50 \
  --output      training/patina/<exp-folder>/bench_summary.json \
  --debug-file  training/patina/<exp-folder>/bench_debug.jsonl
```

`--local-path` loads an HF checkpoint in-process (no server needed). Debug JSONL and summary go
in the experiment folder.

**Baselines** in `bench/results/local*.json`, for calibration:

| model | `core_metric` | `vintage_qa` | `--max-per-task` |
|---|---:|---:|---:|
| x-ai/grok-4.5 | 0.8178 | - | 20 |
| google/gemma-4-26b-a4b-it | 0.6816 | - | 20 |
| qwen3-vl-30b-a3b-thinking | 0.6621 | - | 20 |
| mistralai/mistral-nemo | 0.6035 | - | 20 |
| Falcon-H1-1.5B | 0.5427 | 0.074 | 500 |
| SmolLM2-1.7B | 0.3002 | 0.096 | 500 |
| **SmolLM2-360M** | **0.0139** | 0.052 | 500 |

⚠️ Two caveats. (1) The 360M baseline is at the **noise floor** - centred `boolq` is −0.66.
CORE can detect a *catastrophe* at this size, not fine degradation. (2) These ran at
`--max-per-task 500` via a served endpoint, and the summary does not record base vs Instruct.
**Experiment 1 runs its own `--max-per-task 50` baseline** so deltas are apples-to-apples; the
saved files are loose context only.

## `knowledge/pre1900/detect-sft.py` · `detect.py` · `banned_terms.py`

**They are not interchangeable:**

- **`detect-sft.py`** - understands **`messages` / `conversation` / flat Q&A**. This is the one
  that filtered v1 and v2 (threshold ~0.5). Judges user turns leniently and assistant turns
  strictly. Use for **datasets**.
- **`detect.py`** - reads a flat `--field text`. Use for plain text and for scoring model
  **output**.
- For scoring generations, import rather than shell out - `detect.Scorer.score()` returns
  `p_pre1900`, `n_tokens`, `english_frac` **and `banned_hits`** in a single call, so no separate
  `banned_terms` call is needed:
  ```python
  sys.path.insert(0, 'knowledge/pre1900')   # detect.py does a bare `from banned_terms import ...`
  from detect import Scorer, tokenize
  ```

🚨 **Known limitation - it scores *register*, and cannot tell 1899 from 1600 BC.** Measured on
random 1,200-row samples:

| sample | kept as "vintage" |
|---|---:|
| **`shard1` rows with an Archangel system prompt** (40,637 exist) | **100.00%** |
| `shard1` overall | 88.00% |
| `Talkie-13B.jsonl` (filtered) | 100.00% |
| `TypeWriter-7B.jsonl` | 98.17% |

*"Archangel Grabrail stands before thee in thought and word. Mark my counsel well…"* passes at
**100%**. Biblical/ancient archaic register is indistinguishable to it from late-Victorian
prose. It is a good **filter** (which is what it was built for) and a poor **judge of period**.
**This is why success is never defined by `p_pre1900` alone** - and why a rising `p_pre1900`
next to rising thee/thou/hath is a *failure*, not a success.

## `training/evaluate.py`

Checkpoint evaluation harness. Suites are **`info`, `perplexity`, `embeddings`, `generation`**
— fixed-sentence perplexity, token-probability and entropy stats, embedding drift, n-gram
stats, generation probes.

```bash
python training/evaluate.py \
  --checkpoint training/patina/<exp-folder>/final_merged \
  --suites perplexity embeddings --chat \
  --output training/patina/<exp-folder>/evaluate.json
```

⚠️ Useful for *"did the fine-tune actually move the weights, and by how much"* - it is **not** a
scored vintage-ness eval. Perplexity is comparable across checkpoints only on its fixed
constant sentences. Use it as a secondary read, never as the success criterion.

---

# Experiment folder naming

Encode the hyperparameters **in the folder name**, so the experiment is identifiable from
`ls` alone and two runs can never silently overwrite each other:

```
training/patina/exp/<model>/<arm>/<hyperparams>/
```

`<hyperparams>` is a dot-joined, fixed-order key`value` string:

```
<lora|dora|rslora>.r<R>.a<A>.drop<D>.seqlen<L>.b<B>.grad<G>.lr<LR>.<sched>.warmup<W>.e<E>.attn<IMPL>[.<deviation>…]
```

`seed` is omitted (always 42 - put it in the name only when you actually vary it), and so is any
default you never change. Append a short suffix for each deliberate deviation, e.g. `.fp32`,
`.optadamw`, `.packing`, `.steps50`.

Real examples (these exist on disk):

```
exp/SmolLM2-135M-Instruct/smoke/
  lora.r16.a32.drop0.05.seqlen1024.b2.grad2.lr2e-4.cosine.warmup20.steps50.attnsdpa/

exp/SmolLM2-135M-Instruct/v1-noTypeWriter/
  lora.r16.a32.drop0.05.seqlen1024.b2.grad2.lr2e-4.cosine.warmup100.e1.attneager/     ← ✅ completed
  lora.r16.a32.drop0.05.seqlen1024.b2.grad2.lr2e-4.cosine.warmup100.e1.attnsdpa/      ← ❌ NaN @3800
  lora.r16.a32.drop0.05.seqlen1024.b2.grad2.lr2e-4.cosine.warmup100.e1.attnsdpa.fp32/ ← ❌ NaN @1700
```

Rules: only include a key if it is set; **never rename a folder after a run** (the name is the
record). Every folder contains:

```
<hyperparams>/
  config.toml           the TOML for this run (paths relative to THIS folder)
  NOTES.md              what / how / WHY + wall-clock, tokens/sec, peak VRAM,
                        final train & val loss, and every surprise
  sft_checkpoints/      intermediate checkpoints
  final/                LoRA adapter only - NOT loadable by --local-path
  final_merged/         merged full model - THIS is what you benchmark
  bench_summary.json    run_benchmark.py --output
  bench_debug.jsonl     run_benchmark.py --debug-file
  style_eval.json       style_eval.py metrics
  samples.md            base-vs-tuned generations, side by side
  evaluate.json         training/evaluate.py (secondary read)
```

---

# Tasks / protocol

We have to be honest about this: there's a billion infinities of possibilities that we can play
with. So Experiment 1 changes **one** thing at a time and measures it.

## Step 0 - housekeeping (no GPU)

1. `training/patina/style_prompts.jsonl` - ~50 held-out prompts, with the allowed/banned
   notion taken from `knowledge/README.md` and `banned_terms.py`:
   - ~40 period-neutral (daily life, trade, travel, letters, moral advice, natural philosophy)
   - ~6 knowledge probes drawn from the `## Allowed knowledge` list
   - ~4 **anachronism traps** drawn from the `## Banned knowledge` list
   - Assert **zero normalized overlap** with any v1 user turn. Review before use.
2. `training/patina/style_eval.py` - loads the prompt set, generates with the chat template at
   fixed seed and temperature, scores each generation via `detect.Scorer`, and writes
   `style_eval.json` + a side-by-side `samples.md`. Reports mean `p_pre1900`, share ≥ 0.5,
   `banned_hits` per 1k words, trap-prompt behaviour, an **archaic-fantasy register counter**
   (thee/thou/thy/hath/verily/behold - because `p_pre1900` can't see this), and the Δ against
   the untuned base on identical prompts.
3. Token-length histogram over arm A (`scripts/count_tokens.py`) to set `max_seq_length` at
   ~p95. Expect well under 2048 - median assistant turns run 273–637 characters depending on
   the file.

## Step 1 - baselines (before training anything)

`run_benchmark.py --max-per-task 50` and `style_eval.py` against untuned
`SmolLM2-360M--Instruct` and `SmolLM2-135M--Instruct`, into `training/patina/baselines/`.
Without this, post-training numbers don't mean anything.

## Step 2 - smoke test (this is where LoRA gets proven)

`max_steps = 50`, 135M-Instruct, arm-B data. Its job is to answer: does `method = "lora"` run at
all; does the adapter save; does `merge_and_unload()` produce a `final_merged/` that
`--local-path` can load; what are tokens/sec and peak VRAM. Abort on OOM, `NaN` loss, a
`chat_template` error, or a merge failure. **Nothing else starts until this passes.**

## Step 3 - the two arms

Same model, same config, same seed; only `dataset_path` differs. Reference config:

```toml
[model]
base_model = "../../../SmolLM2-360M--Instruct"   # depth depends on the folder nesting
attn_implementation = "eager"     # sdpa NaNs non-deterministically, flex NaNs at step 1

[data]
dataset_path = [                  # arm A: 15 files · arm B: 14, no TypeWriter-7B
  "../../../vintage-ft-v1/ChatGPT.jsonl",
  # ... never violet_sft_dataset.jsonl until it is converted to `messages`
]
val_fraction   = 0.025
max_seq_length = 1024             # set from the Step 0.4 p95 measurement

[training]
method = "lora"                   # NEVER "full"
output_dir      = "./sft_checkpoints"
final_model_dir = "./final"       # -> ./final_merged is what you benchmark
num_train_epochs = 1

bf16 = true
packing = false                   # sdpa: no document masking
torch_compile = false             # unproven on ROCm / RDNA4
gradient_checkpointing = false    # required: LoRA + checkpointing breaks in this script
neftune_noise_alpha = 0.0         # the shipped 0.1 is meaningless (paper range 5-15);
                                  # zero keeps the A-vs-B ablation single-variable
learning_rate = 2e-4              # LoRA wants ~10x the full-FT rate
per_device_train_batch_size = 2
gradient_accumulation_steps  = 2

save_strategy = "steps"; save_steps = 500
eval_strategy = "steps"; eval_steps  = 500   # must equal save_steps when
load_best_model_at_end = true                # load_best_model_at_end = true
seed = 42

[lora]
r = 32
lora_alpha = 64
lora_dropout = 0.05
target_modules = ["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]
bias = "none"
merge_after_training = true       # REQUIRED for benchmarking
```

## Step 4 - evaluate both arms

Per arm: `bench_summary.json`, `bench_debug.jsonl`, `style_eval.json`, `samples.md`,
`evaluate.json`, `NOTES.md`. Fill in the A-vs-B table: style metrics, `core_metric`,
`vintage_qa`, wall-clock, tokens/sec, peak VRAM, final train and val loss.

## Step 5 - widen, using measured cost

Only after Step 4, one variable at a time:
- 135M-Instruct on the winning arm
- the two base models, with the Instruct `chat_template` copied into a **new**
  `tokenizer_config.json` under the experiment folder (originals untouched)
- **LoRA variants** - requires adding `use_dora` / `use_rslora` / `init_lora_weights`
  passthrough to `build_peft_config`; ask before editing
- **🧪 `attn_implementation = "flex_attention"`**, first at `packing = false` to isolate the
  attention change, then at `packing = true` - and verify document boundaries are actually
  honoured before trusting any result

---

# Success criteria

**Current form:**

Experiment 1 **passes** if, on the held-out prompts, the tuned model beats **its own base** on
all four:

1. mean `p_pre1900` strictly greater than base (Δ > 0). **The "≥ 0.5" clause is dead weight -
   the untuned 135M-Instruct already scores 0.7958** purely from the 1899 system prompt.
2. **absolute `banned_hits` count** lower than base. **NOT `banned_hits` per 1k words** - that
   rate is confounded by output length. exp-01b cut anachronisms 6 → 5 while the per-1k rate
   *rose*, because generations shrank 26% (861 → 640 chars).
3. **archaic-fantasy per 1k words does not rise.** Promoted from "watch" to a hard criterion:
   exp-01b raised it **+44%** (0.86 → 1.24) while `p_pre1900` improved, which is exactly the
   failure `p_pre1900` cannot see.
4. **more trap prompts refused than base.** ⚠️ The current regex detector in `style_eval.py` is
   **unreliable** - it false-positived twice on the baseline (it scored *"I am delighted to
   enlighten you on the subject of vitamins…"* as a disclaimer). **Must be replaced** with a
   `banned_terms.py`-driven check: a trap is failed if the answer engages with the banned
   concept at all. Until then, read trap numbers by hand from `samples.md`.

…and `core_metric` does not collapse against the Step 1 baseline.

`vintage_qa` is reported on its own line as a knowledge/recall signal - valid here because v1
is uncontaminated (0.07%).

🚨 **The circularity is documented.** `p_pre1900` is also the function that filtered
the training data at ~0.5, so criterion 1 is partly self-fulfilling: a model trained on
detector-approved text will score well by construction. **Criteria 2 and 3, the archaic-fantasy
register counter, and the base-vs-tuned Δ are the load-bearing ones.** Remember the detector
rates Archangel prose at 100%. Read `samples.md` before declaring victory.

---

# Inputs / outputs

**Inputs** (what we steam-tune): a base or instruct model · a TOML config with tweaks ·
fine-tuning datasets.

**Outputs** (what we produce): LoRA adapters + merged checkpoints · benchmark results ·
experiment log details - what we tried, how we tried it, and **WHY**.

---

# Deferred work

**`training/patina/triage_v2.py`** - read-only, no GPU. Persona-tags every row across the three
shards from its system prompt, counts rows with no system prompt, counts rows verbatim in
`vintage_qa`, and estimates how many rows are period-appropriate for 1899. Writes
`training/patina/v2_triage_report.md`. **The gate v2 must pass before it earns a single GPU-hour.**

**`training/patina/convert_violet.py`** - read-only in, new file out. Parses
`violet_sft_dataset.jsonl`'s `<|user|>` / `<|violet_mood|>` / `<|assistant|>` template (or
better, rebuilds from `metadata.question` + `multi_turn`), synthesises a per-row system prompt
folding in `mood`, skips the bare-int line 49,981, and writes a proper `messages` JSONL.
**Highest-value deferred item - this is the most on-target content in the project.**

Also deferred: a clean held-out QA eval carved from the 87,277 non-benchmark `shard1`
questions, giving recall *and* generalization numbers separately.

---

# Rules

Never leave the `~/Dev/vintage-LLM/` folder !! but it's OK to use `/tmp/`
Never use git, don't commit, don't pull, don't push. You don't need it.
Never delete anything, ever, just create new files. Including: never overwrite
`training/fine_tune_config.toml`.
Never use `method = "full"` - LoRA and LoRA variants only.
Stay in `training/patina/` and the specific files mentioned above -
there's a ton of other files and folders and you'll get lost.
