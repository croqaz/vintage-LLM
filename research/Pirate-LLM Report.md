# Pirate LLM — Detailed Training Report

> **Date:** May 2026
> **Scope:** Full analysis of the pretraining and SFT pipelines, with notes on correctness, hardening, and what needs to change before scaling to a 10B-parameter multi-GPU run.

---

## 1. Full Pipeline Overview

```
roneneldan/TinyStories (HF dataset)
  └─ piratize.py          → multiprocessing arrr.translate() on every story
  └─ save_to_disk()       → dataset/tiny_stories_pirate/

tokenize_ds.py            → train BPE tokenizer (vocab=8192) on pirate text
  └─ pirate_bpe.json

tokenize_corpus.py        → encode corpus to uint16 flat binary files
  └─ train.bin / val.bin

training/train.py         → base pretraining (causal LM) on train.bin
  └─ out/ckpt.pt          → pushed to HF Hub

training/sft_train.py     → supervised fine-tuning on TeeZee/dolly-15k-pirate-speech
  └─ out/sft_ckpt.pt      → pushed to HF Hub

training/sample.py        → autoregressive generation with temperature + top-k
```

---

## 2. Data Pipeline

### 2.1 Corpus Construction
- **Source:** `roneneldan/TinyStories` — a synthetic children's short-story dataset (≈2.1M stories).
- **Piratization:** Every story is passed through `arrr.translate()`, a rule-based English→Pirate-English translator. Translation is parallelized across `cpu_count()-1` workers with `chunksize=64` via `multiprocessing.Pool.imap`.
- **Output:** A Hugging Face `DatasetDict` saved to disk, preserving train/validation splits.

### 2.2 Tokenizer
- **Type:** Byte-Pair Encoding (BPE) via HuggingFace `tokenizers`.
- **Pre-tokenizer:** `ByteLevel` (no prefix space) — operates on raw UTF-8 bytes, so no unknown tokens are possible (the initial alphabet covers all 256 bytes).
- **Vocab size:** 8,192 — deliberately tiny, GPU-friendly.
- **Special token:** `<|endoftext|>` (token ID 0 by default) is the story separator, appended after every document.
- **Trained on:** Pirate-ized training split only (no data leakage from validation).

### 2.3 Binary Corpus Files
- Stories are tokenized in batches of 1,000, concatenated with an `<|endoftext|>` token between each story, and written as a flat `uint16` NumPy memmap.
- The file is over-allocated at ~500 tokens/story then truncated to the exact byte count at the end.
- `get_batch()` re-creates the memmap on every call (deliberate, per nanoGPT issue #154 — avoids a long-running memory leak at the cost of negligible per-call overhead).

### 2.4 SFT Dataset
- **Source:** `TeeZee/dolly-15k-pirate-speech` — the Databricks Dolly-15k instruction dataset translated to pirate speech.
- **Schema:** `instruction`, `context` (optional), `response`.
- Fully loaded into a Python list in memory during `build_sft_dataset()`.
- Random 2% split used for validation. Shuffled with `random.Random(seed=1337)`.

---

## 3. Model Architecture

### 3.1 High-Level Structure

```
Input token IDs  (B, T)
  → wte: token embeddings    (B, T, 384)
  → wpe: position embeddings (T, 384)   [learned, absolute]
  → input dropout (p=0.1)
  → 6× Block:
      LayerNorm → CausalSelfAttention → residual
      LayerNorm → MLP                 → residual
  → final LayerNorm
  → lm_head Linear (384 → 8192)
  → F.cross_entropy loss
```

### 3.2 Hyperparameters (default / "Tiny" tier)

| Parameter       | Value  | Notes                                      |
|-----------------|--------|--------------------------------------------|
| `vocab_size`    | 8,192  | Matches custom BPE tokenizer               |
| `block_size`    | 256    | Context window in tokens                   |
| `n_layer`       | 6      | Transformer blocks                         |
| `n_head`        | 6      | Attention heads                            |
| `n_embd`        | 384    | Hidden dimension (64 dim/head)             |
| `dropout`       | 0.1    | Applied on input, attention, MLP           |
| `bias`          | False  | No bias in Linear or LayerNorm             |
| **Parameters**  | ~10.6M | (position embeddings excluded by convention)|

### 3.3 Attention
- Uses PyTorch's `nn.MultiheadAttention` with `batch_first=True`.
- A full `(block_size, block_size)` upper-triangular additive mask (`-inf` above diagonal) is registered as a buffer per layer and sliced to `[:T, :T]` at forward time.
- `need_weights=False` avoids materializing the attention weight matrix.
- Residual dropout applied after attention output.

### 3.4 MLP
- `Linear(n_embd → 4*n_embd)` → GELU → `Linear(4*n_embd → n_embd)` → Dropout.
- 4× expansion factor is the GPT-2 standard.

### 3.5 Weight Initialization
- All `nn.Linear` weights: `N(0, 0.02)`, biases (if any): zero.
- All `nn.Embedding` weights: `N(0, 0.02)`.
- LayerNorm: default PyTorch init (weight=1, bias=0).

### 3.6 Weight Tying
- `wte.weight` and `lm_head.weight` are the same tensor.
- Saves `vocab_size × n_embd = 8192 × 384 ≈ 3.1M` parameters — a 30% reduction at this scale.

---

## 4. Base Pretraining (`training/train.py`)

### 4.1 Objective
Standard causal language modeling: predict the next token. Loss is `F.cross_entropy` over the full `(B×T, vocab_size)` logit matrix. **No token is masked** — all positions contribute equally to the loss.

### 4.2 Optimizer

**AdamW** with correct weight decay separation:
- 2D+ parameter tensors (weights): `weight_decay = 0.1`
- 1D parameter tensors (embeddings, LayerNorm scales): `weight_decay = 0.0`
- `β₁ = 0.9`, `β₂ = 0.95` (lower than Adam's 0.999 — the transformer standard)
- Gradient clipping at `‖g‖₂ ≤ 1.0`

### 4.3 LR Schedule: Warmup + Cosine Decay

$$
\text{lr}(t) = \begin{cases}
\text{lr}_{\text{peak}} \cdot \dfrac{t+1}{T_{\text{warm}}+1} & t < T_{\text{warm}} \\[6pt]
\text{lr}_{\text{min}} + \dfrac{1+\cos\!\left(\pi \cdot \dfrac{t - T_{\text{warm}}}{T_{\text{decay}} - T_{\text{warm}}}\right)}{2} \cdot (\text{lr}_{\text{peak}} - \text{lr}_{\text{min}}) & T_{\text{warm}} \le t \le T_{\text{decay}} \\[6pt]
\text{lr}_{\text{min}} & t > T_{\text{decay}}
\end{cases}
$$

| Parameter         | Value  |
|-------------------|--------|
| `learning_rate`   | 3e-4   |
| `min_lr`          | 3e-5   |
| `warmup_iters`    | 200    |
| `lr_decay_iters`  | 20,000 |
| `max_iters`       | 20,000 |

The warmup ramp uses `(it+1) / (warmup_iters+1)` which reaches exactly `lr_peak` one step after warmup, not at warmup — a minor off-by-one but inconsequential.

### 4.4 Mixed Precision
- `bfloat16` path: `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)` + `GradScaler(enabled=False)` (bfloat16 has enough dynamic range, no loss scaling needed).
- `float16` path: `autocast` + `GradScaler(enabled=True)`.
- CPU/MPS: `nullcontext()` — full float32.

### 4.5 Training Loop (per iteration)
```
1. Compute lr = get_lr(iter_num) → set on all param groups
2. If eval step: estimate_loss() → print → checkpoint → push to Hub
3. get_batch("train") → (x, y)
4. forward under autocast context → loss
5. optimizer.zero_grad(set_to_none=True)
6. scaler.scale(loss).backward()
7. scaler.unscale_(optimizer)
8. clip_grad_norm_(model.parameters(), 1.0)
9. scaler.step(optimizer)
10. scaler.update()
```

### 4.6 Checkpointing
- Saves `out/ckpt.pt` on every eval interval (not just when val loss improves).
- Checkpoint contains: `model`, `optimizer`, `config`, `iter_num`, `val_loss`, `best_val_loss`.
- Optionally pushed to HF Hub via `HfApi().upload_file()` with a commit message.
- On resume: loads local `ckpt.pt` first; falls back to HF Hub download if configured.

### 4.7 Evaluation
- `estimate_loss()` runs `eval_iters=100` random batches from both `train.bin` and `val.bin`.
- Model is set to `.eval()` during estimation (disables dropout), then restored to `.train()`.
- `@torch.no_grad()` decorator avoids accumulating gradients.

---

## 5. Supervised Fine-Tuning (`training/sft_train.py` + `training/sft_data.py`)

### 5.1 Prompt Format

**With context:**
```
### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}<|endoftext|>
```

**Without context:**
```
### Instruction:
{instruction}

### Response:
{response}<|endoftext|>
```

### 5.2 Label Masking
- Prompt tokens are set to `IGNORE_INDEX = -100` in the labels tensor.
- Response tokens + `<|endoftext|>` carry real token IDs in labels.
- Padding uses `<|endoftext|>` in `input_ids` and `-100` in labels.
- `F.cross_entropy` silently ignores `-100` by default (`ignore_index=-100` is PyTorch's default), so prompt positions contribute zero to the loss. **This works but only by coincidence of matching PyTorch's default — it is never made explicit in the code.**

### 5.3 Sequence Handling
```
full_sequence = prompt_ids + response_ids + [eos]
```
- Truncated to `block_size` (response tokens are cut, not the prompt).
- If `len(prompt_ids) >= block_size`: example is skipped entirely.
- Right-padded with `eos` / `-100` to exactly `block_size`.
- Causal shift: `x = input_ids[:, :-1]`, `y = labels[:, 1:]` → both length `block_size - 1`.

### 5.4 Weight Loading
- Pretrained `ckpt.pt` is downloaded from HF Hub.
- The architecture config is taken from the checkpoint, not from the SFT config.
- `block_size` mismatch raises a hard error (position embeddings cannot be remapped).
- `dropout` is overridden from the SFT config onto the checkpoint's arch config.

### 5.5 SFT Loop
Identical structure to base training: get_lr → eval → forward → zero_grad → backward → clip → step. Saves `sft_ckpt.pt` with a `"stage": "sft"` marker in the checkpoint dict.

---

## 6. Sampling / Generation (`training/sample.py`)

- Autoregressive loop: forward → take last-position logits → scale by temperature → top-k filter → softmax → multinomial sample → append.
- Context is cropped to the last `block_size` tokens when the sequence grows longer.
- Top-k filtering: all logits below the k-th largest are set to `-inf` before softmax.
- `@torch.no_grad()` throughout.

---

## 7. What the Code Does Well

| Area | Detail |
|------|--------|
| **Pre-norm architecture** | `LayerNorm` applied before (not after) attention and MLP — more stable training, less prone to gradient vanishing |
| **Weight tying** | Input embedding and LM head share parameters — correct practice, significant saving at small scale |
| **AdamW weight decay separation** | Only decays ≥2D tensors; LayerNorm, biases, and embedding vectors are correctly excluded |
| **Gradient clipping** | `clip_grad_norm_` at 1.0 — prevents loss spikes from degenerate batches |
| **`no_bias` design** | Omitting biases in Linear layers is a small but real efficiency win; LayerNorm bias excluded too |
| **memmap data loading** | Avoids loading the full corpus into RAM; handles multi-GB datasets gracefully |
| **`set_to_none=True`** | More memory-efficient than zeroing gradient tensors |
| **Crash-safe checkpointing** | Always saves on each eval interval regardless of improvement — never loses more than `eval_interval` iterations |
| **SFT prompt masking** | Loss correctly computed only on response tokens |
| **Mixed precision** | bfloat16 autocast + GradScaler is set up correctly |
| **Piratization parallelism** | Uses `Pool.imap` with chunksize — efficient multiprocessing |
| **Resumption logic** | Local checkpoint takes priority over Hub; graceful fallback with informative error messages |
| **`torch.compile` opt-in** | Correctly disabled on non-CUDA platforms |

---

## 8. Bugs and Correctness Issues

### 8.1 CRITICAL — Gradient Accumulation Is Configured but Never Implemented

`Config.gradient_accumulation_steps` exists and defaults to `1`, but **neither `train.py` nor `sft_train.py` reads or uses it**. The training loop has no micro-step logic. This means:

- On small GPUs where batch_size must be reduced, you cannot compensate with gradient accumulation.
- The config field is misleading — it gives the impression it works.

**Fix:** wrap the forward+backward in an inner loop of `gradient_accumulation_steps` micro-steps, scale the loss by `1/gradient_accumulation_steps`, and call `optimizer.step()` / `scaler.step()` only after all micro-steps.

### 8.2 CRITICAL — `ignore_index` in SFT Loss Is Implicit, Not Explicit

The model's `forward()` calls:
```python
loss = F.cross_entropy(logits.view(-1, ...), targets.view(-1))
```
This happens to work because PyTorch defaults to `ignore_index=-100` and `IGNORE_INDEX = -100`. But the model has no knowledge it's being used for SFT. If someone changes `IGNORE_INDEX` in `sft_data.py`, the SFT loss will silently become wrong (computing loss on prompt tokens).

**Fix:** Either pass `ignore_index=IGNORE_INDEX` explicitly at the call site, or add a `ignore_index` parameter to `GPT.forward()`.

### 8.3 HIGH — Causal Mask Registered Per Layer, Not Per Model

Each `CausalSelfAttention` registers its own `(block_size, block_size)` float32 buffer. With 6 layers, 6 identical masks are stored. At `block_size=256` this is negligible (256×256×4×6 = 1.5MB), but at `block_size=4096` and `n_layer=32` it becomes 2GB of redundant buffers.

**Fix:** Register the causal mask once in `GPT.__init__` and pass a slice of it into each block's forward.

### 8.4 HIGH — Additive Float Mask Disables Flash Attention Fast Path

`nn.MultiheadAttention` internally calls `F.scaled_dot_product_attention`. When an explicit `attn_mask` is passed as a float tensor (as opposed to a boolean mask or `is_causal=True`), PyTorch's SDPA cannot use the Flash Attention kernel on many backends. Switching to:
```python
self.attn(..., attn_mask=None, is_causal=True)
```
allows SDPA to dispatch to Flash Attention on CUDA, which is 2-4× faster and uses O(√T) memory instead of O(T²).

### 8.5 MEDIUM — `weights_only=False` Is a Security Risk

All three `torch.load()` calls (in `train.py`, `sft_train.py`, `sample.py`) use `weights_only=False`. This unpickles the entire checkpoint including the `Config` dataclass, which can execute arbitrary code if the checkpoint file is untrusted.

**Fix:** Serialize the config separately (e.g., as JSON), then load with `weights_only=True` for the tensors.

### 8.6 MEDIUM — `_init_weights` Missing Scaled Residual Init

GPT-2 and successors initialize the output projection of each attention block and the second linear in each MLP with `std = 0.02 / sqrt(2 * n_layer)` to account for the accumulation of residuals. At 6 layers this is `0.02 / sqrt(12) ≈ 0.00577`. With flat `std=0.02` the residual stream grows too fast at initialization. At 6 layers this barely matters; at 32+ layers it causes slower convergence.

### 8.7 MEDIUM — SFT Validation Samples With Heavy Replacement

`estimate_val_loss` draws `eval_iters=100` random batches from `val_examples` (approx. 300 examples). Each batch has `batch_size=32` examples sampled with replacement. The effective sample size is 3,200 draws from a pool of 300, meaning every example appears ~10 times on average. The variance of the loss estimate is artificially low (it's averaging highly correlated samples), and a single bad example in validation has outsized influence.

**Fix:** For small val sets, do a full sweep instead of random sampling. Track the exact count and switch strategy based on set size.

### 8.8 LOW — Unconventional zero_grad Placement

The training loop in both `train.py` and `sft_train.py` orders operations as:
```
forward → zero_grad → backward → step
```
rather than the conventional:
```
zero_grad → forward → backward → step
```
In non-accumulation mode this is functionally correct (zero_grad before backward is all that matters), but it prevents straightforward addition of gradient accumulation later and confuses readers.

### 8.9 LOW — No `cudnn.benchmark`

For fixed-size inputs (which this always produces — every batch is `(batch_size, block_size)`), `torch.backends.cudnn.benchmark = True` tells cuDNN to benchmark convolution algorithms at startup. While this project has no convolutions, it is still a standard practice for any CUDA training run and costs nothing to enable.

### 8.10 LOW — `num_parameters()` Excludes Position Embeddings

The comment says "by convention," but this is not a universal convention and can mislead users comparing against published parameter counts. The difference is `block_size × n_embd = 256 × 384 = 98,304` parameters — ~1% at this scale.

---

## 9. Design and Scalability Weaknesses

### 9.1 Absolute Learned Position Embeddings

`wpe` is a standard `nn.Embedding(block_size, n_embd)` lookup. This means:
- **Context length is fixed at training time** and cannot be extended at inference.
- The SFT loader explicitly enforces this: `block_size` mismatch is a hard error.
- Extrapolation beyond `block_size=256` produces garbage — there are simply no trained position vectors for positions ≥ 256.

For a 10B model you will want **RoPE** (Rotary Position Embedding) or **ALiBi**, both of which generalize to longer contexts and do not require retraining if you increase the window.

### 9.2 No Data Parallelism (DDP / FSDP)

The training loop has zero awareness of multi-process/multi-GPU execution. There is no:
- `torch.distributed` setup
- `DistributedDataParallel` wrapping
- `DistributedSampler` for the data loader
- Rank-aware checkpointing (only rank 0 should save)

For a 10B model across multiple GPUs, this is the first thing that must be added.

### 9.3 No Activation Checkpointing

All intermediate activations are retained in memory during the forward pass for use in the backward pass. For a 6-layer tiny model this is trivial. For a 32+ layer 10B model, the activation memory alone can exceed GPU VRAM. PyTorch's `torch.utils.checkpoint.checkpoint_sequential` (or per-block checkpointing) trades compute for memory.

### 9.4 No Optimizer State Sharding

With full AdamW, the optimizer stores two additional tensors (first and second moment estimates) per parameter — effectively **3× the model size** in memory (model + m + v). For a 10B model in bf16 that is `3 × 10B × 2 bytes = 60 GB` on a single process. You need **ZeRO-2/3** (DeepSpeed) or **FSDP** to shard optimizer states across GPUs.

### 9.5 Tiny Vocabulary

8,192 tokens is appropriate for this toy project but far too small for a general-purpose 10B model. Modern production models use 32k–128k vocabularies. A larger vocabulary improves tokenization efficiency (fewer tokens per word), which directly reduces sequence lengths and training costs. It also requires a much larger embedding matrix (`vocab_size × n_embd`) that benefits significantly from weight tying.

### 9.6 No Fused Kernels in Optimizer

PyTorch's AdamW supports `fused=True` on CUDA, which fuses the parameter update into a single CUDA kernel per tensor, reducing kernel launch overhead and memory bandwidth usage. This is a free ~5-15% optimizer speedup on large models.

### 9.7 No Gradient Checkpointing in SFT

SFT typically uses longer effective sequences and a smaller batch. Without gradient checkpointing, VRAM peaks are higher during SFT than pretraining with the same model.

### 9.8 SFT Dataset Fully In-Memory

`build_sft_dataset()` loads all `SFTExample` objects (each holding two lists of `block_size` ints) into RAM. At 15k examples × 256 ints × 2 arrays × 8 bytes = ~62MB, this is trivial. At 10B-scale SFT with longer contexts and millions of examples, this approach fails.

### 9.9 Massive Code Duplication Between train.py and sft_train.py

`setup_training`, `build_optimizer`, `get_lr`, and `save_checkpoint` are copy-pasted with minor variations. Any fix to one must be applied to the other. This is a maintainability risk, not a correctness issue.

---

## 10. What Must Change Before Scaling to 10B / Multi-GPU

This is ordered by priority:

### 10.1 Replace the Architecture Core

| Component | Current | Required for 10B |
|-----------|---------|-----------------|
| Position encoding | Absolute learned | RoPE or ALiBi |
| Attention implementation | `nn.MultiheadAttention` + float mask | Flash Attention (`F.sdpa` with `is_causal=True`) |
| MLP activation | GELU | SwiGLU (standard in modern LLMs) |
| Attention | Full MHA | GQA (Grouped Query Attention) to reduce KV cache |
| Vocab size | 8,192 | 32,768–128,000 |
| Block size | 256 | 4,096–32,768 |
| n_layer | 6 | 32–80 |
| n_embd | 384 | 4,096–8,192 |
| n_head | 6 | 32–64 |

### 10.2 Implement Distributed Training

```python
# Minimal DDP skeleton needed in train.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
# Only rank 0 writes checkpoints
if dist.get_rank() == 0:
    save_checkpoint(...)
```

For 10B you will likely need **FSDP** (not DDP) to shard model weights across GPUs, or **Tensor Parallelism** via libraries like Megatron-LM or torchtitan.

### 10.3 Implement Gradient Accumulation (Fix the Config Gap)

```python
for micro_step in range(config.gradient_accumulation_steps):
    x, y = get_batch("train", config)
    with ctx:
        _, loss = model(x, y)
        loss = loss / config.gradient_accumulation_steps
    scaler.scale(loss).backward()
# step + update once outside the micro-step loop
```

This is also required for DDP to work correctly (the `no_sync()` context manager on all but the last micro-step avoids unnecessary gradient synchronization).

### 10.4 Enable Activation Checkpointing

```python
from torch.utils.checkpoint import checkpoint

# In Block.forward:
def forward(self, x):
    x = x + checkpoint(self.attn, self.ln_1(x), use_reentrant=False)
    x = x + checkpoint(self.mlp, self.ln_2(x), use_reentrant=False)
    return x
```

This roughly halves activation memory at the cost of one extra forward pass per block.

### 10.5 Use Fused AdamW

```python
optimizer = torch.optim.AdamW(
    optim_groups,
    lr=config.learning_rate,
    betas=(config.beta1, config.beta2),
    fused=True,  # add this — CUDA only, check availability first
)
```

### 10.6 Adopt a Proper Training Framework

For a 10B model, writing a training loop from scratch is impractical. The options, ranked by control vs. convenience:

| Framework | Best for |
|-----------|----------|
| **torchtitan** | Modern PyTorch-native, supports FSDP2, torch.compile, Flash Attention |
| **DeepSpeed** | ZeRO-3 offloading, most aggressive memory optimization |
| **Megatron-LM** | Tensor + pipeline parallelism, used by most frontier models |
| **Axolotl / LLaMA-Factory** | SFT/RLHF on existing open models, less from-scratch |

### 10.7 Data Pipeline Changes

- Tokenizer: retrain with 32k–128k vocab on a domain-appropriate corpus.
- Binary format: consider using the `datasets` library with memory-mapped arrow files instead of raw binary, which gives you shuffling, interleaving, and streaming for free.
- At 10B scale, pretraining requires hundreds of billions of tokens. The TinyStories corpus (~500M tokens piratized) is insufficient by 2–3 orders of magnitude.

### 10.8 LR Schedule and Batch Size Scaling

The Chinchilla scaling laws suggest:
$$N_{\text{tokens}} \approx 20 \times N_{\text{params}}$$
For a 10B model: ~200B tokens minimum, preferably 1T+ for a well-trained model.

Global batch size should be in the range of 1M–4M tokens (e.g., batch_size=1024 × block_size=4096 = 4M tokens/step). Achieve this through `gradient_accumulation_steps × (per-GPU batch) × num_GPUs`.

---

## 11. Summary Table

| Category | Assessment |
|----------|-----------|
| Model architecture | Correct and clean GPT-2 clone; not modern |
| Weight initialization | Mostly correct; missing scaled residual init |
| Optimizer setup | Correct weight decay separation; missing fused kernels |
| LR schedule | Correct warmup + cosine |
| Mixed precision | Correctly structured; bfloat16 preferred path is right |
| Gradient accumulation | **Configured but never implemented — bug** |
| SFT loss masking | Works but relies on implicit PyTorch default |
| Flash Attention | **Not available — float mask disables fast path** |
| Multi-GPU support | **Not implemented at all** |
| Activation checkpointing | **Not implemented** |
| Resumption / checkpointing | Good; minor security concern with weights_only=False |
| Position encoding | Absolute learned — **not scalable** |
| Context length | Fixed at 256 — **not scalable** |
| Vocab size | 8,192 — too small for general-purpose LLM |
| Code duplication | High between train.py and sft_train.py |
| Data pipeline | Clean; memmap is correct; SFT in-memory won't scale |
