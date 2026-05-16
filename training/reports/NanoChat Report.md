# Nanochat Training Pipeline — Technical Report

## 1. Overview

Nanochat is a minimal full-stack ChatGPT clone that implements the complete pipeline from pretraining through supervised fine-tuning (SFT) to reinforcement learning (RL). The training is organized in three sequential phases:

1. **Base pretraining** (`scripts/base_train.py`) — Language modeling on a large web corpus
2. **Supervised fine-tuning** (`scripts/chat_sft.py`) — Instruction-following on curated conversations
3. **Reinforcement learning** (`scripts/chat_rl.py`) — Policy gradient optimization on GSM8K math

The system targets PyTorch 2.9.1 with CUDA support (bf16 on Ampere+, fp16/fp32 fallback on older hardware, CPU/MPS also supported) and supports single-GPU and multi-GPU (via `torchrun`) training without using PyTorch DDP wrapping — the distributed optimizer handles all communication internally.

---

## 2. Libraries and Dependencies

### Core Dependencies (from `pyproject.toml`)

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.9.1 | Core deep learning framework (CUDA 12.8 or CPU) |
| `tokenizers` | >=0.22.0 | HuggingFace Tokenizers for BPE training |
| `tiktoken` | >=0.11.0 | OpenAI's fast BPE for inference-time tokenization |
| `rustbpe` | >=0.1.0 | Custom Rust-based BPE trainer (paired with tiktoken) |
| `kernels` | >=0.11.7 | Flash Attention 3 kernel loading (Hopper GPUs) |
| `wandb` | >=0.21.3 | Experiment tracking and logging |
| `datasets` | >=4.0.0 | HuggingFace Datasets (for SFT data loading) |
| `fastapi` | >=0.117.1 | Web server for chat interface |
| `uvicorn` | >=0.36.0 | ASGI server for FastAPI |
| `psutil` | >=7.1.0 | System resource monitoring |
| `pyarrow` | (transitive) | Parquet file reading for pretraining data |
| `filelock` | (transitive) | Concurrent download protection in multi-rank |
| `jinja2` | (transitive) | Template rendering for evaluation prompts |

### Dev Dependencies

| Library | Purpose |
|---------|---------|
| `transformers` | HuggingFace model loading for evaluation baselines |
| `matplotlib` | Plotting for scaling law analysis |
| `pytest` | Unit testing |
| `ipykernel` | Jupyter notebook support |

### Missing / Implicit Dependencies

- **`pyarrow`**: Required for Parquet file I/O in the dataloader but not listed as a direct dependency (likely a transitive dependency of `datasets`).
- **`jinja2`**: Required by `core_eval.py` for prompt template rendering but not listed directly (likely transitive via `datasets` or `transformers`).
- **NCCL**: Required for distributed training but provided by the PyTorch CUDA installation.
- **`flash-attn` / `flash-attn-3`**: Not a pip dependency — loaded dynamically via the `kernels` package from HuggingFace Hub at runtime (`kernels.get_kernel('varunneal/flash-attention-3')`). Only works on Hopper (SM 90) GPUs.

---

## 3. Model Architecture

The model is a decoder-only GPT Transformer defined in `nanochat/gpt.py`.

### 3.1 Configuration (`GPTConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sequence_len` | 2048 | Maximum context length |
| `vocab_size` | 32768 | Vocabulary size (padded to multiple of 64 for tensor core efficiency) |
| `n_layer` | 12 | Number of transformer blocks |
| `n_head` | 6 | Number of query attention heads |
| `n_kv_head` | 6 | Number of key/value heads (supports GQA when < n_head) |
| `n_embd` | 768 | Model dimension (= depth × aspect_ratio, nudged to nearest multiple of head_dim) |
| `window_pattern` | "SSSL" | Sliding window attention pattern tiled across layers |

The model dimension is computed as `depth × aspect_ratio` (default aspect_ratio=64), adjusted upward to the nearest multiple of `head_dim` (default 128) so that `n_head = n_embd / head_dim` divides evenly.

### 3.2 Architecture Details

**Notable design choices:**

1. **Rotary Position Embeddings (RoPE)** — No learned positional embeddings. Base theta = 100,000. Precomputed for 10× the sequence length.

2. **QK-Norm** — RMSNorm applied to queries and keys after rotary embedding, with a learned-free scale factor of 1.2 on both Q and K for sharper attention.

3. **No bias** — All linear layers are bias-free.

4. **No learnable RMSNorm parameters** — `F.rms_norm` with no affine parameters, runs in compute dtype.

5. **ReLU² activation** — MLP uses `F.relu(x).square()` instead of GELU/SwiGLU. MLP hidden dimension is 4× model dimension.

6. **Untied embeddings** — Token embedding (`wte`) and output projection (`lm_head`) have separate weight matrices.

7. **Pre-norm architecture** — RMSNorm applied before both attention and MLP within each block (standard pre-LN Transformer).

8. **Logit softcapping** — Output logits are passed through `softcap * tanh(logits / softcap)` with softcap=15 to prevent extreme logit magnitudes.

9. **Weight casting** — `Linear` subclass casts weights to match input dtype in forward pass (replaces autocast; master weights stay fp32 for optimizer precision).

### 3.3 Sliding Window Attention

The `window_pattern` string is tiled across layers. Characters map to window sizes:
- `L` = full context (sequence_len)
- `S` = short window (quarter context, ceil'd to FA3 tile size of 128)

The final layer always uses full context regardless of pattern. Default "SSSL" means 3 layers of short window then 1 layer of full context, repeating.

### 3.4 Value Residual (ResFormer)

Alternating layers include a **value embedding** (`nn.Embedding`) that provides a per-token value residual directly into the attention value computation:

```
v = v + gate * ve(token_ids)
```

where `gate` is a learned input-dependent sigmoid gate with 12 channels, scaled to range (0, 3). This implements the ResFormer idea of mixing stable value embeddings into attention values.

### 3.5 Per-Layer Learnable Scalars

- **`resid_lambdas`** — Scales the residual stream at each layer. Initialized with a decreasing schedule: `1.15 - 0.10 * i / (n_layer - 1)`.
- **`x0_lambdas`** — Blends initial (post-embedding) representation back into each layer's input. Initialized as decaying: `0.20 - 0.15 * i / (n_layer - 1)`.

The effective layer input is: `resid_lambdas[i] * x + x0_lambdas[i] * x0`.

### 3.6 Smear (Bigram Mixing)

A cheap mechanism that mixes the previous token's embedding into the current position:

```python
gate = smear_lambda * sigmoid(smear_gate(x[:, 1:, :24]))
x[:, 1:] = x[:, 1:] + gate * x[:, :-1]
```

This provides bigram-level information without an additional attention layer.

### 3.7 Backout (Mid-Layer Residual Subtraction)

At the halfway layer (`n_layer // 2`), the residual stream is cached. Before the final norm and projection, this mid-layer state is subtracted (scaled by `backout_lambda ≈ 0.2`) to remove low-level features that are not useful for next-token prediction.

### 3.8 Weight Initialization

| Component | Init | Standard Deviation |
|-----------|------|-------------------|
| `wte` (embedding) | Normal | std=0.8 |
| `lm_head` (output) | Normal | std=0.001 |
| `c_q`, `c_k`, `c_v` | Uniform(-s, s) | s = √3 / √n_embd |
| `c_proj`, `mlp.c_proj` | Zeros | 0 (output projections start at zero) |
| `mlp.c_fc` | Uniform(-0.4s, 0.4s) | 0.4× standard |
| Value embeddings | Uniform(-s, s) | Same as attention |

The Uniform distribution with bound `√3 * std` gives the same standard deviation as Normal but avoids outliers.

---

## 4. Tokenizer

### 4.1 Architecture

The system supports two tokenizer backends:

1. **RustBPE + tiktoken** (primary) — `rustbpe` for training, `tiktoken` for fast inference. Uses pickle serialization.
2. **HuggingFace Tokenizer** (alternative) — Full HuggingFace Tokenizers library with BPE model + ByteLevel pre/post processing.

### 4.2 Configuration

- **Vocab size**: 32,768 (default)
- **Split pattern**: GPT-4 style regex with modified numeric grouping (`\p{N}{1,2}` instead of `\p{N}{1,3}` — optimized for smaller vocab sizes)
- **Byte fallback**: Enabled (every byte sequence is representable)
- **Pre-tokenization**: GPT-4 regex split + ByteLevel encoding

### 4.3 Special Tokens

| Token | Purpose |
|-------|---------|
| `<\|bos\|>` | Beginning of sequence / document delimiter |
| `<\|user_start\|>` / `<\|user_end\|>` | User message boundaries |
| `<\|assistant_start\|>` / `<\|assistant_end\|>` | Assistant message boundaries |
| `<\|python_start\|>` / `<\|python_end\|>` | Python REPL tool invocation |
| `<\|output_start\|>` / `<\|output_end\|>` | Tool output boundaries |

### 4.4 Conversation Rendering

The tokenizer implements `render_conversation()` which converts a structured conversation (list of role/content message dicts) into a flat token sequence with a corresponding loss mask. The mask is `1` only for tokens the assistant is expected to generate (user messages, system messages, and special tokens are masked out for loss computation).

### 4.5 Token Bytes Tensor

A `token_bytes` tensor (vocab_size,) maps each token ID to its byte length. Special tokens map to 0 bytes. This enables computing **bits per byte (bpb)** — a tokenization-independent evaluation metric.

### 4.6 Training (`scripts/tok_train.py`)

The tokenizer is trained on the same ClimbMix data used for pretraining, using `rustbpe` for the BPE merge learning.

---

## 5. Optimizer

The optimizer is a **combined MuonAdamW** defined in `nanochat/optim.py`, with two variants:
- `MuonAdamW` — Single-GPU
- `DistMuonAdamW` — Distributed (multi-GPU, ZeRO-2 style sharding)

### 5.1 Parameter Grouping

| Group | Optimizer | Parameters |
|-------|-----------|------------|
| Unembedding | AdamW | `lm_head` weights |
| Embedding | AdamW | `wte` weights |
| Value embeddings | AdamW | All value embedding weights |
| Residual scalars | AdamW | `resid_lambdas` |
| x0 scalars | AdamW | `x0_lambdas` |
| Smear params | AdamW | `smear_gate`, `smear_lambda`, `backout_lambda` |
| Matrix params | Muon | All transformer block parameters (grouped by shape) |

### 5.2 AdamW

Standard fused AdamW implementation (`@torch.compile(dynamic=False, fullgraph=True)`):
- Per-parameter exponential moving averages (first and second moment)
- Decoupled weight decay (applied before the update)
- Bias correction for both moments
- Different `(beta1, beta2)` tuples per group (e.g., embeddings use (0.8, 0.995), lm_head uses (0.8, 0.96))

Learning rates for AdamW groups are scaled by `∝ 1/√(d_model / 768)` (μP-style width scaling).

### 5.3 Muon (Momentum + Orthogonalization)

Muon applies SGD-momentum followed by an orthogonalization step using the **Polar Express Sign Method** (5 iterations, coefficients precomputed for safety_factor=2e-2):

1. **Nesterov Momentum** — `g = lerp(momentum_buffer, grad, 1-β)`, then Nesterov lookahead.
2. **Polar Express Orthogonalization** — Iterative Newton-Schulz-like computation that produces an approximately orthogonal matrix (`US'V^T` where `S'` ≈ Uniform(0.5, 1.5)). Runs in bf16 for speed.
3. **NorMuon Variance Reduction** — Per-neuron adaptive learning rate that normalizes update scales after orthogonalization. Uses a factored second moment buffer (per-row or per-column depending on matrix aspect ratio).
4. **Cautious Weight Decay** — Applies decay only to parameters where the update and parameter have the same sign: `p -= lr * (g + wd * p * mask)` where `mask = (g * p >= 0)`.

All Muon parameters of the same shape are **stacked** into a single tensor for efficient batched computation and communication.

### 5.4 Distributed Communication (DistMuonAdamW)

The distributed optimizer implements a **3-phase async communication pattern**:

1. **Phase 1**: Launch all async `reduce_scatter` / `all_reduce` operations
2. **Phase 2**: Wait for each reduce → compute update → launch `all_gather` (overlap compute with earlier gathers)
3. **Phase 3**: Wait for all gathers → copy updated params back

**AdamW (ZeRO-2 style):**
- Small params (<1024 elements): `all_reduce` gradients, full update on all ranks
- Large params: `reduce_scatter` gradients, each rank updates only its shard, then `all_gather`

**Muon:**
- All same-shape params stacked into one tensor
- Divided across ranks (each rank owns `ceil(K/N)` params)
- `reduce_scatter` → local Muon update → `all_gather`
- Optimizer state (momentum buffers) is sharded

### 5.5 0-D CPU Tensors

All hyperparameters (lr, momentum, beta, weight_decay, etc.) are stored as 0-dimensional CPU tensors. This avoids `torch.compile` recompilation when hyperparameter values change during training (e.g., learning rate warmup/warmdown).

---

## 6. Learning Rate and Momentum Schedules

### 6.1 Learning Rate Schedule (Pretraining)

A **trapezoidal** schedule: linear warmup → constant → linear warmdown.

```
LR multiplier = {
    warmup:   (step + 1) / warmup_steps          (default 40 steps)
    constant: 1.0
    warmdown: linear decay to final_lr_frac       (default 0.05)
}
```

The warmdown phase occupies `warmdown_ratio` (default 0.65) of total iterations.

### 6.2 Muon Momentum Schedule

```
momentum = {
    0..400:   linear ramp 0.85 → 0.97
    400..warmdown_start: constant 0.97
    warmdown: linear decay 0.97 → 0.90
}
```

### 6.3 Weight Decay Schedule

Cosine decay to zero over training: `wd(t) = wd_0 * 0.5 * (1 + cos(πt/T))`

### 6.4 SFT Schedule

Starts at `init_lr_frac` (default 0.8) of the pretrained learning rates, with optional warmup and linear warmdown to `final_lr_frac` (default 0.0).

### 6.5 RL Schedule

Simple linear decay: `lr_mult = 1 - step/num_steps`

---

## 7. Scaling Laws and Hyperparameter Transfer

### 7.1 Training Horizon

The system uses scaling laws to determine the optimal number of training tokens:

```
target_tokens = target_param_data_ratio × scaling_params
```

Where `scaling_params = transformer_matrices + lm_head` parameters, and `target_param_data_ratio` defaults to 12 (Chinchilla used ~20).

### 7.2 Batch Size Scaling

Following the **Power Lines** paper (arXiv:2505.13738):

```
B_opt ∝ D^0.383
```

where `D` = number of training tokens. Reference point: `B_ref = 2^19 ≈ 524,288` tokens for a d12 model.

### 7.3 Learning Rate Scaling with Batch Size

```
η ∝ √(B / B_ref)
```

Square-root scaling is applied to all learning rates (both AdamW and Muon groups).

### 7.4 Weight Decay Scaling

Following the T_epoch framework (arXiv:2405.13698):

```
λ = λ_ref × √(B/B_ref) × (D_ref/D)
```

This keeps the effective regularization strength constant across different training horizons and batch sizes.

### 7.5 μP-Style Depth Transfer

The d12 model serves as the reference. All hyperparameters are tuned at d12, then transferred to larger models via the scaling rules above.

---

## 8. Dataset Format and Data Loading

### 8.1 Pretraining Data Format

- **Dataset**: ClimbMix-400B (hosted on HuggingFace: `karpathy/climbmix-400b-shuffle`)
- **Format**: Sharded Parquet files (`shard_00000.parquet` through `shard_06542.parquet`)
- **Schema**: Each Parquet file contains row groups, each row group has a `'text'` column of document strings
- **Split**: All shards except the last → train; last shard → validation
- **Storage**: `~/.cache/nanochat/base_data_climbmix/`
- **Download**: On-demand with retry logic and parallel multiprocessing support

### 8.2 Pretraining DataLoader (BOS-Aligned Best-Fit)

The dataloader (`nanochat/dataloader.py`) implements a **BOS-aligned best-fit packing** strategy:

**Algorithm for each row in the batch (capacity = T+1 tokens):**
1. Maintain a buffer of tokenized documents (default 1000)
2. Find the **largest document that fits entirely** in remaining space
3. If found, pack it in; repeat
4. If nothing fits, crop the **shortest document** in the buffer to fill exactly

**Properties:**
- Every row starts with a BOS token
- 100% utilization (no padding, every position is trained on)
- ~35% of all tokens are discarded due to cropping
- Documents are never split across rows (every token can attend back to BOS)

**Performance optimizations:**
- Pre-allocated pinned CPU buffers for zero-copy H2D transfer
- Single contiguous `gpu_buffer` (2×B×T) with views for inputs/targets
- Async `non_blocking=True` GPU copy
- Multi-threaded tokenization (default 4 threads)
- Batch tokenization of documents

**DDP sharding**: Each rank reads different row groups (`rg_idx = ddp_rank + i * world_size`), so all ranks see different data without explicit data sharding.

**Resumption**: The dataloader tracks `(pq_idx, rg_idx, epoch)` state and can resume from any point.

### 8.3 SFT DataLoader

For supervised fine-tuning, the dataloader uses a **BOS-aligned best-fit pad** strategy:

- Same best-fit packing as pretraining, but instead of cropping, remaining space is **padded** (target=-1, ignored by cross-entropy loss)
- Conversations are never truncated — ensures no training signal is lost
- Data comes from a `TaskMixture` that deterministically shuffles all examples across tasks

### 8.4 SFT Data Mixture

| Task | Source | Size | Purpose |
|------|--------|------|---------|
| SmolTalk | HuggingFace `smol-smoltalk` | 460K rows | General conversation |
| Identity conversations | Custom JSONL | 1K rows × 2 epochs | Model identity |
| MMLU | Auxiliary train | 100K × 3 epochs | Multiple choice reasoning |
| GSM8K | Main train | 8K × 4 epochs | Math problem solving + tool use |
| SimpleSpelling | Synthetic | 200K rows | Basic spelling tasks |
| SpellingBee | Synthetic | 80K rows | Letter counting tasks |

### 8.5 RL Data

RL uses only **GSM8K train** (7,473 problems). Each optimization step samples `examples_per_step` (default 16) problems, generates `num_samples` (default 16) completions per problem, scores them with a reward function, and computes policy gradient updates.

---

## 9. FP8 Training (Optional)

Enabled with `--fp8` flag on H100+ GPUs. Implemented in `nanochat/fp8.py` as a minimal ~150-line alternative to torchao's ~2000-line Float8Linear.

### 9.1 Approach

A custom `autograd.Function` (`_Float8Matmul`) wraps each matmul:
1. Compute dynamic scale: `scale = FP8_MAX / max(|tensor|)`
2. Quantize to FP8 (saturating clamp)
3. Call `torch._scaled_mm` (cuBLAS FP8 kernel, ~2× faster than bf16)
4. Return full-precision output

**FP8 dtypes:**
- `float8_e4m3fn` — Used for inputs and weights (higher precision, range [-448, 448])
- `float8_e5m2` — Used for gradients (wider range, range [-57344, 57344])

### 9.2 Module Filtering

Not all linear layers are converted:
- Dimensions must be divisible by 16 (hardware requirement)
- Minimum dimension ≥ 128 (small layers aren't worth the overhead)
- Embeddings and small projection layers stay in bf16

### 9.3 Evaluation

FP8 is disabled during evaluation via a context manager (`disable_fp8`) that temporarily swaps Float8Linear modules back to standard Linear for consistent bf16 evaluation.

---

## 10. Flash Attention

### 10.1 Implementation (`nanochat/flash_attention.py`)

A unified interface that automatically selects the backend:

| Backend | Hardware | Loaded via |
|---------|----------|------------|
| Flash Attention 3 | Hopper (SM 90) + bf16 | `kernels` package from HuggingFace Hub |
| PyTorch SDPA | Everything else (Ada, Blackwell, CPU, MPS) | `torch.nn.functional.scaled_dot_product_attention` |

### 10.2 API

Two functions matching the FA3 interface:
- `flash_attn_func(q, k, v, causal, window_size)` — Training (no KV cache)
- `flash_attn_with_kvcache(q, k_cache, v_cache, k, v, cache_seqlens, ...)` — Inference

Tensors use FA3's native `(B, T, H, D)` layout. The SDPA fallback transposes to `(B, H, T, D)` internally.

### 10.3 Sliding Window Support

- FA3: native `window_size=(left, 0)` parameter
- SDPA: constructs an explicit boolean attention mask when sliding window is needed (not supported natively by SDPA)

**Warning**: SDPA without sliding window support leads to poor GPU utilization. The training script warns and recommends `--window-pattern L` when FA3 is unavailable.

---

## 11. Inference Engine

The `Engine` class (`nanochat/engine.py`) provides efficient inference:

- **KV Cache**: Pre-allocated `(n_layers, B, T_max, H_kv, D)` tensors, managed per-batch-element via `cache_seqlens`
- **Batched generation**: `generate_batch()` generates multiple samples in parallel
- **Calculator tool**: Safe `eval()` with timeout for arithmetic expressions and `.count()` string operations, sandboxed with restricted builtins
- **Streaming**: The base model's `generate()` method is a Python generator yielding one token at a time

---

## 12. Evaluation

### 12.1 Base Model Evaluation

**Bits Per Byte (bpb):**
- Tokenization-independent metric
- Sum of per-token NLL weighted by token byte length
- Special tokens (0 bytes) are excluded
- All ranks participate, then `all_reduce` totals

**CORE Metric:**
- In-context learning evaluation from the DCLM paper (arXiv:2406.11794)
- Multiple choice, schema, and language modeling tasks
- Uses few-shot prompting with jinja2 templates
- Compares log-likelihoods of completions

### 12.2 Chat Model Evaluation

**Generative** (pass@k): Generate k samples, check if any passes the task's `evaluate()` method.

**Categorical** (logit-based): Batch multiple-choice questions, compare logits at answer positions.

**Tasks**: ARC-Easy/Challenge, MMLU, GSM8K, HumanEval, SpellingBee.

---

## 13. Distributed Training

### 13.1 Launch

```bash
torchrun --nproc_per_node=8 -m scripts.base_train
```

### 13.2 Communication

- **Backend**: NCCL (with `device_id` initialization for modern PyTorch)
- **No DDP wrapper**: The `DistMuonAdamW` optimizer handles all gradient reduction and parameter synchronization internally
- **GradScaler (fp16)**: When using fp16, `found_inf` flags are `all_reduce`'d with `MAX` so all ranks agree on whether to skip the step

### 13.3 Reproducibility

- Global seed 42 for weight initialization
- Explicit `torch.Generator` objects for sampling during inference
- `torch.set_float32_matmul_precision("high")` for TF32 matmuls on CUDA

---

## 14. Checkpointing

### 14.1 Saved Artifacts

| File | Contents |
|------|----------|
| `model_{step:06d}.pt` | Model `state_dict` (rank 0 only) |
| `optim_{step:06d}_rank{N}.pt` | Optimizer state (per-rank, sharded) |
| `meta_{step:06d}.json` | Metadata: model config, user config, loop state, dataloader state |

### 14.2 Resumption

Training can resume from any checkpoint via `--resume-from-step`. The system restores:
- Model parameters
- Optimizer state (momentum buffers, second moments)
- Dataloader position (parquet file index, row group index, epoch)
- Loop state (step counter, smoothed loss, total training time, min val bpb)

---

## 15. Memory and Performance Optimizations

1. **`torch.compile(dynamic=False)`** — Model is compiled with static shapes for maximum kernel fusion
2. **Expandable segments** — `PYTORCH_ALLOC_CONF=expandable_segments:True` reduces memory fragmentation
3. **Manual GC management** — GC is disabled after first step (`gc.freeze()`, `gc.disable()`), manual collection every 5000 steps
4. **Gradient accumulation** — Total batch size achieved via micro-batching without storing all activations simultaneously
5. **Meta device initialization** — Model built on `meta` device (shapes only), then `to_empty(device)` and `init_weights()` to avoid double allocation
6. **Pin memory** — CPU staging buffers use `pin_memory=True` for async H2D transfers
7. **Buffer reuse** — Distributed optimizer reuses `stacked_grads` buffer as `all_gather` output to reduce peak memory

---

## 16. Reinforcement Learning

### 16.1 Algorithm

A simplified REINFORCE variant (called "GRPO" in the code, but significantly simplified):

1. **No trust region** — No KL regularization to a reference model
2. **On-policy** — No importance sampling ratio or PPO clipping
3. **DAPO-style token-level normalization** — Loss normalized by number of valid tokens
4. **Mean-subtracted advantages** — `advantage = reward - mean(rewards)` (no σ normalization)

### 16.2 Training Loop

For each step:
1. Sample `examples_per_step` problems from GSM8K
2. For each problem, generate `num_samples` completions (batched, with KV cache)
3. Score completions with the task's `reward()` function
4. Compute per-token log-probabilities via a forward pass
5. Compute policy gradient: `loss = -Σ(logp * advantage) / num_valid_tokens`
6. Mask prompt tokens and tool-use tokens (mask=0 from Engine)
7. Aggregate gradients and step optimizer

### 16.3 Loss Masking

The Engine returns `mask=0` for both prompt tokens AND tool-use forced tokens. The RL loss only backpropagates through tokens the model freely generated.

---

## 17. Logging and Reporting

- **Weights & Biases**: Optional (enabled when `--run` is not "dummy"). Logs loss, MFU, token throughput, evaluation metrics.
- **DummyWandb**: No-op wrapper when W&B is disabled (single-GPU debugging or non-master ranks).
- **Report module** (`nanochat/report.py`): Logs training configuration and outcomes for structured reporting.
- **Console logging**: MFU %, tokens/sec, ETA, smoothed training loss (EMA with β=0.9, debiased).

---

## 18. Summary of Training Phases

| Phase | Script | Data | Loss | Key Hyperparameters |
|-------|--------|------|------|---------------------|
| Pretrain | `base_train.py` | ClimbMix-400B (Parquet) | Cross-entropy (next token) | Auto-computed from scaling laws |
| SFT | `chat_sft.py` | SmolTalk + MMLU + GSM8K + Spelling | Masked cross-entropy (assistant tokens only) | Inherited from pretrain, warmdown to 0 |
| RL | `chat_rl.py` | GSM8K (with rollouts) | -REINFORCE (token-level, mean-subtracted) | 0.05× init LR, linear decay |

---

## 19. Known Limitations and Notes

1. **FA3 only on Hopper** — Blackwell (SM 100) and Ada (SM 89) GPUs fall back to SDPA, which does not natively support sliding window attention (requires explicit masking).
2. **Muon scaling assumptions** — The batch size LR scaling and weight decay scaling are derived from AdamW theory; their applicability to Muon is assumed but "not studied carefully" (per code comments).
3. **No FSDP** — The distributed optimizer implements its own sharding; PyTorch's FSDP/FSDP2 is not used.
4. **Tokenizer number grouping** — The `\p{N}{1,2}` regex choice (vs GPT-4's `{1,3}`) is empirically motivated for 32K vocab but acknowledged as not rigorously validated.
5. **Rotary embedding scaling** — Uses 10× over-allocated buffer with an assert; no dynamic growth.
6. **ClimbMix upgrade** — Legacy support for FinewebEdu-100B dataset is still present with migration warnings.
