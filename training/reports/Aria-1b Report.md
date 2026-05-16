# Technical Report: `train-aria` — 1B Parameter LLM Training Pipeline

This report documents the three‑stage training pipeline contained in this folder:

1. **Pretraining** — [train.py](train.py)
2. **Supervised Fine‑Tuning (SFT)** — [train_sft.py](train_sft.py)
3. **Direct Preference Optimization (DPO)** — [train_dpo.py](train_dpo.py)

Auxiliary configuration is provided by [config.json](config.json) (HuggingFace‑style model card, intended for inference / export) and [tokenizer_config.json](tokenizer_config.json) (tokenizer metadata + chat template).

> **Important caveat — missing source.** All three scripts depend on a Python package named `model/` that lives on `sys.path` next to them (see the `sys.path.insert(...)` line at the top of every script). Specifically they import:
> - `model.config.ModelConfig`, `model.config.TrainConfig`
> - `model.transformer.Transformer`
> - `model.data.get_tokenizer`, `model.data.create_dataloader`
> - `model.sft_data.SFTDataset`, `model.sft_data.sft_collate_fn`
> - `model.dpo_data.DPODataset`, `model.dpo_data.dpo_collate_fn`
>
> **None of these files are present in this workspace.** The architectural / data details below are reconstructed from how the symbols are *used* in the training scripts and from [config.json](config.json). Anything that cannot be confirmed from the visible code is explicitly flagged as "not visible in workspace".

---

## 1. Software Stack

Imports observed across the three training scripts:

| Library | Used for |
| --- | --- |
| `torch` | Tensors, autograd, BF16 autocast, `AdamW` optimizer, `clip_grad_norm_`. |
| `torch.distributed` (`dist`) | NCCL process group, `barrier()` for checkpoint sync. |
| `torch.nn.parallel.DistributedDataParallel` | Multi‑GPU data parallelism. |
| `torch.nn.functional` (DPO only) | `log_softmax`, `logsigmoid` for the DPO loss. |
| `torch.utils.data.DataLoader` + `torch.utils.data.distributed.DistributedSampler` | SFT/DPO map‑style datasets. |
| `os`, `sys`, `math`, `time`, `json`, `datetime`, `glob` | Standard library plumbing (paths, schedules, JSONL logs, NCCL timeout, checkpoint discovery). |

Implicit / not directly imported but required by the missing `model/` package:

- A **tokenizer library** — `tokenizer_config.json` declares `"tokenizer_class": "LlamaTokenizerFast"` and `"backend": "tokenizers"`, so `get_tokenizer()` is almost certainly built on top of HuggingFace **`transformers`** (`LlamaTokenizerFast`) backed by the **`tokenizers`** Rust library. *Not directly visible in the training scripts.*
- A **dataset loading library** — pretraining streams "FineWeb‑Edu 10BT" (HuggingFace **`datasets`**, presumably with `streaming=True`), and SFT/DPO reference UltraChat‑200K and UltraFeedback. *Not directly visible.*
- **NCCL** runtime (selected as the DDP backend).

Launcher: **`torchrun --nproc_per_node=8`** (documented in each script's docstring). Target hardware: **8× H100 80 GB**.

No use of `torch.compile`, FSDP, ZeRO, DeepSpeed, FlashAttention wrappers, or mixed‑precision `GradScaler` — the codebase is intentionally minimal: plain DDP + BF16 autocast (BF16 has enough dynamic range that no loss scaler is needed, and the comment in [train.py](train.py) calls this out explicitly).

---

## 2. Model Architecture

### 2.1 What the code instantiates

The training scripts only do:

```python
model_config = ModelConfig()
model = Transformer(model_config)
```

so the architecture itself is defined inside the missing `model/transformer.py`. From the way it is used we can confirm:

- It is a **decoder‑only causal LM**. The forward signature is `logits, loss = model(input_ids, labels=None)` — labels optional, loss returned when present (standard HF‑like convention).
- It exposes `model.tok_embeddings` (an `nn.Embedding`) and `model.output` (an `nn.Linear(hidden_dim, vocab_size, bias=False)`), and these two are **separate tensors** (untied embeddings) — see the SFT vocab‑expansion block in [train_sft.py](train_sft.py#L101-L116).
- It exposes a `model.config` object with a mutable `vocab_size` attribute.
- The DPO script comments mention "shared RoPE buffers" ([train_dpo.py](train_dpo.py#L66-L67)), so the model uses **Rotary Position Embeddings**.

### 2.2 Architecture parameters (from [config.json](config.json))

`config.json` is a HuggingFace `LlamaForCausalLM` card. Although it is not loaded by the training scripts, it is the canonical export of this model and matches the print banner in [train.py](train.py#L70-L77):

| Field | Value | Meaning |
| --- | --- | --- |
| `architectures` | `LlamaForCausalLM` | LLaMA‑family decoder. |
| `model_type` | `llama` | — |
| `vocab_size` | **32003** | 32000 base + 3 chat specials (`<|user|>`, `<|assistant|>`, `<|end|>`). |
| `hidden_size` | **2048** | Model / residual dimension. |
| `intermediate_size` | **5504** | SwiGLU MLP inner dim. |
| `num_hidden_layers` | **22** | Transformer blocks. |
| `num_attention_heads` | **32** | Query heads → head_dim = 64. |
| `num_key_value_heads` | **8** | **Grouped‑Query Attention**, 4 Q heads share each KV head. |
| `max_position_embeddings` | **2048** | Context length. |
| `rope_theta` | `10000.0` | RoPE base frequency. |
| `rms_norm_eps` | `1e-5` | RMSNorm epsilon. |
| `hidden_act` | `silu` | SwiGLU uses SiLU/Swish. |
| `tie_word_embeddings` | `false` | Confirms separate input/output embeddings. |
| `torch_dtype` | `bfloat16` | Storage dtype for export. |
| `bos_token_id` / `eos_token_id` / `pad_token_id` | 1 / 2 / 2 | Pad reuses EOS. |

The pretraining banner ([train.py](train.py#L70-L77)) prints exactly these numbers via `model_config.num_layers`, `hidden_dim`, `num_attention_heads`, `num_kv_heads`, `intermediate_dim`, `max_seq_len`, `vocab_size` — so `ModelConfig` mirrors this `LlamaConfig`. The print at [train.py](train.py#L84) also confirms a parameter count near 1 B (the "1B Transformer" in the docstring).

**Summary:** ~1.0–1.1 B parameter LLaMA‑style decoder — 22 layers, hidden 2048, 32 query heads / 8 KV heads (GQA), SwiGLU MLP with intermediate 5504, RMSNorm, RoPE (θ=10 000), 2048 context, untied embeddings.

---

## 3. Tokenizer

Defined by [tokenizer_config.json](tokenizer_config.json):

- **Class:** `LlamaTokenizerFast` (HuggingFace `transformers`, Rust `tokenizers` backend). The actual SentencePiece / BPE merges file is **not present in the workspace** — only the config.
- **Base specials:** `bos="<s>"` (id 1), `eos="</s>"` (id 2), `unk="<unk>"` (id 0), `pad="</s>"` (pad id = eos id = 2).
- **Added specials (ids 32000–32002):** `<|user|>`, `<|assistant|>`, `<|end|>`. These are added **after** pretraining at the start of SFT.
- **`model_max_length`:** 2048.
- **Chat template** (Jinja2):

  ```
  <|user|>\n{user}\n<|end|>\n<|assistant|>\n{assistant}\n<|end|>\n
  ```

  with an `add_generation_prompt` branch that appends a trailing `<|assistant|>\n` for inference.

`get_tokenizer()` from `model.data` (not visible) is presumably a thin wrapper that returns this tokenizer with `pad_token_id` set to the EOS id. The SFT/DPO scripts rely on `tokenizer.get_vocab()`, `tokenizer.add_tokens(..., special_tokens=True)`, `len(tokenizer)`, and `tokenizer.pad_token_id` — all standard HF tokenizer API.

---

## 4. Stage 1 — Pretraining ([train.py](train.py))

### 4.1 Distributed setup

- `dist.init_process_group("nccl", timeout=30 min)` ([train.py](train.py#L52)).
- Reads `RANK`, `LOCAL_RANK`, `WORLD_SIZE` from env (set by `torchrun`).
- `torch.cuda.set_device(local_rank)`; model wrapped in `DDP(model, device_ids=[local_rank])` ([train.py](train.py#L88)).

### 4.2 Optimizer

`torch.optim.AdamW` with **fused** kernel and **decoupled weight decay** on matrices only:

- `decay_params`  = parameters with `dim() >= 2` → `weight_decay = train_config.weight_decay`.
- `nodecay_params` = parameters with `dim() <  2` (biases, RMSNorm gains) → `weight_decay = 0`.
- Betas: `(train_config.beta1, train_config.beta2)` (values not visible — defined inside `TrainConfig`).

See [train.py](train.py#L91-L96).

### 4.3 Learning‑rate schedule — **Warmup‑Stable‑Decay (WSD)**

Implemented in `get_wsd_lr` ([train.py](train.py#L26-L35)):

- **Warmup:** linear from 0 to `max_lr` over `warmup_steps`.
- **Stable:** constant `max_lr` until `0.8 × total_steps`.
- **Decay:** cosine decay from `max_lr` down to `min_lr` over the final 20 % of steps.

`total_steps` is derived dynamically from `train_config.total_tokens` and the effective batch (see §4.5). `warmup_steps`, `learning_rate`, `min_lr`, `total_tokens` are stored in `TrainConfig` (not visible).

### 4.4 Precision & gradient handling

- Forward + loss inside `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` ([train.py](train.py#L165)).
- No `GradScaler` (BF16 doesn't need one).
- `torch.nn.utils.clip_grad_norm_(..., train_config.grad_clip)` after the accumulation loop ([train.py](train.py#L173)).
- Gradient accumulation: `train_config.gradient_accumulation_steps` micro‑steps per optimizer step, loss divided by accumulation count to keep the gradient unbiased ([train.py](train.py#L155-L171)).

### 4.5 Batch / token math

```
eff_batch       = batch_size_per_gpu * world_size * gradient_accumulation_steps
tokens_per_step = eff_batch * max_seq_len
total_steps     = total_tokens // tokens_per_step
```

([train.py](train.py#L62-L64)). Concrete numbers depend on `TrainConfig`, which is not visible.

### 4.6 Dataset and dataloader

- Function `create_dataloader(tokenizer, train_config, rank, world_size, seed_override)` from `model.data` (not visible).
- Comment ([train.py](train.py#L131)): "streaming FineWeb‑Edu 10BT" → likely `datasets.load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", streaming=True)` packed into 2048‑token chunks.
- Yields `(input_ids, labels)` pairs of shape `[batch_size_per_gpu, max_seq_len]`.
- The seed passed in is `train_config.seed + resume_step`, so a resumed run sees a *different* shuffled stream than the original — important to avoid reusing the exact same micro‑batches after restart ([train.py](train.py#L122-L125)).

**Format details** (sharding strategy, EOS packing, attention masking) are inside `model.data` and are **not visible** here.

### 4.7 Checkpointing & resume

- `find_latest_checkpoint(checkpoint_dir)` globs `step_*.pt` and picks the highest step ([train.py](train.py#L38-L47)).
- On startup, if a checkpoint exists, it is loaded with `torch.load(..., weights_only=False)` and `model.module.load_state_dict(...)` + `optimizer.load_state_dict(...)` ([train.py](train.py#L107-L120)). `resume_step` becomes the new starting `global_step`.
- Periodic save every `train_config.save_interval` steps, gated by a `dist.barrier()` so all ranks reach the save point before rank 0 writes ([train.py](train.py#L207-L220)).
- Saved blob: `step`, `model` state_dict (unwrapped from DDP via `.module`), `optimizer` state_dict, latest `loss`, and a `config` dict snapshotting `model_config.__dict__` and `train_config.__dict__`.
- A `final.pt` is written at the end without optimizer state.

### 4.8 Logging

- Console line every `log_interval` steps with: avg loss, current LR, tokens/s, peak GPU memory (`torch.cuda.max_memory_allocated`), % done, ETA.
- JSONL log appended to `<log_dir>/train_log.jsonl` (mode `"a"`, so it survives resumes) — only on rank 0.

### 4.9 Training loop summary

```
while global_step < total_steps:
    zero_grad
    for micro in range(grad_accum):
        input_ids, labels = next(data_iter)
        with bf16 autocast:
            _, loss = model(input_ids, labels)
            loss = loss / grad_accum
        loss.backward()
    clip_grad_norm_
    set_lr(WSD(global_step))
    optimizer.step()
    log / checkpoint
```

---

## 5. Stage 2 — SFT ([train_sft.py](train_sft.py))

### 5.1 Configuration (hard‑coded constants at the top of the file)

| Setting | Value |
| --- | --- |
| `BASE_CHECKPOINT` | `/checkpoints/step_19000.pt` |
| `SFT_CHECKPOINT_DIR` | `/checkpoints_sft` |
| `LOG_DIR` | `/training/logs` |
| `DATA_CACHE` | `/data` |
| `NUM_EPOCHS` | 2 |
| `BATCH_SIZE_PER_GPU` | 4 |
| `GRADIENT_ACCUMULATION` | 4 (eff. batch 4·8·4 = **128**) |
| `MAX_SEQ_LEN` | 2048 |
| `LEARNING_RATE` | 2e‑5 → `MIN_LR` 2e‑6 |
| `WARMUP_STEPS` | 200 |
| `WEIGHT_DECAY` | 0.01 |
| `GRAD_CLIP` | 1.0 |
| Betas (AdamW) | (0.9, 0.95) |
| `LOG_INTERVAL` / `SAVE_INTERVAL` | 10 / 500 |

### 5.2 Loading the base model + vocab expansion

1. Build a fresh `Transformer(ModelConfig())` and `load_state_dict(ckpt["model"])` from the pretraining checkpoint ([train_sft.py](train_sft.py#L79-L88)).
2. Add `<|user|>`, `<|assistant|>`, `<|end|>` to the tokenizer if missing ([train_sft.py](train_sft.py#L91-L96)).
3. If the new vocab is larger than the model's, **resize both embedding and output head**:
   - Allocate new `nn.Embedding(new_vocab, hidden)` and copy old rows; initialize new rows to the **mean of existing embeddings** (better than random init for warm‑start) ([train_sft.py](train_sft.py#L101-L110)).
   - Allocate new `nn.Linear(hidden, new_vocab, bias=False)` and copy old rows. *Note:* new output rows are left at PyTorch's default Kaiming init (no mean‑copy here — possibly intentional asymmetry, possibly an oversight).
4. Update `model.config.vocab_size`.

### 5.3 Optimizer / scheduler

- Same parameter‑group split as pretraining (decay vs no‑decay), `AdamW(fused=True)`, betas (0.9, 0.95).
- Schedule: **plain cosine with warmup** (`get_cosine_lr`) — no stable plateau ([train_sft.py](train_sft.py#L49-L53)).

### 5.4 Dataset

- `SFTDataset(tokenizer, max_seq_len, split="train_sft", cache_dir=DATA_CACHE)` from `model.sft_data` (not visible).
- File docstring states it is **UltraChat 200K** (HF dataset `HuggingFaceH4/ultrachat_200k`, split `train_sft`).
- Map‑style dataset → wrapped in `DistributedSampler(shuffle=True)` and `DataLoader(num_workers=4, pin_memory=True)` with a custom `sft_collate_fn(batch, pad_id=tokenizer.pad_token_id)` ([train_sft.py](train_sft.py#L132-L141)).
- The collate signature receiving `pad_id` strongly implies right‑padding to the longest item in the batch, returning `(input_ids, labels)` where the prompt portion of `labels` is masked to the ignore index so the loss is computed only on assistant turns. **The masking strategy itself is not visible** — confirm in `model/sft_data.py`.

### 5.5 Loop

Standard epoch loop with `sampler.set_epoch(epoch)` for proper shuffling. Same BF16 autocast + accumulation + clip + step + cosine‑LR pattern as pretraining. Exits inner accumulation early on `StopIteration` and breaks out of the epoch when the dataloader is empty (`if batch_loss == 0: break`) — see [train_sft.py](train_sft.py#L181-L194).

Checkpoints (`sft_step_*.pt`, `sft_final.pt`) save `step`, `model.module.state_dict()`, `model_config.__dict__`, and the new `vocab_size`. **Optimizer state is *not* saved** in SFT — there is no resume path here.

---

## 6. Stage 3 — DPO ([train_dpo.py](train_dpo.py))

### 6.1 Configuration

| Setting | Value |
| --- | --- |
| `SFT_CHECKPOINT` | `/checkpoints_sft/sft_final.pt` |
| `DPO_CHECKPOINT_DIR` | `/checkpoints_dpo` |
| `NUM_EPOCHS` | 1 |
| `BATCH_SIZE_PER_GPU` | 2 |
| `GRADIENT_ACCUMULATION` | 4 (eff. batch 2·8·4 = **64**) |
| `MAX_SEQ_LEN` | 1024 |
| `LEARNING_RATE` | 5e‑7 → `MIN_LR` 1e‑7 |
| `WARMUP_STEPS` | 100 |
| `WEIGHT_DECAY` | 0.01 |
| `GRAD_CLIP` | 1.0 |
| `BETA` | 0.1 (DPO temperature) |
| Betas (AdamW) | (0.9, 0.95) |
| `LOG_INTERVAL` / `SAVE_INTERVAL` | 10 / 200 |

### 6.2 Two models in memory

- **Policy** = trainable copy, FP32 master weights, wrapped in DDP.
- **Reference** = frozen copy of the same SFT weights, cast to **bfloat16**, set to `eval()` and `requires_grad=False` ([train_dpo.py](train_dpo.py#L141-L146)).

Both are loaded from `SFT_CHECKPOINT` *before* DDP wrapping. The tokenizer's chat specials are re‑added and `model_config.vocab_size = len(tokenizer)` is set so the architecture matches the SFT‑expanded checkpoint.

### 6.3 Dataset

- `DPODataset(tokenizer, max_seq_len=1024, split="train", cache_dir=...)` from `model.dpo_data` (not visible).
- File docstring states **UltraFeedback** preference pairs.
- `dpo_collate_fn(batch, pad_id=...)` returns a dict with **`chosen_ids`**, **`rejected_ids`**, **`prompt_lens`** ([train_dpo.py](train_dpo.py#L218-L220)). I.e. for each sample the prompt is concatenated with both the chosen and rejected continuations, and `prompt_lens[b]` marks where the response begins.

### 6.4 Per‑token log‑probability helper

`get_per_token_logps(model, input_ids, prompt_lens)` ([train_dpo.py](train_dpo.py#L60-L83)):

1. Drop the last token (`input_ids[:, :-1]`) and run the model under BF16 autocast.
2. Compute `log_softmax(logits.float(), dim=-1)` (cast to FP32 for numerical stability).
3. Gather the log‑prob of the *next* token (`input_ids[:, 1:]`).
4. Build a per‑sample mask that is `1` only on response positions (`response_start = prompt_len - 1` up to the first padding token) and sum the masked log‑probs → one scalar per sequence.

> **Subtle issue worth flagging:** the code computes `seq_len = (labels[b] != 0).sum()`. This treats token id `0` as the pad position, but the tokenizer's `pad_token_id` is **2** (= EOS). If the collate function pads with `0` this is fine; if it pads with `pad_token_id=2` (which is what is passed into `dpo_collate_fn`), the mask will incorrectly include padding. Worth verifying when `model/dpo_data.py` becomes available.

### 6.5 DPO loss

`dpo_loss(...)` ([train_dpo.py](train_dpo.py#L86-L99)) implements the standard pairwise objective:

$$
\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}\!\left[\log \sigma\!\Big(\beta\big(\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\big)\Big)\right]
$$

with $\beta = 0.1$. Two diagnostics are also returned:

- **`chosen_better`** — fraction of the batch where the chosen reward exceeds the rejected reward (preference accuracy).
- **`reward_margin`** — mean of `chosen_reward − rejected_reward`.

Both are logged alongside loss and LR.

### 6.6 Loop

For each micro‑batch: 4 forward passes (policy/ref × chosen/rejected) → DPO loss → backward; reference forwards are wrapped in `torch.no_grad()`. Same LR (cosine), grad clip, AdamW pattern as SFT.

Checkpoints save `step`, `policy.module.state_dict()`, `model_config.__dict__`, `vocab_size` (no optimizer state).

---

## 7. Logs

The workspace contains [logs/sft_log.jsonl](logs/sft_log.jsonl) and [logs/dpo_log.jsonl](logs/dpo_log.jsonl). They are JSON‑lines with one record per `LOG_INTERVAL` steps, matching the writer code in each script. Pretraining writes to `train_log.jsonl` in *append* mode (resume‑safe); SFT/DPO open with mode `"w"` so re‑runs overwrite.

---

## 8. Pieces explicitly NOT in this workspace

The training scripts cannot run as‑is from what is checked in here. The following are referenced but missing:

| Missing item | Where it is referenced |
| --- | --- |
| `model/__init__.py`, `model/config.py` (`ModelConfig`, `TrainConfig`) | All three scripts |
| `model/transformer.py` (`Transformer`) | All three scripts |
| `model/data.py` (`get_tokenizer`, `create_dataloader`) | [train.py](train.py), [train_sft.py](train_sft.py), [train_dpo.py](train_dpo.py) |
| `model/sft_data.py` (`SFTDataset`, `sft_collate_fn`) | [train_sft.py](train_sft.py) |
| `model/dpo_data.py` (`DPODataset`, `dpo_collate_fn`) | [train_dpo.py](train_dpo.py) |
| Tokenizer artifacts (`tokenizer.model` / `tokenizer.json`) — only `tokenizer_config.json` is present | implicit in `get_tokenizer()` |
| `requirements.txt` / `pyproject.toml` — exact versions of `torch`, `transformers`, `datasets`, `tokenizers` are unspecified | — |
| Checkpoints referenced as starting points (`step_19000.pt`, `sft_final.pt`) live on `/jfs/...` paths and are not in‑repo | [train_sft.py](train_sft.py#L29), [train_dpo.py](train_dpo.py#L34) |
| Launcher / cluster scripts (SLURM / `torchrun` wrappers) | — |

Filling in these files would be required to (a) confirm exact hyper‑parameters in `TrainConfig`, (b) verify the SFT prompt‑masking strategy, and (c) validate the DPO padding assumption noted in §6.4.

---

## 9. End‑to‑End Pipeline Summary

```
                FineWeb-Edu 10BT (streamed)
                            │
                            ▼
   ┌──────────────────────────────────────────────────┐
   │ Stage 1 — Pretrain                               │
   │   train.py  · DDP × 8 H100 · BF16 autocast       │
   │   AdamW(fused) · WSD LR · grad-accum · clip 1.0  │
   │   resume from step_*.pt                          │
   └──────────────────────────────────────────────────┘
                            │  step_*.pt (1B base)
                            ▼
   ┌──────────────────────────────────────────────────┐
   │ Stage 2 — SFT                                    │
   │   train_sft.py · UltraChat-200K                  │
   │   add 3 chat tokens, resize emb + head           │
   │   AdamW · cosine LR 2e-5→2e-6 · 2 epochs         │
   └──────────────────────────────────────────────────┘
                            │  sft_final.pt
                            ▼
   ┌──────────────────────────────────────────────────┐
   │ Stage 3 — DPO                                    │
   │   train_dpo.py · UltraFeedback pairs             │
   │   policy (FP32, DDP) + ref (BF16, frozen)        │
   │   β=0.1 · cosine LR 5e-7→1e-7 · 1 epoch          │
   └──────────────────────────────────────────────────┘
                            │
                            ▼
                       dpo_final.pt
```

Inference / export uses the HuggingFace‑compatible [config.json](config.json) and [tokenizer_config.json](tokenizer_config.json) so the trained checkpoint can be loaded as a standard `LlamaForCausalLM`.
