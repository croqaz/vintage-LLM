# nanowhale — Technical Training Report

This report describes how the training pipeline in this repository works: the
libraries it relies on, the model architecture, the tokenizer, the optimizer
and scheduler, the dataset format, and how data is loaded and consumed by the
trainer.

The analysis is based on the source files in the workspace:

- [configs/main_100m.yaml](configs/main_100m.yaml), [configs/debug.yaml](configs/debug.yaml), [configs/fallback_under_1b.yaml](configs/fallback_under_1b.yaml)
- [configuration_deepseek_v4.py](configuration_deepseek_v4.py)
- [modeling_deepseek_v4.py](modeling_deepseek_v4.py)
- [README.md](README.md), [SUMMARY.md](SUMMARY.md)
- [scripts/prepare_data.py](scripts/prepare_data.py)
- [scripts/train_pretrain.py](scripts/train_pretrain.py)
- [scripts/train_sft.py](scripts/train_sft.py)
- [tokenizer/tokenizer_config.json](tokenizer/tokenizer_config.json)

---

## 1. High-level overview

The repository implements a from-scratch trainer for a custom **DeepSeek-V4**
class causal language model registered into the HuggingFace `Auto*` machinery.
There are two training stages:

1. **Pretraining** ([scripts/train_pretrain.py](scripts/train_pretrain.py)) — random-init the model and train on raw text from `HuggingFaceFW/fineweb-edu` using `trl.SFTTrainer` in *packing / completion-only-disabled* mode (i.e. effectively a causal-LM pretraining loop).
2. **Supervised fine-tuning** ([scripts/train_sft.py](scripts/train_sft.py)) — load the pretrained checkpoint and continue training on the conversational dataset `HuggingFaceTB/smol-smoltalk` with the same `SFTTrainer`.

Both scripts share the same model, tokenizer, optimizer family, scheduler and
distributed-launch story; they differ in dataset, loss formulation
(packed text vs. chat-templated SFT), learning rate, and step counts.

The actual training run used the `main_100m.yaml` config (~110M params) on a
single H100 80GB GPU. The pretrained base model and the SFT chat model have
both been published to the HuggingFace Hub:

| Model | Hub link |
|---|---|
| `cmpatino/nanowhale-100m-base` | Pretrained base (5K steps on FineWeb-Edu) |
| `cmpatino/nanowhale-100m` | SFT chat model (3K steps on SmolTalk) |

Each Hub repo ships `model.safetensors` (~441 MB), `config.json` (with
`auto_map`), the tokenizer files, a `chat_template.jinja`, and the custom
Python modules (`configuration_deepseek_v4.py`, `modeling_deepseek_v4.py`)
for `trust_remote_code=True` loading.

---

## 2. Libraries used

Declared in [requirements.txt](requirements.txt):

| Library | Role |
|---|---|
| `torch` (≥ 2.6) | Core tensor / autograd / `torch.compile` / `scaled_dot_product_attention` (SDPA). |
| `transformers` (≥ 5.0) | `PretrainedConfig`, `PreTrainedModel`, `GenerationMixin`, `AutoConfig`/`AutoModelForCausalLM` registration, `PreTrainedTokenizerFast`, `BaseModelOutputWithPast`, `CausalLMOutputWithPast`. |
| `datasets` (≥ 3.0) | Streaming and in-memory loading of FineWeb-Edu and SmolTalk. |
| `accelerate` (≥ 1.0) | Used implicitly by `trl` for distributed launch (`accelerate launch --num_processes N …`). |
| `trl` (≥ 1.0) | Provides `SFTTrainer` and `SFTConfig` — the actual training loop, gradient accumulation, packing, checkpointing, hub push, etc. |
| `trackio` | Optional experiment tracker; both scripts try to `import trackio` and switch `report_to` to `"trackio"` on success. |
| `safetensors` | Checkpoint serialization (used transitively by `transformers` saves). |
| `pyyaml` | Loads the YAML configs in [configs/](configs/). |
| `huggingface_hub` | Required for `push_to_hub` paths inside `SFTConfig`. |

**Not declared / not used directly**: there is no `flash-attn`, no `apex`,
no `deepspeed`, no `bitsandbytes`, no `wandb`/`tensorboard` dependency in
`requirements.txt`. Attention falls back to PyTorch SDPA (see §4.2).

**Not present in the repo (worth flagging):**

- There is no `accelerate` config file in the repo; the multi-GPU command in
  the docstring (`accelerate launch --num_processes 8 …`) assumes the user
  has run `accelerate config` separately.
- There is no `Dockerfile`, no `.env`, no CI configuration.
- The chat script ([scripts/chat.py](scripts/chat.py)), eval smoke test
  ([scripts/eval_smoke.py](scripts/eval_smoke.py)), parameter counter
  ([scripts/count_params.py](scripts/count_params.py)) and hub uploader
  ([scripts/upload_to_hub.py](scripts/upload_to_hub.py)) exist but were not
  read for this report — they are not part of the training loop itself.

---

## 3. Tokenizer

Defined under [tokenizer/](tokenizer/) and consumed via
`PreTrainedTokenizerFast.from_pretrained("tokenizer")` in
`build_tokenizer()` in [scripts/train_pretrain.py](scripts/train_pretrain.py).

From [tokenizer/tokenizer_config.json](tokenizer/tokenizer_config.json):

- Backend: `tokenizers` (Rust fast tokenizer, loaded as `PreTrainedTokenizerFast`).
- BOS: `<｜begin▁of▁sentence｜>`
- EOS: `<｜end▁of▁sentence｜>`
- PAD: same as EOS (also assigned defensively at runtime: `tok.pad_token = tok.eos_token` if missing).
- `tokenizer_class` is reported as `TokenizersBackend`, which is a
  HF Transformers v5 style declaration.
- The vocabulary itself lives in [tokenizer/tokenizer.json](tokenizer/tokenizer.json) (not inspected line-by-line in this report); the configs assume a vocab size of `129280`, matching the upstream DeepSeek-V4 tokenizer.

For chat-style fine-tuning the script does **not** call
`apply_chat_template` explicitly; it hands the raw `messages` field of the
SmolTalk dataset to `SFTTrainer`, which internally applies the tokenizer's
chat template if one is defined in `tokenizer.json` / `tokenizer_config.json`.
The visible `tokenizer_config.json` does not contain a `chat_template` field;
however, from [SUMMARY.md](SUMMARY.md) we know that a `chat_template.jinja`
file was uploaded alongside the model artifacts to the Hub. Whether the local
tokenizer embeds a template inside `tokenizer.json` itself is not verified;
SFT may rely on TRL's default formatting when run from the repo, while the
Hub-hosted version carries an explicit Jinja template.

---

## 4. Model architecture (DeepSeek-V4)

Two files define the model:

- [configuration_deepseek_v4.py](configuration_deepseek_v4.py) — the
  `DeepseekV4Config` (`model_type = "deepseek_v4"`).
- [modeling_deepseek_v4.py](modeling_deepseek_v4.py) — pure-PyTorch port of
  the upstream DeepSeek-V4 model so it works with `Trainer`/`SFTTrainer`
  without custom kernels (no `tilelang` dependency).

Both are registered into the HF `Auto*` system at the top of each training
script:

```python
AutoConfig.register("deepseek_v4", DeepseekV4Config)
AutoModelForCausalLM.register(DeepseekV4Config, DeepseekV4ForCausalLM)
```

### 4.1 Top-level structure

`DeepseekV4ForCausalLM` wraps `DeepseekV4Model` plus an `lm_head`
(`nn.Linear(hidden_size, vocab_size, bias=False)`). Tied weights are declared
via `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`,
but the configs set `tie_word_embeddings: false`, so they remain untied in
practice.

`DeepseekV4Model` consists of:

1. `nn.Embedding(vocab_size, hidden_size)` — token embeddings.
2. `num_hidden_layers` × `DeepseekV4Block`.
3. A final RMSNorm (`DeepseekV4RMSNorm`).
4. **Hyper-Connection head** (`hc_head`) that contracts the maintained
   `hc_mult` parallel hidden-state copies down to a single tensor before the
   final norm and `lm_head`.
5. A precomputed RoPE buffer `freqs_cis` of shape
   `[2, max_position_embeddings, qk_rope_head_dim/2]`, registered
   non-persistent (not saved with state dict).

### 4.2 Attention — Multi-head Latent Attention (MLA) variant

Implemented in `DeepseekV4Attention`:

- **Q projection** is low-rank: `wq_a` (hidden → `q_lora_rank`) → RMSNorm →
  `wq_b` (→ `num_heads * head_dim`).
- **KV projection** is direct (no low-rank, no separate K/V) and produces a
  **single-head** tensor of size `head_dim`: `wkv` then RMSNorm. It is then
  `unsqueeze(1).expand(-1, num_heads, -1, -1)` so the same K/V is shared
  across all Q heads (extreme MQA / GQA-1 setup; configs use
  `num_key_value_heads: 1`).
- **RoPE** is applied to the last `qk_rope_head_dim` dims of both Q and KV
  (`apply_rotary_emb`, real-valued, compile-friendly). After SDPA, the output
  is **de-rotated** along the same RoPE dims.
- **Output projection** is grouped low-rank: the head-dim output is split
  into `o_groups` chunks; `wo_a` (a `[o_groups, o_lora_rank, …]` weight used
  via `einsum`) compresses each group, and `wo_b` (`o_groups*o_lora_rank →
  hidden_size`) projects back.
- **Attention sink bias** parameter `attn_sink` exists per-head, but the
  forward path uses `F.scaled_dot_product_attention` and the sink bias is
  intentionally not added (commented in code as a minor approximation).
- The causal mask is built explicitly inside `DeepseekV4Model.forward` as a
  4-D `[1,1,S,S]` `-inf`/`0` mask, which is passed as `attn_mask` to SDPA.

The config also carries CSA-related fields (`compress_ratios`,
`compress_rope_theta`, `index_n_heads`, `index_head_dim`, `index_topk`,
`sliding_window`). **Worth flagging:** the actual `DeepseekV4Attention`
implementation in [modeling_deepseek_v4.py](modeling_deepseek_v4.py) does
**not** apply compression, sliding window, or sparse index attention — it is
pure dense SDPA. The configs all set `compress_ratios` to all-zeros, so this
is consistent with how the model is used today, but the *Compressed Sparse
Attention* mentioned in the file header is not actually wired up. This is a
deliberate simplification documented at the top of the module
("Custom kernels (tilelang) are NOT required").

### 4.3 Mixture of Experts

`DeepseekV4MoE` per layer:

- `n_routed_experts` SwiGLU experts (`DeepseekV4Expert`: `w1` gate, `w3` up,
  `w2` down, with optional symmetric `swiglu_limit` clamp).
- One **shared expert** that always runs and is added to the routed sum.
- Gate (`DeepseekV4Gate`) computes scores `F.linear(x, weight)` in fp32 and
  applies the activation chosen by `scoring_func`:
  - `"softmax"` — standard softmax.
  - `"sigmoid"` — independent gate.
  - `"sqrtsoftplus"` — `softplus(x).sqrt()` (the V4 default, used in all
    configs in this repo).
- A learnable per-expert `bias` (only present when `layer_idx >=
  num_hash_layers`) is added before `topk`, but the actual mixing weights
  used are gathered from the **pre-bias** scores; that matches the
  `noaux_tc` scheme declared by `topk_method`.
- Routing weights are renormalized to sum to 1 (when `scoring_func !=
  "softmax"`) and then multiplied by `routed_scaling_factor` (1.5 in all
  configs).
- Dispatch is a Python `for i in range(n_routed_experts)` loop driven by
  `torch.bincount`/`torch.where` — simple and correct, but not throughput
  optimal for many experts.

The first `num_hash_layers` blocks use a **hash-routed** gate (no learnable
bias, fixed by `weight @ x` — there is no separate hash function in this
port, so for `num_hash_layers > 0` the only difference is "no bias"). All
provided configs set `num_hash_layers: 0`.

Because not all experts run on every step, `SFTConfig` is built with
`ddp_find_unused_parameters=True` in both training scripts — explicitly
called out in code with `# DDP: MoE has unused params (inactive experts)`.

### 4.4 Hyper-Connections (HC)

Implemented by `hc_split_sinkhorn`, `DeepseekV4Block.hc_pre`,
`DeepseekV4Block.hc_post`, and the model-level `hc_head`.

Instead of a single residual stream, the model maintains `hc_mult` parallel
copies of the hidden state (`[B, S, hc_mult, D]`). For each sub-layer
(attention or FFN):

1. `hc_pre` reduces the `hc_mult` copies to one tensor `y`, while also
   producing per-token `pre`, `post`, and a `[hc_mult, hc_mult]`
   combination matrix from a single linear projection of the flattened
   state. The combination matrix is doubly-stochastic-normalized via
   `sinkhorn_iters` Sinkhorn iterations (alternating row/column normalization).
2. The sub-layer (norm → attn or norm → MoE) is run on `y`.
3. `hc_post` writes back `post * y + comb @ residual`, restoring the
   `hc_mult` channels.

At the very top of the model, `hc_head` does a *one-shot* contraction
(sigmoid-gated weighted sum) of the `hc_mult` copies into one stream before
the final RMSNorm and `lm_head`.

Init: `hc_*_fn` weights are `N(0, 0.01)`, scales are ones, biases are zeros
(see `_init_weights`).

### 4.5 Multi-Token Prediction (MTP)

The config declares `num_nextn_predict_layers=1`, but the modeling code does
**not** implement MTP heads — the loss in `DeepseekV4ForCausalLM.forward` is
a plain shifted cross-entropy on the next token only. **Worth flagging** if
MTP is expected.

### 4.6 Loss

In `DeepseekV4ForCausalLM.forward`:

```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, vocab_size),
                       shift_labels.view(-1),
                       ignore_index=-100)
```

Standard causal-LM next-token loss with `-100` masking. There is **no MoE
load-balancing auxiliary loss**, **no z-loss**, and no router KL — only the
LM cross-entropy.

### 4.7 KV cache

`DeepseekV4Model.forward` currently **forces `use_cache = False` and
`past_key_values = None`** with the comment "DynamicCache compatibility TBD".
Generation will work but without a cache (re-encoded each step). For
training this is irrelevant.

### 4.8 Configuration matrix

| Field | `debug.yaml` | `main_100m.yaml` | `fallback_under_1b.yaml` |
|---|---|---|---|
| `hidden_size` | 64 | 320 | 896 |
| `num_hidden_layers` | 2 | 8 | 16 |
| `num_attention_heads` / `num_key_value_heads` | 4 / 1 | 8 / 1 | 16 / 1 |
| `head_dim` (rope/nope) | 32 (16/16) | 96 (32/64) | 256 (64/192) |
| `q_lora_rank` | 32 | 160 | 448 |
| `o_groups` / `o_lora_rank` | 2 / 16 | 2 / 80 | 4 / 224 |
| `n_routed_experts` / `shared` / `top-k` | 4 / 1 / 2 | 4 / 1 / 2 | 8 / 1 / 2 |
| `moe_intermediate_size` | 128 | 640 | 1792 |
| `hc_mult` / `hc_sinkhorn_iters` | 4 / 5 | 4 / 2 | 4 / 10 |
| `vocab_size` | 129280 | 129280 | 129280 |
| `max_position_embeddings` | 512 | 2048 | 2048 |
| `tie_word_embeddings` | false | false | false |
| Param target | ~17M | ~110M | ~995M |

---

## 5. Pretraining loop ([scripts/train_pretrain.py](scripts/train_pretrain.py))

### 5.1 Initialization

1. CLI args: `--config`, `--debug`, `--output_dir`, `--hub_model_id`,
   `--resume_from_checkpoint`.
2. `torch.set_float32_matmul_precision('high')` to enable TF32-style matmul
   reductions.
3. `build_model(model_cfg)` instantiates `DeepseekV4ForCausalLM(config)` from
   scratch; weights come from `_init_weights` (Linear/Embedding ~ N(0,
   `initializer_range=0.02`), RMSNorm = 1, HC `_fn` ~ N(0, 0.01), HC scales
   = 1, biases = 0, `attn_sink` = 0).
4. If CUDA is available, `model = torch.compile(model)` is called
   unconditionally. In the actual pretraining run this delivered a **1.77×
   speedup** (122 ms/step → 72 ms/step). **Worth flagging:** with HF
   `Trainer` / `SFTTrainer`, the trainer also enables compilation paths
   internally; double-compiling can produce confusing graph breaks. There
   is no flag to opt out here.
5. `build_tokenizer()` loads from the local [tokenizer/](tokenizer/) folder.

**`torch.compile` checkpoint prefix issue**: `torch.compile` wraps the model
in a `_orig_mod` container, which means all keys in saved `state_dict` get a
`_orig_mod.` prefix. During the actual project the prefix had to be
post-hoc stripped from all 270 keys in `model.safetensors` across every
checkpoint (`final/`, `checkpoint-4000/`, `checkpoint-4500/`,
`checkpoint-5000/`). This is a known `torch.compile` + `Trainer` interaction
that should be handled by unwrapping the compiled model before saving, or
by stripping the prefix at load time.

### 5.2 Output & checkpointing

- `output_dir` defaults to `checkpoints/pretrain_<config_basename>/` and is
  created if missing.
- `SFTConfig` controls saving: `save_steps` (from config or `10` in debug),
  `save_total_limit=3`. Final model is saved to
  `<output_dir>/final/` together with the tokenizer.

### 5.3 Distributed launch

The docstring documents two modes:

- Single GPU: `python scripts/train_pretrain.py --config …`
- Multi-GPU: `accelerate launch --num_processes 8 scripts/train_pretrain.py --config …`

`SFTTrainer` (built on `transformers.Trainer`) handles the DDP wrapping
through `accelerate`. `ddp_find_unused_parameters=True` is set due to MoE.

### 5.4 Training hyperparameters

From [configs/main_100m.yaml](configs/main_100m.yaml) `training:` block,
mapped into `SFTConfig`:

- `per_device_train_batch_size: 8`
- `gradient_accumulation_steps: 4`
- `max_seq_length / max_length: 2048`
- `learning_rate: 6.0e-4`
- `weight_decay: 0.1`
- `adam_beta1: 0.9`, `adam_beta2: 0.95`
- `max_grad_norm: 1.0`
- `lr_scheduler_type: cosine`
- `warmup_ratio: 0.03`
- `max_steps: 5000`
- `save_steps: 500`, `logging_steps: 10`, `logging_first_step: True`
- `bf16: True` (when CUDA is available)
- `gradient_checkpointing: True` with `{"use_reentrant": False}`
- `optim: "adamw_torch_fused"`
- `dataloader_num_workers: 4`, `dataloader_prefetch_factor: 2`
- `seed: 42`
- `disable_tqdm: True`, `report_to: ["none"]` (overridden to `["trackio"]`
  when the import succeeds).

The `debug.yaml` and `fallback_under_1b.yaml` files override these (smaller
batch / shorter seq / fewer steps, etc.).

### 5.5 Optimizer

The optimizer is selected by the `optim="adamw_torch_fused"` SFTConfig
field, which makes `transformers.Trainer` instantiate
`torch.optim.AdamW(..., fused=True)`.

- Betas come from `adam_beta1`, `adam_beta2` (0.9 / 0.95).
- Weight decay is applied to *all* parameters by default in
  `Trainer.create_optimizer`, but `Trainer` excludes biases and
  `LayerNorm.weight` parameters from decay automatically. Note that this
  model uses `DeepseekV4RMSNorm` (`weight` parameter, not `LayerNorm`); the
  default exclusion list typically catches `*norm*.weight`-style names —
  worth verifying for this code if precise WD masking matters.
- Gradient clipping at `max_grad_norm=1.0` is performed by `Trainer`.

### 5.6 LR scheduler

`lr_scheduler_type: cosine` + `warmup_ratio: 0.03` ⇒ `transformers`'
`get_cosine_schedule_with_warmup`: linear warmup over `0.03 * max_steps`
steps from 0 to `learning_rate`, then cosine decay to (effectively) 0 over
the remaining steps. There is no `min_lr` knob — it is the standard HF
cosine that decays to 0.

### 5.7 Mixed precision and the BF16 NaN problem

`bf16=True` instructs `Trainer` to wrap forward/backward in
`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. No `GradScaler`
is needed (and none is configured) because BF16 has FP32-equivalent dynamic
range. Inside the model, several ops (RMSNorm, MoE gate scoring, HC mixing
matrices) are deliberately upcast to fp32 for numerical stability.

**Critical known issue (from [README.md](README.md) and
[SUMMARY.md](SUMMARY.md)):** at this small model scale (~110M), the
Hyper-Connections architecture produces intermediate values that **overflow
the BF16 representable range**, causing NaN losses. Pretraining was
completed in BF16 (it survived), but during SFT the NaN problem surfaced
and the run had to be restarted **in full fp32**. For inference both models
also require fp32 to avoid NaN. This is a scaling / numerical-stability
issue specific to the HC mechanism and the small hidden dimensions involved;
larger models with wider hidden states may not exhibit it.

### 5.8 Gradient checkpointing

Enabled via `gradient_checkpointing=True` with `use_reentrant=False`. The
model honours this in `DeepseekV4Model.forward` by wrapping each block in
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` when
`self.gradient_checkpointing and self.training`. `Trainer` toggles the
`gradient_checkpointing` attribute on the model when the config is set.

### 5.9 Logging

- `logging_steps` default 10 (1 in debug) drives stdout logs through HF
  `Trainer`'s logger.
- `report_to` is `"none"` unless `trackio` imports successfully, in which
  case it becomes `"trackio"` and a `trackio.init(project="smol-deepseek-v4",
  name=…)` call is made before training. There is no W&B / TB integration
  configured.

### 5.10 Hub upload

When `--hub_model_id` is supplied, `push_to_hub=True` and `hub_model_id` are
forwarded to `SFTConfig`. `Trainer` handles the upload at save time using
`huggingface_hub`.

---

## 6. Dataset format and data loading (pretraining)

### 6.1 Source

`HuggingFaceFW/fineweb-edu`, the educational filter of FineWeb. From
[scripts/prepare_data.py](scripts/prepare_data.py)'s preview function, each
sample is a dict with at least:

- `text` — the document text (string).
- `score` — quality score (float).
- `token_count` — pre-computed token count from FineWeb's pipeline.

Only the `text` column is used for training (`dataset_text_field="text"` in
`SFTConfig`).

### 6.2 Loading mode

```python
dataset = load_dataset("HuggingFaceFW/fineweb-edu",
                       split="train", streaming=True)
```

- **Full run**: streaming `IterableDataset`. Nothing is materialized to disk
  beyond the HF datasets cache. The trainer iterates indefinitely until
  `max_steps` is reached.
- **Debug run** (`--debug`): the streaming iterator is `take(200)`-d, the
  rows are listed and converted to an in-memory `Dataset.from_list(rows)`
  for deterministic smoke testing.

### 6.3 Packing and tokenization

Inside `SFTConfig`:

- `dataset_text_field="text"`
- `max_length=max_seq_length` (2048 by default in `main_100m.yaml`)
- `packing=True`

`SFTTrainer` therefore tokenizes documents using the loaded
`PreTrainedTokenizerFast` and concatenates them into fixed-length sequences
of `max_length` tokens (TRL's *packing* mode). For pretraining this is
exactly the desired behaviour (no per-sample padding waste, full causal LM
loss across the packed sequence). Labels are produced by `SFTTrainer` as a
shift of `input_ids`; `-100` masking is unused in pretraining since
`packing=True` and there is no completion-only mask.

### 6.4 DataLoader settings

`SFTConfig` sets `dataloader_num_workers=4`,
`dataloader_prefetch_factor=2`. `Trainer` then constructs a regular
PyTorch `DataLoader`. With a streaming source the dataset is wrapped so
that each rank consumes a sharded slice (handled by `accelerate` /
`Trainer`).

There is **no custom collator** in the script — `SFTTrainer` provides its
own packed-LM collator.

---

## 7. SFT loop ([scripts/train_sft.py](scripts/train_sft.py))

### 7.1 Differences from pretraining

- Model is loaded from a pretrained checkpoint:
  `DeepseekV4ForCausalLM.from_pretrained(args.pretrained_path)`. Tokenizer
  is loaded from the same directory.
- Dataset is `HuggingFaceTB/smol-smoltalk` (~460K conversations),
  **non-streaming** (full in-memory `Dataset` with a known `len`). Each
  sample has a `messages` field — a list of `{"role", "content"}` turns
  (visible in [scripts/prepare_data.py](scripts/prepare_data.py)'s
  `preview_smoltalk`).
- `SFTConfig` is built without `dataset_text_field` and without
  `packing=True`; `SFTTrainer` recognises the conversational schema and
  applies the tokenizer's chat template (see §3 caveat about template
  presence).
- Different hyperparameters (CLI-controlled): `--max_steps 3000`,
  `--batch_size 4`, `--grad_accum 4`, `--lr 2e-5`, `--max_length 2048`,
  `weight_decay=0.01`, `warmup_ratio=0.05`, `save_steps=500`,
  `logging_steps=10`. `lr_scheduler_type` and optimizer are unchanged.
- `bf16` is enabled whenever CUDA is present (no config gate) — however, as
  documented in §5.7, the actual SFT run had to fall back to **fp32** due
  to NaN from the HC architecture.
- Same `ddp_find_unused_parameters=True`, same `gradient_checkpointing`,
  same `optim="adamw_torch_fused"`, same trackio hook.

### 7.2 `from_pretrained` weight re-initialization quirk

From [SUMMARY.md](SUMMARY.md): `from_pretrained` **re-initialized** some
weights in the custom architecture, silently overwriting the pretrained
values. The workaround used in practice was to switch to manual
`load_state_dict` loading instead of `from_pretrained`. This is a known
interaction between HF `PreTrainedModel._init_weights` and custom module
types that do not appear in the default `_keys_to_ignore_on_load_missing`
list — specifically the HC parameters and MoE gate biases.

### 7.3 No torch.compile in SFT

The pretraining script wraps the model in `torch.compile`; the SFT script
does **not**. This is likely intentional — given the `_orig_mod.` prefix
checkpoint issue that had to be fixed after pretraining (§5.1), and the
fp32 fallback needed for SFT, skipping `torch.compile` is the pragmatic
choice.

---

## 8. Effective-batch math (reference)

For [configs/main_100m.yaml](configs/main_100m.yaml) on `N` GPUs:

- Tokens per micro-step per GPU = `per_device_train_batch_size * max_seq_length` = `8 * 2048 = 16,384`.
- Tokens per optimizer step = `16,384 * gradient_accumulation_steps * N` = `65,536 * N`.
- Total tokens at `max_steps=5000`, single GPU = `~327M`. On 8 GPUs it is `~2.6B` tokens. (No `total_tokens` budget is computed inside the script — only `max_steps` drives termination.)

Per [SUMMARY.md](SUMMARY.md), the actual pretraining run processed
**~2.6B tokens** in 5,000 steps, implying the effective batch in tokens was
~524K — this is consistent with an 8-GPU run (`65,536 * 8 = 524,288`) or a
single-GPU run with a higher effective GA. The README states it ran on
**1× H100 80GB**. Given `per_device_train_batch_size=8` and
`gradient_accumulation_steps=4` on 1 GPU the tokens per step are `65,536`
and 5,000 steps yields `327M` tokens. To reach the claimed 2.6B, the
effective batch must have been 8× larger — either via 8 GPUs or a different
GA value not captured in the checked-in config. The discrepancy is noted
but not resolved in the sources.

### 8.1 Embedding parameter overhead

With `vocab_size=129,280` and `hidden_size=320`, the embedding matrix
(`nn.Embedding`) alone is `129,280 * 320 = ~41.4M` parameters. Since
`tie_word_embeddings=false`, the `lm_head` adds another `~41.4M`, totalling
**~82.8M** for input + output embedding layers — **~75%** of the model.
[README.md](README.md) quotes the split as 41M embedding + 69M
non-embedding (110M total), suggesting the `lm_head` is counted separately
or shares storage at the `safetensors` level. Either way, the outsized
vocabulary dominates the parameter budget and limits the capacity available
for language modelling, which is a primary driver of the high perplexity.

---

## 9. Actual training results

The numbers below come from [README.md](README.md) and [SUMMARY.md](SUMMARY.md) — they are the
result of the runs that produced the published Hub checkpoints.

### 9.1 Pretraining (5,000 steps on FineWeb-Edu)

| Metric | Value |
|---|---|
| Tokens seen | ~2.6B |
| Final loss | ~5.3 |
| Token accuracy | 33.8% |
| Hardware | 1× H100 80GB, BF16 |
| Throughput | 72 ms/step (with `torch.compile`; 122 ms without) |
| Checkpoints saved | `checkpoint-4000/`, `checkpoint-4500/`, `checkpoint-5000/`, `final/` |

### 9.2 SFT (3,000 steps on SmolTalk, fp32)

| Metric | Start | End |
|---|---|---|
| Train loss | 15.41 | 10.22 |
| Eval loss | 2.873 | 2.607 |
| Token accuracy | 36.2% | 48.5% |
| Entropy | 3.95 | 2.81 |

Eval loss trajectory: 2.873 → 2.717 → 2.654 → 2.622 → 2.609 → 2.607
(converged, no overfitting).

### 9.3 Perplexity (held-out English text)

| Model | Loss | Perplexity |
|---|---|---|
| Pretrained | 2.612 | 13.62 |
| SFT | 2.558 | 12.90 |

### 9.4 Generation quality

- **Pretrained base**: incoherent multilingual text. The model barely beats
  uniform distribution over the 129K vocabulary.
- **SFT model**: learned conversational structure (paragraphs, explanations,
  lists) but hallucinates facts — expected for 110M params with a large
  fraction of capacity consumed by embeddings.

---

## 10. Known issues and operational notes

These are documented in [README.md](README.md) and [SUMMARY.md](SUMMARY.md) and are worth
highlighting because they affect reproducibility and deployment.

1. **BF16 NaN** (§5.7) — the HC architecture at small scale produces values
   that overflow BF16 range. Pretraining survived in BF16 but SFT required
   fp32. Inference also requires fp32.
2. **`from_pretrained` re-initialization** (§7.2) — the custom architecture
   causes `from_pretrained` to silently reinitialize some weights. The
   workaround is `load_state_dict` (documented in the Hub model cards).
3. **`torch.compile` `_orig_mod.` prefix** (§5.1) — compiled-model
   checkpoints carry a `_orig_mod.` key prefix that must be stripped before
   the weights can be loaded by an uncompiled model.
4. **Embedding overhead** (§8.1) — the 129K vocab embedding table consumes
   37% of all parameters, severely limiting non-embedding capacity.
5. **No KV cache** (§4.7) — generation re-encodes the full sequence at every
   step. Functional but slow.

---

## 11. What is *not* present (explicit gaps)

- **No MoE auxiliary balancing loss / z-loss.** Pure CE on next token.
- **No CSA / sliding-window / sparse index attention** in the model
  forward, despite the config fields existing.
- **No MTP heads**, despite `num_nextn_predict_layers=1`.
- **No `accelerate` or `deepspeed` config files** in the repo.
- **No Flash-Attention dependency or import** — attention runs on PyTorch
  SDPA (which itself can dispatch to a Flash kernel where supported).
- **No explicit chat template** in the local [tokenizer/tokenizer_config.json](tokenizer/tokenizer_config.json)
  (a `chat_template.jinja` exists in the Hub repos but not in the local
  `tokenizer/` directory).
- **No KV cache** during generation (forced off in `DeepseekV4Model.forward`).
- **No `min_lr`** for the cosine schedule; it decays to 0.
- **No evaluation loop** in either training script — `SFTConfig` is created
  without `eval_strategy`/`eval_dataset`.
- **No seed override / deterministic flags** beyond `seed=42` passed into
  `SFTConfig`.

---

## 12. Checkpoint artifacts

From [SUMMARY.md](SUMMARY.md), the local checkpoint layout after both
training stages:

```
checkpoints/
├── pretrain_100m/
│   ├── checkpoint-4000/
│   ├── checkpoint-4500/
│   ├── checkpoint-5000/
│   └── final/           ← pretrained model (uploaded to Hub)
└── sft/
    ├── checkpoint-2000/
    ├── checkpoint-2500/
    ├── checkpoint-3000/
    └── final/           ← SFT model (uploaded to Hub)
```

Each Hub repo contains:

- `model.safetensors` (~441 MB)
- `config.json` (with `auto_map` entries for `AutoConfig` /
  `AutoModelForCausalLM`)
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`
- `configuration_deepseek_v4.py`, `modeling_deepseek_v4.py` (custom code for
  `trust_remote_code=True`)
- `README.md` (model card)

---

## 13. End-to-end summary

A single-command pretraining run on the 110M config looks like:

```bash
accelerate launch --num_processes 8 \
    scripts/train_pretrain.py --config configs/main_100m.yaml \
    [--hub_model_id <user>/<model>]
```

This will:

1. Register `deepseek_v4` with HF Auto classes, build a fresh
   `DeepseekV4ForCausalLM(config)` (~110M params), and `torch.compile` it.
2. Load the local fast tokenizer (vocab 129,280, EOS as PAD).
3. Stream `HuggingFaceFW/fineweb-edu` and feed packed 2048-token sequences
   to `trl.SFTTrainer`.
4. Train with **AdamW (fused)**, `lr=6e-4`, `betas=(0.9, 0.95)`,
   `wd=0.1`, **cosine** schedule with 3% warmup, `grad_clip=1.0`, **BF16
   autocast**, **gradient checkpointing**, MoE-aware DDP, for `max_steps`.
5. Log every 10 steps (to stdout and trackio if available), checkpoint
   every `save_steps` keeping the last 3, and save a `final/` model +
   tokenizer; optionally push to the HF Hub.

SFT then re-uses the same machinery on `HuggingFaceTB/smol-smoltalk`
(~460K conversations) with a much smaller LR (`2e-5`), a slightly larger
warmup ratio, and chat-templated samples instead of packed raw text. In
practice SFT was run in **fp32** due to HC-related BF16 NaN.
