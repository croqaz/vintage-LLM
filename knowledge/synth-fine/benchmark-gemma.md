# Benchmark — throughput tuning

Throughput optimisation of models served by the `synth-vllm` image (vLLM 0.25.1,
OpenAI server) for the synth-data workload. GPU: **one RTX 4090 (24 GB),
device 0 only**. Metric is steady-state **output tokens/s** measured by
`bench_gemma.py` from the API `usage` fields.

## Method
- Client: `bench_gemma.py` — fires a fixed workload at a bounded in-flight
  concurrency, sums exact `completion_tokens`, divides by wall time. A short
  warmup pass (max_tokens=32) precedes timing so cold cudagraph capture / first
  token aren't counted.
- Workload: 5 prompt templates × 40 seed chunks = **200 requests**, chat mode,
  system = `systems/vintage_1850.txt`, `chunk-tokens=700` (~900 prompt tokens
  incl. system), `max-tokens=256`. Prompts: `continue-v2`,
  `lost-manuscript-framing`, `diverse_qa_pairs`, `teacher_student_dialogue`,
  `narrative`.
- Server: `synth.sh serve` equivalent (`docker run … synth-vllm serve`), one
  config per run, restarted between configs. `TEXT_ONLY=1` throughout (image /
  audio / video disabled — this model is omni-modal and we only want text).
- Reproduce one point:
  ```sh
  MODEL_PATH=/models/gemma TEXT_ONLY=1 MAX_MODEL_LEN=4096 GPU_MEM_UTIL=0.92 \
    SERVER_EXTRA_ARGS='--kv-cache-dtype fp8' ./synth.sh serve   # in one shell
  python3 bench_gemma.py prompts/continue-v2.txt prompts/lost-manuscript-framing.txt \
    prompts/diverse_qa_pairs.txt prompts/teacher_student_dialogue.txt prompts/narrative.txt \
    --num-seeds 40 --chunk-tokens 700 --max-tokens 256 --concurrency 128
  ```

## What each lever does (and whether it's applicable here)
- **Continuous batching** — vLLM's core scheduler; always on in V1. Nothing to toggle.
- **Chunked prefill** — on by default in V1 (`--max-num-batched-tokens` caps the
  per-step token budget, mixing prefill + decode). Tuned via `mnbt` below.
- **`max-num-seqs` (mns)** — cap on sequences running concurrently.
- **`max-num-batched-tokens` (mnbt)** — per-step token budget (prefill throughput knob).
- **`--kv-cache-dtype fp8`** — halves KV-cache byte size → ~2× the token budget,
  which is the binding constraint for this 15 GB model on 24 GB.
- **Tensor parallelism (tp)** — **not applicable**: TP>1 needs a second GPU, and
  device 1 is reserved. TP also wouldn't help here — the model already fits on
  one card, so TP would only add cross-GPU communication overhead. Kept at **tp=1**.
- **Speculative decoding** — tested (ngram). Helps single-stream latency but tends
  to *cost* throughput under heavy batching (extra draft/verify compute per step).
  MTP spec-decode for Gemma-4 is nightly-only per the vLLM recipe, so not tested here.

## Model: `models/gemma`
Community **Gemma-4 fine-tune** (behaves like `google/gemma-4-E4B-it`).
- arch `Gemma4ForConditionalGeneration`, `model_type=gemma4`, saved by transformers 5.11.0
- 42 decoder layers, **`num_kv_shared_layers=18`** (last 18 layers reuse earlier
  KV → need `patch_gemma4.py`), dense (no MoE), GQA 8 q-heads / 2 kv-heads,
  `head_dim=256`, `hidden_size=2560`, `sliding_window=512`, ctx up to 131072.
- ~15 GB bf16 weights → on a 24 GB card that leaves little room for KV cache,
  which makes **KV-cache capacity the primary throughput bottleneck**.
- Must be served with the baked-in `patch_gemma4.py` **and** `TEXT_ONLY=1`.

## Results (concurrency 128, 200 requests, max_tokens 256)

| config | KV pool (tokens) | out-tok/s | req/s | p50 lat | p99 lat |
|---|---|---|---|---|---|
| baseline (mml4096, util0.92) | 45,898 | 2,015 | 8.0 | 13.6 s | 20.4 s |
| + `--kv-cache-dtype fp8` | 88,290 | 3,250 | 13.1 | 8.6 s | 14.7 s |
| + fp8, util 0.95 | 124,100 | 3,962 | 15.8 | 6.5 s | 8.7 s |
| + fp8, mml 2048, util 0.92 | 64,414 | 3,280 | 13.1 | 8.6 s | 14.7 s |

Extra levers, all on the **fp8 + util 0.95** base (concurrency 128):

| variant | KV pool | out-tok/s | vs base | verdict |
|---|---|---|---|---|
| base (fp8, util 0.95) | 124,100 | **3,962** | — | best |
| `--async-scheduling` | 124,100 | 3,964 | ±0% | no effect on this workload |
| mml 2048 (util 0.95) | 90,541 | 3,947 | −0.4% | no gain (util, not mml, sets the pool) |
| `--max-num-batched-tokens 8192` | 76,892 | 3,920 | −1% | slightly worse (steals KV) |
| `--max-num-batched-tokens 16384` | 55,939 | 3,177 | −20% | worse (steals KV) |
| ngram spec-decode (k=3) | 79,782 | 2,416 | −39% | hurts under heavy batching |

### Concurrency sweep (out-tok/s) — fp8, two utilisation levels

| client concurrency | util 0.95 (124k KV) | util 0.97 (148k KV) |
|---:|---:|---:|
| 32  | 1,688 | 1,692 |
| 64  | 2,564 | 2,550 |
| 96  | 3,217 | 3,268 |
| 128 | **3,941** | 3,949 |
| 192 | 3,213 ↓ | **4,445** |
| 256 | 3,206 | 4,439 |

At util 0.95 throughput peaks at 128 then **falls** at ≥192 (p99 latency jumps
6.5 s → 15 s): once concurrent demand exceeds what the 124k-token KV pool holds,
vLLM preempts + recomputes sequences and thrashes. The bigger 148k pool at util
0.97 absorbs c≈192, giving the global best **4,445 out-tok/s**.

### `max-num-seqs` (mns) — at util 0.97, concurrency 192

| max-num-seqs | out-tok/s | note |
|---|---|---|
| 64 | 2,674 | caps concurrency *below* KV capacity → −40%, don't do this |
| default | 4,445 | KV pool is the binding limit, not the seq cap |
| 384 | 4,466 | ≈ no change (raising above KV capacity does nothing) |

## Findings
**The workload is KV-cache-capacity bound.** gemma-heretic is ~15 GB on a 24 GB card,
so after weights + activations only ~2–3 GB is left for the KV cache. Throughput
is set by *how many sequences fit in that pool at once*, so every worthwhile win
came from enlarging the pool or matching client concurrency to it.

**What worked (in order of impact):**
1. **`--kv-cache-dtype fp8`** — halves KV bytes, ~doubles the token pool
   (45,898 → 88,290 → 124,100 with util below). Biggest single lever:
   2,015 → 3,250 tok/s. No measurable quality loss (spot-checked continue-v2 +
   diverse_qa_pairs output — coherent period prose and well-formed Q&A).
2. **`--gpu-memory-utilization 0.97`** — pushes the pool to 147,982 tokens
   (0.90→0.95→0.97 all helped). 0.97 ran cleanly (200/200 ok even at c=256); did
   not try 0.98 (OOM risk mid-run isn't worth ~5% for a long bulk job).
3. **Match client `--concurrency` to the pool** — with the 148k pool the knee is
   **~192**. Too low underfeeds (c=32 → 1,688); too high for the pool causes
   preemption/recompute thrashing (util 0.95 at c=192 fell to 3,213, p99 15 s).

**What didn't help / hurt:**
- **`--async-scheduling`**: 0% here.
- **`--max-num-batched-tokens`**: leave at default — raising it (8192/16384)
  *steals* memory from the KV pool and lowers throughput (−1% / −20%).
- **`max-num-seqs`**: default is fine; lowering it below KV capacity throttles
  (mns 64 → −40%); raising it is a no-op.
- **`max-model-len 2048`** vs 4096: no gain at equal util — the *pool* (set by
  util + fp8) is what matters, not the per-request cap. Keep 4096 (≥ chunk+gen).
- **ngram speculative decoding**: −39%. Draft+verify adds per-step compute that
  isn't repaid under heavy batching on non-repetitive prose. (MTP spec-decode is
  vLLM-nightly-only for Gemma-4; not tested.)
- **Tensor parallelism (tp>1)**: not applicable — needs the reserved GPU 1, and
  wouldn't help since the model fits one card (TP would only add comm overhead).
- **Chunked prefill / continuous batching**: always on in vLLM V1; nothing to tune.

**Fastest:** fp8 + `gpu-memory-utilization 0.97` + client concurrency ≈192 →
**~4,450 out-tok/s** (mns 384 nudged 4,466). **Slowest:** the naive baseline
(no fp8, util 0.90) at **2,015**; ngram spec-decode (2,416) and starving the
server with low concurrency (c=32 → 1,688) are the other ways to go slow.
Net: **~2.2× the naive baseline** from flags alone, no quality cost.

### Recommended config for gemma-heretic
```sh
MODEL_PATH=/models/gemma SERVED_MODEL_NAME=gemma-heretic TEXT_ONLY=1 \
  MAX_MODEL_LEN=4096 GPU_MEM_UTIL=0.97 SERVER_EXTRA_ARGS='--kv-cache-dtype fp8' \
  ./synth.sh generate <prompts...> --system systems/vintage_1850.txt \
    --era 'the year 1850' --concurrency 192 --output out/gemma-heretic.jsonl
```
Everything except `--concurrency` is a server setting; the patch + text-only are
already required for gemma-heretic to load at all.
