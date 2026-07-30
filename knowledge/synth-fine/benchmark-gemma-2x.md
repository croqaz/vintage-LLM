# Benchmark - making the gemma generation job ~2× faster

Goal: the production `generate.py` run has been going for ~4 days and is **11.7%
done**; at its current rate it lands in late September. This report validates
ways to at least double throughput. Everything here was measured on **GPU 1**.

## The live job (measured, not assumed)

| | |
|---|---|
| container | `gracious_bouman` (image `synth-vllm`), started 2026-07-24 21:01 |
| server | `MODEL_PATH=/models/gemma TEXT_ONLY=1 MAX_MODEL_LEN=4096 GPU_MEM_UTIL=0.97 SERVER_EXTRA_ARGS='--kv-cache-dtype fp8'` |
| client | 5 prompts (`diverse_qa_pairs`, `extract_knowledge`, `narrative`, `lost-manuscript-framing`, `magazine_feature`), `--concurrency 192`, everything else default |
| effective client defaults | `--chunk-tokens 3000`, `--max-chunks 3`, `--max-tokens 1024`, `temperature 0.8`, no `--system`, no `--clean` |
| progress | 2,825,783 / 24,147,900 tasks (**11.7%**), ok 2,727,087, err 98,696 |
| rate | **8.4 req/s** lifetime; instantaneous samples 7.9–9.4 req/s |
| KV pool | 147,982 tokens → "Maximum concurrency for 4,096 tokens per request: **36.13x**" |
| GPU 0 | 100% util, 435–444 W of 450 W, 67–68 °C, **no throttling** (`clocks_throttle_reasons.active = 0x0`) |

**Remaining: 21,322,117 tasks.** At 8.4 req/s that is **29.4 days**.

### Why the earlier benchmark did not predict this
`benchmark-gemma.md` tuned a *synthetic* workload: ~900-token prompts,
`max_tokens=256`, and it reported ~4,450 out-tok/s. The real job sends prompts
averaging **951 tokens** with `max_tokens=1024`, and spends most of its time in
**prefill**: at baseline it pushes 8,054 prompt-tok/s against only 2,713
out-tok/s - a 3:1 ratio. Conclusions tuned on the short workload (notably "match
concurrency to the KV pool") do not carry over. This report re-measures with a
faithful replica.

## Method
- `bench_real.py` - reproduces the production workload exactly: real seed chunks
  built by `generate.py`'s own `iter_seeds`/`build_tasks`/`chunk_text`, the real
  templates, no system prompt, `max_tokens=1024`, `temperature=0.8`. Reports
  **req/s**, directly comparable to the `N/M done … X/s` line generate.py prints.
- Workload: seeds 0–79 → **420 requests** (mean prompt 951 tokens). A warmup pass
  at `max_tokens=32` precedes timing.
- One server config per run on **GPU 1 only** (`--gpus device=1`, port 4567),
  restarted between configs. Baseline reproduces the live job to within seed
  noise (8.47 req/s on GPU 1 vs 8.4 lifetime / 9.4 instantaneous on GPU 0), so
  the replica is trustworthy.
- Reproduce a point:
  ```bash
  docker run -d --rm --name gbench1 --gpus "device=1" -p 4567:1234 \
    -v "$PWD/models:/models" \
    -e MODEL_PATH=/models/gemma -e SERVED_MODEL_NAME=gemma -e TEXT_ONLY=1 \
    -e MAX_MODEL_LEN=4096 -e GPU_MEM_UTIL=0.97 \
    -e SERVER_EXTRA_ARGS='--kv-cache-dtype fp8 --quantization fp8' synth-vllm serve
  python3 bench_real.py prompts/continue-v2.txt prompts/diverse_qa_pairs.txt \
    prompts/extract_knowledge.txt prompts/narrative.txt prompts/lost-manuscript-framing.txt \
    --base-url http://127.0.0.1:4567/v1 --num-seeds 80 --concurrency 288 --reorder
  ```

## Results (real workload, 420 requests, GPU 1)

| config | conc | req/s | out-tok/s | p50 | ok/420 | vs live |
|---|---:|---:|---:|---:|---:|---:|
| **live_baseline** (bf16 weights, fp8 KV) | 192 | 8.47 | 2,713 | 16.3 s | 396 | 1.00× |
| live_baseline | 96 | 7.88 | 2,549 | 9.3 s | 396 | 0.93× |
| live_baseline | 48 | 6.29 | 2,003 | 6.2 s | 396 | 0.74× |
| fp8 weights | 48 | 7.54 | 2,373 | 5.1 s | 396 | 0.89× |
| fp8 weights | 96 | 9.30 | 2,954 | 8.1 s | 396 | 1.10× |
| **fp8 weights** | 192 | 11.12 | 3,502 | 13.3 s | 396 | **1.31×** |
| fp8 weights | 288 | 10.99 | 3,514 | 19.3 s | 396 | 1.30× |
| fp8 weights | 384 | 10.91 | 3,503 | 23.2 s | 396 | 1.29× |
| **reorder only** (bf16 weights) | 192 | 12.19 | 3,687 | 10.5 s | 396 | **1.44×** |
| fp8 weights + reorder | 192 | 13.82 | 4,280 | 9.5 s | 396 | 1.63× |
| **fp8 weights + reorder** | 288 | **14.83** | **4,524** | 12.2 s | 396 | **1.75×** |
| fp8 weights, `max-model-len 4352` | 192 | 10.17 | 3,289 | 14.3 s | **417** | 1.20× |
| fp8 weights, `mnbt 16384` | 192 | 10.86 | 3,483 | 13.1 s | 396 | 1.28× |

## What worked

### 1. `--quantization fp8` - weight quantization (+31%)
The single config-only change. vLLM quantizes the bf16 weights to FP8 at load
(no pre-quantized checkpoint needed) and the RTX 4090 has native FP8 tensor
cores - the log confirms `Selected CutlassFP8ScaledMMLinearKernel for
Fp8PerTensorOnlineLinearMethod`. Effects:
- weights ~15.4 GB → ~8 GB, so the **KV pool goes 147,982 → 327,525 tokens**
  (max concurrency 36.13× → **79.96×**)
- FP8 GEMMs are faster than bf16 on Ada

**8.47 → 11.12 req/s.** Costs nothing but a flag. Loads cleanly with the existing
`patch_gemma4.py` and `TEXT_ONLY=1`.

### 2. Putting the document *before* the instruction (+44%, no numerics touched)
This is the biggest single lever and it is free of any quantization risk.

All six templates are shaped `<instructions>\n{text}`. The 5 templates applied to
the same seed chunk therefore each have a **different prefix**, so vLLM's prefix
cache can never reuse the expensive ~800-token chunk prefill - the same chunk is
prefilled **5 separate times**. Since the workload is 3:1 prefill-dominated, that
is where the time goes.

Reordering to `{text}\n\n<instructions>` makes the chunk a shared prefix across
all 5 requests for that chunk (and `generate.py` already emits the 5 siblings
adjacently, so they are in flight together). Same tokens submitted - the win is
entirely prefix-cache hits.

**8.47 → 12.19 req/s with bf16 weights**, and it composes with fp8:
**14.83 req/s (1.75×) with fp8 weights at concurrency 288.**

⚠️ **This changes the prompts, so it changes the generated data.** It is not a
numerical-accuracy question but a corpus-consistency one: 2.7 M records already
exist in the old format. Document-first / instruction-last is a well-established
format for long-context prompting, and the spot-check below looks clean - but
mixing formats mid-corpus is your call. See "Quality" below.

### 3. Concurrency: 192 is already right; 288 helps only with reorder
The live job's `--concurrency 192` is **not** over-subscribed, despite the KV
pool holding only 36 full-length sequences. Throughput rises monotonically
48 → 96 → 192 and then flattens (fp8: 192 → 11.12, 288 → 10.99, 384 → 10.91).
With reorder the plateau moves out to 288 (14.83). This directly contradicts the
old report's guidance and is the clearest sign the workload is **compute-bound,
not KV-capacity-bound**, once fp8 has enlarged the pool.

## What didn't work
- **`--max-num-batched-tokens 16384`**: 10.86 vs 11.12 (−2%). Leave at default,
  even though the workload is prefill-heavy.
- **Concurrency above 192** (without reorder): flat-to-slightly-worse, and p50
  latency doubles (13.3 s → 23.2 s).
- **Lower concurrency**: strictly worse (48 → 0.74×). The old report's
  "match concurrency to the KV pool" advice is wrong for this workload.
- **`--max-model-len 4352`**: −9% throughput. It *does* fix a real data-loss bug
  (see below), but the server-side fix is the expensive way to do it.
- **INT4 / AWQ / GPTQ (W4A16)**: not tested, and the data argues against it. Once
  fp8 raised the KV pool to 80× concurrency, throughput **plateaued** - the
  bottleneck moved to compute. W4A16 shrinks weights further but needs a
  dequant step and is typically *slower* than FP8 at large batch sizes, and it
  would require a calibration pass over this fine-tune. Low expected value.
- **Tensor parallelism across the two 4090s**: still the wrong tool. Two
  independent servers (data parallel) is strictly better for throughput than
  splitting one model over a PCIe link, and TP would require stopping GPU 0.

## Bug found: 3.5% of all generations are being silently dropped
98,696 of the job's errors - **every single one sampled** - are:

```
Error code: 400 - This model's maximum context length is 4096 tokens.
However, you requested 1024 output tokens and your prompt contains at least
3073 input tokens, for a total of at least 4097 tokens.
```

`generate.py` chunks with **tiktoken/cl100k** at `--chunk-tokens 3000`, but the
server counts **gemma** tokens; long chunks land at 3073 and 3073 + 1024 = 4097
overshoots `MAX_MODEL_LEN=4096` by one token. Those records are lost, not retried.

Guessing a safe `--chunk-tokens` does **not** work, as a smoke test proved:
gemma emits ~**1.05 tokens per cl100k token** on this OCR corpus, and template
overhead varies a lot (measured with the model tokenizer: `diverse_qa_pairs`
**321**, `extract_knowledge` 191, `continue-v2` 193, `narrative` 90,
`magazine_feature` 83, `lost-manuscript-framing` 70). `--chunk-tokens 2800`
still overflowed on `diverse_qa_pairs`.

Fixed properly in `generate.py`:
- `--tokenizer-path` (defaults to `$MODEL_PATH`) makes chunking use the **model's
  own tokenizer**, so `--chunk-tokens` is exact rather than an estimate;
- a startup budget check warns when `chunk + worst template + max_tokens` cannot
  fit, and prints the exact value to use.

With the real tokenizer the budget is `4096 − 1024 − 321 = 2751`, so
**`--chunk-tokens 2700`** is correct and costs nothing.
Server-side `MAX_MODEL_LEN=4352` also fixes it (ctx errors 24/420 → 3/420) but
costs **−9% throughput**, so it is the wrong lever.

## Quality

40 generations per config, **greedy (`temperature 0`)**, same seeds and templates,
via `quality_check.py`. Both candidate configs were compared against the current
production config (bf16 weights + fp8 KV).

| config | words | repeated 5-grams | non-ASCII | unfinished @512 tok |
|---|---:|---:|---:|---:|
| bf16 (production) | 208 | 0.003 | 0.0001 | 8/40 |
| fp8 weights | 214 | 0.004 | 0.0001 | 6/40 |
| fp8 weights + reorder | 200 | 0.003 | 0.0001 | 4/40 |

Statistically indistinguishable: same output length, no repetition
degeneration, no encoding artefacts, no increase in truncation. Manual reading of
matched samples shows equivalent, coherent period prose in all three.

Raw text similarity between bf16 and fp8 is 0.38, and **that number is not a
quality signal** - under greedy decoding any tiny numerical difference makes the
two continuations diverge at some token and never re-converge. It measures
divergence, not degradation.

### Reordering the templates did initially cost quality - and was fixed
A 2,640-generation run with the reordered templates, compared against 6,000
records from the live corpus, showed real regressions on three of five prompts
(rates, not counts):

| template | mean chars | short <200 | meta-commentary |
|---|---|---|---|
| diverse_qa_pairs | 1483 → 1552 | 0% → 0% | 0% → 0% |
| magazine_feature | 4406 → 4177 | 0% → 0% | 0% → 0% |
| narrative | 2393 → 2281 | 0% → 0.2% | 0.1% → **4.5%** |
| extract_knowledge | 2069 → 1941 | 0% → 0.2% | 4.6% → **10.2%** |
| lost-manuscript-framing | 1051 → **578** | 9.3% → **34.8%** | 0.2% → 0.6% |

With the instruction *after* the document the model treats the excerpt's end as a
natural stopping point (short continuations) and drifts into commenting on the
source. Three templates were then tuned - explicit "begin directly, never open
with 'This document…'" lines for `narrative` and `extract_knowledge`, and
"carry the manuscript forward for two or three substantial paragraphs" for
`lost-manuscript-framing`. Re-measured over another 2,630 generations, the tuned
text-first templates now **match or beat** the live corpus everywhere:

| template | mean chars | short <200 | meta-commentary |
|---|---|---|---|
| diverse_qa_pairs | 1483 → 1548 | 0% → 0% | 0% → 0% |
| extract_knowledge | 2069 → 1955 | 0% → 0% | 4.6% → **1.3%** |
| lost-manuscript-framing | 1051 → **1847** | 9.3% → **0.0%** | 0.2% → 0.2% |
| magazine_feature | 4406 → 4184 | 0% → 0% | 0% → 0% |
| narrative | 2393 → 2250 | 0% → 0% | 0.1% → **0.0%** |

Lesson: reordering is not free - it changes model behaviour and each template
needs checking. Any *further* template converted to text-first should be
re-measured the same way.

Two honest caveats:
- The check is 40 samples. Individual factual details drawn from the source can
  differ between configs (in one sampled pair, "less than two and a half years"
  vs "less than a year and a half"). That is expected from divergent greedy
  paths rather than evidence of fp8 damage, but if factual fidelity to the seed
  is critical, run a larger comparison before committing.
- vLLM logs `Using uncalibrated q_scale 1.0 and/or prob_scale 1.0 with fp8
  attention`. This comes from the **fp8 KV cache, which the live job already
  uses** - it is not new with `--quantization fp8`.

## Options, with completion dates

Remaining: **21,322,117 tasks**. Baseline 8.4 req/s.

| | scenario | req/s | days | finishes |
|---|---|---:|---:|---|
| A | do nothing | 8.4 | 29.4 | ~Aug 27 |
| B | **add GPU 1 running today's exact config** | 16.8 | 14.7 | ~Aug 12 |
| C | add GPU 1 with `--quantization fp8` | 19.4 | 12.7 | ~Aug 10 |
| D | add GPU 1 with fp8 + reorder @ 288 | 23.1 | 10.7 | ~Aug 8 |
| E | D, and restart GPU 0 with fp8 + reorder too | 29.4 | 8.4 | ~Aug 6 |

**The 2× target is met by option B alone** - just putting the idle GPU to work,
with no config change, no quantization, and no prompt change. C and D are
strictly better and carry only the risks noted above. E is the only one that
requires stopping the running job.

### End-to-end validation (GPU 1, full production path)
The final configuration was run through the real `synth.sh generate` path, not
just the benchmark client:

| run | req/s | ok | errors |
|---|---:|---:|---:|
| tuned templates, `--chunk-tokens 2700`, c=288 | **13.9–15.0** | 2,630 / 2,630 | **0** |

vs the live job's 8.4 req/s - **~1.8× per GPU**, with the context-overflow data
loss eliminated entirely.

Three bugs were found and fixed during this validation:
1. `entrypoint.sh` never passed `PORT` to `generate.py`, so any container with
   `PORT != 1234` failed with `APIConnectionError` - it now injects
   `--base-url http://127.0.0.1:$PORT/v1` unless the caller supplies one. This
   blocks running a second worker on another port.
2. `generate.py` materialised every Task up front (~88 GB RSS, long startup on
   4.5 M seeds). It now streams through a bounded queue.
3. Chunking used tiktoken while the server counts gemma tokens - see the
   context-overflow bug above.

## Interference check
The live job was measured repeatedly while GPU 1 was running full-tilt
benchmarks: 9.38 req/s and 7.86 req/s in two 90–120 s windows, against a
lifetime average of 8.4. The control - GPU 1 completely idle (1 MiB, 19 W) after
all benchmarks finished - gives **8.01 req/s**, i.e. within noise of the 7.86
measured under full load. **Running GPU 1 costs GPU 0 nothing measurable.**
GPU 0 showed no throttling at any point
(`clocks_throttle_reasons.active = 0x0`, 2640–2700 MHz, 435–444 W, 67–68 °C) and
the box has 64 cores at load average ~4. The spread is seed-length variance, not
contention. Running both GPUs at once draws ~850 W across the two cards.
