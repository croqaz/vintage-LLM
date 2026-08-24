# TypeWriter seed-completion sampler

Generates creative text completions for **`typewriter-ai/typewriter-1913-7B-base-v2`**
(a 7.24 B Llama-architecture *base* model) from the bundled `seeds.txt`.

This replaces the previous llama.cpp/GGUF pipeline
(`zakarth/talkie-1930-13b-it-vulkan-fixed-GGUF`). The new model ships only as
safetensors, so it's served with a real batching engine (**vLLM** or **SGLang**)
behind an OpenAI-compatible API that `sample.py` drives.

## Files
| file | purpose |
|---|---|
| `sample.py` | client: seeds → completions (API or local-HF backend). Now supports `--concurrency`. |
| `bench.py` | throughput/latency benchmark client (records tokens/s, latency). |
| `Dockerfile.vllm` | image `typewriter-vllm` — vLLM OpenAI server + clients. |
| `Dockerfile.sglang` | image `typewriter-sglang` — SGLang server + clients. |
| `entrypoint.sh` | boots the server, waits for health, runs `sample`/`bench`/`serve`. |
| `run_benchmarks.sh` | sweeps engine flag-configs × concurrency into `reports/`. |
| `clean_completions.py` | post-process a `completions.txt` dump into clean JSONL. |
| `reports/` | benchmark method, results, and the recommendation. |
| `model-cache/` | downloaded model weights (mounted into containers; git-ignored). |

## One-time setup: download the model
The weights (~14.5 GB) live in `model-cache/` and are **mounted** into containers
(not baked into the image). To (re)download:
```bash
mkdir -p model-cache/typewriter-1913-7B-base-v2 && cd $_
BASE=https://huggingface.co/typewriter-ai/typewriter-1913-7B-base-v2/resolve/main
for f in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors; do
  curl -fL -o "$f" "$BASE/$f?download=true"; done
```

## Build
```bash
docker build -f Dockerfile.vllm   -t typewriter-vllm   .
docker build -f Dockerfile.sglang -t typewriter-sglang .
```

## Generate completions (recommended: vLLM, high concurrency)
Only GPU **device 1** is used (`--gpus '"device=1"'`). Output lands in `out/`.
```bash
docker run --rm --gpus '"device=1"' \
  -v "$PWD/model-cache/typewriter-1913-7B-base-v2:/models/typewriter:ro" \
  -v "$PWD/out:/app/complete/out" \
  typewriter-vllm sample --limit 1000 --concurrency 64 --format jsonl
# -> out/completions.txt (JSONL: {"seed":..., "text":...})
```
`--concurrency 64` is the key flag — it's ~35× faster than the default `1`.

For maximum speed (+~50%, minor quality tradeoff) add FP8 weights:
```bash
docker run --rm --gpus '"device=1"' \
  -v "$PWD/model-cache/typewriter-1913-7B-base-v2:/models/typewriter:ro" \
  -v "$PWD/out:/app/complete/out" \
  -e SERVER_CMD='vllm serve /models/typewriter --served-model-name typewriter --host 127.0.0.1 --port 1234 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --quantization fp8' \
  typewriter-vllm sample --limit 1000 --concurrency 64 --format jsonl
```

## talkie-1930-13b-it (13B, custom architecture)
A second model, `lewtun/talkie-1930-13b-it-hf`, is served too — but it's a
**custom architecture** (not vLLM-native) and doesn't fit 24GB in bf16, so it
needed a port + offline FP8 quantization. Full write-up:
`reports/02_talkie_13b.md`. Artifacts in `talkie-port/` (vLLM-compatible
`modeling_talkie.py`, parity check, FP8 quant scripts). Servable model:
`model-cache/talkie-13b-fp8/`. Run it (reuses the `typewriter-vllm` image):
```bash
docker run --rm --gpus '"device=1"' \
  -v "$PWD/model-cache/talkie-13b-fp8:/models/talkie:ro" \
  -v "$PWD/out:/app/complete/out" \
  -e SERVED_MODEL_NAME=talkie -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e HEALTH_TIMEOUT=900 \
  -e SERVER_CMD='vllm serve /models/talkie --served-model-name talkie --model-impl transformers --trust-remote-code --max-model-len 2048 --gpu-memory-utilization 0.92 --host 127.0.0.1 --port 1234' \
  typewriter-vllm sample --concurrency 32 --format jsonl
```
Peak ~1,074 tok/s at concurrency 32 (FP8, one 4090).

## Benchmark
```bash
LIMIT=64 MAXTOK=256 CONC=1,16,64 ./run_benchmarks.sh vllm
LIMIT=64 MAXTOK=256 CONC=1,16,64 ./run_benchmarks.sh sglang
python3 reports/summarize.py reports/results.jsonl
```

## Results summary
See `reports/01_results.md`. Headline on one RTX 4090, steady state at high
concurrency: **~6,800 output tok/s (BF16) and ~9,285 tok/s (FP8, no quality
loss)** — versus ~60 tok/s for the old one-at-a-time client (~110–150× faster).
Throughput scales almost linearly with `--concurrency` (use 128–256 for the bulk
job). SGLang is smoother at moderate concurrency but slower at saturation.

Recommended: **vLLM + `--quantization fp8`, `--concurrency 128–256`**.
