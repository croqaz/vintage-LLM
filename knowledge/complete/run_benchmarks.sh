#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Sweep serving-engine configurations for the TypeWriter model and collect
# throughput/latency numbers into reports/results.jsonl.
#
# Every configuration is one `docker run`: the container boots the server with
# the given SERVER_CMD, waits for health, then bench.py sweeps concurrency
# levels against 127.0.0.1 and appends JSON rows. All inference happens INSIDE
# the image; this script only orchestrates docker.
#
# GPU: physical device 1 only (device 0 is reserved).
#
#   ./run_benchmarks.sh vllm      # requires image typewriter-vllm
#   ./run_benchmarks.sh sglang    # requires image typewriter-sglang
#   LIMIT=32 MAXTOK=128 CONC=1,16,64 ./run_benchmarks.sh vllm
# ---------------------------------------------------------------------------
set -uo pipefail

ENGINE="${1:-vllm}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_MOUNT="${HERE}/model-cache/typewriter-1913-7B-base-v2"
OUT="${HERE}/reports"
GPU='"device=1"'
LIMIT="${LIMIT:-64}"
MAXTOK="${MAXTOK:-256}"
CONC="${CONC:-1,8,32,64}"

mkdir -p "$OUT"

if [ ! -d "$MODEL_MOUNT" ]; then
    echo "ERROR: model not found at $MODEL_MOUNT" >&2
    exit 1
fi

# Base server command per engine (flags appended per-config below).
case "$ENGINE" in
    vllm)
        IMAGE="typewriter-vllm"
        BASE="vllm serve /models/typewriter --served-model-name typewriter --host 127.0.0.1 --port 1234 --dtype bfloat16"
        ;;
    sglang)
        IMAGE="typewriter-sglang"
        BASE="python3 -m sglang.launch_server --model-path /models/typewriter --served-model-name typewriter --host 0.0.0.0 --port 1234 --dtype bfloat16"
        ;;
    *)
        echo "unknown engine: $ENGINE" >&2; exit 1;;
esac

run_cfg() {
    local label="$1"; shift
    local server_cmd="$1"; shift
    echo ""
    echo "=================================================================="
    echo ">>> [$ENGINE] $label"
    echo "    SERVER_CMD=$server_cmd"
    echo "=================================================================="
    docker run --rm --gpus "$GPU" \
        -v "$MODEL_MOUNT:/models/typewriter:ro" \
        -v "$OUT:/app/complete/out" \
        -e SERVER_CMD="$server_cmd" \
        "$IMAGE" bench \
            --limit "$LIMIT" --max-tokens "$MAXTOK" --concurrency "$CONC" \
            --label "${ENGINE}:${label}" -o out/results.jsonl \
        2> >(tee "$OUT/serverlog_${ENGINE}_${label}.txt" >&2)
    if [ $? -ne 0 ]; then
        echo "!!! FAILED: $ENGINE $label (see reports/serverlog_${ENGINE}_${label}.txt)"
    fi
}

echo "engine=$ENGINE image=$IMAGE limit=$LIMIT max_tokens=$MAXTOK conc=$CONC"

if [ "$ENGINE" = "vllm" ]; then
    run_cfg "eager-baseline"   "$BASE --max-model-len 4096 --gpu-memory-utilization 0.90 --enforce-eager"
    run_cfg "default"          "$BASE --max-model-len 4096 --gpu-memory-utilization 0.90"
    run_cfg "ctx2048"          "$BASE --max-model-len 2048 --gpu-memory-utilization 0.90"
    run_cfg "memutil-0.95"     "$BASE --max-model-len 4096 --gpu-memory-utilization 0.95"
    run_cfg "kvcache-fp8"      "$BASE --max-model-len 4096 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8"
    run_cfg "weights-fp8"      "$BASE --max-model-len 4096 --gpu-memory-utilization 0.90 --quantization fp8"
else
    run_cfg "default"          "$BASE --context-length 4096 --mem-fraction-static 0.90"
    run_cfg "no-cuda-graph"    "$BASE --context-length 4096 --mem-fraction-static 0.90 --disable-cuda-graph"
    run_cfg "ctx2048"          "$BASE --context-length 2048 --mem-fraction-static 0.90"
    run_cfg "memfrac-0.95"     "$BASE --context-length 4096 --mem-fraction-static 0.95"
    run_cfg "kvcache-fp8"      "$BASE --context-length 4096 --mem-fraction-static 0.90 --kv-cache-dtype fp8_e5m2"
    run_cfg "triton-attn"      "$BASE --context-length 4096 --mem-fraction-static 0.90 --attention-backend triton"
fi

echo ""
echo "Done. Results in $OUT/results.jsonl"
