#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Thin wrapper around `docker run` for the synth-vllm image. It pins GPU 0,
# mounts the model cache, seeds, out/ and (for live editing) prompts/systems,
# and forwards the rest of the arguments to the container entrypoint.
#
#   HF_TOKEN=hf_... ./synth.sh download google/gemma-4-E4B-it --out /models/gemma
#   MODEL_PATH=/models/gemma TEXT_ONLY=1 \
#     ./synth.sh generate prompts/continue.txt --system systems/vintage_1850.txt \
#       --era 'the year 1850' --concurrency 64 --limit 500 --output out/gemma4.jsonl
#   ./synth.sh serve                      # just the API server on :$PORT
#   ./synth.sh bash                       # shell inside the container
#
# Env overrides: IMAGE, GPU, PORT, MODEL_PATH, SERVED_MODEL_NAME, MAX_MODEL_LEN,
# GPU_MEM_UTIL, DTYPE, TEXT_ONLY, TOKENIZER, SERVER_EXTRA_ARGS, SERVER_CMD, HF_TOKEN.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-synth-vllm}"
GPU="${GPU:-0}"
PORT="${PORT:-1234}"

mkdir -p models out

# Only forward env vars that are actually set, so the image's own defaults win.
env_args=()
for v in MODEL_PATH SERVED_MODEL_NAME MAX_MODEL_LEN GPU_MEM_UTIL DTYPE TEXT_ONLY \
         TOKENIZER SERVER_EXTRA_ARGS SERVER_CMD HEALTH_TIMEOUT HF_TOKEN PORT; do
    if [ -n "${!v:-}" ]; then env_args+=(-e "$v=${!v}"); fi
done

# seeds.jsonl is optional for `download`/`serve`/`bash`, required for generate;
# mount it read-only only when it exists on the host.
seed_args=()
if [ -f seeds.jsonl ]; then seed_args+=(-v "$PWD/seeds.jsonl:/app/seeds.jsonl:ro"); fi

# Allocate a TTY only when we actually have one, so this works the same when
# run interactively, under nohup/background, or in a script.
tty_args=()
if [ -t 0 ] && [ -t 1 ]; then tty_args+=(-it); fi

exec docker run --rm "${tty_args[@]}" \
    --gpus "\"device=${GPU}\"" \
    -p "${PORT}:${PORT}" \
    -v "$PWD/models:/models" \
    -v "$PWD/out:/app/out" \
    -v "$PWD/prompts:/app/prompts:ro" \
    -v "$PWD/systems:/app/systems:ro" \
    "${seed_args[@]}" \
    "${env_args[@]}" \
    "$IMAGE" "$@"
