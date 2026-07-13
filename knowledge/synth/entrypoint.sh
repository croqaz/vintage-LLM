#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Start the local llama-server, wait until it is ready, then run the
# self-instruct generator. Any arguments passed to `docker run` after the
# image name are forwarded verbatim to self_instruct_1900.py.
# ---------------------------------------------------------------------------
set -euo pipefail

LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"   # self_instruct_1900.py defaults to 127.0.0.1:1234
LLAMA_PORT="${LLAMA_PORT:-1234}"

echo "[entrypoint] starting llama-server (model: ${MODEL_PATH})"
/app/llama-server \
    --model "${MODEL_PATH}" \
    --host "${LLAMA_HOST}" \
    --port "${LLAMA_PORT}" \
    --n-gpu-layers "${N_GPU_LAYERS:-999}" \
    --ctx-size "${CTX_SIZE:-8192}" \
    ${LLAMA_EXTRA_ARGS:-} &
SERVER_PID=$!

# Tear the server down when this script exits for any reason.
cleanup() { kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[entrypoint] waiting for llama-server on ${LLAMA_HOST}:${LLAMA_PORT} ..."
until python -c "import urllib.request; urllib.request.urlopen('http://${LLAMA_HOST}:${LLAMA_PORT}/health', timeout=2)" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[entrypoint] llama-server exited before becoming ready" >&2
        wait "${SERVER_PID}" || true
        exit 1
    fi
    sleep 2
done
echo "[entrypoint] llama-server ready; launching self_instruct_1900.py"

python self_instruct_1900.py --out out/instruct.jsonl "$@"
