#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Container entrypoint. Modes (first argument):
#
#   download <hf-args...>   Download a model into /models (no server). Forwards
#                           everything to download_model.py.
#   serve                   Boot the vLLM server in the foreground and stay up
#                           (use it as a plain OpenAI-compatible API).
#   generate <gen-args...>  Boot vLLM in the background, wait for health, then
#                           run generate.py with the remaining arguments.
#   bash | sh <cmd...>      Escape hatch: exec a shell / arbitrary command.
#
# Any first argument that is not one of the words above is treated as the start
# of generate.py's arguments (i.e. `generate` is the default mode), so
#   docker run ... synth-vllm prompts/continue.txt --limit 100
# just works.
# ---------------------------------------------------------------------------
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-1234}"

case "${1:-}" in
    download)  shift; exec python3 /app/download_model.py "$@" ;;
    serve)     shift; MODE=serve ;;
    generate)  shift; MODE=generate ;;
    bash|sh)   exec "$@" ;;
    *)         MODE=generate ;;   # forward all args to generate.py
esac

# --------------------------------------------------------------------------- #
# Build the vLLM server command.                                              #
# SERVER_CMD overrides the whole thing; otherwise it is assembled from the     #
# individual knobs and SERVER_EXTRA_ARGS is appended verbatim (word-split).    #
# --------------------------------------------------------------------------- #
if [ -n "${SERVER_CMD:-}" ]; then
    server_argv=(bash -c "$SERVER_CMD")
else
    server_argv=(
        vllm serve "${MODEL_PATH:-/models/current}"
        --served-model-name "${SERVED_MODEL_NAME:-synth}"
        --host "$HOST" --port "$PORT"
        --dtype "${DTYPE:-auto}"
        --max-model-len "${MAX_MODEL_LEN:-8192}"
        --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}"
        --enable-prefix-caching
    )
    if [ -n "${TOKENIZER:-}" ]; then
        server_argv+=(--tokenizer "$TOKENIZER")
    fi
    if [ "${TEXT_ONLY:-}" = "1" ]; then
        # Disable image/audio/video on multimodal models (e.g. Gemma 3/4): frees
        # KV cache, speeds things up, and skips multimodal profiling. Do NOT set
        # this for text-only models (SmolLM2, etc.) — they reject the flag.
        server_argv+=(--limit-mm-per-prompt '{"image":0,"audio":0,"video":0}')
    fi
    if [ -n "${SERVER_EXTRA_ARGS:-}" ]; then
        # Intentionally unquoted so flags split into separate argv entries.
        # shellcheck disable=SC2206
        server_argv+=(${SERVER_EXTRA_ARGS})
    fi
fi

echo "[entrypoint] starting vLLM: ${server_argv[*]}"
"${server_argv[@]}" &
SERVER_PID=$!

# Tear the server down whenever this script exits, for any reason.
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------- #
# Wait for the server to report healthy (or die trying).                       #
# Always probe over the loopback, regardless of the server's bind address.     #
# --------------------------------------------------------------------------- #
HEALTH_URL="http://127.0.0.1:${PORT}/health"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"
echo "[entrypoint] waiting up to ${HEALTH_TIMEOUT}s for ${HEALTH_URL} ..."
deadline=$(( SECONDS + HEALTH_TIMEOUT ))
until python3 -c "import sys,urllib.request; urllib.request.urlopen('${HEALTH_URL}', timeout=2)" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[entrypoint] vLLM exited before becoming ready" >&2
        wait "$SERVER_PID" || true
        exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "[entrypoint] timed out waiting for vLLM health" >&2
        exit 1
    fi
    sleep 2
done
echo "[entrypoint] vLLM is ready."

if [ "$MODE" = "serve" ]; then
    echo "[entrypoint] serve mode: API up on container ${HOST}:${PORT} (reach it from the host via the published port); press Ctrl-C to stop."
    wait "$SERVER_PID"
    exit $?
fi

# generate mode ------------------------------------------------------------- #
if [ "$#" -eq 0 ]; then
    echo "[entrypoint] no generate.py args given; using default: prompts/continue.txt --limit 20"
    set -- prompts/continue.txt --limit 20
fi

echo "[entrypoint] running: python3 generate.py $*"
python3 generate.py "$@"
