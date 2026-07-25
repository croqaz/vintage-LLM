#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Engine-agnostic launcher for a completion sampler.
#
# 1. Starts an OpenAI-compatible inference server in the background. The full
#    server command comes from $SERVER_CMD (set per-engine by the Dockerfile);
#    extra flags can be appended at run time via $SERVER_EXTRA_ARGS.
# 2. Waits until the server's health endpoint is live.
# 3. Runs a client workload against 127.0.0.1:
#       sample <args...>   -> python3 sample.py seeds.txt --api-url ... <args>
#       bench  <args...>   -> python3 bench.py  seeds.txt --api-url ... <args>
#       serve              -> just keep the server in the foreground
#       <anything else>    -> exec verbatim (server still running)
#    Default (no args) == `sample`.
# ---------------------------------------------------------------------------
set -euo pipefail

HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-1234}"
API_URL="http://${HOST}:${PORT}"
HEALTH_URL="${HEALTH_URL:-${API_URL}/health}"
MODEL_NAME="${SERVED_MODEL_NAME:-typewriter}"

if [ -z "${SERVER_CMD:-}" ]; then
    echo "[entrypoint] ERROR: SERVER_CMD is not set" >&2
    exit 1
fi

echo "[entrypoint] starting inference server:"
echo "             ${SERVER_CMD} ${SERVER_EXTRA_ARGS:-}"
# shellcheck disable=SC2086
eval "${SERVER_CMD} ${SERVER_EXTRA_ARGS:-}" &
SERVER_PID=$!

cleanup() { kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[entrypoint] waiting for server health at ${HEALTH_URL} ..."
ATTEMPTS=0
until curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[entrypoint] server exited before becoming ready" >&2
        wait "${SERVER_PID}" || true
        exit 1
    fi
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ "${ATTEMPTS}" -gt "${HEALTH_TIMEOUT:-600}" ]; then
        echo "[entrypoint] server not healthy after ${ATTEMPTS}s — giving up" >&2
        exit 1
    fi
    sleep 1
done
echo "[entrypoint] server ready after ${ATTEMPTS}s"

MODE="${1:-sample}"
[ "$#" -gt 0 ] && shift || true

case "${MODE}" in
    sample)
        exec python3 sample.py seeds.txt --api-url "${API_URL}" --model "${MODEL_NAME}" \
            -o out/completions.txt "$@"
        ;;
    bench)
        exec python3 bench.py seeds.txt --api-url "${API_URL}" --model "${MODEL_NAME}" "$@"
        ;;
    serve)
        echo "[entrypoint] serve mode — server running on ${API_URL}"
        wait "${SERVER_PID}"
        ;;
    *)
        exec "${MODE}" "$@"
        ;;
esac
