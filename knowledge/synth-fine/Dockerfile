# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Synthetic-data generator — vLLM backend (model-agnostic)
# ---------------------------------------------------------------------------
# Serves a small safetensors instruct model with vLLM's OpenAI-compatible server
# and drives it with generate.py. The model is NOT baked into the image: it is
# downloaded into a mounted models/ dir (-> /models) and selected at run
# time via the MODEL_PATH env var, so switching models never means rebuilding.
#
#   Build:
#     docker build -t synth-vllm .
#
#   Download a model into the mounted models/ dir (one-time, per model):
#     docker run --rm -e HF_TOKEN="$HF_TOKEN" -v "$PWD/models:/models" \
#       synth-vllm download google/gemma-4-E4B-it --out /models/gemma
#
#   Generate (server boots in the background, then generate.py runs):
#     docker run --rm --gpus '"device=0"' \
#       -v "$PWD/models:/models" \
#       -v "$PWD/seeds.jsonl:/app/seeds.jsonl:ro" \
#       -v "$PWD/out:/app/out" \
#       -e MODEL_PATH=/models/gemma -e TEXT_ONLY=1 \
#       synth-vllm prompts/continue.txt --system systems/vintage_1850.txt \
#         --concurrency 64 --limit 500
#
# ---------------------------------------------------------------------------
ARG VLLM_TAG=latest
FROM vllm/vllm-openai:${VLLM_TAG}

# generate.py needs the OpenAI client and (soft) tiktoken. download_model.py
# uses huggingface_hub, which is already in the base image (with the Xet
# high-performance transfer backend).
RUN pip install --no-cache-dir openai tiktoken

# Fix vLLM 0.25.1's native Gemma-4 attention so KV-sharing checkpoints
# (num_kv_shared_layers > 0, e.g. gemma-4-E2B/E4B) load. See patch_gemma4.py.
COPY patch_gemma4.py /app/patch_gemma4.py
RUN python3 /app/patch_gemma4.py

# Server / client defaults. Everything here is overridable with `docker run -e`.
# HOST is the server BIND address; 0.0.0.0 so `serve` mode is reachable via a
# published port. The health check and generate.py always talk to 127.0.0.1.
ENV HOST=0.0.0.0 \
    PORT=1234 \
    SERVED_MODEL_NAME=synth \
    MODEL_PATH=/models/current \
    MAX_MODEL_LEN=8192 \
    GPU_MEM_UTIL=0.90 \
    DTYPE=auto \
    HEALTH_TIMEOUT=1200 \
    HF_XET_HIGH_PERFORMANCE=1

WORKDIR /app
COPY generate.py ocr_clean.py download_model.py entrypoint.sh /app/
COPY prompts/ /app/prompts/
COPY systems/ /app/systems/
RUN chmod +x /app/entrypoint.sh && mkdir -p /app/out /models

# The base vLLM image sets its own ENTRYPOINT (the API server); replace it.
ENTRYPOINT ["/app/entrypoint.sh"]
