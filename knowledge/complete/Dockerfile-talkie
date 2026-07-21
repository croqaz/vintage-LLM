# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Seed-completion sampler
# ---------------------------------------------------------------------------
# Boots a local llama-server (OpenAI-compatible API) and drives it with
# sample.py to generate creative text completions from the bundled seeds.
#
#   Build:
#     cd complete
#     docker build -t vintage-complete .
#
#   Run (any args after the image name are forwarded to sample.py):
#     docker run --gpus all -v "$PWD/out:/app/complete/out" \
#         vintage-complete --limit 100 --samples-per-seed 3 --format jsonl
#
#   Output lands in the mounted out/ folder (out/completions.txt by default).
# ---------------------------------------------------------------------------

# Base image: Llama.cpp server with CUDA 13 support
FROM ghcr.io/ggml-org/llama.cpp:server-cuda13

# Prepended to preserve the CUDA paths.
ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"

# UV package and project manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# curl fetches the model; ca-certificates lets it talk HTTPS to HuggingFace.
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Model -----------------------------------------------------------------
ARG HF_REPO=zakarth/talkie-1930-13b-it-vulkan-fixed-GGUF
ARG HF_FILE=talkie-1930-13b-it.Q8_0.gguf
ENV MODEL_PATH=/models/${HF_FILE}
RUN mkdir -p /models \
    && curl -fL --retry 3 --retry-delay 5 -o "${MODEL_PATH}" \
        "https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILE}?download=true"

# --- Python (uv managed) ---------------------------------------------------
# sample.py's API backend only needs `requests`; torch/transformers are
# lazily imported and unused here, so they're intentionally not installed.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --managed-python /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"
RUN uv pip install requests

# --- Application -----------------------------------------------------------
COPY sample.py /app/complete/sample.py
COPY seeds.txt /app/complete/seeds.txt
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /app/complete
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
