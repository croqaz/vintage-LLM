# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Synthetic-data generator
# ---------------------------------------------------------------------------
# Boots a local llama-server (OpenAI-compatible API) and drives it with
# self_instruct_1900.py to synthesise the 1899 time-capsule dataset.
#
#   Build:
#     cd knowledge
#     docker build -f synth/Dockerfile -t vintage-synth .
#
#   Run (any args after the image name are forwarded to self_instruct_1900.py):
#     docker run --gpus all -v "$PWD/out:/app/synth/out" \
#         vintage-synth --num 50 --temporal-judge --out out/dataset.jsonl
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
# self_instruct_1900.py only needs the standard library.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --managed-python /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# --- Application -----------------------------------------------------------
# The script does `sys.path.append('../pre1900')` relative to its CWD, and
# banned_terms.py loads data/*.txt relative to its own file location, so the
# layout must be:
#   /app/synth/self_instruct_1900.py
#   /app/pre1900/banned_terms.py
#   /app/pre1900/data/*.txt
COPY pre1900/banned_terms.py /app/pre1900/banned_terms.py
COPY pre1900/data/*.txt /app/pre1900/data/
COPY synth/self_instruct_1900.py /app/synth/self_instruct_1900.py
COPY synth/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Run from synth/ so the script's `../pre1900` relative import resolves.
WORKDIR /app/synth
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
