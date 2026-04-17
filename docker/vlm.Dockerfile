# VLM Service: serves Qwen2.5-VL via vLLM's OpenAI-compatible API.
# Requires NVIDIA GPU. For CPU-only dev, the API falls back gracefully.
#
# Usage:
#   docker build -f docker/vlm.Dockerfile -t vortexvision/vlm:latest .
#   docker run --gpus all -p 8080:8080 \
#     -e MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct \
#     vortexvision/vlm:latest

FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
ENV HF_HOME=/models/hf_cache
ENV PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip curl git && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# vLLM provides an OpenAI-compatible server with vision support
RUN pip install --no-cache-dir \
    vllm>=0.4.0 \
    transformers>=4.41.0 \
    accelerate>=0.30.0 \
    Pillow>=10.3.0

VOLUME ["/models"]
EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --max-model-len 4096 \
    --dtype auto \
    --trust-remote-code"]
