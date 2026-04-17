# Use CPU-based PyTorch for portability; swap base image for GPU builds.
# GPU: FROM nvcr.io/nvidia/pytorch:24.01-py3
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY detection/     ./detection/
COPY anomaly/       ./anomaly/
COPY ingestion/     ./ingestion/
COPY config/        ./config/

COPY mlops/__init__.py      ./mlops/__init__.py
COPY mlops/clip_saver.py    ./mlops/clip_saver.py

EXPOSE 8001

# Prometheus metrics endpoint is on 8001 — healthcheck polls it
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -sf http://localhost:8001/ > /dev/null || exit 1

CMD ["python", "-m", "detection"]
