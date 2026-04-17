# MLOps image: used as base for Kubeflow pipeline components.
# Contains all training dependencies.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source needed for training
COPY anomaly/ ./anomaly/
COPY detection/ ./detection/
COPY ingestion/ ./ingestion/
COPY mlops/ ./mlops/
COPY config/ ./config/

# Create required directories
RUN mkdir -p models metrics data/raw data/processed

CMD ["python", "-m", "mlops.train"]
