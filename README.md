# VortexVision

Production-grade real-time multimodal video analytics platform. Detects objects, tracks them across frames, identifies anomalies (fights, accidents, crowd rushes, weapons), and answers natural language queries over live video.

## Architecture

```
Video Sources (RTSP/YouTube/File)
        │
        ▼
  Kafka (video-frames)
        │
        ▼
  Detection Worker
  ├── YOLO26 + ByteTrack (object detection + tracking)
  ├── ConvAutoencoder (frame-level anomaly scoring)
  ├── TemporalTransformer (behavioral sequence anomaly)
  └── AnomalyClassifier (fight/weapon/crowd_rush/accident/trespassing)
        │
        ├── Kafka (annotated-frames) ──► API ──► WebSocket ──► Frontend
        └── Kafka (anomaly-events)  ──► API ──► DB + Alerts
                                          │
                                          ▼
                                   VLM (Qwen2.5-VL)
                                   NL Query Engine
```

## Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLO26 (Ultralytics) + ByteTrack |
| Anomaly Detection | ConvAutoencoder + TemporalTransformer |
| Anomaly Classification | Rule-based + heuristic classifier |
| VLM / NL Query | Qwen2.5-VL (vLLM or mock) |
| Streaming | Apache Kafka |
| Backend | FastAPI + SQLAlchemy (PostgreSQL) |
| Auth | JWT + API Key |
| Rate Limiting | Redis-backed sliding window |
| Frontend | Streamlit + WebSocket live feed |
| Model Serving | KServe + Ray Serve |
| MLOps | MLflow, DVC, Kubeflow Pipelines |
| Infra | Terraform + Kubernetes (EKS/GKE/kind) |
| GitOps | ArgoCD |
| CI/CD | GitHub Actions (lint, test, scan, deploy) |
| Monitoring | Prometheus, Grafana, Evidently AI, Loki, Jaeger |

## Quick Start (Local Dev — No GPU Required)

### Prerequisites
- Docker + Docker Compose
- Python 3.11+
- 8GB RAM minimum

### 1. Clone and set up

```bash
git clone https://github.com/your-org/vortexvision
cd vortexvision

# Install Python deps + create .env
make install

# Initialize random model weights (for dev — no training needed)
make init-models
```

### 2. Start the full stack

```bash
make up
```

This starts: Kafka, PostgreSQL, Redis, API, Detector, Frontend, MLflow, Prometheus, Grafana, Loki, Jaeger, AlertManager, Kafka UI, node-exporter, and a mock VLM server.

### 3. Inject demo video streams (no cameras needed)

```bash
make seed-demo
```

This generates synthetic video streams (normal pedestrians, fight scene, traffic) and injects them into Kafka.

### 4. Access the services

| Service | URL | Credentials |
|---|---|---|
| Streamlit UI | http://localhost:8501 | admin / vortex-admin-pass |
| API Docs | http://localhost:8000/docs | — |
| Grafana | http://localhost:3000 | admin / vortex123 |
| MLflow | http://localhost:5000 | — |
| Kafka UI | http://localhost:8080 | — |
| Jaeger | http://localhost:16686 | — |
| Prometheus | http://localhost:9090 | — |

### One-command setup (install + init + start + seed)

```bash
make setup
```

## Training Your Own Models

```bash
# 1. Place raw videos in data/raw/normal/ and data/raw/anomaly/
# 2. Run the full pipeline
make train

# Or step by step:
make prepare-data
make extract-features
make train-ae
make train-tf
make evaluate
make export
```

Datasets: UCF-Crime, ShanghaiTech Campus, custom traffic feeds.

## Kubernetes Deployment

```bash
# Provision infrastructure (AWS EKS)
make tf-init && make tf-apply

# Deploy application
kubectl apply -k infra/k8s/overlays/production

# GitOps (ArgoCD auto-syncs on git push)
kubectl apply -f infra/argocd/application.yaml
```

## Real Camera / Stream Setup

```bash
# Add an RTSP stream
curl -X POST http://localhost:8000/streams/ \
  -H "X-API-Key: vortex-dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"stream_id": "cam-entrance", "source": "rtsp://192.168.1.100/stream1", "fps_limit": 15}'

# Add a YouTube stream
curl -X POST http://localhost:8000/streams/ \
  -H "X-API-Key: vortex-dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"stream_id": "yt-traffic", "source": "https://youtube.com/watch?v=...", "fps_limit": 10}'
```

## Natural Language Queries

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me all red cars speeding in the last 5 minutes", "time_window_seconds": 300}'
```

## Performance Targets

- Sub-50ms P95 inference latency per frame
- 30+ concurrent video streams
- 99.99% uptime SLA
- Automatic retraining on drift detection (Evidently AI)

## Project Structure

```
vortexvision/
├── api/                    # FastAPI backend (REST + WebSocket)
├── anomaly/                # Autoencoder, Transformer, Classifier
├── config/                 # Pydantic settings, logging
├── detection/              # YOLO26 + ByteTrack + worker
├── docker/                 # Dockerfiles + init scripts
├── frontend/               # Streamlit UI
├── infra/                  # Terraform, K8s, Helm, ArgoCD
├── ingestion/              # Kafka producer/consumer, stream manager
├── migrations/             # Alembic DB migrations
├── mlops/                  # Training, evaluation, export, DVC pipeline
├── monitoring/             # Prometheus, Grafana, Loki, Evidently
├── scripts/                # Bootstrap, demo, setup utilities
├── serving/                # Ray Serve deployment
├── tests/                  # Unit + integration + load tests
└── vlm/                    # Qwen VLM client + NL query engine
```
