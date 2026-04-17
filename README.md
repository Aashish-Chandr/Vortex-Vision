<div align="center">
# ⚡ VortexVision
 
### Real-Time · Multimodal · Production-Ready
 
**Production-grade real-time video analytics platform.**  
Detect objects · Track motion · Identify anomalies · Query in natural language — all over live video.
 
<br/>
  
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Kafka](https://img.shields.io/badge/Apache%20Kafka-Streaming-231F20?style=for-the-badge&logo=apachekafka&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-EKS%2FGKE-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

![CI/CD](https://img.shields.io/github/actions/workflow/status/Aashish-Chandr/Vortex-Vision/ci.yml?style=flat-square&label=CI%2FCD&logo=githubactions)
![Language](https://img.shields.io/github/languages/top/Aashish-Chandr/Vortex-Vision?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/Aashish-Chandr/Vortex-Vision?style=flat-square)
![Stars](https://img.shields.io/github/stars/Aashish-Chandr/Vortex-Vision?style=flat-square)

<br/>

[🚀 Quick Start](#-quick-start) · [🏗 Architecture](#-architecture) · [⚙️ Stack](#%EF%B8%8F-tech-stack) · [📂 Structure](#-project-structure) · [🌐 Deployment](#-kubernetes-deployment) · [🤝 Contributing](#-contributing)

</div>

---

## 🌀 What is VortexVision?

**VortexVision** is an end-to-end, production-ready platform for analyzing live video streams with deep learning. It ingests video from cameras, files, or YouTube, runs it through a multi-stage AI pipeline, and delivers real-time alerts, annotated streams, and natural-language answers via a web UI and REST API.

| Capability | Details |
|---|---|
| 🎯 **Object Detection & Tracking** | YOLO26 + ByteTrack for frame-accurate multi-object tracking |
| 🚨 **Anomaly Detection** | ConvAutoencoder (frame-level) + TemporalTransformer (behavioral) |
| 🏷️ **Anomaly Classification** | Fights · Weapons · Crowd Rushes · Accidents · Trespassing |
| 💬 **Natural Language Queries** | Qwen2.5-VL VLM — ask questions about your live footage in plain English |
| 📡 **Live Streaming** | WebSocket-backed real-time annotated video in the browser |
| 🔐 **Auth & Rate Limiting** | JWT + API Key auth, Redis-backed sliding-window rate limiting |
| 📊 **Full Observability** | Prometheus · Grafana · Loki · Jaeger · Evidently AI drift detection |
| ☸️ **Cloud-Native** | Kubernetes (EKS/GKE/kind), GitOps via ArgoCD, IaC via Terraform |

---

## 🏗 Architecture

```
  ╔══════════════════════════════════════════╗
  ║     Video Sources (RTSP / YouTube / File) ║
  ╚══════════════════╦═══════════════════════╝
                     │
                     ▼
           ┌─────────────────┐
           │  Apache Kafka   │  ← video-frames topic
           └────────┬────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │    Detection Worker   │
        │  ┌─────────────────┐  │
        │  │ YOLO26+ByteTrack│  │  Object Detection + Tracking
        │  ├─────────────────┤  │
        │  │ ConvAutoencoder │  │  Frame-level Anomaly Scoring
        │  ├─────────────────┤  │
        │  │TemporalTransfmr │  │  Behavioral Sequence Analysis
        │  ├─────────────────┤  │
        │  │AnomalyClassifier│  │  fight / weapon / crowd_rush / accident
        │  └─────────────────┘  │
        └────────┬──────────────┘
                 │
       ┌─────────┴──────────┐
       │                    │
       ▼                    ▼
  annotated-frames     anomaly-events
  (Kafka topic)        (Kafka topic)
       │                    │
       ▼                    ▼
  ┌─────────────────────────────┐
  │         FastAPI             │
  │  REST · WebSocket · Auth    │
  └──────┬──────────┬───────────┘
         │          │
         ▼          ▼
    Frontend     DB + Alerts
   (Streamlit)  (PostgreSQL)
         │
         ▼
   VLM Query Engine
   (Qwen2.5-VL / vLLM)
   "Show me red cars speeding"
```

---

## ⚙️ Tech Stack

<details open>
<summary><b>Full Stack Overview</b></summary>

| Layer | Technology |
|---|---|
| 🔍 **Object Detection** | [YOLO26](https://github.com/ultralytics/ultralytics) (Ultralytics) + ByteTrack |
| 🧠 **Anomaly Detection** | ConvAutoencoder + TemporalTransformer (custom PyTorch) |
| 🏷️ **Anomaly Classification** | Rule-based + heuristic classifier |
| 💬 **VLM / NL Queries** | Qwen2.5-VL served via vLLM (or mock for dev) |
| 📨 **Streaming** | Apache Kafka |
| 🌐 **Backend** | FastAPI + SQLAlchemy (PostgreSQL) |
| 🔐 **Auth** | JWT + API Key |
| ⚡ **Rate Limiting** | Redis-backed sliding window |
| 🖥️ **Frontend** | Streamlit + WebSocket live feed |
| 🚀 **Model Serving** | KServe + Ray Serve |
| 🔬 **MLOps** | MLflow · DVC · Kubeflow Pipelines |
| 🏗️ **Infrastructure** | Terraform + Kubernetes (EKS / GKE / kind) |
| 🔄 **GitOps** | ArgoCD |
| 🔁 **CI/CD** | GitHub Actions (lint · test · scan · deploy) |
| 📊 **Monitoring** | Prometheus · Grafana · Evidently AI · Loki · Jaeger |

</details>

---

## 🚀 Quick Start

### Prerequisites

- **Docker** + **Docker Compose**
- **Python 3.11+**
- **8 GB RAM** minimum *(no GPU needed for local dev)*

### 1 · Clone & Install

```bash
git clone https://github.com/Aashish-Chandr/Vortex-Vision.git
cd Vortex-Vision

# Install Python dependencies and generate .env from .env.example
make install

# Initialise random model weights (no training required in dev)
make init-models
```

### 2 · Start the Full Stack

```bash
make up
```

This single command spins up: **Kafka · PostgreSQL · Redis · FastAPI · Detection Worker · Streamlit · MLflow · Prometheus · Grafana · Loki · Jaeger · AlertManager · Kafka UI · node-exporter · mock VLM server**.

### 3 · Seed Demo Streams

```bash
make seed-demo
```

Generates synthetic video streams (normal pedestrians · fight scene · traffic) and pushes them into Kafka. No cameras required.

### 4 · Access Services

| Service | URL | Credentials |
|---|---|---|
| 🖥️ Streamlit UI | http://localhost:8501 | `admin` / `vortex-admin-pass` |
| 📖 API Docs (Swagger) | http://localhost:8000/docs | — |
| 📊 Grafana | http://localhost:3000 | `admin` / `vortex123` |
| 🔬 MLflow | http://localhost:5000 | — |
| 📨 Kafka UI | http://localhost:8080 | — |
| 🔍 Jaeger Tracing | http://localhost:16686 | — |
| 📈 Prometheus | http://localhost:9090 | — |

### 5 · One-Command Full Setup

```bash
# Install + init + start + seed — all in one shot
make setup
```

---

## 📡 Stream Management

### Add an RTSP Camera

```bash
curl -X POST http://localhost:8000/streams/ \
  -H "X-API-Key: vortex-dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "stream_id": "cam-entrance",
    "source": "rtsp://192.168.1.100/stream1",
    "fps_limit": 15
  }'
```

### Add a YouTube Stream

```bash
curl -X POST http://localhost:8000/streams/ \
  -H "X-API-Key: vortex-dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "stream_id": "yt-traffic",
    "source": "https://youtube.com/watch?v=STREAM_ID",
    "fps_limit": 10
  }'
```

---

## 💬 Natural Language Queries

Ask questions about your live footage in plain English via the VLM query engine:

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show me all red cars speeding in the last 5 minutes",
    "time_window_seconds": 300
  }'
```

Example queries:
- *"How many people were in Zone B between 2 PM and 3 PM?"*
- *"Was there a fight detected near the entrance today?"*
- *"List all weapons detected in the last hour."*
- *"Show crowd density trends for camera 3."*

---

## 🧬 Training Your Own Models

```bash
# 1. Place raw videos in:
#    data/raw/normal/   ← normal footage
#    data/raw/anomaly/  ← anomalous footage

# 2. Full pipeline (all steps)
make train

# — or step by step —
make prepare-data       # Preprocess and split datasets
make extract-features   # Extract frame-level embeddings
make train-ae           # Train ConvAutoencoder
make train-tf           # Train TemporalTransformer
make evaluate           # Run evaluation metrics
make export             # Export models for serving
```

**Supported datasets:** UCF-Crime · ShanghaiTech Campus · custom traffic feeds

---

## ☸️ Kubernetes Deployment

### Provision Infrastructure (AWS EKS)

```bash
make tf-init
make tf-apply
```

### Deploy the Application

```bash
kubectl apply -k infra/k8s/overlays/production
```

### GitOps with ArgoCD

ArgoCD automatically syncs on every `git push` to `main`:

```bash
kubectl apply -f infra/argocd/application.yaml
```

---

## 📈 Performance Targets

| Metric | Target |
|---|---|
| ⚡ P95 Inference Latency | **< 50 ms** per frame |
| 📡 Concurrent Streams | **30+** simultaneous video feeds |
| 🟢 Uptime SLA | **99.99%** |
| 🔁 Model Retraining | Automatic on drift detection (Evidently AI) |

---

## 📂 Project Structure

```
vortexvision/
│
├── 🌐 api/                  # FastAPI backend — REST + WebSocket endpoints
├── 🚨 anomaly/              # ConvAutoencoder · TemporalTransformer · Classifier
├── ⚙️  config/              # Pydantic settings, structured logging
├── 🎯 detection/            # YOLO26 + ByteTrack + Detection worker
├── 🐳 docker/               # Dockerfiles + init scripts
├── 🖥️  frontend/            # Streamlit UI + WebSocket live feed
├── 🏗️  infra/               # Terraform · K8s · Helm charts · ArgoCD
├── 📨 ingestion/            # Kafka producer/consumer · Stream manager
├── 🗄️  migrations/          # Alembic DB migrations
├── 🔬 mlops/                # Training pipeline · Evaluation · DVC
├── 📊 monitoring/           # Prometheus · Grafana dashboards · Evidently · Loki
├── 📜 scripts/              # Bootstrap · demo seed · setup utilities
├── 🚀 serving/              # Ray Serve model deployment
├── 🧪 tests/                # Unit · Integration · Load tests
├── 💬 vlm/                  # Qwen2.5-VL client + NL query engine
│
├── docker-compose.yml       # Full local stack
├── Makefile                 # Developer convenience commands
├── pyproject.toml           # Python project config
└── .env.example             # Environment variable template
```

---

## 🧪 Testing

```bash
# Run all tests
make test

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Load tests
make load-test
```

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key variables:

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka broker address |
| `REDIS_URL` | Redis connection for rate limiting |
| `JWT_SECRET_KEY` | Secret for JWT signing |
| `API_KEY` | Default API key for development |
| `VLM_ENDPOINT` | vLLM server URL for Qwen2.5-VL |
| `MLFLOW_TRACKING_URI` | MLflow server address |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feat/your-feature`
3. **Commit** your changes: `git commit -m "feat: add amazing feature"`
4. **Push** to the branch: `git push origin feat/your-feature`
5. **Open** a Pull Request

Please make sure to:
- Follow the existing code style (`make lint`)
- Add tests for new features (`make test`)
- Update documentation as needed

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ by [Aashish Chandra](https://github.com/Aashish-Chandr)

⭐ **Star this repo** if you find it useful!

[![GitHub stars](https://img.shields.io/github/stars/Aashish-Chandr/Vortex-Vision?style=social)](https://github.com/Aashish-Chandr/Vortex-Vision/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Aashish-Chandr/Vortex-Vision?style=social)](https://github.com/Aashish-Chandr/Vortex-Vision/fork)

</div>
