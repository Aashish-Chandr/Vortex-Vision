.PHONY: help install dev test lint format build up down logs clean migrate train deploy

PYTHON := python
PIP := pip
DOCKER_COMPOSE := docker-compose

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Development ───────────────────────────────────────────────────────────────
install:  ## Install all dependencies
	$(PIP) install -r requirements-dev.txt
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example"; fi
	@mkdir -p data/raw/normal data/raw/anomaly models metrics logs

dev:  ## Full dev setup + start stack
	@bash scripts/setup_dev.sh
	$(DOCKER_COMPOSE) up --build

up:  ## Start stack in background
	@if [ ! -f .env ]; then cp .env.example .env; fi
	$(DOCKER_COMPOSE) up -d --build
	@echo ""
	@echo "Stack started:"
	@echo "  UI:        http://localhost:8501"
	@echo "  API docs:  http://localhost:8000/docs"
	@echo "  Grafana:   http://localhost:3000  (admin/vortex123)"
	@echo "  MLflow:    http://localhost:5000"
	@echo "  Kafka UI:  http://localhost:8080"
	@echo ""
	@echo "To demo without cameras: make seed-demo"

down:  ## Stop stack
	$(DOCKER_COMPOSE) down

logs:  ## Tail all service logs
	$(DOCKER_COMPOSE) logs -f

logs-api:  ## Tail API logs
	$(DOCKER_COMPOSE) logs -f api

logs-detector:  ## Tail detector logs
	$(DOCKER_COMPOSE) logs -f detector

# ── Testing ───────────────────────────────────────────────────────────────────
test:  ## Run unit tests
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-fail-under=70

test-fast:  ## Run tests without coverage
	pytest tests/ -v -x

load-test:  ## Run load tests (requires running API)
	locust -f tests/load/locustfile.py --host http://localhost:8000 --headless -u 50 -r 5 --run-time 60s

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:  ## Run linters
	ruff check .
	black --check .
	mypy api/ detection/ anomaly/ vlm/ ingestion/ config/ --ignore-missing-imports

format:  ## Auto-format code
	ruff check --fix .
	black .

security:  ## Run security scan
	bandit -r api/ detection/ anomaly/ vlm/ ingestion/ mlops/ -ll

# ── Database ──────────────────────────────────────────────────────────────────
migrate:  ## Run database migrations
	alembic upgrade head

migrate-create:  ## Create a new migration (usage: make migrate-create MSG="add column")
	alembic revision --autogenerate -m "$(MSG)"

# ── MLOps ─────────────────────────────────────────────────────────────────────
prepare-data:  ## Prepare training data
	$(PYTHON) mlops/prepare_data.py --raw-dir data/raw --out-dir data/processed

extract-features:  ## Extract detection features for transformer training
	$(PYTHON) mlops/extract_features.py --data-dir data/processed/train --out-dir data/processed/sequences/train

train-ae:  ## Train autoencoder
	$(PYTHON) mlops/train.py --data-dir data/processed/normal_frames --epochs 50

train-tf:  ## Train temporal transformer
	$(PYTHON) mlops/train_transformer.py --data-dir data/processed/sequences/train --epochs 30

train:  ## Run full DVC training pipeline
	dvc repro

evaluate:  ## Evaluate models
	$(PYTHON) mlops/evaluate.py --model-dir models --data-dir data/processed/test

export:  ## Export models to ONNX + TorchScript
	$(PYTHON) mlops/model_export.py --model-dir models --out-dir models/exported

compile-pipeline:  ## Compile Kubeflow pipeline
	$(PYTHON) mlops/kubeflow_pipeline.py

# ── Docker ────────────────────────────────────────────────────────────────────
build:  ## Build all Docker images
	docker build -f docker/api.Dockerfile -t vortexvision/api:latest .
	docker build -f docker/detector.Dockerfile -t vortexvision/detector:latest .
	docker build -f docker/frontend.Dockerfile -t vortexvision/frontend:latest .

# ── Kubernetes ────────────────────────────────────────────────────────────────
k8s-apply:  ## Apply K8s manifests (dev)
	kubectl apply -k infra/k8s/overlays/production

k8s-delete:  ## Delete K8s resources
	kubectl delete -k infra/k8s/overlays/production

k8s-status:  ## Check pod status
	kubectl get pods -n vortexvision

# ── Infrastructure ────────────────────────────────────────────────────────────
tf-init:  ## Terraform init
	cd infra/terraform && terraform init

tf-plan:  ## Terraform plan
	cd infra/terraform && terraform plan

tf-apply:  ## Terraform apply
	cd infra/terraform && terraform apply

# ── Utilities ─────────────────────────────────────────────────────────────────
clean:  ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .coverage coverage.xml htmlcov/ .mypy_cache/ .ruff_cache/
	rm -f vortexvision_pipeline.yaml test.db test_vortexvision.db

seed-demo:  ## Inject synthetic demo streams (no cameras needed)
	$(PYTHON) scripts/seed_demo.py --streams cam-01 cam-02 cam-03 --fps 10 --duration 120

init-models:  ## Initialize random model weights for dev (no training needed)
	$(PYTHON) scripts/download_models.py --init-random

download-models:  ## Pull trained model weights from DVC remote
	$(PYTHON) scripts/download_models.py --dvc-pull

setup:  ## One-command dev setup (install + init models + start stack)
	@bash scripts/setup_dev.sh
	$(PYTHON) scripts/download_models.py --init-random
	$(DOCKER_COMPOSE) up -d --build
	@echo "Seeding demo data in 10s..."
	@sleep 10
	$(PYTHON) scripts/seed_demo.py --duration 60

open-grafana:  ## Open Grafana in browser
	open http://localhost:3000

open-mlflow:  ## Open MLflow in browser
	open http://localhost:5000

open-api-docs:  ## Open API docs in browser
	open http://localhost:8000/docs

open-ui:  ## Open Streamlit UI in browser
	open http://localhost:8501
