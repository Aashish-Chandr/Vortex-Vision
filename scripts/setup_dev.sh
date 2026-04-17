#!/usr/bin/env bash
# VortexVision development environment setup script.
# Run once after cloning: bash scripts/setup_dev.sh

set -euo pipefail

echo "=== VortexVision Dev Setup ==="

# 1. Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $PYTHON_VERSION"

# 2. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 3. Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements-dev.txt

# 4. Set up .env
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "  ⚠️  Edit .env with your actual values before running"
fi

# 5. Create required directories
echo "Creating required directories..."
mkdir -p data/raw/normal data/raw/anomaly
mkdir -p models/exported
mkdir -p metrics
mkdir -p logs

# 6. Initialize DVC
if [ ! -f ".dvc/.gitignore" ]; then
    echo "Initializing DVC..."
    dvc init --no-scm 2>/dev/null || true
fi

# 7. Run database migrations (if DB is available)
echo "Running database migrations..."
alembic upgrade head 2>/dev/null || echo "  ⚠️  DB not available — run 'make migrate' after starting docker-compose"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Start the stack:     make up"
echo "  2. Add demo data:       python scripts/seed_demo.py"
echo "  3. Open the UI:         http://localhost:8501"
echo "  4. Open API docs:       http://localhost:8000/docs"
echo "  5. Open Grafana:        http://localhost:3000  (admin/vortex123)"
echo "  6. Open MLflow:         http://localhost:5000"
