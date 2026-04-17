FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY api/           ./api/
COPY anomaly/       ./anomaly/
COPY detection/     ./detection/
COPY vlm/           ./vlm/
COPY ingestion/     ./ingestion/
COPY config/        ./config/
COPY monitoring/    ./monitoring/
COPY migrations/    ./migrations/
COPY alembic.ini    .

# mlops subset needed at runtime (clip_saver + __init__)
COPY mlops/__init__.py      ./mlops/__init__.py
COPY mlops/clip_saver.py    ./mlops/clip_saver.py

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Run DB migrations then start the server
CMD ["sh", "-c", "alembic upgrade head && uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4"]
