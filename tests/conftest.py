"""
Shared pytest fixtures for VortexVision tests.
"""
import os
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio

# ── Env vars must be set before ANY app module is imported ────────────────────
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_vortexvision.db")
os.environ.setdefault("API_SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("KAFKA_BOOTSTRAP", "localhost:9092")
os.environ.setdefault("YOLO_DEVICE", "cpu")
os.environ.setdefault("VLM_MODE", "api")
os.environ.setdefault("MODEL_STORAGE_PATH", "/tmp/vortexvision_test_models")
os.environ.setdefault("CLIP_STORAGE_PATH", "/tmp/vortexvision_test_clips")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("TESTING", "1")


@pytest.fixture
def dummy_frame_bgr() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_app_state() -> MagicMock:
    state = MagicMock()
    state.detector = None
    state.nl_engine = None
    state.anomaly_scorer = None
    state.behavioral_detector = None
    state.clip_buffer = None
    state.load_errors = {}
    state.get_events.return_value = []
    state.pop_alert = AsyncMock(return_value=None)
    state.get_latest_frame.return_value = None
    return state


@pytest_asyncio.fixture
async def api_client(mock_app_state: MagicMock):
    """
    Async HTTP test client backed by a fresh FastAPI app.
    Creates a minimal app that shares routers with the main app
    but uses a no-op lifespan — no Kafka, no background tasks.
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from httpx import ASGITransport, AsyncClient
    from prometheus_client import make_asgi_app

    from config.settings import get_settings

    get_settings.cache_clear()

    # Reset DB engine so test DATABASE_URL is used
    import api.database as _db

    _db._engine = None
    _db._session_factory = None

    from api.database import Base, get_engine
    from api.middleware import (
        ErrorHandlingMiddleware,
        RateLimitMiddleware,
        RequestLoggingMiddleware,
    )
    from api.routers import auth as auth_router
    from api.routers import events, health, query, streams

    settings = get_settings()

    # Set up DB tables
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # Build a test app with a no-op lifespan
    @asynccontextmanager
    async def test_lifespan(app):
        app.state.vv = mock_app_state
        yield

    test_app = FastAPI(lifespan=test_lifespan)
    test_app.add_middleware(ErrorHandlingMiddleware)
    test_app.add_middleware(RateLimitMiddleware, requests_per_minute=1000)
    test_app.add_middleware(RequestLoggingMiddleware)
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    test_app.mount("/metrics", make_asgi_app())
    test_app.include_router(health.router, prefix="/health", tags=["health"])
    test_app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
    test_app.include_router(streams.router, prefix="/streams", tags=["streams"])
    test_app.include_router(events.router, prefix="/events", tags=["events"])
    test_app.include_router(query.router, prefix="/query", tags=["query"])

    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://test",
    ) as client:
        yield client

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    _db._engine = None
    _db._session_factory = None
