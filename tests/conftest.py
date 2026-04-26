"""
Shared pytest fixtures for VortexVision tests.
"""
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio

# Set env vars before ANY app imports — lru_cache must pick these up
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_vortexvision.db"
os.environ["API_SECRET_KEY"] = "test-secret-key-for-pytest-only"
os.environ["KAFKA_BOOTSTRAP"] = "localhost:9092"
os.environ["YOLO_DEVICE"] = "cpu"
os.environ["VLM_MODE"] = "api"
os.environ["MODEL_STORAGE_PATH"] = "/tmp/vortexvision_test_models"
os.environ["CLIP_STORAGE_PATH"] = "/tmp/vortexvision_test_clips"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["TESTING"] = "1"


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
    Async test client.
    - Resets DB engine so test DATABASE_URL is used.
    - Injects mock state before lifespan runs.
    - TESTING=1 env var tells lifespan to skip background tasks.
    """
    from config.settings import get_settings

    get_settings.cache_clear()

    # Reset lazy DB singletons so they pick up the test DATABASE_URL
    import api.database as _db

    _db._engine = None
    _db._session_factory = None

    from api.database import Base, get_engine

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # Patch the app's lifespan to use a no-op that just injects state
    from api.main import app

    @asynccontextmanager
    async def _test_lifespan(_app):
        _app.state.vv = mock_app_state
        yield

    original_router = app.router.lifespan_context
    app.router.lifespan_context = _test_lifespan

    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client

    # Restore original lifespan
    app.router.lifespan_context = original_router

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    _db._engine = None
    _db._session_factory = None
