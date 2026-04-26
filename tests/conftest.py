"""
Shared pytest fixtures for VortexVision tests.
"""
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio

# ── Set test env vars BEFORE any app imports so lru_cache picks them up ───────
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_vortexvision.db"
os.environ["API_SECRET_KEY"] = "test-secret-key-for-pytest-only"
os.environ["KAFKA_BOOTSTRAP"] = "localhost:9092"
os.environ["YOLO_DEVICE"] = "cpu"
os.environ["VLM_MODE"] = "api"
os.environ["MODEL_STORAGE_PATH"] = "/tmp/vortexvision_test_models"
os.environ["CLIP_STORAGE_PATH"] = "/tmp/vortexvision_test_clips"
os.environ["REDIS_URL"] = "redis://localhost:6379"


@pytest.fixture
def dummy_frame_bgr() -> np.ndarray:
    """640x640 BGR frame with random content."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_frame_rgb(dummy_frame_bgr: np.ndarray) -> np.ndarray:
    import cv2  # noqa: PLC0415

    return cv2.cvtColor(dummy_frame_bgr, cv2.COLOR_BGR2RGB)


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
    """Async test client with mocked app state and in-memory SQLite DB."""
    from config.settings import get_settings

    get_settings.cache_clear()

    from api.database import Base, get_engine, init_db
    from api.main import app

    engine = get_engine()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    app.state.vv = mock_app_state

    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
