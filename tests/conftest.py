"""
Shared pytest fixtures for VortexVision tests.
"""
import asyncio
import os
import numpy as np
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

# Set test env vars BEFORE any app imports so lru_cache picks them up
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_vortexvision.db")
os.environ.setdefault("API_SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("KAFKA_BOOTSTRAP", "localhost:9092")
os.environ.setdefault("YOLO_DEVICE", "cpu")
os.environ.setdefault("VLM_MODE", "api")
os.environ.setdefault("MODEL_STORAGE_PATH", "/tmp/vortexvision_test_models")
os.environ.setdefault("CLIP_STORAGE_PATH", "/tmp/vortexvision_test_clips")


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def dummy_frame_bgr():
    """640x640 BGR frame with random content."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_frame_rgb(dummy_frame_bgr):
    import cv2
    return cv2.cvtColor(dummy_frame_bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture
def mock_app_state():
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
async def api_client(mock_app_state):
    """Async test client with mocked app state and in-memory SQLite DB."""
    # Clear lru_cache so test env vars take effect
    from config.settings import get_settings
    get_settings.cache_clear()

    from api.database import init_db, get_engine, Base
    from api.main import app

    # Re-create tables for each test session
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    app.state.vv = mock_app_state

    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
