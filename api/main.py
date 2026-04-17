"""
VortexVision FastAPI backend.
REST + WebSocket endpoints for real-time video analytics.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from api.database import init_db
from api.middleware import ErrorHandlingMiddleware, RateLimitMiddleware, RequestLoggingMiddleware
from api.routers import events, health, query, streams, auth as auth_router
from api.state import AppState
from api.tracing import setup_tracing
from config.logging_config import configure_logging
from config.settings import get_settings

settings = get_settings()
configure_logging(level=settings.log_level)
setup_tracing(service_name="vortexvision-api")
logger = logging.getLogger(__name__)


def _validate_env():
    """Fail fast if critical environment variables are missing or obviously wrong."""
    issues = []
    if settings.api_secret_key in ("change-me-in-production", ""):
        issues.append("API_SECRET_KEY is not set to a secure value")
    if not settings.kafka_bootstrap:
        issues.append("KAFKA_BOOTSTRAP is not set")
    if not settings.database_url:
        issues.append("DATABASE_URL is not set")
    if issues:
        for issue in issues:
            logger.warning("Config warning: %s", issue)


_validate_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VortexVision %s starting up...", settings.app_version)
    await init_db()
    app.state.vv = AppState()
    await app.state.vv.startup()

    # Start background Kafka → DB event consumer
    from api.event_consumer import consume_events
    from api.frame_consumer import consume_frames
    from monitoring.evidently.drift_worker import run_drift_monitor

    consumer_task = asyncio.create_task(consume_events(app.state.vv))
    frame_task = asyncio.create_task(consume_frames(app.state.vv))
    drift_task = asyncio.create_task(run_drift_monitor(app.state.vv))

    yield

    logger.info("VortexVision shutting down...")
    consumer_task.cancel()
    frame_task.cancel()
    drift_task.cancel()
    try:
        await asyncio.gather(consumer_task, frame_task, drift_task, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    await app.state.vv.shutdown()


app = FastAPI(
    title="VortexVision",
    description="Real-time multimodal video analytics platform",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware (order matters: outermost first) ───────────────────────────────
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://frontend:8501"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ── Prometheus metrics endpoint ───────────────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(streams.router, prefix="/streams", tags=["streams"])
app.include_router(events.router, prefix="/events", tags=["events"])
app.include_router(query.router, prefix="/query", tags=["query"])


# ── WebSocket endpoints ───────────────────────────────────────────────────────
@app.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket):
    """Push real-time anomaly alerts to connected clients."""
    await websocket.accept()
    state: AppState = websocket.app.state.vv
    try:
        while True:
            alert = await state.pop_alert()
            if alert:
                await websocket.send_json(alert)
            else:
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logger.info("Alert WebSocket client disconnected")


@app.websocket("/ws/stream/{stream_id}")
async def stream_websocket(websocket: WebSocket, stream_id: str):
    """Push annotated JPEG frames for a specific stream at ~30fps."""
    await websocket.accept()
    state: AppState = websocket.app.state.vv
    try:
        while True:
            frame_bytes = state.get_latest_frame(stream_id)
            if frame_bytes:
                await websocket.send_bytes(frame_bytes)
            await asyncio.sleep(1 / 30)
    except WebSocketDisconnect:
        logger.info("Stream WebSocket disconnected: %s", stream_id)
