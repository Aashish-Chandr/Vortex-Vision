"""
Health check endpoints: liveness, readiness, and deep component health.
"""
import os
import time

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/")
async def liveness():
    """Kubernetes liveness probe — always returns 200 if process is alive."""
    return {"status": "ok", "timestamp": time.time()}


@router.get("/ready")
async def readiness(request: Request):
    """Kubernetes readiness probe — checks if app state is initialized."""
    state = getattr(request.app.state, "vv", None)
    if state is None:
        return {"status": "not_ready", "reason": "app state not initialized"}
    return {"status": "ready", "timestamp": time.time()}


@router.get("/deep")
async def deep_health(request: Request):
    """Deep health check: reports status of every component."""
    state = getattr(request.app.state, "vv", None)
    if state is None:
        return {"status": "degraded", "components": {}}

    db_ok = False
    try:
        from api.database import get_session_factory
        from sqlalchemy import text

        async with get_session_factory()() as session:
            await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    tracing_ok = False
    try:
        from opentelemetry import trace

        tracing_ok = trace.get_tracer_provider().__class__.__name__ != "ProxyTracerProvider"
    except Exception:
        pass

    components = {
        "detector": state.detector is not None,
        "anomaly_scorer": state.anomaly_scorer is not None,
        "behavioral_detector": state.behavioral_detector is not None,
        "vlm_engine": state.nl_engine is not None,
        "clip_buffer": state.clip_buffer is not None,
        "database": db_ok,
        "tracing": tracing_ok,
    }
    load_errors = getattr(state, "load_errors", {})
    return {
        "status": "healthy" if components["database"] else "degraded",
        "components": components,
        "load_errors": load_errors,
        "timestamp": time.time(),
        "version": os.getenv("APP_VERSION", "1.0.0"),
    }
