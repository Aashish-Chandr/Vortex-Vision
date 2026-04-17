"""
FastAPI middleware: request logging, rate limiting, error handling, metrics.
"""
import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter(
    "vortexvision_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "vortexvision_api_request_duration_seconds",
    "API request duration",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        t0 = time.monotonic()

        response = await call_next(request)

        duration_ms = (time.monotonic() - t0) * 1000
        endpoint = request.url.path

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration_ms / 1000)

        logger.info(
            "%s %s %d %.1fms [%s]",
            request.method, endpoint, response.status_code, duration_ms, request_id,
        )
        response.headers["X-Request-ID"] = request_id
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "type": type(exc).__name__},
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter.
    Uses Redis when available (multi-instance safe), falls back to in-process dict.
    """

    def __init__(self, app, requests_per_minute: int = 120):
        super().__init__(app)
        self.rpm = requests_per_minute
        self._redis = None
        self._counts: dict[str, list[float]] = {}
        self._try_connect_redis()

    def _try_connect_redis(self):
        try:
            import redis as redis_lib
            from config.settings import get_settings
            settings = get_settings()
            self._redis = redis_lib.from_url(settings.redis_url, decode_responses=True,
                                             socket_connect_timeout=1)
            self._redis.ping()
            logger.info("Rate limiter using Redis backend")
        except Exception:
            logger.info("Rate limiter using in-process backend (Redis unavailable)")
            self._redis = None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path.startswith("/health") or request.url.path == "/metrics":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"

        if self._redis:
            allowed = await self._check_redis(client_ip)
        else:
            allowed = self._check_local(client_ip)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": "60"},
            )
        return await call_next(request)

    async def _check_redis(self, client_ip: str) -> bool:
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            key = f"rl:{client_ip}"
            pipe = self._redis.pipeline()
            now = time.time()
            window_start = now - 60
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, 60)
            results = await loop.run_in_executor(None, pipe.execute)
            count = results[2]
            return count <= self.rpm
        except Exception:
            return True  # fail open

    def _check_local(self, client_ip: str) -> bool:
        now = time.monotonic()
        timestamps = self._counts.get(client_ip, [])
        timestamps = [t for t in timestamps if now - t < 60.0]
        if len(timestamps) >= self.rpm:
            return False
        timestamps.append(now)
        self._counts[client_ip] = timestamps
        return True
