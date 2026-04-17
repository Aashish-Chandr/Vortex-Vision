"""
Structured JSON logging configuration for production.
"""
import logging
import sys
from typing import Any


def configure_logging(level: str = "INFO", json_logs: bool = True):
    """Configure structured logging. Uses JSON in production, plain text in dev."""
    handlers: list[Any] = [logging.StreamHandler(sys.stdout)]

    if json_logs:
        try:
            from pythonjsonlogger import jsonlogger
            handler = logging.StreamHandler(sys.stdout)
            formatter = jsonlogger.JsonFormatter(
                fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
            handler.setFormatter(formatter)
            handlers = [handler]
        except ImportError:
            pass  # fall back to plain text

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    # Silence noisy third-party loggers
    for noisy in ["uvicorn.access", "kafka", "confluent_kafka"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
