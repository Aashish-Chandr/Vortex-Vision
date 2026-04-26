"""
Structured JSON logging configuration for production.
"""
import logging
import sys


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure structured logging. Uses JSON in production, plain text in dev."""
    handlers = [logging.StreamHandler(sys.stdout)]

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
            pass

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    for noisy in ["uvicorn.access", "kafka", "confluent_kafka"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
