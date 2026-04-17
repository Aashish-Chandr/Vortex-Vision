"""
Background Kafka consumer that reads anomaly-events topic and persists
them to the database + pushes to the in-memory alert queue.
Runs as an asyncio task inside the FastAPI process.
"""
import asyncio
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.state import AppState

logger = logging.getLogger(__name__)


async def consume_events(state: "AppState"):
    """
    Long-running async task: polls Kafka for anomaly events,
    writes to DB, and pushes to the WebSocket alert queue.
    """
    from config.settings import get_settings
    from api.database import AnomalyEventRecord, get_session_factory

    settings = get_settings()

    try:
        from confluent_kafka import Consumer, KafkaError
    except ImportError:
        logger.warning("confluent-kafka not installed — event consumer disabled")
        return

    consumer = Consumer({
        "bootstrap.servers": settings.kafka_bootstrap,
        "group.id": "api-event-consumer",
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([settings.kafka_events_topic])
    logger.info("Event consumer subscribed to '%s'", settings.kafka_events_topic)

    loop = asyncio.get_event_loop()

    try:
        while True:
            # Poll in executor to avoid blocking the event loop
            msg = await loop.run_in_executor(None, lambda: consumer.poll(timeout=0.5))

            if msg is None:
                await asyncio.sleep(0.01)
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error("Kafka error: %s", msg.error())
                continue

            try:
                payload = json.loads(msg.value().decode("utf-8"))
            except Exception as e:
                logger.error("Failed to decode event payload: %s", e)
                continue

            # Persist to DB
            try:
                async with get_session_factory()() as session:
                    record = AnomalyEventRecord(
                        stream_id=payload.get("stream_id", ""),
                        timestamp=payload.get("timestamp", 0.0),
                        frame_id=payload.get("frame_id"),
                        anomaly_type=payload.get("anomaly_type"),
                        confidence=payload.get("confidence"),
                        autoencoder_score=payload.get("autoencoder_score"),
                        transformer_score=payload.get("transformer_score"),
                        clip_path=payload.get("clip_path"),
                        description=payload.get("description"),
                    )
                    session.add(record)
                    await session.commit()
            except Exception as e:
                logger.error("Failed to persist event to DB: %s", e)

            # Push to in-memory alert queue for WebSocket broadcast
            state.push_event(payload)

    except asyncio.CancelledError:
        logger.info("Event consumer cancelled")
    finally:
        consumer.close()
        logger.info("Event consumer closed")
