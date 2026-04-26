"""
Background Kafka consumer that reads annotated-frames topic and stores
the latest JPEG bytes per stream_id for WebSocket delivery.
"""
import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.state import AppState

logger = logging.getLogger(__name__)


async def consume_frames(state: "AppState") -> None:
    """Polls Kafka for annotated JPEG frames, stores latest per stream."""
    from config.settings import get_settings

    settings = get_settings()

    try:
        from confluent_kafka import Consumer, KafkaError
    except ImportError:
        logger.warning("confluent-kafka not installed — frame consumer disabled")
        return

    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap,
            "group.id": "api-frame-consumer",
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
            "fetch.max.bytes": 10_485_760,
        }
    )
    consumer.subscribe([settings.kafka_annotated_topic])
    logger.info("Frame consumer subscribed to '%s'", settings.kafka_annotated_topic)

    loop = asyncio.get_running_loop()

    def _poll():
        return consumer.poll(timeout=0.1)

    try:
        while True:
            msg = await loop.run_in_executor(None, _poll)

            if msg is None:
                await asyncio.sleep(0.001)
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error("Kafka frame error: %s", msg.error())
                continue

            stream_id = msg.key().decode("utf-8") if msg.key() else "unknown"
            state.push_frame(stream_id, msg.value())

    except asyncio.CancelledError:
        logger.info("Frame consumer cancelled")
    finally:
        consumer.close()
