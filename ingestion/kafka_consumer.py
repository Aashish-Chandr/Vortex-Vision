"""
Kafka consumer: reads encoded frames and yields numpy arrays for downstream processing.
"""
import argparse
import logging
from typing import Generator, Tuple

import cv2
import numpy as np
from confluent_kafka import Consumer, KafkaError

logger = logging.getLogger(__name__)


class FrameConsumer:
    """Consumes JPEG-encoded frames from Kafka and decodes them."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "video-frames",
        group_id: str = "detection-group",
    ):
        self.consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "latest",
                "enable.auto.commit": True,
            }
        )
        self.consumer.subscribe([topic])
        self._running = False

    def frames(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        """Yields (stream_id, frame) tuples."""
        self._running = True
        try:
            while self._running:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error("Kafka error: %s", msg.error())
                    continue

                stream_id = msg.key().decode("utf-8") if msg.key() else "unknown"
                buf = np.frombuffer(msg.value(), dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is not None:
                    yield stream_id, frame
        finally:
            self.consumer.close()

    def stop(self):
        self._running = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Debug Kafka frame consumer")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="video-frames")
    parser.add_argument("--group", default="debug-consumer")
    args = parser.parse_args()

    consumer = FrameConsumer(
        bootstrap_servers=args.bootstrap,
        topic=args.topic,
        group_id=args.group,
    )
    logger.info("Consuming from %s/%s ...", args.bootstrap, args.topic)
    for stream_id, frame in consumer.frames():
        logger.info("stream=%s shape=%s", stream_id, frame.shape)
