"""
Stream reader: pulls frames from RTSP/YouTube/file sources and publishes to Kafka.
Supports RTSP, HTTP streams, local files, and YouTube URLs (via yt-dlp).
"""
import cv2
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from confluent_kafka import Producer

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    source: str                        # RTSP URL, YouTube URL, or file path
    stream_id: str                     # Unique identifier for this stream
    kafka_topic: str = "video-frames"
    fps_limit: int = 30
    resize: tuple = (640, 640)


def _resolve_source(source: str) -> str:
    """Resolve YouTube URLs to direct stream URLs via yt-dlp if available."""
    if "youtube.com" in source or "youtu.be" in source:
        try:
            import yt_dlp
            ydl_opts = {"format": "best[ext=mp4]/best", "quiet": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
                url = info.get("url") or info.get("formats", [{}])[-1].get("url", source)
                logger.info("Resolved YouTube URL for stream")
                return url
        except ImportError:
            logger.warning("yt-dlp not installed — using URL directly (may fail for YouTube)")
        except Exception as e:
            logger.warning("yt-dlp resolution failed: %s — using URL directly", e)
    return source


class StreamReader:
    """Reads video frames and publishes JPEG-encoded frames to Kafka."""

    def __init__(self, config: StreamConfig, kafka_bootstrap: str = "localhost:9092"):
        self.config = config
        self.producer = Producer({
            "bootstrap.servers": kafka_bootstrap,
            "queue.buffering.max.ms": 5,
            "compression.type": "lz4",
        })
        self._running = False
        self._frames_sent = 0

    def _delivery_report(self, err, msg):
        if err:
            logger.error("Frame delivery failed for stream %s: %s", self.config.stream_id, err)

    def start(self):
        self._running = True
        resolved_source = _resolve_source(self.config.source)
        cap = cv2.VideoCapture(resolved_source)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {self.config.source}")

        # Try to set buffer size to reduce latency on RTSP
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_interval = 1.0 / self.config.fps_limit
        logger.info("Stream '%s' started → %s", self.config.stream_id, self.config.source)

        try:
            while self._running:
                t0 = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Stream '%s' ended or frame dropped", self.config.stream_id)
                    break

                frame = cv2.resize(frame, self.config.resize)
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                self.producer.produce(
                    self.config.kafka_topic,
                    key=self.config.stream_id.encode("utf-8"),
                    value=buf.tobytes(),
                    callback=self._delivery_report,
                )
                self.producer.poll(0)
                self._frames_sent += 1

                elapsed = time.monotonic() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            cap.release()
            self.producer.flush(timeout=5)
            logger.info("Stream '%s' stopped. Frames sent: %d",
                        self.config.stream_id, self._frames_sent)

    def stop(self):
        self._running = False


if __name__ == "__main__":
    import argparse
    from config.settings import get_settings

    parser = argparse.ArgumentParser(description="Start a single video stream reader")
    parser.add_argument("--source", required=True, help="RTSP URL, YouTube URL, or file path")
    parser.add_argument("--stream-id", required=True, help="Unique stream identifier")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=logging.INFO)

    config = StreamConfig(
        source=args.source,
        stream_id=args.stream_id,
        fps_limit=args.fps,
        kafka_topic=settings.kafka_frames_topic,
    )
    reader = StreamReader(config, kafka_bootstrap=settings.kafka_bootstrap)
    reader.start()
