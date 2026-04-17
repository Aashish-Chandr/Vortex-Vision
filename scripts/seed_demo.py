"""
Demo seed script: generates synthetic video frames and injects them into Kafka
so you can demo VortexVision without real cameras.

Simulates:
  - Normal pedestrian traffic (cam-01)
  - A fight scene (cam-02)
  - Traffic with a speeding car (cam-03)

Usage:
  python scripts/seed_demo.py
  python scripts/seed_demo.py --streams cam-01 cam-02 --fps 10 --duration 60
"""
import argparse
import logging
import math
import random
import time
import threading

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _draw_person(frame: np.ndarray, x: int, y: int, color=(0, 200, 0), size=40):
    """Draw a simple stick figure person."""
    # Head
    cv2.circle(frame, (x, y - size), size // 4, color, -1)
    # Body
    cv2.line(frame, (x, y - size + size // 4), (x, y + size // 2), color, 3)
    # Arms
    cv2.line(frame, (x - size // 2, y), (x + size // 2, y), color, 3)
    # Legs
    cv2.line(frame, (x, y + size // 2), (x - size // 3, y + size), color, 3)
    cv2.line(frame, (x, y + size // 2), (x + size // 3, y + size), color, 3)


def _draw_car(frame: np.ndarray, x: int, y: int, color=(0, 100, 255), w=80, h=40):
    """Draw a simple car rectangle."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(frame, (x + 10, y - 20), (x + w - 10, y), color, -1)
    cv2.circle(frame, (x + 15, y + h), 12, (50, 50, 50), -1)
    cv2.circle(frame, (x + w - 15, y + h), 12, (50, 50, 50), -1)


def generate_normal_frame(frame_idx: int, width=640, height=480) -> np.ndarray:
    """Normal pedestrian scene."""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 80  # gray background
    # Ground
    cv2.rectangle(frame, (0, height - 100), (width, height), (60, 60, 60), -1)
    # Timestamp
    cv2.putText(frame, f"CAM-01 NORMAL  t={frame_idx}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # Walking people
    for i in range(3):
        x = int((frame_idx * 2 + i * 200) % width)
        y = height - 120
        _draw_person(frame, x, y, color=(0, 200, 0))
    return frame


def generate_fight_frame(frame_idx: int, width=640, height=480) -> np.ndarray:
    """Fight scene: two people overlapping with red tint."""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 80
    cv2.rectangle(frame, (0, height - 100), (width, height), (60, 60, 60), -1)
    cv2.putText(frame, f"CAM-02 ANOMALY  t={frame_idx}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # Two people very close together (fight)
    cx = width // 2
    cy = height - 130
    jitter = int(10 * math.sin(frame_idx * 0.5))
    _draw_person(frame, cx - 20 + jitter, cy, color=(0, 0, 255))
    _draw_person(frame, cx + 20 - jitter, cy, color=(0, 0, 200))
    # Red overlay to simulate anomaly
    overlay = frame.copy()
    cv2.rectangle(overlay, (cx - 80, cy - 80), (cx + 80, cy + 80), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    return frame


def generate_traffic_frame(frame_idx: int, width=640, height=480) -> np.ndarray:
    """Traffic scene with a speeding car."""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 100
    # Road
    cv2.rectangle(frame, (0, height // 2), (width, height), (70, 70, 70), -1)
    cv2.line(frame, (0, height // 2 + 50), (width, height // 2 + 50), (255, 255, 0), 2)
    cv2.putText(frame, f"CAM-03 TRAFFIC  t={frame_idx}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # Normal cars
    for i in range(2):
        x = int((frame_idx * 3 + i * 300) % (width + 100)) - 100
        _draw_car(frame, x, height // 2 + 20, color=(0, 100, 200))
    # Speeding car (faster)
    fast_x = int((frame_idx * 12) % (width + 100)) - 100
    _draw_car(frame, fast_x, height // 2 + 70, color=(0, 50, 255), w=100)
    cv2.putText(frame, "SPEED!", (max(0, fast_x), height // 2 + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame


SCENE_GENERATORS = {
    "cam-01": generate_normal_frame,
    "cam-02": generate_fight_frame,
    "cam-03": generate_traffic_frame,
}


def stream_frames(stream_id: str, kafka_bootstrap: str, fps: int, duration: int):
    """Produce synthetic frames to Kafka for a given stream."""
    try:
        from confluent_kafka import Producer
    except ImportError:
        logger.error("confluent-kafka not installed. Run: pip install confluent-kafka")
        return

    producer = Producer({"bootstrap.servers": kafka_bootstrap})
    generator = SCENE_GENERATORS.get(stream_id, generate_normal_frame)

    frame_interval = 1.0 / fps
    total_frames = fps * duration
    logger.info("Streaming %s: %d frames at %d fps for %ds", stream_id, total_frames, fps, duration)

    for frame_idx in range(total_frames):
        t0 = time.monotonic()
        frame = generator(frame_idx)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        producer.produce("video-frames", key=stream_id.encode(), value=buf.tobytes())
        producer.poll(0)

        elapsed = time.monotonic() - t0
        sleep = frame_interval - elapsed
        if sleep > 0:
            time.sleep(sleep)

    producer.flush()
    logger.info("Stream %s complete", stream_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed VortexVision with demo video streams")
    parser.add_argument("--streams", nargs="+", default=["cam-01", "cam-02", "cam-03"])
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds")
    parser.add_argument("--kafka", default="localhost:9092")
    args = parser.parse_args()

    logger.info("Starting demo streams: %s", args.streams)
    logger.info("Kafka: %s | FPS: %d | Duration: %ds", args.kafka, args.fps, args.duration)
    logger.info("Open http://localhost:8501 to view the live feed")

    threads = []
    for stream_id in args.streams:
        t = threading.Thread(
            target=stream_frames,
            args=(stream_id, args.kafka, args.fps, args.duration),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    logger.info("Demo complete.")
