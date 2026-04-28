"""
Detection worker: consumes frames from Kafka, runs YOLO26 + anomaly detection,
saves clips, persists events to DB, and pushes metrics to Prometheus.
Handles SIGTERM gracefully.
"""
import json
import logging
import signal
import time
import uuid

import cv2
import numpy as np
from confluent_kafka import Producer
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from anomaly.autoencoder import AnomalyScorer, ConvAutoencoder
from anomaly.classifier import AnomalyClassifier, ClassificationContext
from anomaly.event_types import AnomalyEvent, AnomalyType
from anomaly.transformer_detector import BehavioralAnomalyDetector, TemporalTransformer
from config.logging_config import configure_logging
from config.settings import get_settings
from detection.annotator import annotate_frame
from detection.detector import YOLODetector
from ingestion.kafka_consumer import FrameConsumer
from mlops.clip_saver import ClipBuffer

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

INFERENCE_LATENCY = Histogram(
    "vortexvision_inference_duration_seconds",
    "YOLO inference duration",
    ["stream_id"],
    buckets=[0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5],
)
FRAMES_PROCESSED = Counter(
    "vortexvision_frames_processed_total", "Frames processed", ["stream_id"]
)
ANOMALY_EVENTS = Counter(
    "vortexvision_anomaly_events_total", "Anomaly events", ["stream_id", "anomaly_type"]
)
STREAM_ACTIVE = Gauge("vortexvision_stream_active", "Active streams", ["stream_id"])
KAFKA_LAG = Gauge("vortexvision_kafka_consumer_lag", "Estimated Kafka consumer lag")

_running = True


def _handle_sigterm(signum, frame):
    global _running
    logger.info("SIGTERM received — shutting down gracefully...")
    _running = False


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


def _load_models():
    import torch

    detector = YOLODetector(
        model_path=settings.yolo_model,
        device=settings.yolo_device,
        conf_threshold=settings.yolo_conf,
        iou_threshold=settings.yolo_iou,
    )

    ae_model = ConvAutoencoder()
    try:
        ae_model.load_state_dict(
            torch.load(
                f"{settings.model_storage_path}/autoencoder_best.pt",
                map_location=settings.yolo_device,
            )
        )
        logger.info("Autoencoder weights loaded")
    except FileNotFoundError:
        logger.warning("Autoencoder weights not found — using random init (dev mode)")

    anomaly_scorer = AnomalyScorer(
        ae_model, threshold=settings.ae_threshold, device=settings.yolo_device
    )

    tf_model = TemporalTransformer(seq_len=settings.seq_len)
    try:
        tf_model.load_state_dict(
            torch.load(
                f"{settings.model_storage_path}/transformer_best.pt",
                map_location=settings.yolo_device,
            )
        )
        logger.info("Transformer weights loaded")
    except FileNotFoundError:
        logger.warning("Transformer weights not found — using random init (dev mode)")

    behavioral_detector = BehavioralAnomalyDetector(
        tf_model,
        seq_len=settings.seq_len,
        threshold=settings.transformer_threshold,
        device=settings.yolo_device,
    )

    return detector, anomaly_scorer, behavioral_detector


def main():
    start_http_server(8001)
    logger.info("Metrics server on :8001")

    detector, anomaly_scorer, behavioral_detector = _load_models()
    classifier = AnomalyClassifier()

    clip_buffer = ClipBuffer(
        storage_path=settings.clip_storage_path,
        s3_bucket=settings.s3_bucket,
    )

    producer = Producer(
        {
            "bootstrap.servers": settings.kafka_bootstrap,
            "queue.buffering.max.ms": 10,
            "compression.type": "lz4",
        }
    )

    consumer = FrameConsumer(
        bootstrap_servers=settings.kafka_bootstrap,
        topic=settings.kafka_frames_topic,
        group_id=settings.kafka_consumer_group,
    )

    frame_counters: dict[str, int] = {}
    logger.info("Detection worker running. Consuming from '%s'...", settings.kafka_frames_topic)

    for stream_id, frame in consumer.frames():
        if not _running:
            break

        STREAM_ACTIVE.labels(stream_id=stream_id).set(1)
        frame_id = frame_counters.get(stream_id, 0)
        frame_counters[stream_id] = frame_id + 1

        try:
            with INFERENCE_LATENCY.labels(stream_id=stream_id).time():
                result = detector.infer(frame, stream_id=stream_id, frame_id=frame_id)
        except Exception as exc:
            logger.error("Inference error on stream %s frame %d: %s", stream_id, frame_id, exc)
            continue

        FRAMES_PROCESSED.labels(stream_id=stream_id).inc()

        try:
            ae_score, ae_anomaly = anomaly_scorer.score(frame)
            tf_score, tf_anomaly = behavioral_detector.update(result)
        except Exception as exc:
            logger.error("Anomaly scoring error: %s", exc)
            ae_score, ae_anomaly, tf_score, tf_anomaly = 0.0, False, 0.0, False

        clip_buffer.push_frame(stream_id, frame)

        if ae_anomaly or tf_anomaly:
            event_id = str(uuid.uuid4())
            ctx = ClassificationContext(result=result, ae_score=ae_score, tf_score=tf_score)
            anomaly_type = classifier.classify(ctx)
            clip_path = clip_buffer.trigger(stream_id, event_id, anomaly_type.value)

            event = AnomalyEvent(
                stream_id=stream_id,
                timestamp=result.timestamp,
                frame_id=frame_id,
                anomaly_type=anomaly_type,
                confidence=max(ae_score, tf_score),
                autoencoder_score=ae_score,
                transformer_score=tf_score,
                clip_path=clip_path,
            )

            ANOMALY_EVENTS.labels(
                stream_id=stream_id, anomaly_type=event.anomaly_type.value
            ).inc()

            producer.produce(
                settings.kafka_events_topic,
                key=stream_id,
                value=json.dumps(
                    {
                        "event_id": event_id,
                        "stream_id": event.stream_id,
                        "timestamp": event.timestamp,
                        "frame_id": event.frame_id,
                        "anomaly_type": event.anomaly_type.value,
                        "confidence": round(event.confidence, 4),
                        "autoencoder_score": round(event.autoencoder_score, 4),
                        "transformer_score": round(event.transformer_score, 4),
                        "clip_path": event.clip_path,
                    }
                ),
            )
            logger.info(
                "Anomaly detected: %s on %s (ae=%.3f tf=%.3f)",
                anomaly_type.value,
                stream_id,
                ae_score,
                tf_score,
            )

        annotated = annotate_frame(frame, result.detections)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        producer.produce(settings.kafka_annotated_topic, key=stream_id, value=buf.tobytes())
        producer.poll(0)

    consumer.stop()
    producer.flush(timeout=10)
    logger.info("Detection worker stopped cleanly.")


if __name__ == "__main__":
    main()
