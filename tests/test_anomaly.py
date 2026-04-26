"""Unit tests for anomaly detection models — no real torch required."""
import numpy as np
import pytest

from anomaly.event_types import AnomalyEvent, AnomalyType
from anomaly.transformer_detector import detections_to_feature
from detection.detector import Detection, DetectionResult


# ── Feature extraction (pure numpy, no torch) ─────────────────────────────────

def test_detections_to_feature_empty():
    result = DetectionResult(stream_id="test", timestamp=0.0, frame_id=0)
    feat = detections_to_feature(result)
    assert feat.shape == (128,)
    assert feat.sum() == 0.0


def test_detections_to_feature_with_detections():
    result = DetectionResult(
        stream_id="test",
        timestamp=0.0,
        frame_id=0,
        detections=[
            Detection(
                track_id=1,
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=[10.0, 10.0, 100.0, 200.0],
            ),
            Detection(
                track_id=2,
                class_id=2,
                class_name="car",
                confidence=0.75,
                bbox=[200.0, 100.0, 400.0, 300.0],
            ),
        ],
    )
    feat = detections_to_feature(result)
    assert feat.shape == (128,)
    assert feat[0] > 0, "Count feature should be non-zero"
    assert feat[1] > 0, "Confidence feature should be non-zero"


def test_detections_to_feature_normalized():
    result = DetectionResult(
        stream_id="test",
        timestamp=0.0,
        frame_id=0,
        detections=[
            Detection(
                track_id=i,
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=[0.0, 0.0, 640.0, 640.0],
            )
            for i in range(50)
        ],
    )
    feat = detections_to_feature(result)
    assert feat[0] <= 1.0, "Count should be normalized"


def test_detections_to_feature_bbox_values():
    result = DetectionResult(
        stream_id="test",
        timestamp=0.0,
        frame_id=0,
        detections=[
            Detection(
                track_id=0,
                class_id=0,
                class_name="person",
                confidence=0.8,
                bbox=[64.0, 128.0, 320.0, 480.0],
            )
        ],
    )
    feat = detections_to_feature(result)
    # bbox values should be normalized by 640
    assert abs(feat[2] - 64.0 / 640) < 1e-5
    assert abs(feat[3] - 128.0 / 640) < 1e-5


# ── Event types ───────────────────────────────────────────────────────────────

def test_anomaly_event_creation():
    event = AnomalyEvent(
        stream_id="cam-01",
        timestamp=1234567890.0,
        frame_id=100,
        anomaly_type=AnomalyType.FIGHT,
        confidence=0.85,
        autoencoder_score=0.12,
        transformer_score=0.91,
    )
    assert event.anomaly_type == AnomalyType.FIGHT
    assert event.confidence == 0.85
    assert event.stream_id == "cam-01"


def test_anomaly_type_string_values():
    assert AnomalyType.FIGHT == "fight"
    assert AnomalyType.CROWD_RUSH == "crowd_rush"
    assert AnomalyType.WEAPON == "weapon"
    assert AnomalyType.ACCIDENT == "accident"
    assert AnomalyType.TRESPASSING == "trespassing"
    assert AnomalyType.UNKNOWN == "unknown"


def test_anomaly_event_defaults():
    event = AnomalyEvent(
        stream_id="cam-02",
        timestamp=0.0,
        frame_id=0,
        anomaly_type=AnomalyType.UNKNOWN,
        confidence=0.0,
        autoencoder_score=0.0,
        transformer_score=0.0,
    )
    assert event.clip_path is None
    assert event.description is None
