"""Unit tests for detection pipeline."""
import numpy as np
import pytest
from unittest.mock import patch

from detection.detector import Detection, DetectionResult


@pytest.fixture
def sample_detections():
    return [
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
            confidence=0.85,
            bbox=[200.0, 150.0, 400.0, 300.0],
        ),
        Detection(
            track_id=None,
            class_id=5,
            class_name="bus",
            confidence=0.7,
            bbox=[0.0, 0.0, 50.0, 50.0],
        ),
    ]


def test_detection_result_defaults():
    result = DetectionResult(stream_id="cam-01", timestamp=1234567890.0, frame_id=42)
    assert result.stream_id == "cam-01"
    assert result.frame_id == 42
    assert result.detections == []
    assert result.inference_ms == 0.0


def test_detection_dataclass_fields(sample_detections):
    det = sample_detections[0]
    assert det.track_id == 1
    assert det.class_name == "person"
    assert det.confidence == 0.9
    assert len(det.bbox) == 4


def test_detection_no_track_id(sample_detections):
    det = sample_detections[2]
    assert det.track_id is None
    assert det.class_name == "bus"


def test_detection_result_with_detections(sample_detections):
    result = DetectionResult(
        stream_id="cam-01",
        timestamp=0.0,
        frame_id=0,
        detections=sample_detections,
        inference_ms=12.5,
    )
    assert len(result.detections) == 3
    assert result.inference_ms == 12.5


@patch("detection.detector.YOLO")
def test_yolo_detector_init(mock_yolo):
    from detection.detector import YOLODetector

    detector = YOLODetector(model_path="yolo26n.pt", device="cpu")
    mock_yolo.assert_called_once_with("yolo26n.pt")
    assert detector.device == "cpu"
    assert detector.conf == 0.4
    assert detector.iou == 0.5


@patch("detection.detector.YOLO")
def test_yolo_detector_custom_thresholds(mock_yolo):
    from detection.detector import YOLODetector

    detector = YOLODetector(model_path="yolo26s.pt", device="cpu", conf_threshold=0.6, iou_threshold=0.3)
    assert detector.conf == 0.6
    assert detector.iou == 0.3
