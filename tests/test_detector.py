"""Unit tests for detection pipeline."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from detection.detector import DetectionResult, Detection
from detection.annotator import annotate_frame


@pytest.fixture
def sample_detections():
    return [
        Detection(track_id=1, class_id=0, class_name="person", confidence=0.9,
                  bbox=[10.0, 10.0, 100.0, 200.0]),
        Detection(track_id=2, class_id=2, class_name="car", confidence=0.85,
                  bbox=[200.0, 150.0, 400.0, 300.0]),
        Detection(track_id=None, class_id=5, class_name="bus", confidence=0.7,
                  bbox=[0.0, 0.0, 50.0, 50.0]),
    ]


def test_annotate_frame_preserves_shape(dummy_frame_bgr, sample_detections):
    result = annotate_frame(dummy_frame_bgr, sample_detections)
    assert result.shape == dummy_frame_bgr.shape
    assert result.dtype == np.uint8


def test_annotate_frame_no_detections(dummy_frame_bgr):
    result = annotate_frame(dummy_frame_bgr, [])
    assert result.shape == dummy_frame_bgr.shape


def test_annotate_frame_no_track_id(dummy_frame_bgr):
    dets = [Detection(track_id=None, class_id=0, class_name="person",
                      confidence=0.8, bbox=[10.0, 10.0, 100.0, 200.0])]
    result = annotate_frame(dummy_frame_bgr, dets)
    assert result.shape == dummy_frame_bgr.shape


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


@patch("detection.detector.YOLO")
def test_yolo_detector_init(mock_yolo):
    from detection.detector import YOLODetector
    detector = YOLODetector(model_path="yolo26n.pt", device="cpu")
    mock_yolo.assert_called_once_with("yolo26n.pt")
    assert detector.device == "cpu"
    assert detector.conf == 0.4
