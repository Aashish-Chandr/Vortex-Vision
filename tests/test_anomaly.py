"""Unit tests for anomaly detection models."""
import numpy as np
import pytest
import torch

from anomaly.autoencoder import ConvAutoencoder
from anomaly.transformer_detector import (
    BehavioralAnomalyDetector,
    TemporalTransformer,
    detections_to_feature,
)
from anomaly.event_types import AnomalyEvent, AnomalyType
from detection.detector import Detection, DetectionResult


# ── Autoencoder ───────────────────────────────────────────────────────────────

def test_autoencoder_output_shape():
    model = ConvAutoencoder()
    x = torch.randn(2, 3, 640, 640)
    x_hat, z = model(x)
    assert x_hat.shape == x.shape, "Reconstruction shape must match input"
    assert z.shape == (2, 256), "Latent dim should be 256"


def test_autoencoder_reconstruction_range():
    model = ConvAutoencoder()
    x = torch.rand(1, 3, 640, 640)  # [0, 1] range
    x_hat, _ = model(x)
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0, "Sigmoid output must be in [0, 1]"


def test_autoencoder_different_inputs_different_outputs():
    model = ConvAutoencoder()
    x1 = torch.zeros(1, 3, 640, 640)
    x2 = torch.ones(1, 3, 640, 640)
    x1_hat, _ = model(x1)
    x2_hat, _ = model(x2)
    assert not torch.allclose(x1_hat, x2_hat)


# ── Transformer ───────────────────────────────────────────────────────────────

def test_transformer_output_shape():
    model = TemporalTransformer(feature_dim=128, seq_len=16)
    x = torch.randn(4, 16, 128)
    out = model(x)
    assert out.shape == (4, 1)


def test_transformer_output_probability():
    model = TemporalTransformer(feature_dim=128, seq_len=16)
    x = torch.randn(8, 16, 128)
    out = model(x)
    assert (out >= 0).all() and (out <= 1).all(), "Output must be probability in [0, 1]"


def test_transformer_batch_consistency():
    model = TemporalTransformer(feature_dim=128, seq_len=16)
    model.eval()
    x = torch.randn(1, 16, 128)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2), "Deterministic in eval mode"


# ── Feature extraction ────────────────────────────────────────────────────────

def test_detections_to_feature_empty():
    result = DetectionResult(stream_id="test", timestamp=0.0, frame_id=0)
    feat = detections_to_feature(result)
    assert feat.shape == (128,)
    assert feat.sum() == 0.0


def test_detections_to_feature_with_detections():
    result = DetectionResult(
        stream_id="test", timestamp=0.0, frame_id=0,
        detections=[
            Detection(track_id=1, class_id=0, class_name="person",
                      confidence=0.9, bbox=[10.0, 10.0, 100.0, 200.0]),
            Detection(track_id=2, class_id=2, class_name="car",
                      confidence=0.75, bbox=[200.0, 100.0, 400.0, 300.0]),
        ],
    )
    feat = detections_to_feature(result)
    assert feat.shape == (128,)
    assert feat[0] > 0, "Count feature should be non-zero"
    assert feat[1] > 0, "Confidence feature should be non-zero"


def test_detections_to_feature_normalized():
    result = DetectionResult(
        stream_id="test", timestamp=0.0, frame_id=0,
        detections=[
            Detection(track_id=i, class_id=0, class_name="person",
                      confidence=0.9, bbox=[0.0, 0.0, 640.0, 640.0])
            for i in range(50)
        ],
    )
    feat = detections_to_feature(result)
    assert feat[0] <= 1.0, "Count should be normalized"


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


def test_anomaly_type_values():
    assert AnomalyType.FIGHT == "fight"
    assert AnomalyType.CROWD_RUSH == "crowd_rush"
    assert AnomalyType.WEAPON == "weapon"
