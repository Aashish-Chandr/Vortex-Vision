"""
Transformer-based temporal anomaly detector.
Operates on sequences of detection embeddings to catch behavioral anomalies.
"""
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from detection.detector import DetectionResult


class TemporalTransformer(nn.Module):
    """
    Encodes a sequence of frame-level feature vectors and predicts anomaly probability.
    Input: (batch, seq_len, feature_dim) — Output: (batch, 1)
    """

    def __init__(
        self, feature_dim: int = 128, seq_len: int = 16, nhead: int = 4, num_layers: int = 2
    ):
        super().__init__()
        self.pos_embedding = nn.Embedding(seq_len, feature_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))


def detections_to_feature(result: DetectionResult, feature_dim: int = 128) -> np.ndarray:
    """Converts a DetectionResult into a fixed-size feature vector."""
    feat = np.zeros(feature_dim, dtype=np.float32)
    if not result.detections:
        return feat

    feat[0] = len(result.detections) / 50.0
    feat[1] = float(np.mean([d.confidence for d in result.detections]))

    for i, det in enumerate(result.detections[:20]):
        base = 2 + i * 6
        if base + 5 < feature_dim:
            x1, y1, x2, y2 = det.bbox
            feat[base] = x1 / 640
            feat[base + 1] = y1 / 640
            feat[base + 2] = x2 / 640
            feat[base + 3] = y2 / 640
            feat[base + 4] = det.confidence
            feat[base + 5] = det.class_id / 80.0

    return feat


class BehavioralAnomalyDetector:
    """Maintains a rolling window of frame features and scores sequences."""

    def __init__(
        self,
        model: TemporalTransformer,
        seq_len: int = 16,
        threshold: float = 0.7,
        device: str = "cuda",
    ):
        self.model = model.to(device).eval()
        self.seq_len = seq_len
        self.threshold = threshold
        self.device = device
        self.buffer: deque = deque(maxlen=seq_len)

    @torch.no_grad()
    def update(self, result: DetectionResult) -> Tuple[float, bool]:
        feat = detections_to_feature(result)
        self.buffer.append(feat)

        if len(self.buffer) < self.seq_len:
            return 0.0, False

        seq = torch.tensor(np.stack(self.buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
        prob = self.model(seq).item()
        return prob, prob > self.threshold
