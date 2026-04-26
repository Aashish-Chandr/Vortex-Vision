"""
Convolutional Autoencoder for frame-level anomaly detection.
Trained on normal behavior; high reconstruction error = anomaly.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 40 * 40, latent_dim),
        )
        self.decoder_fc = nn.Linear(latent_dim, 256 * 40 * 40)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(self.decoder_fc(z).view(-1, 256, 40, 40))
        return x_hat, z


class AnomalyScorer:
    """Computes per-frame anomaly scores using reconstruction error."""

    def __init__(self, model: ConvAutoencoder, threshold: float = 0.05, device: str = "cuda"):
        self.model = model.to(device).eval()
        self.threshold = threshold
        self.device = device

    @torch.no_grad()
    def score(self, frame: np.ndarray) -> Tuple[float, bool]:
        """Returns (anomaly_score, is_anomaly)."""
        import torchvision.transforms.functional as TF
        from PIL import Image

        img = Image.fromarray(frame[..., ::-1])
        tensor = TF.to_tensor(TF.resize(img, [640, 640])).unsqueeze(0).to(self.device)
        x_hat, _ = self.model(tensor)
        score = nn.functional.mse_loss(x_hat, tensor).item()
        return score, score > self.threshold
