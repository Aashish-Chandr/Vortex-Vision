"""
Training pipeline for the TemporalTransformer behavioral anomaly detector.
Operates on sequences of detection feature vectors.
Tracked with MLflow.
"""
import argparse
import logging
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from anomaly.transformer_detector import TemporalTransformer, detections_to_feature
from detection.detector import DetectionResult, Detection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class SequenceDataset(Dataset):
    """
    Loads pre-extracted feature sequences from .npy files.
    Each file: (seq_len, feature_dim) float32 array + label (0/1).
    """

    def __init__(self, data_dir: str, seq_len: int = 16, feature_dim: int = 128):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.samples: list[tuple[np.ndarray, int]] = []

        root = Path(data_dir)
        for label, subdir in [(0, "normal"), (1, "anomaly")]:
            for npy_file in sorted((root / subdir).glob("*.npy")):
                seq = np.load(str(npy_file)).astype(np.float32)
                # Pad or truncate to seq_len
                if len(seq) >= seq_len:
                    seq = seq[:seq_len]
                else:
                    pad = np.zeros((seq_len - len(seq), feature_dim), dtype=np.float32)
                    seq = np.vstack([seq, pad])
                self.samples.append((seq, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.tensor(seq), torch.tensor(label, dtype=torch.float32)


def train_transformer(
    data_dir: str,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    seq_len: int = 16,
    feature_dim: int = 128,
    device: str = "cuda",
    experiment_name: str = "vortexvision-transformer",
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "seq_len": seq_len, "feature_dim": feature_dim, "device": device,
        })

        dataset = SequenceDataset(data_dir, seq_len=seq_len, feature_dim=feature_dim)
        if len(dataset) == 0:
            logger.error("No sequence data found in %s. Run feature extraction first.", data_dir)
            return

        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

        model = TemporalTransformer(feature_dim=feature_dim, seq_len=seq_len).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCELoss()

        best_val_loss = float("inf")
        Path("models").mkdir(exist_ok=True)

        for epoch in range(epochs):
            model.train()
            train_loss, correct, total = 0.0, 0, 0
            for seqs, labels in train_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                preds = model(seqs).squeeze(1)
                loss = criterion(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                correct += ((preds > 0.5) == labels.bool()).sum().item()
                total += len(labels)

            scheduler.step()
            train_acc = correct / total
            train_loss /= len(train_loader)

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for seqs, labels in val_loader:
                    seqs, labels = seqs.to(device), labels.to(device)
                    preds = model(seqs).squeeze(1)
                    val_loss += criterion(preds, labels).item()
                    val_correct += ((preds > 0.5) == labels.bool()).sum().item()
                    val_total += len(labels)

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss,
                "train_acc": train_acc, "val_acc": val_acc,
            }, step=epoch)
            logger.info("Epoch %d/%d — train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f",
                        epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                mlflow.pytorch.log_model(model, "transformer-best")
                torch.save(model.state_dict(), "models/transformer_best.pt")

        mlflow.log_metric("best_val_loss", best_val_loss)
        logger.info("Training complete. Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/sequences")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train_transformer(
        data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seq_len=args.seq_len, feature_dim=args.feature_dim, device=args.device,
    )
