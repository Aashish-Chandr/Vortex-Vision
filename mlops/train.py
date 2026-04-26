"""
Training pipeline for anomaly detection models.
Tracked with MLflow. Data versioned with DVC.
"""
import argparse
import json
import logging
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from anomaly.autoencoder import ConvAutoencoder
from mlops.dataset import AnomalyDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_autoencoder(
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda",
    experiment_name: str = "vortexvision-autoencoder",
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr, "device": device})

        dataset = AnomalyDataset(data_dir, mode="normal")
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

        model = ConvAutoencoder().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("metrics").mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                x_hat, _ = model(batch)
                loss = criterion(x_hat, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    x_hat, _ = model(batch)
                    val_loss += criterion(x_hat, batch).item()
            val_loss /= len(val_loader)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            logger.info(
                "Epoch %d/%d — train: %.4f  val: %.4f", epoch + 1, epochs, train_loss, val_loss
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                mlflow.pytorch.log_model(model, "autoencoder-best")
                torch.save(model.state_dict(), "models/autoencoder_best.pt")

        mlflow.log_metric("best_val_loss", best_val_loss)
        with open("metrics/autoencoder_metrics.json", "w") as f:
            json.dump({"best_val_loss": best_val_loss}, f)
        logger.info("Training complete. Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/normal_frames")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_autoencoder(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
