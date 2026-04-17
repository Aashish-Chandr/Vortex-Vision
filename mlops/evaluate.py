"""
Model evaluation: computes AUC-ROC, F1, precision, recall on test set.
Logs results to MLflow and writes metrics/eval_results.json for DVC.
"""
import json
import argparse
import logging
import torch
import numpy as np
import mlflow
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from torch.utils.data import DataLoader

from anomaly.autoencoder import ConvAutoencoder, AnomalyScorer
from mlops.dataset import AnomalyDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def evaluate(model_dir: str, data_dir: str, device: str = "cuda"):
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error("Test data directory not found: %s", data_dir)
        logger.error("Run: make prepare-data  to generate training data first.")
        raise FileNotFoundError(f"Test data not found: {data_dir}")

    model_path = Path(model_dir) / "autoencoder_best.pt"
    if not model_path.exists():
        logger.error("Model weights not found: %s", model_path)
        logger.error("Run: make train-ae  or  dvc pull  to get model weights.")
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = ConvAutoencoder()
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    scorer = AnomalyScorer(model, device=device)

    dataset = AnomalyDataset(data_dir, mode="labeled")
    if len(dataset) == 0:
        logger.error("No labeled frames found in %s", data_dir)
        logger.error("Expected structure: %s/{normal,anomaly}/*.jpg", data_dir)
        raise ValueError(f"Empty dataset at {data_dir}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    scores, labels = [], []
    for frame_tensor, label in loader:
        frame_np = (frame_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        score, _ = scorer.score(frame_np)
        scores.append(score)
        labels.append(int(label.item()))

    scores = np.array(scores)
    labels = np.array(labels)

    # Find optimal threshold via Youden's J
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_thresh = thresholds[np.argmax(j_scores)]

    preds = (scores > best_thresh).astype(int)
    auc = roc_auc_score(labels, scores)
    f1 = f1_score(labels, preds)

    metrics = {
        "auc_roc": round(auc, 4),
        "f1_score": round(f1, 4),
        "best_threshold": round(float(best_thresh), 6),
    }

    logger.info("Evaluation results: %s", metrics)
    logger.info("\n%s", classification_report(labels, preds, target_names=["normal", "anomaly"]))

    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/eval_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with mlflow.start_run():
        mlflow.log_metrics(metrics)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--data-dir", default="data/processed/test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(args.model_dir, args.data_dir, args.device)
