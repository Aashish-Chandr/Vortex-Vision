"""
Model bootstrap script.
Downloads or initializes model weights for VortexVision.

Options:
  --init-random   Create random-weight models for dev/testing (no GPU needed)
  --dvc-pull      Pull trained weights from DVC remote (requires AWS credentials)
  --hf-download   Download pre-trained YOLO weights from Ultralytics

Usage:
  python scripts/download_models.py --init-random
  python scripts/download_models.py --dvc-pull
"""
import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def init_random_weights():
    """Create random-weight model files for dev/testing. No GPU required."""
    import torch
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing random-weight autoencoder...")
    from anomaly.autoencoder import ConvAutoencoder
    ae = ConvAutoencoder()
    torch.save(ae.state_dict(), MODELS_DIR / "autoencoder_best.pt")
    logger.info("  Saved: models/autoencoder_best.pt")

    logger.info("Initializing random-weight transformer...")
    from anomaly.transformer_detector import TemporalTransformer
    tf = TemporalTransformer()
    torch.save(tf.state_dict(), MODELS_DIR / "transformer_best.pt")
    logger.info("  Saved: models/transformer_best.pt")

    logger.info("Random weights initialized. These are for dev only — train real models with: make train")


def dvc_pull():
    """Pull trained model weights from DVC remote storage."""
    logger.info("Pulling models from DVC remote...")
    result = subprocess.run(
        ["dvc", "pull", "models/autoencoder_best.pt", "models/transformer_best.pt"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        logger.info("DVC pull successful")
        logger.info(result.stdout)
    else:
        logger.error("DVC pull failed: %s", result.stderr)
        logger.error("Make sure AWS credentials are set and DVC remote is configured")
        raise RuntimeError("DVC pull failed")


def download_yolo():
    """Download YOLO26 weights from Ultralytics."""
    logger.info("Downloading YOLO26 weights...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolo26n.pt")  # auto-downloads from Ultralytics
        logger.info("YOLO26n weights downloaded to: %s", model.ckpt_path)
    except Exception as e:
        logger.error("YOLO download failed: %s", e)
        logger.error("Install ultralytics: pip install ultralytics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download/initialize VortexVision model weights")
    parser.add_argument("--init-random", action="store_true",
                        help="Create random-weight models for dev/testing")
    parser.add_argument("--dvc-pull", action="store_true",
                        help="Pull trained weights from DVC remote")
    parser.add_argument("--yolo", action="store_true",
                        help="Download YOLO26 weights from Ultralytics")
    args = parser.parse_args()

    if not any([args.init_random, args.dvc_pull, args.yolo]):
        parser.print_help()
        print("\nFor quick dev setup, run: python scripts/download_models.py --init-random")
    else:
        if args.init_random:
            init_random_weights()
        if args.dvc_pull:
            dvc_pull()
        if args.yolo:
            download_yolo()
