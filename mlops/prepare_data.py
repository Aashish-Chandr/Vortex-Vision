"""
Data preparation pipeline:
  1. Extract frames from raw video files (UCF-Crime, ShanghaiTech, custom)
  2. Normalize and resize frames
  3. Split into train / val / test sets
  4. Save frame index CSV for DVC tracking
"""
import argparse
import csv
import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SPLITS = {"train": 0.75, "val": 0.15, "test": 0.10}
FRAME_SIZE = (640, 640)
FRAME_SKIP = 5          # extract every Nth frame
JPEG_QUALITY = 90


def extract_frames(video_path: Path, out_dir: Path, label: str) -> list[dict]:
    """Extract frames from a video file, return list of frame metadata."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open %s", video_path)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SKIP == 0:
            frame = cv2.resize(frame, FRAME_SIZE)
            fname = f"{video_path.stem}_{frame_idx:06d}.jpg"
            fpath = out_dir / fname
            cv2.imwrite(str(fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            records.append({"path": str(fpath), "label": label, "source": video_path.name})
            saved += 1
        frame_idx += 1

    cap.release()
    logger.info("  %s → %d frames extracted", video_path.name, saved)
    return records


def prepare(raw_dir: str, out_dir: str, seed: int = 42):
    raw = Path(raw_dir)
    out = Path(out_dir)
    random.seed(seed)

    all_records: list[dict] = []

    # Expect raw_dir/{normal,anomaly}/**/*.mp4
    for label in ["normal", "anomaly"]:
        label_dir = raw / label
        if not label_dir.exists():
            logger.warning("Directory not found: %s — skipping", label_dir)
            continue

        videos = list(label_dir.glob("**/*.mp4")) + list(label_dir.glob("**/*.avi"))
        logger.info("Processing %d %s videos...", len(videos), label)

        for video in tqdm(videos, desc=label):
            frame_out = out / "frames_raw" / label
            records = extract_frames(video, frame_out, label)
            all_records.extend(records)

    if not all_records:
        logger.error("No records extracted. Check raw_dir structure: raw_dir/{normal,anomaly}/*.mp4")
        return

    random.shuffle(all_records)
    n = len(all_records)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    splits = {
        "train": all_records[:n_train],
        "val": all_records[n_train:n_train + n_val],
        "test": all_records[n_train + n_val:],
    }

    for split_name, records in splits.items():
        split_dir = out / split_name
        for rec in tqdm(records, desc=f"Copying {split_name}"):
            src = Path(rec["path"])
            label = rec["label"]
            dst_dir = split_dir / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / src.name)

        # Write index CSV
        csv_path = out / f"{split_name}_index.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "source"])
            writer.writeheader()
            writer.writerows(records)
        logger.info("Split '%s': %d records → %s", split_name, len(records), csv_path)

    # Also create normal_frames dir for autoencoder training (normal only)
    normal_frames_dir = out / "normal_frames"
    normal_frames_dir.mkdir(parents=True, exist_ok=True)
    for rec in all_records:
        if rec["label"] == "normal":
            src = Path(rec["path"])
            shutil.copy2(src, normal_frames_dir / src.name)

    logger.info("Done. Total frames: %d (train=%d, val=%d, test=%d)",
                n, len(splits["train"]), len(splits["val"]), len(splits["test"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare VortexVision training data")
    parser.add_argument("--raw-dir", required=True, help="Path to raw video directory")
    parser.add_argument("--out-dir", required=True, help="Output directory for processed frames")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.raw_dir, args.out_dir, args.seed)
