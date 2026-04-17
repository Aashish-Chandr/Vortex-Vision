"""
Feature extraction pipeline: runs YOLO26 on processed frames and saves
detection feature sequences as .npy files for transformer training.
"""
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def extract_features(data_dir: str, out_dir: str, seq_len: int = 16, device: str = "cpu"):
    from detection.detector import YOLODetector, DetectionResult
    from anomaly.transformer_detector import detections_to_feature
    import cv2

    detector = YOLODetector(device=device, enable_tracking=False)
    root = Path(data_dir)
    out = Path(out_dir)

    for label in ["normal", "anomaly"]:
        label_dir = root / label
        if not label_dir.exists():
            continue

        out_label_dir = out / label
        out_label_dir.mkdir(parents=True, exist_ok=True)

        # Group frames by video source
        frames_by_video: dict[str, list[Path]] = defaultdict(list)
        for img_path in sorted(label_dir.glob("*.jpg")):
            # Filename format: {video_stem}_{frame_idx:06d}.jpg
            parts = img_path.stem.rsplit("_", 1)
            video_key = parts[0] if len(parts) == 2 else img_path.stem
            frames_by_video[video_key].append(img_path)

        logger.info("Extracting features for %d %s videos...", len(frames_by_video), label)

        for video_key, frame_paths in tqdm(frames_by_video.items(), desc=label):
            features = []
            for fp in sorted(frame_paths):
                frame = cv2.imread(str(fp))
                if frame is None:
                    continue
                result = detector.infer(frame, stream_id=video_key)
                feat = detections_to_feature(result)
                features.append(feat)

            if len(features) < seq_len:
                continue

            # Slide a window over the feature sequence
            for i in range(0, len(features) - seq_len + 1, seq_len // 2):
                window = np.array(features[i:i + seq_len], dtype=np.float32)
                out_path = out_label_dir / f"{video_key}_seq{i:06d}.npy"
                np.save(str(out_path), window)

    logger.info("Feature extraction complete → %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/train")
    parser.add_argument("--out-dir", default="data/processed/sequences/train")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    extract_features(args.data_dir, args.out_dir, args.seq_len, args.device)
