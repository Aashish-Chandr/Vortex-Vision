"""
Object detection using YOLO26 (Ultralytics 2026).
Supports detection, tracking, segmentation, and pose estimation.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    track_id: Optional[int]
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]
    mask: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None


@dataclass
class DetectionResult:
    stream_id: str
    timestamp: float
    frame_id: int
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0


class YOLODetector:
    """Wraps YOLO26 for real-time detection + ByteTrack tracking."""

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        device: str = "cuda",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        enable_tracking: bool = True,
        tracker_config: str = "bytetrack.yaml",
    ):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.enable_tracking = enable_tracking
        self.tracker_config = tracker_config
        logger.info("YOLO26 loaded: %s on %s", model_path, device)

    def infer(self, frame: np.ndarray, stream_id: str = "", frame_id: int = 0) -> DetectionResult:
        t0 = time.monotonic()

        if self.enable_tracking:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                tracker=self.tracker_config,
                verbose=False,
            )
        else:
            results = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )

        inference_ms = (time.monotonic() - t0) * 1000
        return DetectionResult(
            stream_id=stream_id,
            timestamp=time.time(),
            frame_id=frame_id,
            detections=self._parse_results(results),
            inference_ms=inference_ms,
        )

    def _parse_results(self, results) -> List[Detection]:
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for i, box in enumerate(r.boxes):
                track_id = int(box.id[0]) if box.id is not None else None
                class_id = int(box.cls[0])
                mask = None
                if r.masks is not None and i < len(r.masks):
                    mask = r.masks[i].data.cpu().numpy()
                keypoints = None
                if r.keypoints is not None and i < len(r.keypoints):
                    keypoints = r.keypoints[i].data.cpu().numpy()
                detections.append(
                    Detection(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=self.model.names[class_id],
                        confidence=float(box.conf[0]),
                        bbox=box.xyxy[0].tolist(),
                        mask=mask,
                        keypoints=keypoints,
                    )
                )
        return detections
