"""
Frame annotation: draws bounding boxes, track IDs, labels, and masks.
"""
from typing import List

import cv2
import numpy as np

from detection.detector import Detection

PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def annotate_frame(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        color = PALETTE[det.class_id % len(PALETTE)]
        x1, y1, x2, y2 = map(int, det.bbox)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence:.2f}"
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if det.mask is not None:
            mask_resized = cv2.resize(det.mask[0], (frame.shape[1], frame.shape[0]))
            mask_bool = mask_resized > 0.5
            overlay = annotated.copy()
            overlay[mask_bool] = (overlay[mask_bool] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            annotated = overlay

    return annotated
