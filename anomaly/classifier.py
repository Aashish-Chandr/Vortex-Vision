"""
Rule-based + heuristic anomaly classifier.
Maps detection results and anomaly scores to specific AnomalyType labels.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from anomaly.event_types import AnomalyType
from detection.detector import DetectionResult

logger = logging.getLogger(__name__)

WEAPON_CLASSES = {"knife", "gun", "pistol", "rifle", "firearm", "weapon"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}
PERSON_CLASSES = {"person"}


@dataclass
class ClassificationContext:
    result: DetectionResult
    ae_score: float
    tf_score: float
    prev_person_count: int = 0
    restricted_zones: List[tuple] = field(default_factory=list)


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _point_in_box(cx: float, cy: float, box: tuple) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2


class AnomalyClassifier:
    def __init__(
        self,
        fight_iou_threshold: float = 0.15,
        fight_min_persons: int = 2,
        crowd_rush_delta: int = 5,
        accident_ae_threshold: float = 0.08,
        weapon_conf_threshold: float = 0.5,
        restricted_zones: Optional[List[tuple]] = None,
    ) -> None:
        self.fight_iou_threshold = fight_iou_threshold
        self.fight_min_persons = fight_min_persons
        self.crowd_rush_delta = crowd_rush_delta
        self.accident_ae_threshold = accident_ae_threshold
        self.weapon_conf_threshold = weapon_conf_threshold
        self.restricted_zones = restricted_zones or []
        self._prev_person_count: dict[str, int] = {}

    def classify(self, ctx: ClassificationContext) -> AnomalyType:
        detections = ctx.result.detections
        stream_id = ctx.result.stream_id

        persons = [d for d in detections if d.class_name.lower() in PERSON_CLASSES]
        vehicles = [d for d in detections if d.class_name.lower() in VEHICLE_CLASSES]
        weapons = [
            d for d in detections
            if d.class_name.lower() in WEAPON_CLASSES and d.confidence >= self.weapon_conf_threshold
        ]

        if weapons:
            logger.warning("Weapon detected on stream %s: %s", stream_id, [w.class_name for w in weapons])
            return AnomalyType.WEAPON

        if self.restricted_zones and persons:
            for person in persons:
                cx = (person.bbox[0] + person.bbox[2]) / 2
                cy = (person.bbox[1] + person.bbox[3]) / 2
                for zone in self.restricted_zones:
                    if _point_in_box(cx, cy, zone):
                        return AnomalyType.TRESPASSING

        if len(persons) >= self.fight_min_persons and ctx.tf_score > 0.6:
            overlap_count = sum(
                1
                for i in range(len(persons))
                for j in range(i + 1, len(persons))
                if _bbox_iou(persons[i].bbox, persons[j].bbox) > self.fight_iou_threshold
            )
            if overlap_count > 0:
                return AnomalyType.FIGHT

        prev_count = self._prev_person_count.get(stream_id, 0)
        current_count = len(persons)
        self._prev_person_count[stream_id] = current_count
        if current_count - prev_count >= self.crowd_rush_delta and current_count >= 5:
            return AnomalyType.CROWD_RUSH

        if vehicles and ctx.ae_score > self.accident_ae_threshold:
            return AnomalyType.ACCIDENT

        if ctx.ae_score > self.accident_ae_threshold and not persons and not vehicles:
            return AnomalyType.ABANDONED_OBJECT

        return AnomalyType.UNKNOWN
