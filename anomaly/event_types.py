"""
Anomaly event taxonomy and alert payload definitions.
"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class AnomalyType(str, Enum):
    FIGHT = "fight"
    CROWD_RUSH = "crowd_rush"
    ACCIDENT = "accident"
    WEAPON = "weapon"
    TRESPASSING = "trespassing"
    ABANDONED_OBJECT = "abandoned_object"
    UNKNOWN = "unknown"


@dataclass
class AnomalyEvent:
    stream_id: str
    timestamp: float
    frame_id: int
    anomaly_type: AnomalyType
    confidence: float
    autoencoder_score: float
    transformer_score: float
    clip_path: Optional[str] = None      # path to saved video clip
    description: Optional[str] = None   # VLM-generated description
    metadata: dict = field(default_factory=dict)
