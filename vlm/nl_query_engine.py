"""
Natural language query engine over indexed video events.
Example: "Show me all red cars speeding in the last 5 minutes"
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from anomaly.event_types import AnomalyEvent
from vlm.qwen_client import QwenVLClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a video surveillance analyst. You are given frames from a security camera. "
    "Answer questions precisely about what you observe. "
    "Include timestamps and object details when relevant."
)


@dataclass
class VideoClip:
    stream_id: str
    start_time: float
    end_time: float
    frames: List[np.ndarray] = field(default_factory=list)
    description: str = ""
    anomaly_events: List[AnomalyEvent] = field(default_factory=list)


@dataclass
class QueryResult:
    query: str
    clips: List[VideoClip]
    answer: str
    processing_ms: float


class NLQueryEngine:
    """Indexes video clips and answers natural language queries using Qwen VLM."""

    def __init__(self, vlm_client: QwenVLClient, clip_window_seconds: float = 10.0):
        self.vlm = vlm_client
        self.clip_window = clip_window_seconds
        self._clips: List[VideoClip] = []

    def index_clip(self, clip: VideoClip) -> None:
        """Add a clip to the searchable index."""
        if clip.frames:
            clip.description = self.vlm.query_frames(
                clip.frames[:4],
                "Briefly describe what is happening in these frames in one sentence.",
                SYSTEM_PROMPT,
            )
        self._clips.append(clip)
        logger.debug("Indexed clip from stream %s at t=%.1f", clip.stream_id, clip.start_time)

    def query(self, question: str, time_window_seconds: Optional[float] = None) -> QueryResult:
        """Answer a natural language question, optionally filtering to recent clips."""
        t0 = time.monotonic()
        now = time.time()

        candidates = self._clips
        if time_window_seconds:
            cutoff = now - time_window_seconds
            candidates = [c for c in self._clips if c.end_time >= cutoff]

        if not candidates:
            return QueryResult(
                query=question,
                clips=[],
                answer="No video clips available for the requested time window.",
                processing_ms=(time.monotonic() - t0) * 1000,
            )

        sample_frames: List[np.ndarray] = []
        for clip in candidates[:5]:
            sample_frames.extend(clip.frames[:2])

        answer = self.vlm.query_frames(sample_frames, question, SYSTEM_PROMPT)

        return QueryResult(
            query=question,
            clips=candidates,
            answer=answer,
            processing_ms=(time.monotonic() - t0) * 1000,
        )
