"""
Clip saver: buffers frames around anomaly events and saves them as MP4 clips.
Supports local filesystem and S3 upload.
"""
import io
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ClipBuffer:
    """
    Maintains a rolling pre-event frame buffer per stream.
    When an anomaly is detected, saves pre + post frames as an MP4 clip.
    """

    def __init__(
        self,
        pre_event_seconds: float = 5.0,
        post_event_seconds: float = 5.0,
        fps: int = 15,
        storage_path: str = "/data/clips",
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "clips/",
    ):
        self.pre_frames = int(pre_event_seconds * fps)
        self.post_frames = int(post_event_seconds * fps)
        self.fps = fps
        self.storage_path = Path(storage_path)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix

        self._buffers: dict[str, deque] = {}
        self._recording: dict[str, list] = {}
        self._record_remaining: dict[str, int] = {}

        self.storage_path.mkdir(parents=True, exist_ok=True)

    def push_frame(self, stream_id: str, frame: np.ndarray):
        """Add a frame to the rolling buffer. Saves clip if recording."""
        if stream_id not in self._buffers:
            self._buffers[stream_id] = deque(maxlen=self.pre_frames)

        self._buffers[stream_id].append(frame.copy())

        if stream_id in self._recording:
            self._recording[stream_id].append(frame.copy())
            self._record_remaining[stream_id] -= 1
            if self._record_remaining[stream_id] <= 0:
                self._finalize_clip(stream_id)

    def trigger(self, stream_id: str, event_id: str, anomaly_type: str) -> Optional[str]:
        """Trigger clip recording for a stream. Returns expected clip path."""
        if stream_id in self._recording:
            return None  # already recording

        pre_frames = list(self._buffers.get(stream_id, []))
        self._recording[stream_id] = pre_frames
        self._record_remaining[stream_id] = self.post_frames

        ts = int(time.time())
        clip_name = f"{stream_id}_{anomaly_type}_{ts}_{event_id[:8]}.mp4"
        self._clip_names = getattr(self, "_clip_names", {})
        self._clip_names[stream_id] = clip_name

        logger.info("Clip recording triggered: %s", clip_name)
        return str(self.storage_path / clip_name)

    def _finalize_clip(self, stream_id: str):
        frames = self._recording.pop(stream_id)
        self._record_remaining.pop(stream_id)
        clip_name = self._clip_names.pop(stream_id)
        clip_path = self.storage_path / clip_name

        if not frames:
            return

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, self.fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

        logger.info("Clip saved: %s (%d frames)", clip_path, len(frames))

        if self.s3_bucket:
            self._upload_to_s3(clip_path, clip_name)

    def _upload_to_s3(self, local_path: Path, clip_name: str):
        try:
            import boto3
            s3 = boto3.client("s3")
            s3_key = f"{self.s3_prefix}{clip_name}"
            s3.upload_file(str(local_path), self.s3_bucket, s3_key)
            logger.info("Clip uploaded to s3://%s/%s", self.s3_bucket, s3_key)
        except Exception as e:
            logger.error("S3 upload failed: %s", e)
