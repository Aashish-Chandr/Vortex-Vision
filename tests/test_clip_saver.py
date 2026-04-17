"""Unit tests for clip saving logic."""
import numpy as np
import pytest
import tempfile
from pathlib import Path

from mlops.clip_saver import ClipBuffer


@pytest.fixture
def tmp_clip_dir(tmp_path):
    return str(tmp_path / "clips")


@pytest.fixture
def clip_buffer(tmp_clip_dir):
    return ClipBuffer(
        pre_event_seconds=1.0,
        post_event_seconds=1.0,
        fps=5,
        storage_path=tmp_clip_dir,
        s3_bucket=None,
    )


def test_clip_buffer_push_frame(clip_buffer, dummy_frame_bgr):
    clip_buffer.push_frame("cam-01", dummy_frame_bgr)
    assert "cam-01" in clip_buffer._buffers
    assert len(clip_buffer._buffers["cam-01"]) == 1


def test_clip_buffer_trigger_returns_path(clip_buffer, dummy_frame_bgr):
    for _ in range(10):
        clip_buffer.push_frame("cam-01", dummy_frame_bgr)
    path = clip_buffer.trigger("cam-01", "event-abc123", "fight")
    assert path is not None
    assert "cam-01" in path
    assert "fight" in path


def test_clip_buffer_saves_file(clip_buffer, dummy_frame_bgr, tmp_clip_dir):
    fps = clip_buffer.fps
    post_frames = clip_buffer.post_frames

    # Fill pre-event buffer
    for _ in range(clip_buffer.pre_frames):
        clip_buffer.push_frame("cam-01", dummy_frame_bgr)

    clip_buffer.trigger("cam-01", "event-xyz", "accident")

    # Push post-event frames to finalize
    for _ in range(post_frames):
        clip_buffer.push_frame("cam-01", dummy_frame_bgr)

    clips = list(Path(tmp_clip_dir).glob("*.mp4"))
    assert len(clips) == 1, f"Expected 1 clip, found {len(clips)}"


def test_clip_buffer_no_double_trigger(clip_buffer, dummy_frame_bgr):
    for _ in range(10):
        clip_buffer.push_frame("cam-01", dummy_frame_bgr)

    path1 = clip_buffer.trigger("cam-01", "event-1", "fight")
    path2 = clip_buffer.trigger("cam-01", "event-2", "fight")

    assert path1 is not None
    assert path2 is None  # already recording
