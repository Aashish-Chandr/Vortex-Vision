"""Unit tests for clip saving logic."""
import numpy as np
import pytest
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


def test_clip_buffer_push_multiple_frames(clip_buffer, dummy_frame_bgr):
    for _ in range(5):
        clip_buffer.push_frame("cam-01", dummy_frame_bgr)
    assert len(clip_buffer._buffers["cam-01"]) == 5


def test_clip_buffer_trigger_returns_path(clip_buffer, dummy_frame_bgr):
    for _ in range(10):
        clip_buffer.push_frame("cam-01", dummy_frame_bgr)
    path = clip_buffer.trigger("cam-01", "event-abc123", "fight")
    assert path is not None
    assert "cam-01" in path
    assert "fight" in path
    assert path.endswith(".mp4")


def test_clip_buffer_trigger_path_contains_event_prefix(clip_buffer, dummy_frame_bgr):
    for _ in range(5):
        clip_buffer.push_frame("cam-02", dummy_frame_bgr)
    path = clip_buffer.trigger("cam-02", "abcdef1234", "accident")
    assert path is not None
    # event_id[:8] = "abcdef12"
    assert "abcdef12" in path


def test_clip_buffer_no_double_trigger(clip_buffer, dummy_frame_bgr):
    for _ in range(10):
        clip_buffer.push_frame("cam-01", dummy_frame_bgr)

    path1 = clip_buffer.trigger("cam-01", "event-1", "fight")
    path2 = clip_buffer.trigger("cam-01", "event-2", "fight")

    assert path1 is not None
    assert path2 is None  # already recording


def test_clip_buffer_storage_dir_created(tmp_path):
    storage = str(tmp_path / "new" / "nested" / "clips")
    buf = ClipBuffer(storage_path=storage, s3_bucket=None)
    assert Path(storage).exists()


def test_clip_buffer_multiple_streams(clip_buffer, dummy_frame_bgr):
    clip_buffer.push_frame("cam-01", dummy_frame_bgr)
    clip_buffer.push_frame("cam-02", dummy_frame_bgr)
    assert "cam-01" in clip_buffer._buffers
    assert "cam-02" in clip_buffer._buffers
