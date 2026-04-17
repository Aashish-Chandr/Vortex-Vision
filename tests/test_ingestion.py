"""Unit tests for ingestion components."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from ingestion.stream_reader import StreamConfig, StreamReader
from ingestion.stream_manager import StreamManager, StreamStatus


def test_stream_config_defaults():
    config = StreamConfig(source="rtsp://test", stream_id="cam-01")
    assert config.fps_limit == 30
    assert config.kafka_topic == "video-frames"
    assert config.resize == (640, 640)


def test_stream_manager_add_and_list():
    manager = StreamManager(kafka_bootstrap="localhost:9092")
    config = StreamConfig(source="rtsp://test", stream_id="test-stream")

    with patch.object(StreamReader, "start", return_value=None):
        result = manager.add_stream(config)

    assert result is True
    statuses = manager.list_streams()
    assert any(s["stream_id"] == "test-stream" for s in statuses)


def test_stream_manager_duplicate_stream():
    manager = StreamManager()
    config = StreamConfig(source="rtsp://test", stream_id="dup-stream")

    with patch.object(StreamReader, "start", return_value=None):
        manager.add_stream(config)
        result = manager.add_stream(config)

    assert result is False


def test_stream_manager_remove_stream():
    manager = StreamManager()
    config = StreamConfig(source="rtsp://test", stream_id="remove-me")

    with patch.object(StreamReader, "start", return_value=None):
        manager.add_stream(config)

    result = manager.remove_stream("remove-me")
    assert result is True
    assert manager.get_status("remove-me") is None


def test_stream_manager_remove_nonexistent():
    manager = StreamManager()
    result = manager.remove_stream("does-not-exist")
    assert result is False


def test_stream_manager_max_streams():
    manager = StreamManager(max_streams=2)
    for i in range(2):
        config = StreamConfig(source=f"rtsp://test{i}", stream_id=f"stream-{i}")
        with patch.object(StreamReader, "start", return_value=None):
            manager.add_stream(config)

    config = StreamConfig(source="rtsp://overflow", stream_id="overflow")
    with patch.object(StreamReader, "start", return_value=None):
        result = manager.add_stream(config)
    assert result is False
