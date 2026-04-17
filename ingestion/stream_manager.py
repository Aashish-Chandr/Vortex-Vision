"""
Stream manager: lifecycle management for multiple concurrent video streams.
Handles start/stop/restart and stream health monitoring.
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from ingestion.stream_reader import StreamConfig, StreamReader

logger = logging.getLogger(__name__)


class StreamStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ManagedStream:
    config: StreamConfig
    reader: StreamReader
    thread: Optional[threading.Thread] = None
    status: StreamStatus = StreamStatus.STARTING
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    frames_sent: int = 0


class StreamManager:
    """
    Manages multiple concurrent video stream readers.
    Provides health monitoring and automatic restart on failure.
    """

    def __init__(self, kafka_bootstrap: str = "localhost:9092", max_streams: int = 64):
        self.kafka_bootstrap = kafka_bootstrap
        self.max_streams = max_streams
        self._streams: Dict[str, ManagedStream] = {}
        self._lock = threading.Lock()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def add_stream(self, config: StreamConfig) -> bool:
        with self._lock:
            if config.stream_id in self._streams:
                logger.warning("Stream %s already exists", config.stream_id)
                return False
            if len(self._streams) >= self.max_streams:
                logger.error("Max streams (%d) reached", self.max_streams)
                return False

            reader = StreamReader(config, kafka_bootstrap=self.kafka_bootstrap)
            managed = ManagedStream(config=config, reader=reader)
            self._streams[config.stream_id] = managed

        thread = threading.Thread(
            target=self._run_stream,
            args=(config.stream_id,),
            daemon=True,
            name=f"stream-{config.stream_id}",
        )
        managed.thread = thread
        managed.status = StreamStatus.RUNNING
        thread.start()
        logger.info("Stream started: %s", config.stream_id)
        return True

    def remove_stream(self, stream_id: str) -> bool:
        with self._lock:
            managed = self._streams.get(stream_id)
            if not managed:
                return False
            managed.reader.stop()
            managed.status = StreamStatus.STOPPED
            del self._streams[stream_id]
        logger.info("Stream removed: %s", stream_id)
        return True

    def get_status(self, stream_id: str) -> Optional[dict]:
        managed = self._streams.get(stream_id)
        if not managed:
            return None
        return {
            "stream_id": stream_id,
            "status": managed.status.value,
            "source": managed.config.source,
            "started_at": managed.started_at,
            "uptime_seconds": time.time() - managed.started_at,
            "error": managed.error,
        }

    def list_streams(self) -> list[dict]:
        return [self.get_status(sid) for sid in list(self._streams.keys())]

    def _run_stream(self, stream_id: str):
        managed = self._streams.get(stream_id)
        if not managed:
            return
        try:
            managed.reader.start()
        except Exception as e:
            logger.error("Stream %s failed: %s", stream_id, e)
            with self._lock:
                if stream_id in self._streams:
                    self._streams[stream_id].status = StreamStatus.ERROR
                    self._streams[stream_id].error = str(e)

    def _monitor_loop(self):
        """Restart failed streams after a backoff period."""
        while True:
            time.sleep(30)
            with self._lock:
                for stream_id, managed in list(self._streams.items()):
                    if managed.status == StreamStatus.ERROR:
                        logger.info("Restarting failed stream: %s", stream_id)
                        managed.error = None
                        managed.status = StreamStatus.STARTING
                        reader = StreamReader(managed.config, kafka_bootstrap=self.kafka_bootstrap)
                        managed.reader = reader
                        thread = threading.Thread(
                            target=self._run_stream, args=(stream_id,), daemon=True
                        )
                        managed.thread = thread
                        managed.status = StreamStatus.RUNNING
                        thread.start()
