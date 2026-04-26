"""
Shared application state: thread-safe frame store, async alert queue,
lazy-loaded model references, and load status tracking.
"""
import asyncio
import logging
import traceback
from collections import deque
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self) -> None:
        self._alert_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._latest_frames: Dict[str, bytes] = {}
        self._events: deque = deque(maxlen=50_000)
        self.detector = None
        self.anomaly_scorer = None
        self.behavioral_detector = None
        self.vlm_client = None
        self.nl_engine = None
        self.clip_buffer = None
        self.load_errors: Dict[str, str] = {}

    async def startup(self) -> None:
        asyncio.create_task(self._load_models())

    async def _load_models(self) -> None:
        import torch

        from config.settings import get_settings

        settings = get_settings()

        try:
            from detection.detector import YOLODetector

            self.detector = YOLODetector(
                model_path=settings.yolo_model,
                device=settings.yolo_device,
                conf_threshold=settings.yolo_conf,
                iou_threshold=settings.yolo_iou,
            )
            logger.info("YOLO26 detector ready")
        except Exception as exc:
            self.load_errors["detector"] = str(exc)
            logger.warning("YOLO26 not loaded: %s", exc)

        try:
            from anomaly.autoencoder import AnomalyScorer, ConvAutoencoder

            ae = ConvAutoencoder()
            weights_path = f"{settings.model_storage_path}/autoencoder_best.pt"
            try:
                ae.load_state_dict(torch.load(weights_path, map_location=settings.yolo_device))
                logger.info("Autoencoder weights loaded from %s", weights_path)
            except FileNotFoundError:
                logger.warning("Autoencoder weights not found — using random init")
            self.anomaly_scorer = AnomalyScorer(
                ae, threshold=settings.ae_threshold, device=settings.yolo_device
            )
        except Exception as exc:
            self.load_errors["anomaly_scorer"] = str(exc)
            logger.error("Anomaly scorer failed: %s\n%s", exc, traceback.format_exc())

        try:
            from anomaly.transformer_detector import BehavioralAnomalyDetector, TemporalTransformer

            tf = TemporalTransformer(seq_len=settings.seq_len)
            weights_path = f"{settings.model_storage_path}/transformer_best.pt"
            try:
                tf.load_state_dict(torch.load(weights_path, map_location=settings.yolo_device))
                logger.info("Transformer weights loaded from %s", weights_path)
            except FileNotFoundError:
                logger.warning("Transformer weights not found — using random init")
            self.behavioral_detector = BehavioralAnomalyDetector(
                tf,
                seq_len=settings.seq_len,
                threshold=settings.transformer_threshold,
                device=settings.yolo_device,
            )
        except Exception as exc:
            self.load_errors["behavioral_detector"] = str(exc)
            logger.error("Behavioral detector failed: %s\n%s", exc, traceback.format_exc())

        try:
            from vlm.nl_query_engine import NLQueryEngine
            from vlm.qwen_client import QwenVLClient

            self.vlm_client = QwenVLClient(
                mode=settings.vlm_mode,
                model_name=settings.vlm_model,
                api_base=settings.vlm_api_base,
                api_key=settings.vlm_api_key,
            )
            self.nl_engine = NLQueryEngine(self.vlm_client)
            logger.info("VLM query engine ready (mode=%s)", settings.vlm_mode)
        except Exception as exc:
            self.load_errors["vlm_engine"] = str(exc)
            logger.warning("VLM not loaded: %s", exc)

        try:
            from mlops.clip_saver import ClipBuffer

            s3 = settings.s3_bucket if settings.s3_bucket not in ("", "vortexvision-data") else None
            self.clip_buffer = ClipBuffer(
                storage_path=settings.clip_storage_path,
                s3_bucket=s3,
            )
            logger.info("Clip buffer ready → %s", settings.clip_storage_path)
        except Exception as exc:
            self.load_errors["clip_buffer"] = str(exc)
            logger.error("Clip buffer failed: %s", exc)

        loaded = [
            k
            for k in (
                "detector",
                "anomaly_scorer",
                "behavioral_detector",
                "vlm_engine",
                "clip_buffer",
            )
            if k not in self.load_errors
        ]
        logger.info("Models ready: %s | Errors: %s", loaded, list(self.load_errors.keys()))

    async def shutdown(self) -> None:
        logger.info("AppState shutting down")

    def push_frame(self, stream_id: str, frame_bytes: bytes) -> None:
        self._latest_frames[stream_id] = frame_bytes

    def get_latest_frame(self, stream_id: str) -> Optional[bytes]:
        return self._latest_frames.get(stream_id)

    def push_event(self, event: dict) -> None:
        self._events.append(event)
        try:
            self._alert_queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._alert_queue.get_nowait()
                self._alert_queue.put_nowait(event)
            except Exception:
                pass

    async def pop_alert(self) -> Optional[dict]:
        try:
            return self._alert_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def get_events(self, stream_id: Optional[str] = None, limit: int = 100) -> List[dict]:
        events = list(self._events)
        if stream_id:
            events = [e for e in events if e.get("stream_id") == stream_id]
        return events[-limit:]
