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
    def __init__(self):
        self._alert_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._latest_frames: Dict[str, bytes] = {}
        self._events: deque = deque(maxlen=50_000)
        self.detector = None
        self.anomaly_scorer = None
        self.behavioral_detector = None
        self.vlm_client = None
        self.nl_engine = None
        self.clip_buffer = None
        # Tracks load errors for /health/deep
        self.load_errors: Dict[str, str] = {}

    async def startup(self):
        asyncio.create_task(self._load_models())

    async def _load_models(self):
        from config.settings import get_settings
        import torch
        settings = get_settings()

        # ── YOLO detector ─────────────────────────────────────────────────────
        try:
            from detection.detector import YOLODetector
            self.detector = YOLODetector(
                model_path=settings.yolo_model,
                device=settings.yolo_device,
                conf_threshold=settings.yolo_conf,
                iou_threshold=settings.yolo_iou,
            )
            logger.info("YOLO26 detector ready")
        except Exception as e:
            self.load_errors["detector"] = str(e)
            logger.warning("YOLO26 not loaded (expected in dev without GPU): %s", e)

        # ── Autoencoder anomaly scorer ─────────────────────────────────────────
        try:
            from anomaly.autoencoder import ConvAutoencoder, AnomalyScorer
            ae = ConvAutoencoder()
            weights_path = f"{settings.model_storage_path}/autoencoder_best.pt"
            try:
                ae.load_state_dict(torch.load(weights_path, map_location=settings.yolo_device))
                logger.info("Autoencoder weights loaded from %s", weights_path)
            except FileNotFoundError:
                logger.warning("Autoencoder weights not found at %s — using random init (dev mode)", weights_path)
            self.anomaly_scorer = AnomalyScorer(
                ae, threshold=settings.ae_threshold, device=settings.yolo_device
            )
        except Exception as e:
            self.load_errors["anomaly_scorer"] = str(e)
            logger.error("Anomaly scorer failed to load: %s\n%s", e, traceback.format_exc())

        # ── Temporal transformer ───────────────────────────────────────────────
        try:
            from anomaly.transformer_detector import TemporalTransformer, BehavioralAnomalyDetector
            tf = TemporalTransformer(seq_len=settings.seq_len)
            weights_path = f"{settings.model_storage_path}/transformer_best.pt"
            try:
                tf.load_state_dict(torch.load(weights_path, map_location=settings.yolo_device))
                logger.info("Transformer weights loaded from %s", weights_path)
            except FileNotFoundError:
                logger.warning("Transformer weights not found at %s — using random init (dev mode)", weights_path)
            self.behavioral_detector = BehavioralAnomalyDetector(
                tf, seq_len=settings.seq_len,
                threshold=settings.transformer_threshold,
                device=settings.yolo_device,
            )
        except Exception as e:
            self.load_errors["behavioral_detector"] = str(e)
            logger.error("Behavioral detector failed to load: %s\n%s", e, traceback.format_exc())

        # ── VLM query engine ───────────────────────────────────────────────────
        try:
            from vlm.qwen_client import QwenVLClient
            from vlm.nl_query_engine import NLQueryEngine
            self.vlm_client = QwenVLClient(
                mode=settings.vlm_mode,
                model_name=settings.vlm_model,
                api_base=settings.vlm_api_base,
                api_key=settings.vlm_api_key,
            )
            self.nl_engine = NLQueryEngine(self.vlm_client)
            logger.info("VLM query engine ready (mode=%s)", settings.vlm_mode)
        except Exception as e:
            self.load_errors["vlm_engine"] = str(e)
            logger.warning("VLM not loaded: %s", e)

        # ── Clip buffer ────────────────────────────────────────────────────────
        try:
            from mlops.clip_saver import ClipBuffer
            # Only pass S3 bucket if it's been explicitly configured
            s3 = settings.s3_bucket if settings.s3_bucket not in ("", "vortexvision-data") else None
            self.clip_buffer = ClipBuffer(
                storage_path=settings.clip_storage_path,
                s3_bucket=s3,
            )
            logger.info("Clip buffer ready → %s", settings.clip_storage_path)
        except Exception as e:
            self.load_errors["clip_buffer"] = str(e)
            logger.error("Clip buffer failed to initialize: %s", e)

        loaded = [k for k in ("detector", "anomaly_scorer", "behavioral_detector", "vlm_engine", "clip_buffer")
                  if k not in self.load_errors]
        logger.info("Model loading complete. Ready: %s | Errors: %s",
                    loaded, list(self.load_errors.keys()))

    async def shutdown(self):
        logger.info("AppState shutting down")

    def push_frame(self, stream_id: str, frame_bytes: bytes):
        self._latest_frames[stream_id] = frame_bytes

    def get_latest_frame(self, stream_id: str) -> Optional[bytes]:
        return self._latest_frames.get(stream_id)

    def push_event(self, event: dict):
        self._events.append(event)
        try:
            self._alert_queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop oldest to make room
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
