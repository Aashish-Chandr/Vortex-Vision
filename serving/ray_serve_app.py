"""
Ray Serve deployment for distributed, high-throughput inference.
Handles multiple concurrent video streams with autoscaling.

Start with:
  serve run serving/ray_serve_app.py
"""
import logging
import numpy as np
from typing import List

import ray
from ray import serve
from fastapi import FastAPI

logger = logging.getLogger(__name__)

app = FastAPI(title="VortexVision Ray Serve")


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_num_ongoing_requests_per_replica": 4,
    },
)
@serve.ingress(app)
class DetectorDeployment:
    def __init__(self):
        from detection.detector import YOLODetector
        from config.settings import get_settings
        settings = get_settings()
        self.detector = YOLODetector(
            model_path=settings.yolo_model,
            device=settings.yolo_device,
        )
        logger.info("DetectorDeployment initialized")

    @app.post("/detect")
    async def detect(self, payload: dict) -> dict:
        import base64, cv2
        img_bytes = base64.b64decode(payload["image_b64"])
        buf = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        result = self.detector.infer(frame, stream_id=payload.get("stream_id", ""))
        return {
            "stream_id": result.stream_id,
            "timestamp": result.timestamp,
            "frame_id": result.frame_id,
            "inference_ms": result.inference_ms,
            "detections": [
                {
                    "track_id": d.track_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                }
                for d in result.detections
            ],
        }


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 2,
    },
)
class VLMDeployment:
    def __init__(self):
        from vlm.qwen_client import QwenVLClient
        from config.settings import get_settings
        settings = get_settings()
        self.client = QwenVLClient(
            mode=settings.vlm_mode,
            model_name=settings.vlm_model,
            api_base=settings.vlm_api_base,
        )
        logger.info("VLMDeployment initialized")

    async def query(self, frames_b64: List[str], question: str) -> str:
        import base64, cv2
        frames = []
        for b64 in frames_b64:
            buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
        return self.client.query_frames(frames, question)


# Deployment graph
detector_app = DetectorDeployment.bind()
vlm_app = VLMDeployment.bind()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start VortexVision Ray Serve")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--deployment", choices=["detector", "vlm", "all"], default="all")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    serve.start(http_options={"host": args.host, "port": args.port})

    if args.deployment in ("detector", "all"):
        serve.run(detector_app, name="detector", route_prefix="/")
        logger.info("DetectorDeployment running on %s:%d/detect", args.host, args.port)

    if args.deployment in ("vlm", "all"):
        serve.run(vlm_app, name="vlm", route_prefix="/vlm")
        logger.info("VLMDeployment running on %s:%d/vlm", args.host, args.port)

    logger.info("Ray Serve running. Press Ctrl+C to stop.")
    import signal, sys
    signal.signal(signal.SIGINT, lambda *_: (serve.shutdown(), sys.exit(0)))
    signal.pause()
