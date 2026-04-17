"""
Centralized Pydantic settings — reads from environment variables / .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "VortexVision"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # Kafka
    kafka_bootstrap: str = "localhost:9092"
    kafka_frames_topic: str = "video-frames"
    kafka_annotated_topic: str = "annotated-frames"
    kafka_events_topic: str = "anomaly-events"
    kafka_consumer_group: str = "detection-group"

    # Detection
    yolo_model: str = "yolo26n.pt"
    yolo_device: str = "cuda"
    yolo_conf: float = 0.4
    yolo_iou: float = 0.5
    fps_limit: int = 30

    # Anomaly
    ae_threshold: float = 0.05
    transformer_threshold: float = 0.7
    seq_len: int = 16

    # VLM
    vlm_mode: str = "api"           # "local" | "api"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_api_base: str = "http://localhost:8080/v1"
    vlm_api_key: str = "EMPTY"
    vlm_max_tokens: int = 512

    # Storage
    clip_storage_path: str = "/data/clips"
    model_storage_path: str = "/data/models"
    s3_bucket: str = "vortexvision-data"
    s3_region: str = "us-east-1"

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment: str = "vortexvision"

    # Auth
    api_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_key_header: str = "X-API-Key"

    # Rate limiting
    rate_limit_per_minute: int = 120

    # Database
    database_url: str = "sqlite:///./vortexvision.db"

    # Redis (for rate limiting / caching)
    redis_url: str = "redis://localhost:6379"


@lru_cache
def get_settings() -> Settings:
    return Settings()
