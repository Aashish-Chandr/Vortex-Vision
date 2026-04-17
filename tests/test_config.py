"""Unit tests for configuration and settings."""
import os
import pytest
from unittest.mock import patch


def test_settings_defaults():
    from config.settings import Settings
    s = Settings()
    assert s.app_name == "VortexVision"
    assert s.fps_limit == 30
    assert s.yolo_conf == 0.4
    assert s.jwt_expire_minutes == 60


def test_settings_env_override():
    with patch.dict(os.environ, {"FPS_LIMIT": "15", "LOG_LEVEL": "DEBUG"}):
        from config.settings import Settings
        s = Settings()
        assert s.fps_limit == 15
        assert s.log_level == "DEBUG"


def test_get_settings_cached():
    from config.settings import get_settings
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2  # lru_cache returns same instance
