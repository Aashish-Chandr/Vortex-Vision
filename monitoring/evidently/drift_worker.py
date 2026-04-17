"""
Background drift monitoring worker.
Runs periodically, compares recent production predictions against reference data,
and triggers retraining via GitHub Actions webhook if drift is detected.
"""
import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REFERENCE_PATH = os.getenv("DRIFT_REFERENCE_PATH", "data/processed/reference_stats.csv")
_CHECK_INTERVAL_SECONDS = int(os.getenv("DRIFT_CHECK_INTERVAL", "3600"))  # 1 hour
_DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
_RETRAIN_WEBHOOK = os.getenv("RETRAIN_WEBHOOK_URL", "")


def _trigger_retrain_webhook(drift_score: float):
    """POST to GitHub Actions workflow_dispatch to trigger retraining."""
    if not _RETRAIN_WEBHOOK:
        logger.info("No RETRAIN_WEBHOOK_URL set — skipping retrain trigger")
        return
    try:
        import requests
        resp = requests.post(
            _RETRAIN_WEBHOOK,
            json={"ref": "main", "inputs": {"drift_score": str(round(drift_score, 4))}},
            headers={"Authorization": f"token {os.getenv('GITHUB_TOKEN', '')}",
                     "Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
        if resp.ok:
            logger.info("Retrain webhook triggered (drift=%.3f)", drift_score)
        else:
            logger.error("Retrain webhook failed: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logger.error("Retrain webhook error: %s", e)


async def run_drift_monitor(state):
    """
    Async background task: periodically checks for data drift.
    Reads recent anomaly scores from AppState events and compares to reference.
    """
    from monitoring.evidently.drift_monitor import DriftMonitor

    logger.info("Drift monitor started (interval=%ds, threshold=%.2f)",
                _CHECK_INTERVAL_SECONDS, _DRIFT_THRESHOLD)

    # Load reference data if available
    reference_df: Optional[pd.DataFrame] = None
    if Path(_REFERENCE_PATH).exists():
        try:
            reference_df = pd.read_csv(_REFERENCE_PATH)
            logger.info("Reference data loaded: %d rows", len(reference_df))
        except Exception as e:
            logger.warning("Could not load reference data: %s", e)

    while True:
        await asyncio.sleep(_CHECK_INTERVAL_SECONDS)

        if reference_df is None:
            logger.debug("No reference data — skipping drift check")
            continue

        # Build current data from recent events
        recent_events = state.get_events(limit=1000)
        if len(recent_events) < 50:
            logger.debug("Not enough events (%d) for drift check", len(recent_events))
            continue

        try:
            current_df = pd.DataFrame([
                {
                    "autoencoder_score": e.get("autoencoder_score", 0.0),
                    "transformer_score": e.get("transformer_score", 0.0),
                    "confidence": e.get("confidence", 0.0),
                    "label": 1 if e.get("anomaly_type") != "unknown" else 0,
                    "prediction": 1,
                }
                for e in recent_events
            ])

            monitor = DriftMonitor(
                reference_data=reference_df,
                drift_threshold=_DRIFT_THRESHOLD,
                retrain_callback=_trigger_retrain_webhook,
            )
            result = monitor.check(current_df)
            logger.info("Drift check: score=%.3f detected=%s",
                        result["drift_score"], result["drift_detected"])

        except Exception as e:
            logger.error("Drift check failed: %s", e)
