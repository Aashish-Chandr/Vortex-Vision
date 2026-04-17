"""
Data and prediction drift monitoring using Evidently AI.
Triggers retraining pipeline when drift is detected.
"""
import logging
import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping

logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Compares reference (training) distribution against current production data.
    Triggers retraining if drift score exceeds threshold.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        drift_threshold: float = 0.3,
        retrain_callback=None,
    ):
        self.reference = reference_data
        self.drift_threshold = drift_threshold
        self.retrain_callback = retrain_callback

    def check(self, current_data: pd.DataFrame) -> dict:
        report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
        column_mapping = ColumnMapping(
            target="label",
            prediction="prediction",
        )

        report.run(
            reference_data=self.reference,
            current_data=current_data,
            column_mapping=column_mapping,
        )

        result = report.as_dict()
        drift_score = result["metrics"][0]["result"].get("dataset_drift_share", 0.0)

        logger.info("Drift score: %.3f (threshold: %.3f)", drift_score, self.drift_threshold)

        if drift_score > self.drift_threshold:
            logger.warning("Drift detected! Score %.3f > threshold %.3f. Triggering retraining.", drift_score, self.drift_threshold)
            if self.retrain_callback:
                self.retrain_callback(drift_score=drift_score)

        return {"drift_score": drift_score, "drift_detected": drift_score > self.drift_threshold}


def trigger_retraining(drift_score: float):
    """Trigger Kubeflow / DVC retraining pipeline."""
    import subprocess
    logger.info("Triggering retraining pipeline (drift_score=%.3f)...", drift_score)
    subprocess.run(["dvc", "repro", "train_autoencoder"], check=True)
