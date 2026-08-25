"""Evaluation artifact writers and publication-package validators."""

from .persistence import save_role_artifacts
from .publication import validate_gate7_package, validate_gate8_package
from .shhs_zero_shot import (
    confusion_matrix,
    ensemble_probabilities,
    load_prediction_artifact,
    metrics_from_confusion,
)
from .tables import PredictionTable, load_prediction_table, save_prediction_table

__all__ = [
    "PredictionTable",
    "load_prediction_table",
    "save_prediction_table",
    "save_role_artifacts",
    "validate_gate7_package",
    "validate_gate8_package",
    "confusion_matrix",
    "ensemble_probabilities",
    "load_prediction_artifact",
    "metrics_from_confusion",
]
