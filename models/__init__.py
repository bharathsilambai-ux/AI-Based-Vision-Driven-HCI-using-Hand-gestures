"""
Models Package: Training, evaluation, comparative benchmarking, and runtime inference.
"""

from .predict import DEFAULT_MODEL_PATH, GesturePredictor
from .train_model import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_SAVE_PATH,
    DEFAULT_REPORT_PATH,
    ModelTrainer,
)

__all__ = [
    "ModelTrainer",
    "GesturePredictor",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_MODEL_SAVE_PATH",
    "DEFAULT_REPORT_PATH",
]
