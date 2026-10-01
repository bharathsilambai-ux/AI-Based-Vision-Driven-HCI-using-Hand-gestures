"""
Machine Learning Gesture Inference Module
=========================================
Loads trained Scikit-learn classification models (Random Forest, SVM, KNN)
and performs real-time inference on normalized 63D landmark vectors.
"""

import os
import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import config
from src.vision.landmark_utils import LandmarkProcessor

logger = logging.getLogger(__name__)


class MLGestureClassifier:
    """Wrapper for trained scikit-learn landmark classifier."""

    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        label_encoder_path: Path = config.LABEL_ENCODER_PATH,
        min_probability: float = 0.65,
    ):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.min_probability = min_probability

        self.model = None
        self.label_encoder = None
        self.is_ready = False

        self.load_model()

    def load_model(self) -> bool:
        """Attempts to load serialized model and label encoder artifacts."""
        if not self.model_path.exists() or not self.label_encoder_path.exists():
            logger.warning(
                "Trained model not found at %s. Run 'python ml/train_model.py' to train one.",
                self.model_path
            )
            self.is_ready = False
            return False

        try:
            self.model = joblib.load(self.model_path)
            self.label_encoder = joblib.load(self.label_encoder_path)
            self.is_ready = True
            logger.info("Trained model successfully loaded from %s.", self.model_path)
            return True
        except Exception as e:
            logger.error("Error loading model artifacts: %s", str(e))
            self.is_ready = False
            return False

    def predict(self, landmarks_norm: list) -> Tuple[str, float]:
        """
        Predicts gesture class from 21 MediaPipe landmarks.
        
        Args:
            landmarks_norm: List of 21 (x, y, z) normalized landmarks.
            
        Returns:
            (gesture_label, confidence_score)
        """
        if not self.is_ready or self.model is None or self.label_encoder is None:
            return config.GESTURE_IDLE, 0.0

        try:
            # 1. Compute translation- and scale-invariant 63D vector
            features = LandmarkProcessor.normalize_landmarks(landmarks_norm)
            features = features.reshape(1, -1)  # Shape (1, 63)

            # 2. Predict probability distribution if supported
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)[0]
                max_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[max_idx])
                if confidence < self.min_probability:
                    return config.GESTURE_IDLE, confidence
                pred_label = self.label_encoder.inverse_transform([max_idx])[0]
                return str(pred_label), confidence
            else:
                pred_encoded = self.model.predict(features)[0]
                pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
                return str(pred_label), 0.90
        except Exception as e:
            logger.error("Inference exception: %s", str(e))
            return config.GESTURE_IDLE, 0.0
