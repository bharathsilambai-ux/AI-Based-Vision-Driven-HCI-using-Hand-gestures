"""
Phase 6: Real-Time Machine Learning Gesture Predictor Subsystem
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Production Model Ingestion (Module 3 & Module 4):
   - Loads serialized model bundle containing winning classifier, fitted StandardScaler,
     and bijective label encoders.
   - Provides calibrated confidence scoring (0.0 to 1.0) via posterior probability distributions.

2. Flexible Input Formats:
   - Accepts 92-D NumPy feature vectors directly.
   - Accepts raw HandData / HandDetectionResult instances, automatically applying Phase 4
     LandmarkProcessor for end-to-end inference.

3. Top-K Multi-Class Candidate Ranking:
   - Evaluates probability distribution across all 14 gesture classes.
   - Returns top-K candidates for confidence gating and ambiguous gesture disambiguation.

4. Graceful Error Handling & Fallbacks (Module 17):
   - Handles missing model files, corrupted inputs, zero arrays, and NaN/Inf vectors safely.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset import GESTURE_CLASSES, ID_TO_LABEL, LABEL_TO_ID
from features.landmark_processor import LandmarkProcessor
from utils.logger import logger

DEFAULT_MODEL_PATH = os.path.join(_ROOT, "models", "trained_model.pkl")


class GesturePredictor:
    """
    Production runtime predictor for classifying 21-landmark hand gestures
    using trained Scikit-Learn models with probability estimates.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        processor: Optional[LandmarkProcessor] = None
    ):
        self.model_path = model_path
        self.processor = processor if processor is not None else LandmarkProcessor()

        # Model bundle components
        self.classifier: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.classes: List[str] = GESTURE_CLASSES.copy()
        self.label_to_id: Dict[str, int] = LABEL_TO_ID.copy()
        self.id_to_label: Dict[int, str] = ID_TO_LABEL.copy()
        self.model_name: str = "Uninitialized"
        self.metrics: Dict[str, Any] = {}
        self.is_loaded: bool = False

        self.load_model()

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Loads trained model bundle from disk."""
        target_path = model_path or self.model_path
        if not os.path.exists(target_path):
            logger.warning(f"[GesturePredictor] Model file not found at: {target_path}")
            self.is_loaded = False
            return False

        try:
            bundle = joblib.load(target_path)
            self.classifier = bundle.get("classifier")
            self.scaler = bundle.get("scaler")
            self.classes = bundle.get("classes", GESTURE_CLASSES)
            self.label_to_id = bundle.get("label_to_id", LABEL_TO_ID)
            self.id_to_label = bundle.get("id_to_label", ID_TO_LABEL)
            self.model_name = bundle.get("model_name", "Unknown")
            self.metrics = bundle.get("metrics", {})
            self.is_loaded = self.classifier is not None
            logger.info(f"[GesturePredictor] Successfully loaded model '{self.model_name}' from: {target_path}")
            return True
        except Exception as e:
            logger.error(f"[GesturePredictor] Failed to load model bundle: {e}")
            self.is_loaded = False
            return False

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        """
        Predicts gesture class and calibrated confidence score from a 92-D feature vector.
        
        Args:
            feature_vector: 1D NumPy array with 92 elements.
            
        Returns:
            Tuple of (predicted_gesture_name: str, confidence: float in [0.0, 1.0]).
        """
        if not self.is_loaded or self.classifier is None:
            return "NONE", 0.0

        if feature_vector is None or len(feature_vector) == 0:
            return "NONE", 0.0

        # Validate and format input
        feat = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        if feat.shape[1] != 92:
            logger.warning(f"[GesturePredictor] Input feature dim is {feat.shape[1]}, expected 92.")
            return "NONE", 0.0

        if np.isnan(feat).any() or np.isinf(feat).any():
            return "NONE", 0.0

        # Apply StandardScaler if available
        if self.scaler is not None:
            try:
                feat = self.scaler.transform(feat)
            except Exception as e:
                logger.error(f"[GesturePredictor] Scaling error: {e}")

        # Compute prediction and probabilities
        try:
            if hasattr(self.classifier, "predict_proba"):
                probs = self.classifier.predict_proba(feat)[0]
                best_idx = int(np.argmax(probs))
                confidence = float(probs[best_idx])
            else:
                pred_val = self.classifier.predict(feat)[0]
                best_idx = int(pred_val)
                confidence = 0.95

            # Map index to class label
            gesture_name = self.id_to_label.get(best_idx, "UNKNOWN")
            return gesture_name, round(confidence, 4)
        except Exception as e:
            logger.error(f"[GesturePredictor] Inference error: {e}")
            return "NONE", 0.0

    def predict_top_k(self, feature_vector: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """
        Returns top-K candidate predictions sorted by descending confidence.
        
        Args:
            feature_vector: 92-D feature vector.
            k: Number of top candidates to return.
            
        Returns:
            List of (gesture_name, confidence) tuples.
        """
        if not self.is_loaded or self.classifier is None:
            return [("NONE", 0.0)]

        feat = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        if self.scaler is not None:
            feat = self.scaler.transform(feat)

        if not hasattr(self.classifier, "predict_proba"):
            top_name, top_conf = self.predict(feature_vector)
            return [(top_name, top_conf)]

        probs = self.classifier.predict_proba(feat)[0]
        top_indices = np.argsort(probs)[::-1][:k]

        candidates: List[Tuple[str, float]] = []
        for idx in top_indices:
            label = self.id_to_label.get(int(idx), f"CLASS_{idx}")
            conf = float(probs[idx])
            candidates.append((label, round(conf, 4)))

        return candidates

    def predict_from_landmarks(self, hand: Any) -> Tuple[str, float]:
        """
        Convenience method: Extracts 92-D features from a hand object (HandData/Landmarks)
        and predicts the gesture class in a single call.
        """
        features = self.processor.extract_single_hand(hand)
        ml_vector = features.to_ml_vector()
        return self.predict(ml_vector)

    def get_model_info(self) -> Dict[str, Any]:
        """Returns structured metadata about the loaded classifier."""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
            "classes_count": len(self.classes),
            "classes": self.classes,
            "metrics": self.metrics
        }
