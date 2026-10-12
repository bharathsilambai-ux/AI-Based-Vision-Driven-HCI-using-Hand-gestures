"""
Phase 7: Static Hand Gesture Recognition Subsystem
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. ML-Driven Static Classification (Module 3 & Module 4):
   - Integrates the selected production ML classifier (SVM_RBF) via GesturePredictor.
   - Evaluates the 10 static gesture classes:
     OPEN_PALM, FIST, THUMBS_UP, THUMBS_DOWN, POINT, PEACE, OK, PINCH, THREE_FINGERS, FOUR_FINGERS.
   - Delivers calibrated posterior confidence scores per prediction.

2. Biomechanical & Geometric Cross-Validation (Sanity Checks):
   - Computes anatomical finger extension/curl states, thumb orientation vectors,
     and pinch tip-to-tip distances to cross-validate ML predictions.
   - Prevents biomechanically impossible classifications (e.g. FIST when fingers are extended).

3. Top-K Multi-Candidate Disambiguation:
   - Returns ranked candidate distribution for downstream temporal filters.

4. Robust Input Ingestion & Error Handling (Module 17):
   - Accepts raw HandData, 21x3 landmark NumPy arrays, or 92-D feature vectors.
   - Safely handles None, empty, NaN, or out-of-bounds inputs without crashing.
"""

import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset import GESTURE_CLASSES
from features.landmark_processor import HandFeatures, LandmarkProcessor
from models.predict import DEFAULT_MODEL_PATH, GesturePredictor
from utils.logger import logger

# 10 Canonical Static Gestures
STATIC_GESTURE_CLASSES = [
    "OPEN_PALM",
    "FIST",
    "THUMBS_UP",
    "THUMBS_DOWN",
    "POINT",
    "PEACE",
    "OK",
    "PINCH",
    "THREE_FINGERS",
    "FOUR_FINGERS",
]


class StaticGestureResult:
    """Structured result returned by StaticGestureRecognizer."""

    def __init__(
        self,
        gesture: str = "NONE",
        confidence: float = 0.0,
        is_static: bool = False,
        top_k: Optional[List[Tuple[str, float]]] = None,
        finger_states: Optional[List[int]] = None,
        pinch_distance: float = 1.0,
        is_geometrically_valid: bool = True,
        latency_ms: float = 0.0,
        status: str = "IDLE",
    ):
        self.gesture = gesture
        self.confidence = float(confidence)
        self.is_static = is_static
        self.top_k = top_k or []
        self.finger_states = finger_states or [0, 0, 0, 0, 0]  # [Thumb, Index, Middle, Ring, Pinky]
        self.pinch_distance = float(pinch_distance)
        self.is_geometrically_valid = is_geometrically_valid
        self.latency_ms = float(latency_ms)
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gesture": self.gesture,
            "confidence": round(self.confidence, 4),
            "is_static": self.is_static,
            "top_k": [(g, round(c, 4)) for g, c in self.top_k],
            "finger_states": self.finger_states,
            "pinch_distance": round(self.pinch_distance, 4),
            "is_geometrically_valid": self.is_geometrically_valid,
            "latency_ms": round(self.latency_ms, 3),
            "status": self.status,
        }

    def __repr__(self) -> str:
        return f"<StaticGestureResult gesture='{self.gesture}' conf={self.confidence:.2f} status='{self.status}'>"


class StaticGestureRecognizer:
    """
    Evaluates 21-landmark hand data using the trained ML classifier,
    optionally cross-referenced with geometric biomechanics.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        predictor: Optional[GesturePredictor] = None,
        processor: Optional[LandmarkProcessor] = None,
        enable_geometric_validation: bool = True,
        min_confidence: float = 0.50,
    ):
        self.processor = processor if processor is not None else LandmarkProcessor()
        self.predictor = predictor if predictor is not None else GesturePredictor(model_path=model_path, processor=self.processor)
        self.enable_geometric_validation = enable_geometric_validation
        self.min_confidence = min_confidence

    @property
    def is_ready(self) -> bool:
        """Returns True if the underlying ML model is loaded and ready."""
        return self.predictor.is_loaded

    def classify_static(
        self,
        hand_input: Union[np.ndarray, Any, None],
        handedness: str = "Right",
        top_k_count: int = 3,
    ) -> StaticGestureResult:
        """
        Classifies static hand gesture from input landmarks or HandData.

        Args:
            hand_input: 21x3 landmarks NumPy array, HandData instance, or None.
            handedness: 'Right' or 'Left'.
            top_k_count: Number of top candidate predictions to return.

        Returns:
            StaticGestureResult containing predicted gesture, confidence, and geometry stats.
        """
        t0 = time.perf_counter()

        if hand_input is None:
            return StaticGestureResult(
                gesture="NONE",
                confidence=0.0,
                is_static=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                status="NO_HAND_DETECTED",
            )

        # Extract normalized features and raw landmark coordinates
        features: Optional[HandFeatures] = None
        raw_lms: Optional[np.ndarray] = None

        try:
            if isinstance(hand_input, HandFeatures):
                features = hand_input
                raw_lms = hand_input.normalized_landmarks
            elif isinstance(hand_input, np.ndarray):
                if hand_input.size == 0 or np.isnan(hand_input).any() or np.isinf(hand_input).any():
                    return StaticGestureResult(
                        gesture="NONE",
                        confidence=0.0,
                        is_static=False,
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        status="INVALID_INPUT",
                    )
                if hand_input.shape == (21, 3):
                    raw_lms = hand_input
                    features = self.processor.extract_single_hand(hand_input)
                elif hand_input.ndim == 1 and hand_input.shape[0] == 92:
                    # Direct 92-D feature vector passed
                    pred_label, pred_conf = self.predictor.predict(hand_input)
                    top_k = self.predictor.predict_top_k(hand_input, k=top_k_count)
                    latency = (time.perf_counter() - t0) * 1000.0
                    is_static = pred_label in STATIC_GESTURE_CLASSES
                    return StaticGestureResult(
                        gesture=pred_label,
                        confidence=pred_conf,
                        is_static=is_static,
                        top_k=top_k,
                        latency_ms=latency,
                        status="SUCCESS" if pred_conf >= self.min_confidence else "LOW_CONFIDENCE",
                    )
                else:
                    return StaticGestureResult(
                        gesture="NONE",
                        confidence=0.0,
                        is_static=False,
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        status="INVALID_SHAPE",
                    )
            else:
                # HandData or object with .landmarks attribute
                lms = getattr(hand_input, "landmarks", None)
                if lms is None or len(lms) < 21:
                    return StaticGestureResult(
                        gesture="NONE",
                        confidence=0.0,
                        is_static=False,
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        status="NO_HAND_DETECTED",
                    )
                features = self.processor.extract_single_hand(hand_input)
                raw_lms = np.asarray(lms, dtype=np.float32)
        except Exception as e:
            logger.error(f"[StaticGestureRecognizer] Error processing input: {e}")
            return StaticGestureResult(
                gesture="NONE",
                confidence=0.0,
                is_static=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                status=f"PROCESSING_ERROR: {e}",
            )

        if features is None or features.feature_vector_92d is None:
            return StaticGestureResult(
                gesture="NONE",
                confidence=0.0,
                is_static=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                status="FEATURE_EXTRACTION_FAILED",
            )

        # 1. Primary Inference via ML Predictor
        feat_vec = features.feature_vector_92d
        pred_label, pred_conf = self.predictor.predict(feat_vec)
        top_k = self.predictor.predict_top_k(feat_vec, k=top_k_count)

        # 2. Geometric analysis (if features available)
        finger_states = features.finger_states if features is not None else [0, 0, 0, 0, 0]
        pinch_dist = 1.0
        if raw_lms is not None and len(raw_lms) >= 21:
            # Thumb tip (4) to Index tip (8) distance normalized by palm reference
            p4 = raw_lms[4]
            p8 = raw_lms[8]
            p0 = raw_lms[0]
            p9 = raw_lms[9]
            d_ref = np.linalg.norm(p9 - p0)
            if d_ref > 1e-4:
                pinch_dist = float(np.linalg.norm(p8 - p4) / d_ref)
            else:
                pinch_dist = float(np.linalg.norm(p8 - p4))

        # 3. Geometric Cross-Validation
        is_geom_valid = True
        if self.enable_geometric_validation:
            is_geom_valid = self._validate_biomechanics(pred_label, finger_states, pinch_dist, raw_lms)

        # Adjust status
        status = "SUCCESS"
        if pred_conf < self.min_confidence:
            status = "LOW_CONFIDENCE"
        elif not is_geom_valid:
            status = "GEOMETRIC_MISMATCH"

        is_static = pred_label in STATIC_GESTURE_CLASSES
        latency = (time.perf_counter() - t0) * 1000.0

        return StaticGestureResult(
            gesture=pred_label,
            confidence=pred_conf,
            is_static=is_static,
            top_k=top_k,
            finger_states=finger_states,
            pinch_distance=pinch_dist,
            is_geometrically_valid=is_geom_valid,
            latency_ms=latency,
            status=status,
        )

    def _validate_biomechanics(
        self,
        gesture: str,
        finger_states: List[int],
        pinch_dist: float,
        landmarks: Optional[np.ndarray],
    ) -> bool:
        """
        Validates if the predicted static gesture is biomechanically plausible
        given the extracted finger extension states and pinch distance.
        """
        # finger_states: [Thumb, Index, Middle, Ring, Pinky], 1 = extended, 0 = curled
        num_extended = sum(finger_states)

        if gesture == "FIST":
            # In a fist, fingers must be mostly curled (at most 1 extended, e.g. thumb resting)
            return num_extended <= 1

        elif gesture == "OPEN_PALM":
            # In an open palm, at least 4 fingers must be extended
            return num_extended >= 4

        elif gesture == "POINT":
            # Index extended (index=1), Middle, Ring, Pinky curled
            index_ext = finger_states[1] == 1
            middle_curled = finger_states[2] == 0
            ring_curled = finger_states[3] == 0
            pinky_curled = finger_states[4] == 0
            return index_ext and (middle_curled and ring_curled and pinky_curled)

        elif gesture == "PEACE":
            # Index and Middle extended, Ring and Pinky curled
            index_ext = finger_states[1] == 1
            middle_ext = finger_states[2] == 1
            ring_curled = finger_states[3] == 0
            pinky_curled = finger_states[4] == 0
            return index_ext and middle_ext and ring_curled and pinky_curled

        elif gesture == "PINCH":
            # Thumb and Index tips must be close
            return pinch_dist < 0.65

        elif gesture == "OK":
            # Thumb and Index tips close, Middle, Ring, Pinky extended
            middle_ext = finger_states[2] == 1
            return pinch_dist < 0.70 and middle_ext

        elif gesture == "THREE_FINGERS":
            # Exactly 3 or 4 fingers extended
            return 2 <= num_extended <= 4

        elif gesture == "FOUR_FINGERS":
            # 4 non-thumb fingers extended
            four_ext = (
                finger_states[1] == 1 and
                finger_states[2] == 1 and
                finger_states[3] == 1 and
                finger_states[4] == 1
            )
            return four_ext or num_extended >= 4

        elif gesture == "THUMBS_UP" or gesture == "THUMBS_DOWN":
            # Thumb extended, other fingers curled
            others_curled = sum(finger_states[1:]) <= 1
            if not others_curled:
                return False
            if landmarks is not None and len(landmarks) >= 21:
                # Check thumb vector orientation (tip Y vs MCP Y)
                thumb_tip_y = landmarks[4][1]
                thumb_mcp_y = landmarks[2][1]
                if gesture == "THUMBS_UP":
                    # Y decreases upwards in image coordinates
                    return thumb_tip_y <= thumb_mcp_y + 0.10
                else:
                    return thumb_tip_y >= thumb_mcp_y - 0.10
            return True

        # For swipe / dynamic labels classified statically, return True
        return True
