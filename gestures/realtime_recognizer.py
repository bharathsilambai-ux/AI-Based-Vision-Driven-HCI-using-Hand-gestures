"""
Phase 7 & 8: Real-Time Hand Gesture Recognition Pipeline & Stabilization Engine
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. End-to-End Live Video & Landmark Ingestion (Phase 7):
   - Accepts raw HandData / landmark arrays from HandDetector (Phase 3).
   - Automatically processes 21 coordinates into canonical 92-D invariant feature vectors
     via LandmarkProcessor (Phase 4).
   - Runs high-throughput ML inference via GesturePredictor (Phase 6, SVM_RBF).

2. Dynamic & Static Motion Fusion:
   - Simultaneously evaluates static poses (StaticGestureRecognizer) and continuous motion trajectories
     (DynamicGestureRecognizer) to support the full 14-gesture vocabulary.

3. Confidence-Based Decision Gating (Module 4):
   - Configurable confidence threshold (CONFIDENCE_THRESHOLD = 0.70).
   - Strictly suppresses command execution when confidence falls below the threshold,
     logging "LOW CONFIDENCE — COMMAND IGNORED".

4. Multi-Frame Temporal Stabilization (Module 5):
   - Deque buffer storing recent temporal predictions.
   - Frequency-based majority voting filter eliminates single-frame misclassifications.
   - Configurable minimum stable frame requirement (e.g. 3-5 frames) before gesture confirmation.
   - Smooth gesture transition debounce preventing command jitter.

5. Intelligent Command Cooldown (Module 6):
   - Per-action and per-gesture cooldown timers.
   - Distinguishes continuous actions (e.g. CURSOR_MOVE, DRAG_DROP) from discrete triggers (e.g. LEFT_CLICK, SCREENSHOT).
   - Prevents repeated runaway command firing.

6. Operational Mode & Application Context Mapping (Module 7 & 13):
   - Integrates GestureMappingManager to dispatch mode-aware commands.
"""

import collections
import os
import sys
import time
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset import GESTURE_CLASSES
from features.landmark_processor import HandFeatures, LandmarkProcessor
from gestures.dynamic_gestures import DynamicGestureRecognizer, DynamicGestureResult
from gestures.gesture_mapping import GestureMappingManager, OperationalMode
from gestures.static_gestures import StaticGestureRecognizer, StaticGestureResult
from models.predict import DEFAULT_MODEL_PATH, GesturePredictor
from utils.logger import logger

# Canonical Cooldown Configurations in Seconds (Module 6)
DEFAULT_COOLDOWNS: Dict[str, float] = {
    "LEFT_CLICK": 0.25,
    "RIGHT_CLICK": 0.35,
    "DOUBLE_CLICK": 0.45,
    "SCREENSHOT": 1.20,
    "LOCK_SCREEN": 1.50,
    "CLOSE_TAB": 0.80,
    "REFRESH_PAGE": 0.80,
    "NEW_TAB": 0.60,
    "VOLUME_UP": 0.20,
    "VOLUME_DOWN": 0.20,
    "MUTE": 0.50,
    "PLAY_PAUSE": 0.40,
    "NEXT_TRACK": 0.40,
    "PREV_TRACK": 0.40,
    "NEXT_SLIDE": 0.40,
    "PREV_SLIDE": 0.40,
    "START_PRESENTATION": 1.00,
    "EXIT_PRESENTATION": 1.00,
    "SWIPE": 0.40,
    "DEFAULT_DISCRETE": 0.35,
}

# Continuous Actions that bypass discrete single-trigger cooldown
CONTINUOUS_ACTIONS: Set = {
    "CURSOR_MOVE",
    "HOVER",
    "DRAG_DROP",
    "SCROLL_UP",
    "SCROLL_DOWN",
    "LASER_POINTER",
}


class GestureRecognitionResult:
    """Complete structured output of the real-time gesture recognition pipeline."""

    def __init__(
        self,
        raw_gesture: str = "NONE",
        confirmed_gesture: str = "NONE",
        confidence: float = 0.0,
        is_confident: bool = False,
        is_stabilized: bool = False,
        is_dynamic: bool = False,
        active_mode: str = OperationalMode.MOUSE,
        mapped_action: Optional[str] = None,
        can_execute: bool = False,
        finger_states: Optional[List[int]] = None,
        pinch_distance: float = 1.0,
        tracking_point: Optional[Tuple[float, float]] = None,
        hand_count: int = 0,
        primary_handedness: str = "None",
        latency_ms: float = 0.0,
        status_message: str = "IDLE",
    ):
        self.raw_gesture = raw_gesture
        self.confirmed_gesture = confirmed_gesture
        self.confidence = float(confidence)
        self.is_confident = is_confident
        self.is_stabilized = is_stabilized
        self.is_dynamic = is_dynamic
        self.active_mode = active_mode
        self.mapped_action = mapped_action
        self.can_execute = can_execute
        self.finger_states = finger_states or [0, 0, 0, 0, 0]
        self.pinch_distance = float(pinch_distance)
        self.tracking_point = tracking_point
        self.hand_count = hand_count
        self.primary_handedness = primary_handedness
        self.latency_ms = float(latency_ms)
        self.status_message = status_message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_gesture": self.raw_gesture,
            "confirmed_gesture": self.confirmed_gesture,
            "confidence": round(self.confidence, 4),
            "is_confident": self.is_confident,
            "is_stabilized": self.is_stabilized,
            "is_dynamic": self.is_dynamic,
            "active_mode": self.active_mode,
            "mapped_action": self.mapped_action,
            "can_execute": self.can_execute,
            "finger_states": self.finger_states,
            "pinch_distance": round(self.pinch_distance, 4),
            "tracking_point": (
                (round(self.tracking_point[0], 2), round(self.tracking_point[1], 2))
                if self.tracking_point else None
            ),
            "hand_count": self.hand_count,
            "primary_handedness": self.primary_handedness,
            "latency_ms": round(self.latency_ms, 3),
            "status_message": self.status_message,
        }

    def __repr__(self) -> str:
        act = f" -> {self.mapped_action}" if self.mapped_action else ""
        return f"<GestureResult {self.confirmed_gesture} ({self.confidence*100:.1f}%){act} [{self.status_message}]>"


class RealtimeGestureRecognizer:
    """
    Central real-time gesture recognition engine.
    Orchestrates ML classification, temporal majority smoothing,
    confidence gating, dynamic motion analysis, and cooldown dispatch.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.70,
        temporal_window_size: int = 6,
        min_stable_frames: int = 3,
        initial_mode: str = OperationalMode.MOUSE,
        processor: Optional[LandmarkProcessor] = None,
        predictor: Optional[GesturePredictor] = None,
        mapping_manager: Optional[GestureMappingManager] = None,
    ):
        self.processor = processor if processor is not None else LandmarkProcessor()
        self.predictor = predictor if predictor is not None else GesturePredictor(model_path=model_path, processor=self.processor)
        self.static_recognizer = StaticGestureRecognizer(
            model_path=model_path,
            predictor=self.predictor,
            processor=self.processor,
            min_confidence=confidence_threshold * 0.7,
        )
        self.dynamic_recognizer = DynamicGestureRecognizer()
        self.mapping_manager = mapping_manager if mapping_manager is not None else GestureMappingManager(initial_mode=initial_mode)

        # Confidence & Stabilization Parameters (Modules 4 & 5)
        self.confidence_threshold = confidence_threshold
        self.temporal_window_size = temporal_window_size
        self.min_stable_frames = min_stable_frames

        # Temporal Rolling History Buffer: Deque of (gesture_name, confidence)
        self._history: Deque[Tuple[str, float]] = collections.deque(maxlen=temporal_window_size)
        self._last_confirmed_gesture: str = "NONE"

        # Command Cooldown Timers (Module 6)
        self.cooldown_durations: Dict[str, float] = dict(DEFAULT_COOLDOWNS)
        self._action_cooldown_timers: Dict[str, float] = {}
        self._last_command_time: float = 0.0
        self._last_executed_command: Optional[str] = None

    def reset(self):
        """Flushes history and resets state."""
        self._history.clear()
        self.dynamic_recognizer.reset()
        self._last_confirmed_gesture = "NONE"
        self._action_cooldown_timers.clear()
        self._last_command_time = 0.0
        self._last_executed_command = None

    def set_confidence_threshold(self, threshold: float):
        """Sets the active confidence threshold."""
        self.confidence_threshold = max(0.01, min(0.99, float(threshold)))
        self.static_recognizer.min_confidence = self.confidence_threshold * 0.7

    def set_mode(self, mode: str) -> bool:
        """Changes the current operating mode."""
        return self.mapping_manager.set_mode(mode)

    def get_mode(self) -> str:
        """Returns the active operating mode."""
        return self.mapping_manager.get_mode()

    def process_frame(
        self,
        hand_detection_result: Any,
        timestamp: Optional[float] = None,
    ) -> GestureRecognitionResult:
        """
        Processes a single frame's hand detection data through the complete pipeline.

        Args:
            hand_detection_result: HandDetectionResult or list of HandData or raw (21,3) NumPy array.
            timestamp: Optional frame timestamp.

        Returns:
            GestureRecognitionResult with confirmed gesture, action, confidence, and status.
        """
        t0 = time.perf_counter()
        now = timestamp if timestamp is not None else t0

        # 1. Parse Hand Detection Data (Module 17: Error Handling)
        primary_hand = None
        hand_count = 0
        handedness = "None"
        raw_landmarks: Optional[np.ndarray] = None
        tracking_point: Optional[Tuple[float, float]] = None

        if hand_detection_result is None:
            self._history.append(("NONE", 0.0))
            self.dynamic_recognizer.reset()
            return GestureRecognitionResult(
                status_message="NO HAND DETECTED",
                active_mode=self.mapping_manager.get_mode(),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Handle various input formats
        if hasattr(hand_detection_result, "hands") and isinstance(hand_detection_result.hands, list):
            hand_count = len(hand_detection_result.hands)
            if hand_count > 0:
                primary_hand = hand_detection_result.hands[0]
                handedness = getattr(primary_hand, "handedness", "Right")
                raw_landmarks = getattr(primary_hand, "landmarks", None)
        elif isinstance(hand_detection_result, list):
            hand_count = len(hand_detection_result)
            if hand_count > 0:
                primary_hand = hand_detection_result[0]
                handedness = getattr(primary_hand, "handedness", "Right")
                raw_landmarks = getattr(primary_hand, "landmarks", None)
        elif isinstance(hand_detection_result, np.ndarray) and hand_detection_result.shape == (21, 3):
            hand_count = 1
            primary_hand = hand_detection_result
            handedness = "Right"
            raw_landmarks = hand_detection_result
        elif hasattr(hand_detection_result, "landmarks"):
            hand_count = 1
            primary_hand = hand_detection_result
            handedness = getattr(primary_hand, "handedness", "Right")
            raw_landmarks = getattr(primary_hand, "landmarks", None)

        if primary_hand is None or raw_landmarks is None:
            self._history.append(("NONE", 0.0))
            self.dynamic_recognizer.reset()
            return GestureRecognitionResult(
                status_message="NO HAND DETECTED",
                hand_count=hand_count,
                active_mode=self.mapping_manager.get_mode(),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Tracking point: Landmark 8 (Index tip) or Landmark 9 (Middle MCP)
        if len(raw_landmarks) > 8:
            tracking_point = (float(raw_landmarks[8][0]), float(raw_landmarks[8][1]))

        # 2. Run Static Gesture Recognition (ML Predictor)
        static_res: StaticGestureResult = self.static_recognizer.classify_static(
            hand_input=primary_hand,
            handedness=handedness,
        )

        is_pinching = (static_res.gesture == "PINCH")

        # 3. Run Dynamic Motion Recognition (Trajectory Tracker)
        dynamic_res: DynamicGestureResult = self.dynamic_recognizer.update(
            tracking_point=tracking_point,
            is_pinching=is_pinching,
            timestamp=now,
        )

        # 4. Arbitration: Prioritize Dynamic Gestures if Triggered
        is_dynamic = False
        if dynamic_res.is_dynamic and dynamic_res.gesture != "NONE":
            candidate_gesture = dynamic_res.gesture
            candidate_confidence = dynamic_res.confidence
            is_dynamic = True
        else:
            candidate_gesture = static_res.gesture
            candidate_confidence = static_res.confidence

        # 5. Temporal Stabilization (Module 5)
        # Append candidate to rolling history
        self._history.append((candidate_gesture, candidate_confidence))

        # Majority Voting across history window
        history_labels = [g for g, _ in self._history]
        counts = collections.Counter(history_labels)
        dominant_gesture, count = counts.most_common(1)[0]

        # Calculate average confidence for the dominant gesture
        matching_confs = [c for g, c in self._history if g == dominant_gesture]
        avg_confidence = float(np.mean(matching_confs)) if matching_confs else 0.0

        is_stabilized = (count >= self.min_stable_frames) and (dominant_gesture != "NONE")
        confirmed_gesture = dominant_gesture if is_stabilized else "NONE"

        # 6. Confidence Gating (Module 4)
        is_confident = (avg_confidence >= self.confidence_threshold)
        if not is_confident and confirmed_gesture != "NONE":
            status_msg = f"LOW CONFIDENCE ({avg_confidence*100:.1f}% < {self.confidence_threshold*100:.0f}%) — COMMAND IGNORED"
            confirmed_gesture = "NONE"
        elif confirmed_gesture == "NONE":
            status_msg = "STABILIZING" if dominant_gesture != "NONE" else "IDLE"
        else:
            status_msg = f"CONFIRMED: {confirmed_gesture} ({avg_confidence*100:.1f}%)"

        # 7. Operational Mode Mapping (Module 7)
        mapped_action = None
        if confirmed_gesture != "NONE":
            mapped_action = self.mapping_manager.get_action(confirmed_gesture)

        # 8. Command Cooldown Evaluation (Module 6)
        can_execute = False
        if mapped_action is not None and is_confident and is_stabilized:
            can_execute = self._check_cooldown(mapped_action, now)

            if can_execute:
                self._record_execution(mapped_action, now)
                status_msg = f"EXECUTING: {mapped_action} [{confirmed_gesture}]"
            elif mapped_action not in CONTINUOUS_ACTIONS:
                status_msg = f"COOLDOWN ACTIVE: {mapped_action}"

        total_latency = (time.perf_counter() - t0) * 1000.0

        return GestureRecognitionResult(
            raw_gesture=candidate_gesture,
            confirmed_gesture=confirmed_gesture,
            confidence=avg_confidence,
            is_confident=is_confident,
            is_stabilized=is_stabilized,
            is_dynamic=is_dynamic,
            active_mode=self.mapping_manager.get_mode(),
            mapped_action=mapped_action,
            can_execute=can_execute,
            finger_states=static_res.finger_states,
            pinch_distance=static_res.pinch_distance,
            tracking_point=tracking_point,
            hand_count=hand_count,
            primary_handedness=handedness,
            latency_ms=total_latency,
            status_message=status_msg,
        )

    def _check_cooldown(self, action: str, now: float) -> bool:
        """Determines if the action is cleared to execute based on its cooldown timer."""
        # Continuous actions (e.g. mouse cursor tracking) bypass discrete cooldowns
        if action in CONTINUOUS_ACTIONS:
            return True

        last_time = self._action_cooldown_timers.get(action, 0.0)
        cooldown_needed = self.cooldown_durations.get(action, self.cooldown_durations.get("DEFAULT_DISCRETE", 0.35))

        return (now - last_time) >= cooldown_needed

    def _record_execution(self, action: str, now: float):
        """Records timestamp of executed command."""
        self._action_cooldown_timers[action] = now
        self._last_command_time = now
        self._last_executed_command = action
