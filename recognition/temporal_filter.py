"""
Temporal Filtering and False-Action Prevention Module.
Applies confidence gating, multi-frame debouncing, gesture cooldown timers,
double-pinch detection, and coordinate EMA smoothing.
"""

from collections import defaultdict
import os
import sys
import time
from typing import Dict, Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from utils.logger import logger

try:
    from .gesture_rules import GestureID
except ImportError:
    from gesture_rules import GestureID


class FilteredGesture:
    """Represents a validated, debounced gesture ready for action dispatch."""

    def __init__(
        self,
        gesture_id: str,
        confidence: float,
        is_discrete_trigger: bool = False,
        is_double_pinch: bool = False,
        smoothed_point: Optional[Tuple[int, int]] = None,
        delta_x: float = 0.0,
        delta_y: float = 0.0
    ):
        self.gesture_id = gesture_id
        self.confidence = confidence
        self.is_discrete_trigger = is_discrete_trigger
        self.is_double_pinch = is_double_pinch
        self.smoothed_point = smoothed_point
        self.delta_x = delta_x
        self.delta_y = delta_y


class TemporalFilter:
    """
    Robust stability state machine preventing false triggers and accidental actions.
    """

    def __init__(
        self,
        debounce_frames: int = config.GESTURE_DEBOUNCE_FRAMES,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD_DEFAULT,
        double_pinch_window: float = config.DOUBLE_PINCH_WINDOW
    ):
        self.debounce_frames = debounce_frames
        self.confidence_threshold = confidence_threshold
        self.double_pinch_window = double_pinch_window

        # Debounce state
        self.candidate_gesture = GestureID.IDLE
        self.candidate_count = 0
        self.confirmed_gesture = GestureID.IDLE

        # Action cooldown timestamps: { gesture_id: last_trigger_timestamp }
        self._cooldowns: Dict[str, float] = defaultdict(float)

        # Pinch click timing state (for single vs double pinch)
        self.last_pinch_release_time = 0.0
        self.last_pinch_trigger_time = 0.0
        self.is_currently_pinching = False

        # Continuous coordinate EMA state
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None

    def set_parameters(
        self,
        debounce_frames: Optional[int] = None,
        confidence_threshold: Optional[float] = None
    ):
        if debounce_frames is not None:
            self.debounce_frames = max(1, min(10, debounce_frames))
        if confidence_threshold is not None:
            self.confidence_threshold = max(0.4, min(0.99, confidence_threshold))

    def reset(self):
        """Clears temporal tracking states."""
        self.candidate_gesture = GestureID.IDLE
        self.candidate_count = 0
        self.confirmed_gesture = GestureID.IDLE
        self.is_currently_pinching = False
        self.smooth_x = None
        self.smooth_y = None

    def update(
        self,
        raw_gesture: str,
        confidence: float,
        raw_point: Optional[Tuple[int, int]] = None,
        smoothing_alpha: float = config.SMOOTHING_ALPHA_DEFAULT
    ) -> FilteredGesture:
        """
        Executes the False-Action Prevention pipeline:
        1. Confidence gating
        2. Consecutive-frame confirmation (debouncing)
        3. Double-pinch evaluation
        4. Cooldown verification
        5. Coordinate smoothing
        """
        now = time.time()

        # 1. Confidence Check
        if confidence < self.confidence_threshold:
            # Drop low-confidence detections into IDLE
            raw_gesture = GestureID.IDLE

        # 2. Consecutive-Frame Debounce Filter
        if raw_gesture == self.candidate_gesture:
            self.candidate_count += 1
        else:
            self.candidate_gesture = raw_gesture
            self.candidate_count = 1

        # Immediate bypass for urgent stops (Open Palm)
        required = 1 if raw_gesture == GestureID.OPEN_PALM else self.debounce_frames
        if self.candidate_count >= required:
            self.confirmed_gesture = self.candidate_gesture

        # 3. Coordinate Smoothing (for cursor / pointing)
        smoothed_pt = None
        delta_x, delta_y = 0.0, 0.0

        if raw_point is not None:
            rx, ry = raw_point
            if self.smooth_x is None or self.smooth_y is None:
                self.smooth_x = float(rx)
                self.smooth_y = float(ry)
            else:
                prev_x, prev_y = self.smooth_x, self.smooth_y
                self.smooth_x = smoothing_alpha * float(rx) + (1.0 - smoothing_alpha) * self.smooth_x
                self.smooth_y = smoothing_alpha * float(ry) + (1.0 - smoothing_alpha) * self.smooth_y
                delta_x = self.smooth_x - prev_x
                delta_y = self.smooth_y - prev_y
            smoothed_pt = (int(self.smooth_x), int(self.smooth_y))

        # 4. Discrete Trigger & Cooldown Check
        is_discrete = False
        is_double_pinch = False

        # Evaluate Pinch Trigger
        if self.confirmed_gesture == GestureID.PINCH_INDEX:
            if not self.is_currently_pinching:
                # Check for double pinch within window
                if (now - self.last_pinch_trigger_time) <= self.double_pinch_window:
                    is_double_pinch = True
                    is_discrete = True
                    self.last_pinch_trigger_time = 0.0  # Reset
                else:
                    # Single click trigger
                    if (now - self._cooldowns[GestureID.PINCH_INDEX]) >= config.COOLDOWN_CLICK:
                        is_discrete = True
                        self._cooldowns[GestureID.PINCH_INDEX] = now
                        self.last_pinch_trigger_time = now
                self.is_currently_pinching = True
        else:
            if self.is_currently_pinching:
                self.last_pinch_release_time = now
            self.is_currently_pinching = False

        # Evaluate other discrete gestures
        discrete_cooldowns = {
            GestureID.PINCH_MIDDLE: config.COOLDOWN_CLICK,
            GestureID.PINCH_RING: config.COOLDOWN_SHORTCUT,
            GestureID.PINCH_PINKY: config.COOLDOWN_SHORTCUT,
            GestureID.PEACE_SIGN: config.COOLDOWN_SHORTCUT,
            GestureID.THREE_FINGERS: config.COOLDOWN_SHORTCUT,
            GestureID.FOUR_FINGERS: config.COOLDOWN_SHORTCUT,
            GestureID.FIVE_FINGERS: config.COOLDOWN_SHORTCUT,
            GestureID.FINGER_GUN: config.COOLDOWN_SHORTCUT,
            GestureID.ROCK_SIGN: config.COOLDOWN_SHORTCUT,
            GestureID.SWIPE_LEFT: config.COOLDOWN_SWIPE,
            GestureID.SWIPE_RIGHT: config.COOLDOWN_SWIPE,
            GestureID.SWIPE_UP: config.COOLDOWN_SWIPE,
            GestureID.SWIPE_DOWN: config.COOLDOWN_SWIPE,
            GestureID.TWO_HAND_ROTATE: config.COOLDOWN_SHORTCUT,
        }

        if self.confirmed_gesture in discrete_cooldowns:
            cd = discrete_cooldowns[self.confirmed_gesture]
            if (now - self._cooldowns[self.confirmed_gesture]) >= cd:
                is_discrete = True
                self._cooldowns[self.confirmed_gesture] = now

        return FilteredGesture(
            gesture_id=self.confirmed_gesture,
            confidence=confidence,
            is_discrete_trigger=is_discrete,
            is_double_pinch=is_double_pinch,
            smoothed_point=smoothed_pt,
            delta_x=delta_x,
            delta_y=delta_y
        )
