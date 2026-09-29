"""
Gesture Recognition Module.
Translates detected hand landmarks into robust, stabilized gesture states:
- ONE_FINGER_UP: Index finger raised pointing up + upward motion (Scroll Up)
- ONE_FINGER_DOWN: Index finger pointing down / downward motion (Scroll Down)
- TWO_FINGERS_UP: Index & Middle raised (Scrolling Mode Ready)
- OPEN_PALM: All 5 fingers extended (Halt/Stop scrolling immediately)
- PINCH: Thumb & Index tips in close proximity (Single Click with debounce)
- FIST / IDLE: Neutral resting state
- NO_HAND: Hand absent from camera viewport

Applies Exponential Moving Average (EMA) coordinate smoothing and
temporal multi-frame debouncing to eliminate camera jitter.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from hand_detector import HandDetector, HandLandmarkData


class GestureType:
    NO_HAND = "NO_HAND"
    IDLE = "IDLE"
    OPEN_PALM = "OPEN_PALM"
    TWO_FINGERS_UP = "TWO_FINGERS_UP"
    SCROLL_UP = "SCROLL_UP"
    SCROLL_DOWN = "SCROLL_DOWN"
    PINCH_CLICK = "PINCH_CLICK"


class GestureResult:
    """Structured container for gesture detection output."""
    def __init__(
        self,
        gesture_type: str,
        display_label: str,
        status_color: str,
        delta_x: float = 0.0,
        delta_y: float = 0.0,
        smoothed_tip: Optional[Tuple[int, int]] = None,
        is_pinch: bool = False,
        pinch_distance: float = 1.0,
        is_click_event: bool = False,
        fingers_up: Optional[List[int]] = None
    ):
        self.gesture_type = gesture_type
        self.display_label = display_label
        self.status_color = status_color
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.smoothed_tip = smoothed_tip
        self.is_pinch = is_pinch
        self.pinch_distance = pinch_distance
        self.is_click_event = is_click_event
        self.fingers_up = fingers_up or [0, 0, 0, 0, 0]


class GestureRecognizer:
    """
    Stabilized state machine for gesture recognition with EMA smoothing
    and frame-based debouncing.
    """

    def __init__(
        self,
        smoothing_alpha: float = config.SMOOTHING_ALPHA_DEFAULT,
        debounce_frames: int = config.GESTURE_DEBOUNCE_FRAMES,
        pinch_threshold: float = config.PINCH_THRESHOLD_DEFAULT,
        pinch_cooldown: float = config.PINCH_CLICK_COOLDOWN
    ):
        self.smoothing_alpha = smoothing_alpha
        self.debounce_frames = debounce_frames
        self.pinch_threshold = pinch_threshold
        self.pinch_cooldown = pinch_cooldown

        # Smoothing state
        self.smoothed_x: Optional[float] = None
        self.smoothed_y: Optional[float] = None
        self.prev_smoothed_x: Optional[float] = None
        self.prev_smoothed_y: Optional[float] = None

        # Debouncing state
        self.candidate_gesture = GestureType.NO_HAND
        self.candidate_count = 0
        self.active_gesture = GestureType.NO_HAND

        # Pinch click timing state
        self.last_click_time = 0.0
        self.is_pinching = False

        # State tracking
        self.scrolling_armed = False

    def reset(self):
        """Resets all internal temporal tracking states."""
        self.smoothed_x = None
        self.smoothed_y = None
        self.prev_smoothed_x = None
        self.prev_smoothed_y = None
        self.candidate_gesture = GestureType.NO_HAND
        self.candidate_count = 0
        self.active_gesture = GestureType.NO_HAND
        self.scrolling_armed = False
        self.is_pinching = False

    def update_parameters(self, smoothing_alpha: float = None, pinch_threshold: float = None):
        """Allows real-time tuning from the GUI sliders."""
        if smoothing_alpha is not None:
            self.smoothing_alpha = max(0.1, min(1.0, smoothing_alpha))
        if pinch_threshold is not None:
            self.pinch_threshold = max(0.01, min(0.2, pinch_threshold))

    def process(
        self,
        hand: Optional[HandLandmarkData],
        deadzone_pixels: float = config.DEADZONE_PIXELS_DEFAULT
    ) -> GestureResult:
        """
        Analyzes hand landmarks, updates smoothing filters, and determines gesture state.
        """
        # Case 1: No hand detected in frame
        if hand is None:
            self.reset()
            return GestureResult(
                gesture_type=GestureType.NO_HAND,
                display_label="No Hand Detected",
                status_color=config.COLOR_DANGER
            )

        # 1. Determine finger positions
        fingers = HandDetector.count_fingers_up(hand)
        thumb, index, middle, ring, pinky = fingers
        num_fingers_up = sum(fingers)

        # 2. Extract landmark positions for Index Finger Tip (id 8) and Thumb Tip (id 4)
        raw_idx_x, raw_idx_y = hand.get_px(8)
        norm_idx = hand.get_norm(8)
        norm_thumb = hand.get_norm(4)
        norm_idx_mcp = hand.get_norm(5)

        # 3. Calculate Pinch Distance (Thumb Tip vs Index Tip)
        pinch_dist = math.hypot(norm_thumb[0] - norm_idx[0], norm_thumb[1] - norm_idx[1])
        pinch_detected = pinch_dist < self.pinch_threshold

        # 4. Apply Exponential Moving Average (EMA) smoothing to Index Tip
        if self.smoothed_x is None or self.smoothed_y is None:
            self.smoothed_x = float(raw_idx_x)
            self.smoothed_y = float(raw_idx_y)
            self.prev_smoothed_x = float(raw_idx_x)
            self.prev_smoothed_y = float(raw_idx_y)
            delta_x = 0.0
            delta_y = 0.0
        else:
            self.prev_smoothed_x = self.smoothed_x
            self.prev_smoothed_y = self.smoothed_y

            # EMA formula: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
            alpha = self.smoothing_alpha
            self.smoothed_x = alpha * float(raw_idx_x) + (1.0 - alpha) * self.smoothed_x
            self.smoothed_y = alpha * float(raw_idx_y) + (1.0 - alpha) * self.smoothed_y

            # Calculate raw displacement
            raw_dx = self.smoothed_x - self.prev_smoothed_x
            raw_dy = self.smoothed_y - self.prev_smoothed_y

            # Apply deadzone filter to remove resting micro-tremors
            delta_x = 0.0 if abs(raw_dx) < deadzone_pixels else raw_dx
            delta_y = 0.0 if abs(raw_dy) < deadzone_pixels else raw_dy

        smoothed_point = (int(self.smoothed_x), int(self.smoothed_y))

        # 5. Check Index Finger Explicit Pointing Direction
        # In image coords: lower Y = pointing UP, higher Y = pointing DOWN
        index_pointing_up = (norm_idx[1] < norm_idx_mcp[1] - config.FINGER_POINTING_THRESHOLD)
        index_pointing_down = (norm_idx[1] > norm_idx_mcp[1] + config.FINGER_POINTING_THRESHOLD)

        # 6. Raw Gesture Classification Logic
        detected_raw = GestureType.IDLE

        # Priority A: Open Palm (All 5 fingers extended or 4 fingers with thumb relaxed)
        if num_fingers_up >= 4 and index and middle and ring and pinky:
            detected_raw = GestureType.OPEN_PALM
            self.scrolling_armed = False

        # Priority B: Pinch Click (Thumb and Index fingertips meet)
        elif pinch_detected and not (middle and ring and pinky):
            detected_raw = GestureType.PINCH_CLICK

        # Priority C: Two Fingers Up (Index + Middle raised, others down) -> Arms Scrolling Mode
        elif index and middle and not ring and not pinky:
            detected_raw = GestureType.TWO_FINGERS_UP
            self.scrolling_armed = True

        # Priority D: One Finger Gestures (Index active, others down)
        elif index and not middle and not ring and not pinky:
            # Check motion direction or explicit pointing orientation
            if index_pointing_up:
                if delta_y < -1.0:
                    detected_raw = GestureType.SCROLL_UP
                elif delta_y > 1.0:
                    detected_raw = GestureType.SCROLL_DOWN
                else:
                    # Finger is held up statically
                    detected_raw = GestureType.SCROLL_UP if self.scrolling_armed else GestureType.IDLE
            elif index_pointing_down or delta_y > 1.0:
                detected_raw = GestureType.SCROLL_DOWN
            else:
                detected_raw = GestureType.IDLE

        # Explicit downward pointing check even if tip is below MCP
        elif index_pointing_down and not middle and not ring and not pinky:
            detected_raw = GestureType.SCROLL_DOWN

        # Priority E: Closed fist / neutral
        elif num_fingers_up == 0:
            detected_raw = GestureType.IDLE
        else:
            detected_raw = GestureType.IDLE

        # 7. Debounce Filter (temporal confirmation across consecutive frames)
        if detected_raw == self.candidate_gesture:
            self.candidate_count += 1
        else:
            self.candidate_gesture = detected_raw
            self.candidate_count = 1

        # Quick response for immediate stops (Open Palm) and movement gestures
        immediate_gestures = (GestureType.OPEN_PALM, GestureType.SCROLL_UP, GestureType.SCROLL_DOWN)
        required_frames = 1 if detected_raw in immediate_gestures else self.debounce_frames

        if self.candidate_count >= required_frames:
            self.active_gesture = self.candidate_gesture

        # 8. Pinch Click Trigger Evaluation (Discrete single click with cooldown)
        is_click_event = False
        current_time = time.time()
        if self.active_gesture == GestureType.PINCH_CLICK:
            if not self.is_pinching:
                if (current_time - self.last_click_time) >= self.pinch_cooldown:
                    is_click_event = True
                    self.last_click_time = current_time
                    self.is_pinching = True
        else:
            self.is_pinching = False

        # 9. Format display labels & color codes
        label, color = self._get_display_info(self.active_gesture, self.scrolling_armed, delta_y)

        return GestureResult(
            gesture_type=self.active_gesture,
            display_label=label,
            status_color=color,
            delta_x=delta_x,
            delta_y=delta_y,
            smoothed_tip=smoothed_point,
            is_pinch=pinch_detected,
            pinch_distance=pinch_dist,
            is_click_event=is_click_event,
            fingers_up=fingers
        )

    def _get_display_info(self, gesture: str, armed: bool, delta_y: float) -> Tuple[str, str]:
        """Maps gesture type to human-readable UI text and status colors."""
        if gesture == GestureType.OPEN_PALM:
            return "Open Palm (Scrolling Stopped)", config.COLOR_WARNING
        elif gesture == GestureType.TWO_FINGERS_UP:
            return "Scrolling Mode Ready (2 Fingers)", config.COLOR_CYAN
        elif gesture == GestureType.SCROLL_UP:
            motion_desc = "Scrolling UP" if delta_y < 0 else "Pointing UP"
            return f"One Finger Up ({motion_desc})", config.COLOR_SUCCESS
        elif gesture == GestureType.SCROLL_DOWN:
            motion_desc = "Scrolling DOWN" if delta_y > 0 else "Pointing DOWN"
            return f"One Finger Down ({motion_desc})", config.COLOR_SUCCESS
        elif gesture == GestureType.PINCH_CLICK:
            return "Pinch Gesture (Click Action)", config.COLOR_ACCENT
        elif gesture == GestureType.IDLE:
            if armed:
                return "Ready (Move index to scroll)", config.COLOR_CYAN
            return "Neutral / Idle", config.COLOR_TEXT_SECONDARY
        else:
            return "No Hand Detected", config.COLOR_DANGER
