"""
Comprehensive Deterministic Gesture Rule Engine.
Evaluates 21 hand landmarks, joint angles, and kinematics for all 48 required operations
across Mouse, Navigation, Media, System, Keyboard, Zoom, and Dual-Hand modes.
"""

import math
import os
import sys
from typing import List, Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from vision.feature_extractor import DualHandFeatures, HandFeatures
from vision.hand_tracker import HandData


class GestureID:
    NONE = "NONE"
    IDLE = "IDLE"

    # Mouse
    INDEX_POINTING = "INDEX_POINTING"
    PINCH_INDEX = "PINCH_INDEX"
    DOUBLE_PINCH_INDEX = "DOUBLE_PINCH_INDEX"
    PINCH_MIDDLE = "PINCH_MIDDLE"
    PINCH_RING = "PINCH_RING"
    PINCH_PINKY = "PINCH_PINKY"
    CLOSED_FIST = "CLOSED_FIST"
    OPEN_PALM = "OPEN_PALM"
    INDEX_HOLD = "INDEX_HOLD"

    # Navigation & Media
    TWO_FINGERS_UP = "TWO_FINGERS_UP"
    TWO_FINGERS_DOWN = "TWO_FINGERS_DOWN"
    PEACE_SIGN = "PEACE_SIGN"
    THUMBS_UP = "THUMBS_UP"
    THUMBS_DOWN = "THUMBS_DOWN"
    THUMB_PINKY = "THUMB_PINKY"
    SWIPE_LEFT = "SWIPE_LEFT"
    SWIPE_RIGHT = "SWIPE_RIGHT"
    SWIPE_UP = "SWIPE_UP"
    SWIPE_DOWN = "SWIPE_DOWN"
    HAND_MOVE_UP = "HAND_MOVE_UP"
    HAND_MOVE_DOWN = "HAND_MOVE_DOWN"

    # System & Keyboard
    OK_SIGN = "OK_SIGN"
    PINCH_HOLD = "PINCH_HOLD"
    THREE_FINGERS = "THREE_FINGERS"
    FOUR_FINGERS = "FOUR_FINGERS"
    FIVE_FINGERS = "FIVE_FINGERS"
    FIVE_FINGERS_HOLD = "FIVE_FINGERS_HOLD"
    FINGER_GUN = "FINGER_GUN"
    ROCK_SIGN = "ROCK_SIGN"
    ROCK_SIGN_HOLD = "ROCK_SIGN_HOLD"

    # Zoom & Dual-Hand
    PINCH_EXPAND = "PINCH_EXPAND"
    PINCH_CONTRACT = "PINCH_CONTRACT"
    TWO_HAND_EXPAND = "TWO_HAND_EXPAND"
    TWO_HAND_CONTRACT = "TWO_HAND_CONTRACT"
    TWO_HAND_ROTATE = "TWO_HAND_ROTATE"
    TWO_HAND_PALMS = "TWO_HAND_PALMS"
    TWO_HAND_SWIPE_LEFT = "TWO_HAND_SWIPE_LEFT"
    TWO_HAND_SWIPE_RIGHT = "TWO_HAND_SWIPE_RIGHT"


class GestureRules:
    """Evaluates geometric criteria to classify hand gestures with calibrated confidence ratings."""

    def __init__(self, pinch_threshold: float = config.PINCH_THRESHOLD_DEFAULT):
        self.pinch_threshold = pinch_threshold

    def set_pinch_threshold(self, value: float):
        self.pinch_threshold = max(0.02, min(0.15, float(value)))

    def classify_single_hand(self, feat: HandFeatures) -> Tuple[str, float]:
        """
        Classifies single hand landmark geometry, kinematics, and orientation for HCI operations.
        Returns: (gesture_id, confidence_score)
        """
        hand = feat.hand
        fingers = feat.fingers_up
        thumb, index, middle, ring, pinky = fingers
        num_up = sum(fingers)

        # Scale factor based on palm span (wrist 0 to middle MCP 9)
        palm_size = hand.get_distance_norm(0, 9)
        scale_factor = palm_size / 0.20 if palm_size > 0.04 else 1.0
        effective_pinch_thresh = self.pinch_threshold * 1.25 * max(0.70, min(1.45, scale_factor))

        d_thumb_idx = hand.get_distance_norm(4, 8)
        d_thumb_mid = hand.get_distance_norm(4, 12)
        d_thumb_ring = hand.get_distance_norm(4, 16)
        pinch_thresh = effective_pinch_thresh

        # -------------------------------------------------------------
        # 1. OK SIGN (📸 Take Screenshot):
        # Thumb + Index touch with middle and ring extended upright (classic OK)
        # -------------------------------------------------------------
        if d_thumb_idx < pinch_thresh and middle and ring and num_up >= 3:
            return GestureID.OK_SIGN, 0.98

        # -------------------------------------------------------------
        # 2. INDEX PINCH (🤏 Left Click):
        # Thumb + Index tips touching, other fingers curled down (NOT an OK sign)
        # -------------------------------------------------------------
        if d_thumb_idx < pinch_thresh and d_thumb_idx <= d_thumb_mid and not (middle and ring and pinky):
            conf = 1.0 - (d_thumb_idx / pinch_thresh) * 0.30
            return GestureID.PINCH_INDEX, min(0.99, max(0.80, conf))

        # -------------------------------------------------------------
        # 3. MIDDLE PINCH (👌 Right Click):
        # Thumb + Middle tips touching, middle is closer than index
        # -------------------------------------------------------------
        if d_thumb_mid < pinch_thresh and d_thumb_mid < d_thumb_idx:
            conf = 1.0 - (d_thumb_mid / pinch_thresh) * 0.30
            return GestureID.PINCH_MIDDLE, min(0.98, max(0.80, conf))

        # 4. Ring Pinch (Secondary)
        if d_thumb_ring < pinch_thresh and not pinky and not index:
            return GestureID.PINCH_RING, 0.88

        # -------------------------------------------------------------
        # 5. OPEN PALM (🖐️ Release Drag / Drop & Halt Scroll):
        # All 5 fingers extended spread flat
        # -------------------------------------------------------------
        if num_up == 5 or (thumb and index and middle and ring and pinky):
            return GestureID.OPEN_PALM, 0.98

        # Four fingers (Index, Middle, Ring, Pinky with thumb tucked)
        if not thumb and index and middle and ring and pinky:
            return GestureID.FOUR_FINGERS, 0.95

        # -------------------------------------------------------------
        # 6. CLOSED FIST (✊ Drag & Drop / Hold):
        # All 5 fingers curled into fist
        # -------------------------------------------------------------
        if num_up == 0 or (num_up == 1 and thumb and not (index or middle or ring or pinky) and hand.get_norm(4)[1] >= hand.get_norm(2)[1] - 0.01):
            return GestureID.CLOSED_FIST, 0.98

        # -------------------------------------------------------------
        # 7. THUMBS UP / THUMBS DOWN (🔊 Volume Control):
        # Only Thumb extended, other 4 fingers curled
        # -------------------------------------------------------------
        if thumb and not (index or middle or ring or pinky):
            thumb_tip = hand.get_norm(4)
            thumb_mcp = hand.get_norm(2)
            if thumb_tip[1] < thumb_mcp[1] - 0.02:
                return GestureID.THUMBS_UP, 0.96
            elif thumb_tip[1] > thumb_mcp[1] + 0.02:
                return GestureID.THUMBS_DOWN, 0.96
            return GestureID.THUMBS_UP, 0.90

        # -------------------------------------------------------------
        # 8. TWO FINGERS UP vs TWO FINGERS DOWN (✌️ Scroll Up / 👇 Scroll Down):
        # Index & Middle extended, Ring & Pinky curled
        # -------------------------------------------------------------
        if index and middle and not ring and not pinky:
            p_idx = hand.get_norm(8)
            p_wrist = hand.get_norm(0)
            if p_idx[1] > p_wrist[1] - 0.02:
                return GestureID.TWO_FINGERS_DOWN, 0.96
            p_mid = hand.get_norm(12)
            d_tips = math.hypot(p_idx[0] - p_mid[0], p_idx[1] - p_mid[1]) / (scale_factor or 1.0)
            if d_tips > 0.11:
                return GestureID.PEACE_SIGN, 0.94
            return GestureID.TWO_FINGERS_UP, 0.96

        # -------------------------------------------------------------
        # 9. SINGLE INDEX POINTING (☝️ Move Cursor):
        # Index extended, other fingers curled down, not pinching
        # -------------------------------------------------------------
        if index and not (middle or ring or pinky):
            return GestureID.INDEX_POINTING, 0.98

        # Dynamic High-Speed Swipes (speed > 400 px/s)
        if feat.speed > 420.0 and num_up >= 1:
            deg = feat.direction_deg
            if -35.0 <= deg <= 35.0:
                return GestureID.SWIPE_RIGHT, 0.92
            elif deg >= 145.0 or deg <= -145.0:
                return GestureID.SWIPE_LEFT, 0.92
            elif -125.0 <= deg <= -55.0:
                return GestureID.SWIPE_UP, 0.90
            elif 55.0 <= deg <= 125.0:
                return GestureID.SWIPE_DOWN, 0.90

        # Additional secondary helpers (Rock sign, Three/Four fingers, Gun)
        if index and pinky and not (middle or ring):
            return GestureID.ROCK_SIGN, 0.93

        if index and middle and ring and not pinky:
            return GestureID.THREE_FINGERS, 0.93

        if index and middle and ring and pinky and not thumb:
            return GestureID.FOUR_FINGERS, 0.93

        if thumb and index and not (middle or ring or pinky):
            d_thumb_idx_mcp = hand.get_distance_norm(4, 5)
            if d_thumb_idx_mcp > 0.14:
                return GestureID.FINGER_GUN, 0.92

        if thumb and pinky and not (index or middle or ring):
            return GestureID.THUMB_PINKY, 0.94

        return GestureID.IDLE, 0.60

    def classify_dual_hands(self, dual: DualHandFeatures) -> Tuple[str, float]:
        """
        Classifies two-hand interactions.
        Returns: (gesture_id, confidence_score)
        """
        # 1. Both hands showing open palms (Show Desktop)
        l_fingers = sum(dual.left_hand.fingers_up)
        r_fingers = sum(dual.right_hand.fingers_up)
        if l_fingers >= 4 and r_fingers >= 4 and abs(dual.delta_distance) < 0.02 and abs(dual.delta_angle_deg) < 10.0:
            return GestureID.TWO_HAND_PALMS, 0.92

        # 2. Two-Hand Distance Expand / Contract (Zoom In / Out)
        if dual.delta_distance > 0.020:
            return GestureID.TWO_HAND_EXPAND, min(0.98, 0.70 + abs(dual.delta_distance) * 5.0)
        elif dual.delta_distance < -0.020:
            return GestureID.TWO_HAND_CONTRACT, min(0.98, 0.70 + abs(dual.delta_distance) * 5.0)

        # 3. Two-Hand Rotation (Rotate Document)
        if abs(dual.delta_angle_deg) > 10.0:
            return GestureID.TWO_HAND_ROTATE, min(0.98, 0.70 + abs(dual.delta_angle_deg) / 35.0)

        if l_fingers >= 4 and r_fingers >= 4:
            return GestureID.TWO_HAND_PALMS, 0.90

        return GestureID.NONE, 0.0


# Backward compatibility aliases
RuleGestureEngine = GestureRules
