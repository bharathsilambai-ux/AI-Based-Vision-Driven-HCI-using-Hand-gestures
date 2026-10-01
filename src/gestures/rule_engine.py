"""
Rule-Based Deterministic Gesture Recognition Engine
===================================================
Classifies hand postures using geometric rules, Euclidean distances,
and finger extension states with zero training requirements.
"""

from typing import List, Tuple, Dict, Any
import config
from src.vision.landmark_utils import LandmarkProcessor


class RuleGestureEngine:
    """Evaluates geometric criteria to classify predefined hand gestures."""

    def __init__(
        self,
        pinch_threshold: float = config.PINCH_THRESHOLD,
        middle_pinch_threshold: float = config.MIDDLE_PINCH_THRESHOLD,
    ):
        self.pinch_threshold = pinch_threshold
        self.middle_pinch_threshold = middle_pinch_threshold

    def classify(
        self,
        landmarks: List[Tuple[float, float, float]],
        handedness: str = "Right"
    ) -> Tuple[str, float]:
        """
        Classifies current hand pose into a gesture label.
        
        Returns:
            (gesture_label, confidence_score)
        """
        if not landmarks or len(landmarks) != 21:
            return config.GESTURE_IDLE, 0.0

        proc = LandmarkProcessor
        fingers = proc.get_finger_states(landmarks, handedness)
        # fingers: [Thumb, Index, Middle, Ring, Pinky]

        pinch_dist = proc.get_pinch_distance(landmarks)
        middle_pinch_dist = proc.get_middle_pinch_distance(landmarks)

        thumb_tip = landmarks[proc.THUMB_TIP]
        thumb_ip = landmarks[proc.THUMB_IP]
        thumb_mcp = landmarks[proc.THUMB_MCP]
        wrist = landmarks[proc.WRIST]

        # 1. Check Fist and Thumb Gestures (When all 4 fingers are curled)
        four_curled = (fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0)
        if four_curled:
            dist_thumb_wrist = proc.euclidean_distance_2d(thumb_tip[:2], wrist[:2])

            # Thumb Up: Thumb extended upright significantly beyond the curled fist
            if thumb_tip[1] < thumb_ip[1] < thumb_mcp[1] and dist_thumb_wrist > 0.28:
                return config.GESTURE_THUMB_UP, 0.95

            # Thumb Down: Thumb extended downward significantly below the fist
            if thumb_tip[1] > thumb_ip[1] > thumb_mcp[1] and dist_thumb_wrist > 0.25:
                return config.GESTURE_THUMB_DOWN, 0.95

            # Otherwise, all curled into palm -> Closed Fist
            return config.GESTURE_FIST, 0.96

        # 2. Check Pinch Gestures (Only active if at least one finger is extended)
        # Middle finger pinch -> Right Click
        if middle_pinch_dist < self.middle_pinch_threshold and pinch_dist > self.pinch_threshold * 1.2:
            conf = min(0.99, max(0.70, 1.0 - (middle_pinch_dist / self.middle_pinch_threshold) * 0.3))
            return config.GESTURE_MIDDLE_PINCH, conf

        # Index + Thumb pinch -> Left Click / Drag
        if pinch_dist < self.pinch_threshold:
            conf = min(0.99, max(0.70, 1.0 - (pinch_dist / self.pinch_threshold) * 0.3))
            return config.GESTURE_PINCH, conf

        # 3. Check Rock On / Screenshot Gesture (Index and Pinky up, Middle and Ring curled)
        if fingers[1] == 1 and fingers[4] == 1 and fingers[2] == 0 and fingers[3] == 0:
            return config.GESTURE_SCREENSHOT, 0.94

        # 4. Check Three Fingers Up (Index, Middle, Ring up; Pinky curled)
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
            return config.GESTURE_THREE_FINGERS, 0.93

        # 5. Check Two Fingers Up (Index and Middle up, Ring and Pinky curled)
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            # Check if pointing upward
            idx_tip = landmarks[proc.INDEX_TIP]
            idx_pip = landmarks[proc.INDEX_PIP]
            if idx_tip[1] < idx_pip[1]:
                return config.GESTURE_TWO_FINGERS_UP, 0.94
            else:
                return config.GESTURE_TWO_FINGERS_DOWN, 0.90

        # 6. Check Two Fingers Down explicitly if tips are below PIPs
        idx_tip = landmarks[proc.INDEX_TIP]
        mid_tip = landmarks[proc.MIDDLE_TIP]
        idx_pip = landmarks[proc.INDEX_PIP]
        mid_pip = landmarks[proc.MIDDLE_PIP]
        if (idx_tip[1] > idx_pip[1]) and (mid_tip[1] > mid_pip[1]) and fingers[3] == 0 and fingers[4] == 0:
            # Both index and middle extended downwards
            dist_idx = proc.euclidean_distance_2d(idx_tip[:2], wrist[:2])
            if dist_idx > 0.2:
                return config.GESTURE_TWO_FINGERS_DOWN, 0.91

        # 7. Open Palm (All 5 fingers extended)
        if sum(fingers) >= 4 and fingers[1] == 1 and fingers[2] == 1:
            return config.GESTURE_OPEN_PALM, 0.97

        # Default fallback
        return config.GESTURE_IDLE, 0.50
