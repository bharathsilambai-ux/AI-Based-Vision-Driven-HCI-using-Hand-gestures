"""
Root Hand Detector Bridge - Connects to detection.hand_detector.
Maintains backward compatibility for single-hand unpack calls (gui.py, gesture_recognition.py).
"""

import numpy as np
from typing import Tuple, Optional, List

from detection.hand_detector import (
    HandDetector as BaseHandDetector,
    HandData,
    HandLandmarkData,
    HAND_CONNECTIONS,
    FINGER_TIPS,
    FINGER_DIPS,
    FINGER_PIPS,
    FINGER_MCPS
)


class HandDetector(BaseHandDetector):
    """
    Subclass that provides single-hand return for find_hands() to support legacy unpacks,
    while also exposing detect() for multi-hand lists.
    """

    def find_hands(
        self,
        frame: np.ndarray,
        draw: bool = True
    ) -> Tuple[Optional[HandData], np.ndarray]:
        """Returns (primary_hand_or_none, annotated_frame)."""
        hands, annotated = self.detect(frame, draw=draw)
        primary = hands[0] if hands else None
        return primary, annotated


__all__ = [
    "HandDetector",
    "HandData",
    "HandLandmarkData",
    "HAND_CONNECTIONS",
    "FINGER_TIPS",
    "FINGER_DIPS",
    "FINGER_PIPS",
    "FINGER_MCPS"
]
