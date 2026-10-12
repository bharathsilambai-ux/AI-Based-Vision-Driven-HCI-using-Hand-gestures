"""
Vision Hand Tracker Module - Bridge to detection.hand_detector.
Maintains backward compatibility across all modules while providing
unified multi-hand tracking, rotation estimation, and sci-fi HUD rendering.
"""

from detection.hand_detector import (
    HandDetector,
    HandData,
    HandLandmarkData,
    HandTracker,
    HAND_CONNECTIONS,
    FINGER_TIPS,
    FINGER_DIPS,
    FINGER_PIPS,
    FINGER_MCPS
)

__all__ = [
    "HandDetector",
    "HandData",
    "HandLandmarkData",
    "HandTracker",
    "HAND_CONNECTIONS",
    "FINGER_TIPS",
    "FINGER_DIPS",
    "FINGER_PIPS",
    "FINGER_MCPS"
]
