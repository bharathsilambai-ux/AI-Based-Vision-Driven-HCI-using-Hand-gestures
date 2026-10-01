"""
Landmark Mathematics & Geometric Feature Extraction Module
==========================================================
Provides translation-invariant and scale-invariant normalization,
Euclidean distance functions, finger curl angles, and orientation analysis.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any


class LandmarkProcessor:
    """Mathematical transforms and feature extraction for 21 MediaPipe hand landmarks."""

    # Landmark indices for fingers
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
    FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

    @staticmethod
    def euclidean_distance_3d(pt1: Tuple[float, float, float], pt2: Tuple[float, float, float]) -> float:
        """Calculates 3D Euclidean distance between two points."""
        return math.sqrt(
            (pt1[0] - pt2[0]) ** 2 +
            (pt1[1] - pt2[1]) ** 2 +
            (pt1[2] - pt2[2]) ** 2
        )

    @staticmethod
    def euclidean_distance_2d(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """Calculates 2D Euclidean distance between two points."""
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    @classmethod
    def normalize_landmarks(
        cls, landmarks: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        """
        Normalizes 21 3D landmarks to be:
        1. Translation-invariant (wrist centered at (0, 0, 0))
        2. Scale-invariant (divided by distance between wrist and middle MCP)

        Returns:
            1D NumPy array of shape (63,) containing [x0, y0, z0, ..., x20, y20, z20]
        """
        if not landmarks or len(landmarks) != 21:
            raise ValueError(f"Expected 21 landmarks, got {len(landmarks) if landmarks else 0}")

        # Landmark 0 is the wrist
        wrist = np.array(landmarks[cls.WRIST], dtype=np.float32)
        middle_mcp = np.array(landmarks[cls.MIDDLE_MCP], dtype=np.float32)

        # 1. Translation invariance: shift all landmarks so wrist is origin
        translated = np.array(landmarks, dtype=np.float32) - wrist

        # 2. Scale invariance: normalize by reference distance (wrist to middle MCP)
        ref_distance = np.linalg.norm(middle_mcp - wrist)
        if ref_distance < 1e-6:
            ref_distance = 1.0  # Prevent division by zero

        normalized = translated / ref_distance
        return normalized.flatten()

    @classmethod
    def get_finger_states(
        cls,
        landmarks: List[Tuple[float, float, float]],
        handedness: str = "Right"
    ) -> List[int]:
        """
        Determines whether each of the 5 fingers is extended (1) or curled (0).
        Fingers: [Thumb, Index, Middle, Ring, Pinky]

        Uses orientation-robust distance and directional vectors.
        """
        if not landmarks or len(landmarks) != 21:
            return [0, 0, 0, 0, 0]

        states = [0, 0, 0, 0, 0]
        wrist = landmarks[cls.WRIST]

        # 1. Thumb State:
        # Distance from thumb tip to pinky MCP compared to thumb IP to pinky MCP
        thumb_tip = landmarks[cls.THUMB_TIP]
        thumb_ip = landmarks[cls.THUMB_IP]
        pinky_mcp = landmarks[cls.PINKY_MCP]

        dist_thumb_tip = cls.euclidean_distance_2d(thumb_tip[:2], pinky_mcp[:2])
        dist_thumb_ip = cls.euclidean_distance_2d(thumb_ip[:2], pinky_mcp[:2])

        # Thumb extended if tip is further from pinky MCP than IP joint
        if dist_thumb_tip > dist_thumb_ip * 1.15:
            states[0] = 1

        # 2. Four Fingers (Index, Middle, Ring, Pinky):
        # Finger is extended if distance from wrist to tip is greater than wrist to PIP
        # AND tip.y is above pip.y (for typical upright hands)
        for i in range(1, 5):
            tip_idx = cls.FINGER_TIPS[i]
            pip_idx = cls.FINGER_PIPS[i]
            mcp_idx = cls.FINGER_MCPS[i]

            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]

            dist_tip_wrist = cls.euclidean_distance_2d(tip[:2], wrist[:2])
            dist_pip_wrist = cls.euclidean_distance_2d(pip[:2], wrist[:2])

            # Check both distance from base and vertical orientation
            if dist_tip_wrist > dist_pip_wrist * 1.1 and tip[1] < pip[1]:
                states[i] = 1

        return states

    @classmethod
    def get_pinch_distance(cls, landmarks: List[Tuple[float, float, float]]) -> float:
        """Normalized Euclidean distance between Thumb Tip (4) and Index Tip (8)."""
        if not landmarks or len(landmarks) != 21:
            return 1.0
        return cls.euclidean_distance_2d(
            landmarks[cls.THUMB_TIP][:2],
            landmarks[cls.INDEX_TIP][:2]
        )

    @classmethod
    def get_middle_pinch_distance(cls, landmarks: List[Tuple[float, float, float]]) -> float:
        """Normalized Euclidean distance between Thumb Tip (4) and Middle Tip (12)."""
        if not landmarks or len(landmarks) != 21:
            return 1.0
        return cls.euclidean_distance_2d(
            landmarks[cls.THUMB_TIP][:2],
            landmarks[cls.MIDDLE_TIP][:2]
        )

    @classmethod
    def calculate_angle_3pts(
        cls,
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float]
    ) -> float:
        """
        Calculates angle in degrees at vertex b formed by rays ba and bc.
        """
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.hypot(ba[0], ba[1])
        mag_bc = math.hypot(bc[0], bc[1])

        if mag_ba * mag_bc < 1e-6:
            return 0.0

        cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
        return math.degrees(math.acos(cosine))

    @classmethod
    def get_palm_center(cls, landmarks: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        """Calculates approximate center of palm from Wrist, Index MCP, and Pinky MCP."""
        if not landmarks or len(landmarks) != 21:
            return (0.0, 0.0)
        w = landmarks[cls.WRIST]
        im = landmarks[cls.INDEX_MCP]
        pm = landmarks[cls.PINKY_MCP]
        return ((w[0] + im[0] + pm[0]) / 3.0, (w[1] + im[1] + pm[1]) / 3.0)
