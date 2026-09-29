"""
Comprehensive Hand Feature Extractor Module.
Extracts translation- and scale-invariant landmark coordinates, joint angles,
pairwise fingertip distances, palm orientation, velocities, and two-hand geometry.
Produces high-dimensional feature vectors ready for ML classification.
"""

import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

try:
    from .hand_tracker import HandData
except ImportError:
    from hand_tracker import HandData


def compute_angle_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float], p3: Tuple[float, float, float]) -> float:
    """
    Computes angle at vertex p2 formed by (p1 - p2) and (p3 - p2) in radians.
    Returns angle in [0, pi].
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine))


class HandFeatures:
    """Container for normalized single-hand features."""

    def __init__(
        self,
        hand: HandData,
        normalized_lms: np.ndarray,
        joint_angles: List[float],
        pairwise_distances: List[float],
        palm_roll_deg: float,
        palm_facing_camera: bool,
        velocity_xy: Tuple[float, float] = (0.0, 0.0),
        speed: float = 0.0,
        direction_deg: float = 0.0
    ):
        self.hand = hand
        self.handedness = hand.handedness
        self.fingers_up = hand.fingers_up
        self.normalized_lms = normalized_lms  # Shape (21, 3)
        self.flattened_coords = normalized_lms.flatten()  # 63 values
        self.joint_angles = joint_angles
        self.pairwise_distances = pairwise_distances
        self.palm_roll_deg = palm_roll_deg
        self.palm_facing_camera = palm_facing_camera
        self.velocity_xy = velocity_xy
        self.speed = speed
        self.direction_deg = direction_deg

    def to_ml_vector(self) -> np.ndarray:
        """
        Concatenates features into a compact 1D vector for ML classifiers.
        Total features: 63 (coords) + 14 (distances) + 8 (angles) + 5 (fingers) + 2 (roll & facing) = 92 features.
        """
        feats = []
        feats.extend(self.flattened_coords.tolist())
        feats.extend(self.pairwise_distances)
        feats.extend(self.joint_angles)
        feats.extend([float(f) for f in self.fingers_up])
        feats.append(self.palm_roll_deg / 180.0)
        feats.append(1.0 if self.palm_facing_camera else -1.0)
        return np.array(feats, dtype=np.float32)


class DualHandFeatures:
    """Container for simultaneous two-hand geometric interaction features."""

    def __init__(
        self,
        left_hand: HandData,
        right_hand: HandData,
        hand_distance_norm: float,
        angle_between_hands_deg: float,
        delta_distance: float = 0.0,
        delta_angle_deg: float = 0.0
    ):
        self.left_hand = left_hand
        self.right_hand = right_hand
        self.distance = hand_distance_norm
        self.angle_deg = angle_between_hands_deg
        self.delta_distance = delta_distance
        self.delta_angle_deg = delta_angle_deg


class FeatureExtractor:
    """
    Extracts geometric, kinematics, and orientation features from tracked hands.
    Maintains temporal state for velocity and two-hand delta calculations.
    """

    def __init__(self):
        # State tracking for single hand velocities: { handedness: (last_tip_x, last_tip_y, last_time) }
        self._history: Dict[str, Tuple[float, float, float]] = {}

        # State tracking for two hands
        self._prev_dual_dist: Optional[float] = None
        self._prev_dual_angle: Optional[float] = None

    def reset(self):
        """Clears temporal velocity history."""
        self._history.clear()
        self._prev_dual_dist = None
        self._prev_dual_angle = None

    def extract_single_hand(self, hand: HandData) -> HandFeatures:
        """
        Computes scale-invariant landmarks, joint angles, pairwise distances,
        palm orientation, and fingertip velocity.
        """
        now = time.time()
        w_lms = [hand.get_norm(i) for i in range(21)]

        # 1. Translation Invariance: Center wrist (id 0) to (0, 0, 0)
        wrist = np.array(w_lms[0])
        centered = np.array([np.array(p) - wrist for p in w_lms])

        # 2. Scale Invariance: Divide by wrist-to-middle-MCP (0 to 9) distance
        scale = np.linalg.norm(centered[9])
        if scale < 1e-4:
            scale = 0.2  # Guard against division by zero
        normalized_lms = centered / scale

        # 3. Key Pairwise Distances (Normalized)
        # Thumb(4) to Index(8), Middle(12), Ring(16), Pinky(20)
        # Index(8) to Middle(12), etc.
        pairs = [
            (4, 8), (4, 12), (4, 16), (4, 20),
            (8, 12), (12, 16), (16, 20),
            (4, 2), (8, 5), (12, 9), (16, 13), (20, 17),
            (8, 0), (12, 0)
        ]
        pairwise_dists = [
            float(np.linalg.norm(normalized_lms[i] - normalized_lms[j]))
            for i, j in pairs
        ]

        # 4. Joint Angles (MCP and PIP joints for fingers)
        # Thumb: (2-3-4), Index: (5-6-7, 6-7-8), Middle: (9-10-11, 10-11-12),
        # Ring: (13-14-15), Pinky: (17-18-19, 18-19-20)
        angle_triplets = [
            (2, 3, 4),   # Thumb IP
            (0, 5, 6),   # Index MCP
            (5, 6, 7),   # Index PIP
            (6, 7, 8),   # Index DIP
            (0, 9, 10),  # Middle MCP
            (9, 10, 11), # Middle PIP
            (0, 13, 14), # Ring MCP
            (0, 17, 18)  # Pinky MCP
        ]
        joint_angles = [
            compute_angle_3d(w_lms[p1], w_lms[p2], w_lms[p3])
            for p1, p2, p3 in angle_triplets
        ]

        # 5. Palm Orientation (Cross Product of Wrist->IndexMCP and Wrist->PinkyMCP)
        v_idx = np.array(w_lms[5]) - wrist
        v_pnk = np.array(w_lms[17]) - wrist
        normal = np.cross(v_idx, v_pnk)
        # In normalized coords, negative Z points into camera
        palm_facing_camera = normal[2] < 0 if hand.handedness == "Right" else normal[2] > 0

        # Roll angle: orientation of index MCP relative to pinky MCP in screen plane
        dx = w_lms[5][0] - w_lms[17][0]
        dy = w_lms[5][1] - w_lms[17][1]
        palm_roll_deg = math.degrees(math.atan2(dy, dx))

        # 6. Dynamics / Velocity Calculation
        h_key = hand.handedness
        idx_px = hand.get_px(8)
        vel_x, vel_y, speed, direction_deg = 0.0, 0.0, 0.0, 0.0

        if h_key in self._history:
            prev_x, prev_y, prev_t = self._history[h_key]
            dt = max(1e-3, now - prev_t)
            if dt < 0.5:  # Valid recent sample
                vel_x = (idx_px[0] - prev_x) / dt  # px/sec
                vel_y = (idx_px[1] - prev_y) / dt
                speed = math.hypot(vel_x, vel_y)
                direction_deg = math.degrees(math.atan2(vel_y, vel_x))

        self._history[h_key] = (float(idx_px[0]), float(idx_px[1]), now)

        return HandFeatures(
            hand=hand,
            normalized_lms=normalized_lms,
            joint_angles=joint_angles,
            pairwise_distances=pairwise_dists,
            palm_roll_deg=palm_roll_deg,
            palm_facing_camera=palm_facing_camera,
            velocity_xy=(vel_x, vel_y),
            speed=speed,
            direction_deg=direction_deg
        )

    def extract_dual_hands(self, hands: List[HandData]) -> Optional[DualHandFeatures]:
        """
        Extracts distance, relative angle, and displacement rates between two hands.
        """
        if len(hands) < 2:
            self._prev_dual_dist = None
            self._prev_dual_angle = None
            return None

        # Sort into Left and Right hands (or first two if ambiguous)
        h1, h2 = hands[0], hands[1]
        left = h1 if h1.handedness == "Left" else h2
        right = h2 if h1.handedness == "Left" else h1

        # Normalized coordinates of wrist and center
        p_left = left.get_norm(0)
        p_right = right.get_norm(0)

        # 1. 3D distance between wrists
        dist = math.sqrt((p_left[0] - p_right[0])**2 + (p_left[1] - p_right[1])**2 + (p_left[2] - p_right[2])**2)

        # 2. Angle in screen plane between hand centers
        c_left = left.center
        c_right = right.center
        angle_deg = math.degrees(math.atan2(c_right[1] - c_left[1], c_right[0] - c_left[0]))

        # 3. Delta calculations
        delta_dist = 0.0
        delta_angle = 0.0

        if self._prev_dual_dist is not None:
            delta_dist = dist - self._prev_dual_dist

        if self._prev_dual_angle is not None:
            delta_angle = angle_deg - self._prev_dual_angle
            # Normalize angle wrap around [-180, 180]
            if delta_angle > 180:
                delta_angle -= 360
            elif delta_angle < -180:
                delta_angle += 360

        self._prev_dual_dist = dist
        self._prev_dual_angle = angle_deg

        return DualHandFeatures(
            left_hand=left,
            right_hand=right,
            hand_distance_norm=dist,
            angle_between_hands_deg=angle_deg,
            delta_distance=delta_dist,
            delta_angle_deg=delta_angle
        )
