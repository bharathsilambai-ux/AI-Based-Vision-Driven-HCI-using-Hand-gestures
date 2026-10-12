"""
Phase 4: Landmark Processing, Normalization & Feature Extraction Subsystem
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Coordinate Invariance Transformations (Translation, Scale, Rotation):
   - Translation Invariance: Centers hand relative to wrist L0: P'_i = P_i - P_0 (L0 = [0, 0, 0]).
   - Scale Invariance: Normalizes coordinates by palm span d_ref = ||P'_9 - P'_0||_2 (wrist to middle MCP L9).
     Guarantees ||hat(P)_9||_2 = 1.0000.
   - Rotation Alignment: Computes 2D/3D rotation matrix aligning palm vector (L0 -> L9) with vertical reference
     axis to generate rotation-invariant landmarks for robust orientation-agnostic ML models.

2. Comprehensive Joint Kinematics & Spatial Geometry:
   - 3D Joint Angles (MCP, PIP, DIP) computed via vector dot products: theta = arccos((u . v) / (||u|| ||v||)).
   - Pairwise normalized Euclidean distances across 14 canonical anatomical pairs.
   - Palm 3D normal vector via cross product (P_5 - P_0) x (P_17 - P_0) for palm facing camera determination.
   - Palm orientation Euler angles: Roll, Pitch, Yaw in degrees.

3. Kinematic Dynamics & Temporal Tracking:
   - Real-time linear velocity (vx, vy in px/sec), scalar speed, and movement direction angle.
   - Linear acceleration (ax, ay in px/sec^2) and acceleration magnitude.
   - Exponential moving average (EMA) velocity smoothing to eliminate tracking jitter.

4. Multi-Hand Spatial Interaction Features:
   - Inter-wrist 3D Euclidean distance and normalized distance.
   - Screen plane relative angle between hand centers.
   - Dynamic temporal delta distance (zoom/expand/contract rate).
   - Dynamic temporal delta angle (two-hand rotation rate with -180 to +180 deg wrap handling).
   - Two-hand gesture symmetry score.

5. Machine Learning Feature Vectorization:
   - Canonical 92-dimensional vector format for Scikit-Learn classifiers and custom gesture models:
     [63 normalized coords] + [14 pairwise dists] + [8 joint angles] + [5 finger states] + [roll] + [camera facing].
   - Rotation-invariant 92-dimensional feature vector.
   - Extended 128-dimensional multi-modal feature vector including velocities, accelerations, and 3D normal.

6. Interoperability & Backward Compatibility:
   - Seamlessly ingests HandData (vision/detection), HandDetectionResult, raw NumPy arrays, or landmark lists.
   - Full drop-in replacement and alias for FeatureExtractor, HandFeatures, and DualHandFeatures.
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

from utils.logger import logger

# Canonical 14 Landmark Pairs for Pairwise Distance Calculation
CANONICAL_DISTANCE_PAIRS: List[Tuple[int, int]] = [
    (4, 8),    # Thumb tip to Index tip (Pinch Index / OK)
    (4, 12),   # Thumb tip to Middle tip (Pinch Middle)
    (4, 16),   # Thumb tip to Ring tip (Pinch Ring)
    (4, 20),   # Thumb tip to Pinky tip (Pinch Pinky)
    (8, 12),   # Index tip to Middle tip (Two fingers / Scissors)
    (12, 16),  # Middle tip to Ring tip
    (16, 20),  # Ring tip to Pinky tip
    (4, 2),    # Thumb tip to Thumb MCP (Thumb curl)
    (8, 5),    # Index tip to Index MCP (Index curl)
    (12, 9),   # Middle tip to Middle MCP (Middle curl)
    (16, 13),  # Ring tip to Ring MCP (Ring curl)
    (20, 17),  # Pinky tip to Pinky MCP (Pinky curl)
    (8, 0),    # Index tip to Wrist (Reach / Span)
    (12, 0)    # Middle tip to Wrist (Hand length)
]

# Canonical 8 Angle Triplets for 92-D ML Vector (vertex is middle index)
CANONICAL_ANGLE_TRIPLETS: List[Tuple[int, int, int]] = [
    (2, 3, 4),    # Thumb IP
    (0, 5, 6),    # Index MCP
    (5, 6, 7),    # Index PIP
    (6, 7, 8),    # Index DIP
    (0, 9, 10),   # Middle MCP
    (9, 10, 11),  # Middle PIP
    (0, 13, 14),  # Ring MCP
    (0, 17, 18)   # Pinky MCP
]

# Extended 15 Finger Joint Triplets (covers all joints)
EXTENDED_ANGLE_TRIPLETS: Dict[str, Tuple[int, int, int]] = {
    "thumb_cmc": (0, 1, 2),
    "thumb_mcp": (1, 2, 3),
    "thumb_ip": (2, 3, 4),
    "index_mcp": (0, 5, 6),
    "index_pip": (5, 6, 7),
    "index_dip": (6, 7, 8),
    "middle_mcp": (0, 9, 10),
    "middle_pip": (9, 10, 11),
    "middle_dip": (10, 11, 12),
    "ring_mcp": (0, 13, 14),
    "ring_pip": (13, 14, 15),
    "ring_dip": (14, 15, 16),
    "pinky_mcp": (0, 17, 18),
    "pinky_pip": (17, 18, 19),
    "pinky_dip": (18, 19, 20),
}


def compute_angle_3d(
    p1: Union[Tuple[float, float, float], np.ndarray, List[float]],
    p2: Union[Tuple[float, float, float], np.ndarray, List[float]],
    p3: Union[Tuple[float, float, float], np.ndarray, List[float]]
) -> float:
    """
    Computes the 3D joint angle at vertex p2 formed by rays (p1 - p2) and (p3 - p2) in radians.
    
    Formula:
        cos(theta) = (v1 . v2) / (||v1|| * ||v2||)
        theta = arccos(clip(cos(theta), -1.0, 1.0))
        
    Returns:
        Angle in radians [0.0, pi]. Returns 0.0 if either vector has near-zero magnitude.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]], dtype=np.float64)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]], dtype=np.float64)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-7 or norm2 < 1e-7:
        return 0.0

    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine))


class HandFeatures:
    """
    Container for normalized single-hand features, joint angles, kinematics,
    palm orientation, and ML vector representations.
    """

    def __init__(
        self,
        hand: Any,
        normalized_lms: np.ndarray,
        joint_angles: List[float],
        pairwise_distances: List[float],
        palm_roll_deg: float,
        palm_facing_camera: bool,
        velocity_xy: Tuple[float, float] = (0.0, 0.0),
        speed: float = 0.0,
        direction_deg: float = 0.0,
        acceleration_xy: Tuple[float, float] = (0.0, 0.0),
        acceleration: float = 0.0,
        palm_normal: Optional[np.ndarray] = None,
        palm_pitch_deg: float = 0.0,
        palm_yaw_deg: float = 0.0,
        rotation_aligned_lms: Optional[np.ndarray] = None,
        extended_angles: Optional[Dict[str, float]] = None
    ):
        self.hand = hand
        self.handedness: str = getattr(hand, "handedness", "Right")
        if hasattr(hand, "fingers_up") and hand.fingers_up is not None:
            self.fingers_up = hand.fingers_up
        else:
            # Dynamically compute fingers_up from normalized_lms
            t_tip = normalized_lms[4]
            t_mcp = normalized_lms[2]
            idx_mcp = normalized_lms[5]
            d_thumb_idx = math.hypot(t_tip[0] - idx_mcp[0], t_tip[1] - idx_mcp[1])
            thumb_up = (t_tip[1] < t_mcp[1] - 0.05 and d_thumb_idx > 0.25) or (d_thumb_idx > 0.45)

            index_up = normalized_lms[8, 1] < normalized_lms[6, 1]
            middle_up = normalized_lms[12, 1] < normalized_lms[10, 1]
            ring_up = normalized_lms[16, 1] < normalized_lms[14, 1]
            pinky_up = normalized_lms[20, 1] < normalized_lms[18, 1]
            self.fingers_up = (bool(thumb_up), bool(index_up), bool(middle_up), bool(ring_up), bool(pinky_up))
        self.normalized_lms: np.ndarray = normalized_lms  # Shape (21, 3)
        self.flattened_coords: np.ndarray = normalized_lms.flatten()  # 63 values
        self.joint_angles: List[float] = joint_angles  # 8 canonical angles
        self.pairwise_distances: List[float] = pairwise_distances  # 14 canonical distances
        self.palm_roll_deg: float = float(palm_roll_deg)
        self.palm_facing_camera: bool = bool(palm_facing_camera)
        self.velocity_xy: Tuple[float, float] = velocity_xy
        self.speed: float = float(speed)
        self.direction_deg: float = float(direction_deg)
        self.acceleration_xy: Tuple[float, float] = acceleration_xy
        self.acceleration: float = float(acceleration)

        # Advanced orientation and rotation-invariant geometry
        self.palm_normal: np.ndarray = (
            palm_normal if palm_normal is not None else np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )
        self.palm_pitch_deg: float = float(palm_pitch_deg)
        self.palm_yaw_deg: float = float(palm_yaw_deg)
        self.rotation_aligned_lms: np.ndarray = (
            rotation_aligned_lms if rotation_aligned_lms is not None else normalized_lms.copy()
        )
        self.extended_angles: Dict[str, float] = extended_angles if extended_angles is not None else {}

    def to_ml_vector(self) -> np.ndarray:
        """
        Concatenates features into the standard 92-dimensional vector for ML classifiers.
        
        Vector Layout (92 dimensions):
        - 0..62  (63 values): Translation- and scale-normalized 3D coordinates (21 * 3)
        - 63..76 (14 values): Canonical pairwise Euclidean distances
        - 77..84  (8 values): Canonical 3D joint angles in radians
        - 85..89  (5 values): Finger extension states (1.0 or 0.0)
        - 90      (1 value):  Normalized palm roll (palm_roll_deg / 180.0)
        - 91      (1 value):  Palm camera facing (+1.0 facing camera, -1.0 facing away)
        
        Total: 63 + 14 + 8 + 5 + 1 + 1 = 92 elements.
        """
        feats: List[float] = []
        feats.extend(self.flattened_coords.tolist())
        feats.extend(self.pairwise_distances)
        feats.extend(self.joint_angles)
        feats.extend([1.0 if f else 0.0 for f in self.fingers_up])
        feats.append(self.palm_roll_deg / 180.0)
        feats.append(1.0 if self.palm_facing_camera else -1.0)
        return np.array(feats, dtype=np.float32)

    @property
    def feature_vector_92d(self) -> np.ndarray:
        """Convenience property returning the 92-D ML vector."""
        return self.to_ml_vector()

    @property
    def finger_states(self) -> List[int]:
        """Convenience property returning [1/0, 1/0, 1/0, 1/0, 1/0] for fingers."""
        return [1 if f else 0 for f in self.fingers_up]

    @property
    def normalized_landmarks(self) -> np.ndarray:
        """Alias for normalized_lms."""
        return self.normalized_lms

    def to_rotation_invariant_vector(self) -> np.ndarray:
        """
        Generates a 92-dimensional ML vector using rotation-aligned landmarks.
        Ideal for gestures whose semantics are independent of hand tilt/rotation.
        """
        feats: List[float] = []
        feats.extend(self.rotation_aligned_lms.flatten().tolist())
        feats.extend(self.pairwise_distances)
        feats.extend(self.joint_angles)
        feats.extend([1.0 if f else 0.0 for f in self.fingers_up])
        feats.append(0.0)  # Roll zeroed in aligned representation
        feats.append(1.0 if self.palm_facing_camera else -1.0)
        return np.array(feats, dtype=np.float32)

    def to_extended_ml_vector(self) -> np.ndarray:
        """
        Produces an extended 128-dimensional multi-modal vector including kinematics
        and full 15-joint angles for high-capacity neural network classifiers.
        """
        feats: List[float] = []
        # Standard 92 features
        feats.extend(self.to_ml_vector().tolist())
        # Kinematic velocities and accelerations (6 features)
        feats.append(self.velocity_xy[0] / 1000.0)
        feats.append(self.velocity_xy[1] / 1000.0)
        feats.append(self.speed / 1000.0)
        feats.append(self.direction_deg / 180.0)
        feats.append(self.acceleration_xy[0] / 5000.0)
        feats.append(self.acceleration_xy[1] / 5000.0)
        # Normal vector and 3D angles (5 features)
        feats.extend(self.palm_normal.tolist())
        feats.append(self.palm_pitch_deg / 180.0)
        feats.append(self.palm_yaw_deg / 180.0)
        # Remaining extended joint angles (7 features to make 110+, pad to 128)
        ext_angles_list = list(self.extended_angles.values())
        feats.extend(ext_angles_list[:15])
        # Pad or trim to exactly 128
        if len(feats) < 128:
            feats.extend([0.0] * (128 - len(feats)))
        else:
            feats = feats[:128]
        return np.array(feats, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes hand feature metrics into a dictionary for JSON reporting."""
        return {
            "handedness": self.handedness,
            "fingers_up": [bool(f) for f in self.fingers_up],
            "palm_roll_deg": round(self.palm_roll_deg, 2),
            "palm_pitch_deg": round(self.palm_pitch_deg, 2),
            "palm_yaw_deg": round(self.palm_yaw_deg, 2),
            "palm_facing_camera": self.palm_facing_camera,
            "velocity_xy": [round(v, 2) for v in self.velocity_xy],
            "speed": round(self.speed, 2),
            "direction_deg": round(self.direction_deg, 2),
            "acceleration": round(self.acceleration, 2),
            "ml_vector_len": len(self.to_ml_vector())
        }


class DualHandFeatures:
    """
    Container for simultaneous dual-hand geometric interaction features:
    inter-hand distance, relative angle, displacement rates, and symmetry.
    """

    def __init__(
        self,
        left_hand: Any,
        right_hand: Any,
        hand_distance_norm: float,
        angle_between_hands_deg: float,
        delta_distance: float = 0.0,
        delta_angle_deg: float = 0.0,
        center_distance_px: float = 0.0,
        symmetry_score: float = 1.0,
        zoom_factor: float = 1.0
    ):
        self.left_hand = left_hand
        self.right_hand = right_hand

        # Distance between hands (aliases for flexible compatibility)
        self.hand_distance_norm: float = float(hand_distance_norm)
        self.distance: float = float(hand_distance_norm)

        # Angle between hands in screen plane (aliases for flexible compatibility)
        self.angle_between_hands_deg: float = float(angle_between_hands_deg)
        self.angle_deg: float = float(angle_between_hands_deg)

        # Temporal delta rates
        self.delta_distance: float = float(delta_distance)
        self.delta_angle_deg: float = float(delta_angle_deg)

        # Metrics
        self.center_distance_px: float = float(center_distance_px)
        self.symmetry_score: float = float(symmetry_score)
        self.zoom_factor: float = float(zoom_factor)

    def to_vector(self) -> np.ndarray:
        """Encodes dual hand geometric interaction into a compact 10D vector."""
        return np.array([
            self.hand_distance_norm,
            self.angle_between_hands_deg / 180.0,
            self.delta_distance * 10.0,
            self.delta_angle_deg / 180.0,
            self.center_distance_px / 1000.0,
            self.symmetry_score,
            self.zoom_factor,
            float(sum(getattr(self.left_hand, "fingers_up", [0]))),
            float(sum(getattr(self.right_hand, "fingers_up", [0]))),
            1.0 if getattr(self.left_hand, "fingers_up", None) == getattr(self.right_hand, "fingers_up", None) else 0.0
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes dual-hand features to a dictionary."""
        return {
            "hand_distance_norm": round(self.hand_distance_norm, 4),
            "angle_between_hands_deg": round(self.angle_between_hands_deg, 2),
            "delta_distance": round(self.delta_distance, 4),
            "delta_angle_deg": round(self.delta_angle_deg, 2),
            "center_distance_px": round(self.center_distance_px, 1),
            "symmetry_score": round(self.symmetry_score, 3),
            "zoom_factor": round(self.zoom_factor, 3)
        }


class LandmarkProcessor:
    """
    Subsystem for normalising 21 3D hand landmarks, calculating joint angles,
    pairwise distances, velocity kinematics, and dual-hand geometry.
    Maintains temporal state for velocity, acceleration, and two-hand tracking.
    """

    def __init__(self, velocity_ema_alpha: float = 0.65):
        """
        Initializes LandmarkProcessor.
        
        Args:
            velocity_ema_alpha: Smoothing factor for exponential moving average velocity (0.0 to 1.0).
        """
        self.velocity_ema_alpha = max(0.1, min(1.0, float(velocity_ema_alpha)))

        # Velocity tracking state per handedness:
        # { handedness: (last_px_x, last_px_y, last_time, last_vx, last_vy) }
        self._history: Dict[str, Tuple[float, float, float, float, float]] = {}

        # Dual hand temporal tracking state
        self._prev_dual_dist: Optional[float] = None
        self._prev_dual_angle: Optional[float] = None
        self._initial_dual_dist: Optional[float] = None

    def reset(self):
        """Clears all temporal velocity and dual-hand state history."""
        self._history.clear()
        self._prev_dual_dist = None
        self._prev_dual_angle = None
        self._initial_dual_dist = None

    def _extract_landmark_coords(self, hand: Any) -> Tuple[List[Tuple[float, float, float]], Tuple[int, int]]:
        """
        Uniformly extracts 21 3D normalized coordinates and index tip pixel coords
        from HandData, HandDetectionResult, NumPy array, or raw landmark list.
        """
        # Case 1: HandData / HandDetectionResult with get_norm() and get_px()
        if hasattr(hand, "get_norm") and hasattr(hand, "get_px"):
            lms_3d = [hand.get_norm(i) for i in range(21)]
            idx_px = hand.get_px(8)
            return lms_3d, idx_px

        # Case 2: Object with landmarks attribute
        if hasattr(hand, "landmarks") and isinstance(hand.landmarks, (list, tuple)):
            lms_raw = hand.landmarks
            lms_3d: List[Tuple[float, float, float]] = []
            idx_px = (0, 0)
            for i, p in enumerate(lms_raw[:21]):
                if hasattr(p, "x") and hasattr(p, "y"):
                    z = getattr(p, "z", 0.0)
                    lms_3d.append((float(p.x), float(p.y), float(z)))
                    if i == 8:
                        idx_px = (int(p.x * 640), int(p.y * 480))
                elif isinstance(p, (list, tuple)) and len(p) >= 3:
                    lms_3d.append((float(p[0]), float(p[1]), float(p[2])))
                    if i == 8:
                        px = int(p[3]) if len(p) >= 5 else int(p[0] * 640)
                        py = int(p[4]) if len(p) >= 5 else int(p[1] * 480)
                        idx_px = (px, py)
                else:
                    lms_3d.append((0.0, 0.0, 0.0))
            return lms_3d, idx_px

        # Case 3: NumPy array of shape (21, 3) or (21, 2)
        if isinstance(hand, np.ndarray) and hand.ndim == 2:
            lms_3d = []
            for i in range(min(21, hand.shape[0])):
                z = float(hand[i, 2]) if hand.shape[1] >= 3 else 0.0
                lms_3d.append((float(hand[i, 0]), float(hand[i, 1]), z))
            while len(lms_3d) < 21:
                lms_3d.append((0.0, 0.0, 0.0))
            idx_px = (int(lms_3d[8][0] * 640), int(lms_3d[8][1] * 480))
            return lms_3d, idx_px

        # Case 4: List or tuple of 21 landmark coordinate points/tuples
        if isinstance(hand, (list, tuple)) and len(hand) >= 21:
            lms_3d = []
            idx_px = (0, 0)
            for i, p in enumerate(hand[:21]):
                if hasattr(p, "x") and hasattr(p, "y"):
                    z = getattr(p, "z", 0.0)
                    lms_3d.append((float(p.x), float(p.y), float(z)))
                    if i == 8:
                        idx_px = (int(p.x * 640), int(p.y * 480))
                elif isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 3:
                    lms_3d.append((float(p[0]), float(p[1]), float(p[2])))
                    if i == 8:
                        px = int(p[3]) if len(p) >= 5 else int(p[0] * 640)
                        py = int(p[4]) if len(p) >= 5 else int(p[1] * 480)
                        idx_px = (px, py)
                else:
                    lms_3d.append((0.0, 0.0, 0.0))
            return lms_3d, idx_px

        # Fallback: Zero array
        return [(0.0, 0.0, 0.0)] * 21, (0, 0)

    def extract_single_hand(self, hand: Any) -> HandFeatures:
        """
        Processes a single hand:
        1. Translates wrist to (0, 0, 0).
        2. Normalizes scale by wrist-to-middle-MCP distance (||P'_9|| = 1.0).
        3. Computes rotation alignment.
        4. Computes 3D joint angles and pairwise distances.
        5. Computes palm orientation, normal, roll, pitch, yaw.
        6. Computes velocity, speed, acceleration, and direction.
        
        Args:
            hand: HandData, HandDetectionResult, or compatible hand structure.
            
        Returns:
            HandFeatures instance.
        """
        now = time.time()
        w_lms, idx_px = self._extract_landmark_coords(hand)

        # -------------------------------------------------------------
        # 1. Translation Invariance: Center wrist (L0) to origin (0, 0, 0)
        # -------------------------------------------------------------
        wrist = np.array(w_lms[0], dtype=np.float64)
        centered = np.array([np.array(p, dtype=np.float64) - wrist for p in w_lms], dtype=np.float64)

        # -------------------------------------------------------------
        # 2. Scale Invariance: Divide by wrist-to-middle-MCP (L0 to L9) distance
        # -------------------------------------------------------------
        scale = np.linalg.norm(centered[9])
        if scale < 1e-4:
            scale = 0.20  # Safe fallback to prevent division by zero
        normalized_lms = centered / scale  # Shape (21, 3)

        # -------------------------------------------------------------
        # 3. Rotation Alignment: In-plane alignment of palm axis to vertical
        # -------------------------------------------------------------
        # Palm vector from wrist (0, 0) to middle MCP (L9) in 2D
        v_palm_x = normalized_lms[9, 0]
        v_palm_y = normalized_lms[9, 1]
        palm_angle = math.atan2(v_palm_y, v_palm_x)

        # Rotate so palm axis points vertically up (-pi/2 in image coords where Y points down)
        rot_angle = -math.pi / 2.0 - palm_angle
        cos_a = math.cos(rot_angle)
        sin_a = math.sin(rot_angle)
        rot_matrix_2d = np.array([
            [cos_a, -sin_a, 0.0],
            [sin_a,  cos_a, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float64)

        rotation_aligned_lms = np.dot(normalized_lms, rot_matrix_2d.T)

        # -------------------------------------------------------------
        # 4. Canonical Pairwise Distances (14 anatomical pairs)
        # -------------------------------------------------------------
        pairwise_dists = [
            float(np.linalg.norm(normalized_lms[i] - normalized_lms[j]))
            for i, j in CANONICAL_DISTANCE_PAIRS
        ]

        # -------------------------------------------------------------
        # 5. 3D Joint Angles (Canonical 8 + Extended 15)
        # -------------------------------------------------------------
        joint_angles = [
            compute_angle_3d(w_lms[p1], w_lms[p2], w_lms[p3])
            for p1, p2, p3 in CANONICAL_ANGLE_TRIPLETS
        ]

        extended_angles: Dict[str, float] = {
            name: compute_angle_3d(w_lms[p1], w_lms[p2], w_lms[p3])
            for name, (p1, p2, p3) in EXTENDED_ANGLE_TRIPLETS.items()
        }

        # -------------------------------------------------------------
        # 6. Palm Orientation & Normal Vector
        # -------------------------------------------------------------
        v_idx = np.array(w_lms[5], dtype=np.float64) - wrist
        v_pnk = np.array(w_lms[17], dtype=np.float64) - wrist
        normal = np.cross(v_idx, v_pnk)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-7:
            unit_normal = normal / norm_len
        else:
            unit_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        # Palm camera facing: In normalized coords, negative Z points into camera
        handedness = getattr(hand, "handedness", "Right")
        palm_facing_camera = unit_normal[2] < 0 if handedness == "Right" else unit_normal[2] > 0

        # Roll angle: Orientation of Index MCP (5) relative to Pinky MCP (17) in screen plane
        dx_roll = w_lms[5][0] - w_lms[17][0]
        dy_roll = w_lms[5][1] - w_lms[17][1]
        palm_roll_deg = math.degrees(math.atan2(dy_roll, dx_roll))

        # Pitch and Yaw angles from 3D unit normal
        palm_pitch_deg = math.degrees(math.atan2(unit_normal[1], max(1e-4, abs(unit_normal[2]))))
        palm_yaw_deg = math.degrees(math.atan2(unit_normal[0], max(1e-4, abs(unit_normal[2]))))

        # -------------------------------------------------------------
        # 7. Kinematic Velocity, Speed & Acceleration
        # -------------------------------------------------------------
        h_key = handedness
        curr_x, curr_y = float(idx_px[0]), float(idx_px[1])
        vel_x, vel_y, speed, direction_deg = 0.0, 0.0, 0.0, 0.0
        acc_x, acc_y, acceleration = 0.0, 0.0, 0.0

        if h_key in self._history:
            prev_x, prev_y, prev_t, prev_vx, prev_vy = self._history[h_key]
            dt = max(1e-3, now - prev_t)
            if dt < 0.5:  # Valid continuous tracking sample (< 500 ms)
                raw_vx = (curr_x - prev_x) / dt  # px/sec
                raw_vy = (curr_y - prev_y) / dt

                # Apply Exponential Moving Average (EMA) smoothing
                alpha = self.velocity_ema_alpha
                vel_x = alpha * raw_vx + (1.0 - alpha) * prev_vx
                vel_y = alpha * raw_vy + (1.0 - alpha) * prev_vy
                speed = math.hypot(vel_x, vel_y)
                direction_deg = math.degrees(math.atan2(vel_y, vel_x))

                # Acceleration
                acc_x = (vel_x - prev_vx) / dt  # px/sec^2
                acc_y = (vel_y - prev_vy) / dt
                acceleration = math.hypot(acc_x, acc_y)

        self._history[h_key] = (curr_x, curr_y, now, vel_x, vel_y)

        return HandFeatures(
            hand=hand,
            normalized_lms=normalized_lms,
            joint_angles=joint_angles,
            pairwise_distances=pairwise_dists,
            palm_roll_deg=palm_roll_deg,
            palm_facing_camera=palm_facing_camera,
            velocity_xy=(vel_x, vel_y),
            speed=speed,
            direction_deg=direction_deg,
            acceleration_xy=(acc_x, acc_y),
            acceleration=acceleration,
            palm_normal=unit_normal.astype(np.float32),
            palm_pitch_deg=palm_pitch_deg,
            palm_yaw_deg=palm_yaw_deg,
            rotation_aligned_lms=rotation_aligned_lms,
            extended_angles=extended_angles
        )

    def extract_dual_hands(self, hands: List[Any]) -> Optional[DualHandFeatures]:
        """
        Computes 3D distance, relative screen angle, delta displacement rates,
        and symmetry metric between two simultaneously tracked hands.
        
        Args:
            hands: List containing at least 2 hand instances.
            
        Returns:
            DualHandFeatures if len(hands) >= 2, else None.
        """
        if len(hands) < 2:
            self._prev_dual_dist = None
            self._prev_dual_angle = None
            self._initial_dual_dist = None
            return None

        # Sort into Left and Right hands (or first two if ambiguous)
        h1, h2 = hands[0], hands[1]
        h1_side = getattr(h1, "handedness", "Right")
        left = h1 if h1_side == "Left" else h2
        right = h2 if h1_side == "Left" else h1

        # Extract normalized wrist coordinates
        w_left, _ = self._extract_landmark_coords(left)
        w_right, _ = self._extract_landmark_coords(right)
        p_left = w_left[0]
        p_right = w_right[0]

        # 1. 3D distance between wrists
        dist = math.sqrt(
            (p_left[0] - p_right[0]) ** 2 +
            (p_left[1] - p_right[1]) ** 2 +
            (p_left[2] - p_right[2]) ** 2
        )

        # 2. Hand Centers
        c_left = getattr(left, "center", (int(p_left[0] * 640), int(p_left[1] * 480)))
        c_right = getattr(right, "center", (int(p_right[0] * 640), int(p_right[1] * 480)))
        center_dist_px = math.hypot(c_right[0] - c_left[0], c_right[1] - c_left[1])

        # 3. Angle in screen plane between hand centers
        angle_deg = math.degrees(math.atan2(c_right[1] - c_left[1], c_right[0] - c_left[0]))

        # 4. Delta calculations
        delta_dist = 0.0
        delta_angle = 0.0

        if self._prev_dual_dist is not None:
            delta_dist = dist - self._prev_dual_dist

        if self._prev_dual_angle is not None:
            delta_angle = angle_deg - self._prev_dual_angle
            # Normalize angle wrap around [-180, 180]
            if delta_angle > 180.0:
                delta_angle -= 360.0
            elif delta_angle < -180.0:
                delta_angle += 360.0

        # Zoom factor relative to initial continuous distance
        if self._initial_dual_dist is None or self._initial_dual_dist < 1e-4:
            self._initial_dual_dist = max(1e-4, dist)
        zoom_factor = dist / self._initial_dual_dist

        self._prev_dual_dist = dist
        self._prev_dual_angle = angle_deg

        # 5. Symmetry metric between hands (finger extension matching)
        fingers_l = getattr(left, "fingers_up", [0, 0, 0, 0, 0])
        fingers_r = getattr(right, "fingers_up", [0, 0, 0, 0, 0])
        matches = sum(1 for fl, fr in zip(fingers_l, fingers_r) if fl == fr)
        symmetry_score = matches / 5.0

        return DualHandFeatures(
            left_hand=left,
            right_hand=right,
            hand_distance_norm=dist,
            angle_between_hands_deg=angle_deg,
            delta_distance=delta_dist,
            delta_angle_deg=delta_angle,
            center_distance_px=center_dist_px,
            symmetry_score=symmetry_score,
            zoom_factor=zoom_factor
        )

    # Aliases for flexible method names
    extract_hand_features = extract_single_hand


# Backward Compatibility Aliases
FeatureExtractor = LandmarkProcessor
