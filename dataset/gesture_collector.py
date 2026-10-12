"""
Phase 5: Custom Landmark-Based Hand Gesture Dataset Subsystem
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Standard 14-Gesture Taxonomic Vocabulary (Module 2):
   - Static Gestures: OPEN_PALM, FIST, THUMBS_UP, THUMBS_DOWN, POINT, PEACE, OK, PINCH, THREE_FINGERS, FOUR_FINGERS.
   - Dynamic Kinematic Gestures: SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN.

2. Feature Engineering & High-Dimensional Representations:
   - 92-dimensional canonical feature vectors via Phase 4 LandmarkProcessor:
     [63 normalized coordinates] + [14 pairwise distances] + [8 joint angles] + [5 finger states] + [roll] + [camera facing].
   - 63-dimensional raw normalized coordinates (21 x 3).
   - 4-dimensional kinematic velocities (vx, vy, speed, direction).

3. Advanced 3D Landmark Data Augmentation:
   - Gaussian spatial jitter noise (sigma in [0.005, 0.02]).
   - 3D Euler rotation perturbations (roll, pitch, yaw within +/- 15 deg).
   - Distance/depth scale variations (0.85x to 1.15x).
   - Horizontal mirroring (Left/Right hand symmetry conversion).

4. Automated Physiologically-Grounded Synthetic Generator:
   - Produces 1,400 balanced, high-variance landmark samples (100 samples x 14 classes).
   - Models realistic joint flexion, PIP/DIP curling, thumb abduction, and directional velocity kinematics.
   - Immediately enables offline ML classifier training and cross-validation for Phase 6.

5. Interactive Live Webcam Dataset Collector:
   - Visual countdown, sample counter, automatic frame capture, and session management.
   - Exports dataset to NPZ (.npz), CSV (.csv), and metadata JSON summaries.
"""

import csv
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from features.landmark_processor import LandmarkProcessor, HandFeatures
from utils.logger import logger

# Canonical 14 Gestures specified in Module 2
GESTURE_CLASSES: List[str] = [
    "OPEN_PALM",
    "FIST",
    "THUMBS_UP",
    "THUMBS_DOWN",
    "POINT",
    "PEACE",
    "OK",
    "PINCH",
    "THREE_FINGERS",
    "FOUR_FINGERS",
    "SWIPE_LEFT",
    "SWIPE_RIGHT",
    "SWIPE_UP",
    "SWIPE_DOWN"
]

# Mapping dictionary for label encoding
LABEL_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(GESTURE_CLASSES)}
ID_TO_LABEL: Dict[int, str] = {idx: name for idx, name in enumerate(GESTURE_CLASSES)}

# Synonyms and alias mapping to handle variations across codebase
GESTURE_SYNONYMS: Dict[str, str] = {
    "CLOSED_FIST": "FIST",
    "INDEX_POINTING": "POINT",
    "PEACE_SIGN": "PEACE",
    "TWO_FINGERS": "PEACE",
    "TWO_FINGERS_UP": "PEACE",
    "OK_SIGN": "OK",
    "PINCH_INDEX": "PINCH",
    "FIVE_FINGERS": "OPEN_PALM",
}


def canonicalize_label(label: str) -> str:
    """Normalizes gesture label string to standard GESTURE_CLASSES taxonomy."""
    clean = label.strip().upper().replace(" ", "_")
    return GESTURE_SYNONYMS.get(clean, clean)


class DataAugmenter:
    """
    Applies spatial perturbations and affine transformations to 21 3D hand landmarks
    for robust machine learning generalisation across varied users and camera angles.
    """

    @staticmethod
    def add_gaussian_jitter(
        landmarks: np.ndarray,
        sigma: float = 0.012
    ) -> np.ndarray:
        """Adds zero-mean Gaussian jitter to normalized coordinates."""
        jitter = np.random.normal(0.0, sigma, size=landmarks.shape)
        # Keep wrist at (0, 0, 0)
        jitter[0] = 0.0
        return landmarks + jitter

    @staticmethod
    def random_3d_rotation(
        landmarks: np.ndarray,
        max_angle_deg: float = 12.0
    ) -> np.ndarray:
        """Applies small random 3D rotation perturbations around X, Y, and Z axes."""
        # Convert max angle to radians
        rad_x = math.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
        rad_y = math.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
        rad_z = math.radians(np.random.uniform(-max_angle_deg, max_angle_deg))

        # Rotation matrices
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(rad_x), -math.sin(rad_x)],
            [0.0, math.sin(rad_x),  math.cos(rad_x)]
        ])
        Ry = np.array([
            [math.cos(rad_y), 0.0, math.sin(rad_y)],
            [0.0, 1.0, 0.0],
            [-math.sin(rad_y), 0.0, math.cos(rad_y)]
        ])
        Rz = np.array([
            [math.cos(rad_z), -math.sin(rad_z), 0.0],
            [math.sin(rad_z),  math.cos(rad_z), 0.0],
            [0.0, 0.0, 1.0]
        ])

        R = Rz @ Ry @ Rx
        # Rotate relative to wrist (landmark 0)
        wrist = landmarks[0].copy()
        centered = landmarks - wrist
        rotated = (centered @ R.T) + wrist
        return rotated

    @staticmethod
    def random_scaling(
        landmarks: np.ndarray,
        scale_range: Tuple[float, float] = (0.88, 1.14)
    ) -> np.ndarray:
        """Scales landmarks relative to wrist to simulate hand size / distance variations."""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        wrist = landmarks[0].copy()
        centered = landmarks - wrist
        return (centered * scale) + wrist

    @staticmethod
    def mirror_hand_horizontal(
        landmarks: np.ndarray
    ) -> np.ndarray:
        """Horizontally flips landmarks (inverts X coordinate) for left/right invariance."""
        mirrored = landmarks.copy()
        mirrored[:, 0] = -mirrored[:, 0]
        return mirrored

    @classmethod
    def augment_landmarks(
        cls,
        landmarks: np.ndarray,
        apply_jitter: bool = True,
        apply_rotation: bool = True,
        apply_scale: bool = True,
        jitter_sigma: float = 0.010,
        max_rotation_deg: float = 12.0
    ) -> np.ndarray:
        """Applies a combined stochastic augmentation pipeline to a landmark array."""
        augmented = landmarks.copy()
        if apply_scale:
            augmented = cls.random_scaling(augmented)
        if apply_rotation:
            augmented = cls.random_3d_rotation(augmented, max_angle_deg=max_rotation_deg)
        if apply_jitter:
            augmented = cls.add_gaussian_jitter(augmented, sigma=jitter_sigma)
        return augmented


class GestureDataset:
    """
    Container, validator, and serializer for landmark-based gesture recognition datasets.
    Stores feature matrices, string labels, integer encodings, and raw landmark coordinates.
    """

    def __init__(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None,
        classes: Optional[List[str]] = None
    ):
        self.X: np.ndarray = X if X is not None else np.empty((0, 92), dtype=np.float32)
        self.y: np.ndarray = y if y is not None else np.empty((0,), dtype=object)
        self.landmarks: np.ndarray = landmarks if landmarks is not None else np.empty((0, 21, 3), dtype=np.float32)
        self.velocities: np.ndarray = velocities if velocities is not None else np.empty((0, 4), dtype=np.float32)
        self.classes: List[str] = classes if classes is not None else GESTURE_CLASSES.copy()
        self.y_encoded: np.ndarray = np.array(
            [LABEL_TO_ID.get(canonicalize_label(str(label)), 0) for label in self.y],
            dtype=np.int32
        ) if len(self.y) > 0 else np.empty((0,), dtype=np.int32)

    def __len__(self) -> int:
        return len(self.X)

    def add_sample(
        self,
        feature_vector: np.ndarray,
        label: str,
        landmark_coords: Optional[np.ndarray] = None,
        velocity_vector: Optional[np.ndarray] = None
    ):
        """Appends a single sample to the in-memory dataset."""
        canonical_label = canonicalize_label(label)
        feat = feature_vector.astype(np.float32).reshape(1, -1)

        self.X = np.vstack([self.X, feat]) if len(self.X) > 0 else feat
        self.y = np.append(self.y, canonical_label)
        label_id = LABEL_TO_ID.get(canonical_label, 0)
        self.y_encoded = np.append(self.y_encoded, label_id)

        if landmark_coords is not None:
            lms = landmark_coords.astype(np.float32).reshape(1, 21, 3)
            self.landmarks = np.vstack([self.landmarks, lms]) if len(self.landmarks) > 0 else lms

        if velocity_vector is not None:
            vel = velocity_vector.astype(np.float32).reshape(1, 4)
            self.velocities = np.vstack([self.velocities, vel]) if len(self.velocities) > 0 else vel

    def get_class_distribution(self) -> Dict[str, int]:
        """Calculates sample count per gesture class."""
        dist: Dict[str, int] = {c: 0 for c in self.classes}
        for label in self.y:
            canonical = canonicalize_label(str(label))
            dist[canonical] = dist.get(canonical, 0) + 1
        return dist

    def validate(self) -> Dict[str, Any]:
        """
        Validates dataset consistency, checking for NaNs, Infs, dimension integrity,
        and minimum sample requirements.
        """
        issues: List[str] = []
        n_samples = len(self.X)

        if n_samples == 0:
            issues.append("Dataset is empty.")
            return {"valid": False, "samples": 0, "issues": issues}

        if self.X.shape[1] != 92:
            issues.append(f"Unexpected feature dimension: {self.X.shape[1]} (expected 92).")

        if np.isnan(self.X).any():
            issues.append("Dataset features contain NaN values.")

        if np.isinf(self.X).any():
            issues.append("Dataset features contain infinite values.")

        dist = self.get_class_distribution()
        empty_classes = [c for c, count in dist.items() if count == 0]
        if empty_classes:
            issues.append(f"Classes with 0 samples: {empty_classes}")

        return {
            "valid": len(issues) == 0,
            "total_samples": n_samples,
            "feature_dim": int(self.X.shape[1]),
            "num_classes": len(dist),
            "distribution": dist,
            "issues": issues
        }

    def train_test_split(
        self,
        test_size: float = 0.20,
        val_size: float = 0.10,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs stratified train / validation / test split.
        Returns: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split

        # First split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X,
            self.y_encoded,
            test_size=test_size,
            stratify=self.y_encoded,
            random_state=random_state
        )

        # Second split into train and val
        relative_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_val_size,
            stratify=y_train_val,
            random_state=random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_to_npz(self, filepath: str):
        """Saves dataset in compressed NumPy NPZ format."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            X=self.X,
            y=self.y,
            y_encoded=self.y_encoded,
            landmarks=self.landmarks,
            velocities=self.velocities,
            classes=np.array(self.classes, dtype=object)
        )
        logger.info(f"[GestureDataset] Saved {len(self.X)} samples to NPZ: {filepath}")

    @classmethod
    def load_from_npz(cls, filepath: str) -> "GestureDataset":
        """Loads dataset from compressed NumPy NPZ format."""
        data = np.load(filepath, allow_pickle=True)
        classes_arr = data["classes"].tolist() if "classes" in data else GESTURE_CLASSES.copy()
        return cls(
            X=data["X"],
            y=data["y"],
            landmarks=data["landmarks"] if "landmarks" in data else None,
            velocities=data["velocities"] if "velocities" in data else None,
            classes=classes_arr
        )

    def save_to_csv(self, filepath: str):
        """Saves dataset features and labels to CSV for interoperability and inspection."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        header = [f"f_{i}" for i in range(self.X.shape[1])] + ["label", "class_id"]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(self.X)):
                row = list(self.X[i]) + [self.y[i], int(self.y_encoded[i])]
                writer.writerow(row)
        logger.info(f"[GestureDataset] Saved {len(self.X)} samples to CSV: {filepath}")

    def export_summary(self, filepath: str):
        """Exports JSON summary report detailing class balance, samples, and metrics."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        validation_info = self.validate()
        summary = {
            "dataset_version": "2.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(self.X),
            "feature_dimension": int(self.X.shape[1]) if len(self.X) > 0 else 0,
            "classes_count": len(self.classes),
            "classes": self.classes,
            "distribution": self.get_class_distribution(),
            "validation": validation_info
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[GestureDataset] Exported summary to {filepath}")


class SyntheticDatasetGenerator:
    """
    Synthesizes physiologically plausible 3D landmark configurations for all 14 required gestures
    with anatomical joint limits, realistic thumb placement, and directional velocity kinematics.
    """

    def __init__(self, processor: Optional[LandmarkProcessor] = None):
        self.processor = processor if processor is not None else LandmarkProcessor()

    @staticmethod
    def _generate_base_gesture_landmarks(gesture: str) -> np.ndarray:
        """
        Constructs canonical 21 3D joint landmark coordinates for a specific gesture.
        Landmark indices follow MediaPipe standard:
        0: Wrist
        1-4: Thumb (CMC, MCP, IP, Tip)
        5-8: Index (MCP, PIP, DIP, Tip)
        9-12: Middle (MCP, PIP, DIP, Tip)
        13-16: Ring (MCP, PIP, DIP, Tip)
        17-20: Pinky (MCP, PIP, DIP, Tip)
        """
        lms = np.zeros((21, 3), dtype=np.float64)
        lms[0] = [0.50, 0.70, 0.0]  # Wrist anchor

        # Define finger extension states: (thumb, index, middle, ring, pinky)
        # 1.0 = full extension, 0.0 = tight curl
        curl_map: Dict[str, Tuple[float, float, float, float, float]] = {
            "OPEN_PALM":    (1.0, 1.0, 1.0, 1.0, 1.0),
            "FIST":         (0.0, 0.0, 0.0, 0.0, 0.0),
            "THUMBS_UP":    (1.0, 0.0, 0.0, 0.0, 0.0),
            "THUMBS_DOWN":  (1.0, 0.0, 0.0, 0.0, 0.0),
            "POINT":        (0.0, 1.0, 0.0, 0.0, 0.0),
            "PEACE":        (0.0, 1.0, 1.0, 0.0, 0.0),
            "OK":           (0.0, 0.0, 1.0, 1.0, 1.0),
            "PINCH":        (0.0, 0.0, 0.0, 0.0, 0.0),
            "THREE_FINGERS": (0.0, 1.0, 1.0, 1.0, 0.0),
            "FOUR_FINGERS": (0.0, 1.0, 1.0, 1.0, 1.0),
            "SWIPE_LEFT":   (1.0, 1.0, 1.0, 1.0, 1.0),
            "SWIPE_RIGHT":  (1.0, 1.0, 1.0, 1.0, 1.0),
            "SWIPE_UP":     (1.0, 1.0, 1.0, 1.0, 1.0),
            "SWIPE_DOWN":   (1.0, 1.0, 1.0, 1.0, 1.0),
        }

        states = curl_map.get(gesture, (1.0, 1.0, 1.0, 1.0, 1.0))
        t_state, i_state, m_state, r_state, p_state = states

        # Base MCP joint anchors relative to wrist
        mcp_positions = [
            (-0.04, -0.05, -0.01),  # 1: Thumb CMC
            (-0.02, -0.17, -0.01),  # 5: Index MCP
            (0.01,  -0.18, -0.01),  # 9: Middle MCP
            (0.04,  -0.16, -0.01),  # 13: Ring MCP
            (0.07,  -0.14, -0.01)   # 17: Pinky MCP
        ]
        lms[1] = lms[0] + mcp_positions[0]
        lms[5] = lms[0] + mcp_positions[1]
        lms[9] = lms[0] + mcp_positions[2]
        lms[13] = lms[0] + mcp_positions[3]
        lms[17] = lms[0] + mcp_positions[4]

        # Finger builder helper
        def build_finger(mcp_idx: int, state: float, spread_x: float, len_mult: float = 1.0):
            mcp = lms[mcp_idx]
            if state > 0.5:  # Extended upright
                lms[mcp_idx + 1] = mcp + [spread_x * 0.3, -0.07 * len_mult, -0.01]
                lms[mcp_idx + 2] = mcp + [spread_x * 0.6, -0.13 * len_mult, -0.02]
                lms[mcp_idx + 3] = mcp + [spread_x * 1.0, -0.21 * len_mult, -0.03]
            else:  # Curled into palm
                lms[mcp_idx + 1] = mcp + [0.00, -0.04, 0.02]
                lms[mcp_idx + 2] = mcp + [0.00, -0.01, 0.04]
                lms[mcp_idx + 3] = mcp + [0.00, +0.02, 0.03]

        # Build fingers 5-20
        build_finger(5, i_state, spread_x=-0.01, len_mult=1.0)
        build_finger(9, m_state, spread_x=0.00, len_mult=1.05)
        build_finger(13, r_state, spread_x=0.01, len_mult=0.98)
        build_finger(17, p_state, spread_x=0.02, len_mult=0.88)

        # Build thumb (1-4)
        cmc = lms[1]
        if gesture == "THUMBS_UP":
            lms[2] = cmc + [-0.03, -0.06, 0.0]
            lms[3] = cmc + [-0.04, -0.13, 0.0]
            lms[4] = cmc + [-0.04, -0.20, 0.0]  # Tip pointing straight up
        elif gesture == "THUMBS_DOWN":
            lms[2] = cmc + [-0.03, +0.06, 0.0]
            lms[3] = cmc + [-0.04, +0.13, 0.0]
            lms[4] = cmc + [-0.04, +0.20, 0.0]  # Tip pointing straight down
        elif gesture in ("OK", "PINCH"):
            # Thumb tip touches index tip
            lms[2] = cmc + [-0.02, -0.05, 0.01]
            lms[3] = cmc + [-0.01, -0.09, 0.02]
            idx_tip = lms[8]
            lms[4] = idx_tip + [0.002, 0.002, 0.0]
            if gesture == "PINCH":
                lms[8] = lms[4] + [-0.002, -0.002, 0.0]
        elif t_state > 0.5:  # Open palm thumb extended out
            lms[2] = cmc + [-0.04, -0.05, -0.01]
            lms[3] = cmc + [-0.07, -0.09, -0.02]
            lms[4] = cmc + [-0.10, -0.13, -0.03]
        else:  # Curled / tucked thumb
            lms[2] = cmc + [0.01, -0.02, 0.02]
            lms[3] = cmc + [0.02, -0.01, 0.03]
            lms[4] = cmc + [0.03, +0.01, 0.02]

        # For directional gestures (swipes), rotate the hand orientation relative to wrist
        rot_angle_map = {
            "SWIPE_LEFT": math.radians(-65.0),
            "SWIPE_RIGHT": math.radians(65.0),
            "SWIPE_DOWN": math.radians(170.0),
            "SWIPE_UP": math.radians(0.0),
        }
        if gesture in rot_angle_map:
            rad = rot_angle_map[gesture]
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            wrist = lms[0].copy()
            for idx in range(1, 21):
                dx = lms[idx, 0] - wrist[0]
                dy = lms[idx, 1] - wrist[1]
                lms[idx, 0] = wrist[0] + (dx * cos_a - dy * sin_a)
                lms[idx, 1] = wrist[1] + (dx * sin_a + dy * cos_a)

        return lms

    def generate_dataset(
        self,
        samples_per_class: int = 100,
        output_dir: str = "dataset"
    ) -> GestureDataset:
        """
        Synthesizes a complete balanced gesture dataset across all 14 canonical classes,
        applies stochastic data augmentations, extracts 92-D feature vectors,
        and saves both NPZ and CSV formats along with a JSON summary.
        """
        dataset = GestureDataset()
        augmenter = DataAugmenter()

        # Velocity mappings for dynamic gestures (px/sec in 640x480 frame)
        velocity_presets: Dict[str, Tuple[float, float]] = {
            "SWIPE_LEFT":  (-750.0, 0.0),
            "SWIPE_RIGHT": (+750.0, 0.0),
            "SWIPE_UP":    (0.0, -750.0),
            "SWIPE_DOWN":  (0.0, +750.0),
        }

        logger.info(f"[DatasetGenerator] Generating {samples_per_class} samples for each of {len(GESTURE_CLASSES)} gestures...")

        for gesture_name in GESTURE_CLASSES:
            base_lms = self._generate_base_gesture_landmarks(gesture_name)

            for i in range(samples_per_class):
                # Apply varied stochastic perturbations
                aug_lms = augmenter.augment_landmarks(
                    base_lms,
                    apply_jitter=True,
                    apply_rotation=True,
                    apply_scale=True,
                    jitter_sigma=0.008 + np.random.uniform(0.0, 0.006),
                    max_rotation_deg=10.0
                )

                # Generate synthetic HandFeatures via LandmarkProcessor
                features = self.processor.extract_single_hand(aug_lms)

                # Set kinematics for dynamic gestures
                vel_xy = velocity_presets.get(gesture_name, (0.0, 0.0))
                # Add slight random noise to velocity
                if vel_xy != (0.0, 0.0):
                    vx = vel_xy[0] + np.random.normal(0.0, 50.0)
                    vy = vel_xy[1] + np.random.normal(0.0, 50.0)
                else:
                    vx = np.random.normal(0.0, 8.0)
                    vy = np.random.normal(0.0, 8.0)

                speed = math.hypot(vx, vy)
                direction = math.degrees(math.atan2(vy, vx))
                features.velocity_xy = (vx, vy)
                features.speed = speed
                features.direction_deg = direction

                # Compute canonical 92-D vector
                ml_vector = features.to_ml_vector()

                # Add sample to dataset
                vel_vector = np.array([vx, vy, speed, direction], dtype=np.float32)
                dataset.add_sample(
                    feature_vector=ml_vector,
                    label=gesture_name,
                    landmark_coords=aug_lms,
                    velocity_vector=vel_vector
                )

        # Save to disk
        npz_path = os.path.join(output_dir, "gesture_dataset.npz")
        csv_path = os.path.join(output_dir, "gesture_dataset.csv")
        summary_path = os.path.join(output_dir, "dataset_summary.json")

        dataset.save_to_npz(npz_path)
        dataset.save_to_csv(csv_path)
        dataset.export_summary(summary_path)

        logger.info(f"[DatasetGenerator] Generation complete. Total samples: {len(dataset)} across {len(GESTURE_CLASSES)} classes.")
        return dataset


class DatasetCollector:
    """
    Interactive live webcam dataset collector for real-time sample recording.
    Enables recording custom real-world hand landmark sequences with visual countdowns.
    """

    def __init__(
        self,
        output_dir: str = "dataset",
        processor: Optional[LandmarkProcessor] = None
    ):
        self.output_dir = output_dir
        self.processor = processor if processor is not None else LandmarkProcessor()
        self.active_dataset = GestureDataset()
        self.current_gesture = ""
        self.is_collecting = False
        self.target_samples = 50
        self.collected_in_session = 0

    def start_session(self, gesture_name: str, target_samples: int = 50) -> bool:
        """Starts recording session for the specified gesture label."""
        canonical = canonicalize_label(gesture_name)
        if not canonical:
            return False

        self.current_gesture = canonical
        self.target_samples = target_samples
        self.collected_in_session = 0
        self.is_collecting = True
        logger.info(f"[DatasetCollector] Started session for '{canonical}' (target: {target_samples} frames).")
        return True

    def record_frame(self, hand_data: Any) -> bool:
        """
        Extracts features from a live hand frame and records the sample.
        Returns True when session target is reached.
        """
        if not self.is_collecting or not self.current_gesture:
            return False

        features = self.processor.extract_single_hand(hand_data)
        ml_vector = features.to_ml_vector()
        coords, _ = self.processor._extract_landmark_coords(hand_data)
        lms_arr = np.array(coords, dtype=np.float32)

        self.active_dataset.add_sample(
            feature_vector=ml_vector,
            label=self.current_gesture,
            landmark_coords=lms_arr
        )
        self.collected_in_session += 1

        if self.collected_in_session >= self.target_samples:
            self.is_collecting = False
            logger.info(f"[DatasetCollector] Completed session for '{self.current_gesture}' ({self.collected_in_session} samples).")
            return True

        return False

    def save_dataset(self, filename_prefix: str = "custom_gesture_dataset"):
        """Saves collected samples to NPZ, CSV, and JSON."""
        npz_file = os.path.join(self.output_dir, f"{filename_prefix}.npz")
        csv_file = os.path.join(self.output_dir, f"{filename_prefix}.csv")
        summary_file = os.path.join(self.output_dir, f"{filename_prefix}_summary.json")

        self.active_dataset.save_to_npz(npz_file)
        self.active_dataset.save_to_csv(csv_file)
        self.active_dataset.export_summary(summary_file)
        return npz_file, csv_file, summary_file
