"""
Universal MediaPipe Hand Detector Module
========================================
Supports both modern MediaPipe Tasks API (MediaPipe >= 0.10.x / Python 3.13)
and legacy MediaPipe Solutions API with automatic asset management.
"""

import os
import cv2
import urllib.request
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import mediapipe as mp
import config

logger = logging.getLogger(__name__)

# Standard 21 Hand Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


class HandDetector:
    """Universal hand detector supporting Tasks API and Solutions API."""

    def __init__(
        self,
        max_num_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.use_tasks_api = False
        self.landmarker = None
        self.hands_solution = None

        self.last_landmarks = []
        self.last_handedness = []

        self._initialize_backend()

    def _initialize_backend(self):
        """Initializes Tasks API or legacy Solutions API."""
        # 1. Try legacy solutions API first if present
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self.mp_hands = mp.solutions.hands
                self.hands_solution = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                logger.info("Initialized MediaPipe Solutions Hands backend.")
                return
            except Exception as e:
                logger.warning("Solutions API init failed (%s), falling back to Tasks API...", e)

        # 2. Use modern MediaPipe Tasks API
        self.use_tasks_api = True
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # Locate model asset
            possible_paths = [
                Path(__file__).resolve().parent / "hand_landmarker.task",
                config.BASE_DIR / "hand_landmarker.task",
                Path("hand_landmarker.task").resolve(),
            ]
            model_path = None
            for p in possible_paths:
                if p.exists() and p.stat().st_size > 1000000:
                    model_path = p
                    break

            if not model_path:
                target_path = Path(__file__).resolve().parent / "hand_landmarker.task"
                logger.info("Downloading hand_landmarker.task asset from Google CDN...")
                urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, str(target_path))
                model_path = target_path

            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_tracking_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            logger.info("Initialized MediaPipe Tasks HandLandmarker successfully.")
        except Exception as e:
            logger.error("Failed to initialize MediaPipe HandLandmarker: %s", str(e))
            raise

    def find_hands(self, frame: np.ndarray):
        """Processes a BGR frame and caches detected landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.use_tasks_api and self.landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.landmarker.detect(mp_image)
            self.last_landmarks = result.hand_landmarks if result else []
            self.last_handedness = result.handedness if result else []
            return result
        elif self.hands_solution:
            rgb_frame.flags.writeable = False
            result = self.hands_solution.process(rgb_frame)
            rgb_frame.flags.writeable = True
            if result and result.multi_hand_landmarks:
                self.last_landmarks = result.multi_hand_landmarks
                self.last_handedness = result.multi_handedness if result.multi_handedness else []
            else:
                self.last_landmarks = []
                self.last_handedness = []
            return result
        return None

    def get_hand_data(self, frame_shape: Tuple[int, int], hand_index: int = 0) -> Optional[Dict[str, Any]]:
        """
        Extracts landmark coordinates and bounding box for the specified hand.
        """
        if not self.last_landmarks or hand_index >= len(self.last_landmarks):
            return None

        h, w = frame_shape[:2]
        hand_lms = self.last_landmarks[hand_index]

        landmarks_norm: List[Tuple[float, float, float]] = []
        landmarks_pixel: List[Tuple[int, int]] = []
        x_pixels: List[int] = []
        y_pixels: List[int] = []

        # Handle both Tasks API (list of landmark objects) and Solutions API (NormalizedLandmarkList)
        lms_iterable = hand_lms.landmark if hasattr(hand_lms, "landmark") else hand_lms

        for lm in lms_iterable:
            landmarks_norm.append((lm.x, lm.y, lm.z))
            px, py = int(lm.x * w), int(lm.y * h)
            landmarks_pixel.append((px, py))
            x_pixels.append(px)
            y_pixels.append(py)

        padding = 15
        xmin = max(0, min(x_pixels) - padding)
        ymin = max(0, min(y_pixels) - padding)
        xmax = min(w, max(x_pixels) + padding)
        ymax = min(h, max(y_pixels) + padding)

        handedness = "Right"
        confidence = 0.95
        if self.last_handedness and hand_index < len(self.last_handedness):
            hand_entry = self.last_handedness[hand_index]
            # Tasks API entry is list of Category
            if isinstance(hand_entry, list) and len(hand_entry) > 0:
                cat = hand_entry[0]
                handedness = cat.category_name or cat.display_name or "Right"
                confidence = float(cat.score)
            elif hasattr(hand_entry, "classification") and len(hand_entry.classification) > 0:
                cls_obj = hand_entry.classification[0]
                handedness = cls_obj.label
                confidence = float(cls_obj.score)

        return {
            "landmarks_norm": landmarks_norm,
            "landmarks_pixel": landmarks_pixel,
            "bbox": (xmin, ymin, xmax, ymax),
            "handedness": handedness,
            "confidence": confidence,
        }

    def draw_hand_mesh(self, frame: np.ndarray, hand_index: int = 0, modern_style: bool = True) -> np.ndarray:
        """Renders aesthetic cybernetic hand skeleton and landmarks onto frame."""
        if not self.last_landmarks or hand_index >= len(self.last_landmarks):
            return frame

        h, w = frame.shape[:2]
        hand_lms = self.last_landmarks[hand_index]
        lms_iterable = hand_lms.landmark if hasattr(hand_lms, "landmark") else hand_lms

        points = []
        for lm in lms_iterable:
            points.append((int(lm.x * w), int(lm.y * h)))

        # Draw connecting bones
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                pt1 = points[start_idx]
                pt2 = points[end_idx]
                cv2.line(frame, pt1, pt2, (200, 240, 100), 2, cv2.LINE_AA)

        # Draw key joints
        fingertip_indices = {4, 8, 12, 16, 20}
        for idx, (px, py) in enumerate(points):
            if idx in fingertip_indices:
                cv2.circle(frame, (px, py), 7, (0, 225, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)
            elif idx == 0:  # Wrist
                cv2.circle(frame, (px, py), 8, (255, 120, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 10, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 4, (100, 255, 100), -1, cv2.LINE_AA)

        return frame

    def close(self):
        """Releases detector resources."""
        if self.landmarker and hasattr(self.landmarker, "close"):
            self.landmarker.close()
        if self.hands_solution and hasattr(self.hands_solution, "close"):
            self.hands_solution.close()
