"""
Multi-Hand Landmark Tracker Module using MediaPipe.
Extracts 21 3D landmarks per hand (up to 2 hands simultaneously),
computes finger states, bounding geometry, and renders a modern visual HUD.
"""

import math
import os
import sys
from typing import Dict, List, Optional, Tuple
import urllib.request

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import mediapipe as mp
import numpy as np

import config
from utils.logger import logger

# 21 Hand joint connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                                 # Palm base
]

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_MCPS = [2, 5, 9, 13, 17]


class HandData:
    """Encapsulates landmarks and computed geometric metrics for a single hand."""

    def __init__(self, raw_landmarks, frame_w: int, frame_h: int, handedness: str = "Right", score: float = 1.0):
        self.handedness = handedness
        self.detection_score = score
        self.landmarks: List[Tuple[float, float, float, int, int]] = []
        # Each landmark stored as: (norm_x, norm_y, norm_z, px_x, px_y)

        x_pixels = []
        y_pixels = []

        for lm in raw_landmarks:
            nx = float(lm.x)
            ny = float(lm.y)
            nz = float(lm.z)
            px = int(nx * frame_w)
            py = int(ny * frame_h)
            self.landmarks.append((nx, ny, nz, px, py))
            x_pixels.append(px)
            y_pixels.append(py)

        pad = 20
        self.bbox = (
            max(0, min(x_pixels) - pad),
            max(0, min(y_pixels) - pad),
            min(frame_w, max(x_pixels) + pad),
            min(frame_h, max(y_pixels) + pad)
        )
        self.center = (
            int(sum(x_pixels) / len(x_pixels)),
            int(sum(y_pixels) / len(y_pixels))
        )

        # Precompute finger extended states: [thumb, index, middle, ring, pinky]
        self.fingers_up = self._calculate_fingers_up()

    def get_norm(self, lm_id: int) -> Tuple[float, float, float]:
        """Returns (norm_x, norm_y, norm_z)."""
        lm = self.landmarks[lm_id]
        return lm[0], lm[1], lm[2]

    def get_px(self, lm_id: int) -> Tuple[int, int]:
        """Returns (px_x, px_y)."""
        lm = self.landmarks[lm_id]
        return lm[3], lm[4]

    def get_distance_norm(self, id1: int, id2: int) -> float:
        """Normalized 3D Euclidean distance between two landmark IDs."""
        p1 = self.get_norm(id1)
        p2 = self.get_norm(id2)
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    def get_distance_px(self, id1: int, id2: int) -> float:
        """Pixel 2D Euclidean distance between two landmark IDs."""
        p1 = self.get_px(id1)
        p2 = self.get_px(id2)
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _calculate_fingers_up(self) -> List[int]:
        """Determines which of the 5 fingers are currently extended."""
        fingers = [0, 0, 0, 0, 0]

        # 1. Thumb: Check lateral & vertical extension relative to MCP and Index MCP
        thumb_tip = self.get_norm(4)
        thumb_ip = self.get_norm(3)
        thumb_mcp = self.get_norm(2)
        idx_mcp = self.get_norm(5)

        d_thumb_idx_mcp = math.hypot(thumb_tip[0] - idx_mcp[0], thumb_tip[1] - idx_mcp[1])
        if thumb_tip[1] < thumb_mcp[1] - 0.02 and d_thumb_idx_mcp > 0.08:
            fingers[0] = 1
        elif d_thumb_idx_mcp > 0.14:
            fingers[0] = 1

        # 2. Fingers (Index, Middle, Ring, Pinky):
        # 3D distance from wrist (0) to tip vs wrist to PIP, plus vertical elevation
        for i, (tip_id, pip_id, mcp_id) in enumerate(zip(FINGER_TIPS[1:], FINGER_PIPS[1:], FINGER_MCPS[1:]), start=1):
            d_tip_wrist = self.get_distance_norm(tip_id, 0)
            d_pip_wrist = self.get_distance_norm(pip_id, 0)
            tip = self.get_norm(tip_id)
            pip = self.get_norm(pip_id)

            if d_tip_wrist > d_pip_wrist * 1.10:
                fingers[i] = 1
            elif tip[1] < pip[1]:
                fingers[i] = 1

        return fingers


class HandTracker:
    """
    Robust dual-hand landmark tracker with MediaPipe Tasks and Solutions APIs.
    """

    def __init__(
        self,
        max_hands: int = config.MAX_NUM_HANDS,
        min_detection_conf: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_conf: float = config.MIN_TRACKING_CONFIDENCE
    ):
        self.max_hands = max_hands
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf
        self.backend = "tasks"
        self.landmarker = None

        self._init_detector()

    def _ensure_model_file(self) -> str:
        """Ensures modern MediaPipe Tasks model exists locally."""
        model_path = config.MODEL_PATH
        if not os.path.exists(model_path):
            logger.info(f"Model missing. Downloading MediaPipe hand landmarker from {config.MODEL_URL}...")
            urllib.request.urlretrieve(config.MODEL_URL, model_path)
            logger.success(f"Model downloaded: {model_path}")
        return model_path

    def _init_detector(self):
        """Initializes detector with Tasks API or Solutions fallback."""
        # 1. Try modern MediaPipe Tasks Vision API
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = self._ensure_model_file()
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.min_detection_conf,
                min_tracking_confidence=self.min_tracking_conf
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            self.backend = "tasks"
            logger.info(f"HandTracker initialized (MediaPipe Tasks, max_hands={self.max_hands}).")
            return
        except Exception as e:
            logger.warning(f"MediaPipe Tasks API init notice: {e}. Trying legacy Solutions API...")

        # 2. Try legacy MediaPipe Solutions API
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.mp_hands = mp.solutions.hands
            self.landmarker = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.min_detection_conf,
                min_tracking_confidence=self.min_tracking_conf
            )
            self.backend = "solutions"
            logger.info("HandTracker initialized (Legacy MediaPipe Solutions).")
            return

        raise RuntimeError("Failed to initialize MediaPipe hand tracker backend.")

    def find_hands(
        self,
        frame: np.ndarray,
        draw: bool = True
    ) -> Tuple[List[HandData], np.ndarray]:
        """
        Processes BGR frame and returns a list of detected HandData instances
        along with an annotated visualization frame.
        """
        h, w = frame.shape[:2]
        annotated = frame.copy() if draw else frame
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hands_found: List[HandData] = []

        if self.backend == "tasks":
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = self.landmarker.detect(mp_image)
                if res and res.hand_landmarks:
                    for i, raw_lms in enumerate(res.hand_landmarks):
                        handedness = "Right"
                        score = 1.0
                        if res.handedness and i < len(res.handedness) and len(res.handedness[i]) > 0:
                            cat = res.handedness[i][0]
                            handedness = cat.display_name or cat.category_name
                            score = float(cat.score)
                            if config.CAMERA_MIRROR:
                                handedness = "Right" if handedness == "Left" else "Left"
                        hand = HandData(raw_lms, w, h, handedness=handedness, score=score)
                        hands_found.append(hand)
            except Exception as e:
                logger.error(f"Inference error in Tasks backend: {e}")
        else:
            try:
                res = self.landmarker.process(rgb)
                if res.multi_hand_landmarks:
                    for i, raw_lms in enumerate(res.multi_hand_landmarks):
                        handedness = "Right"
                        score = 1.0
                        if res.multi_handedness and i < len(res.multi_handedness):
                            cat = res.multi_handedness[i].classification[0]
                            handedness = cat.label
                            score = float(cat.score)
                            if config.CAMERA_MIRROR:
                                handedness = "Right" if handedness == "Left" else "Left"
                        hand = HandData(raw_lms.landmark, w, h, handedness=handedness, score=score)
                        hands_found.append(hand)
            except Exception as e:
                logger.error(f"Inference error in Solutions backend: {e}")

        # Render skeleton on detected hands if requested
        if draw and hands_found:
            for hand in hands_found:
                self.draw_skeleton(annotated, hand)

        return hands_found, annotated

    def draw_skeleton(self, frame: np.ndarray, hand: HandData):
        """Draws aesthetic sci-fi hand skeleton, joints, and bounding box."""
        is_right = hand.handedness == "Right"
        bone_color = config.CV_COLOR_BONE if is_right else (220, 100, 180) # Cyan or Violet

        # 1. Bones
        for p1_id, p2_id in HAND_CONNECTIONS:
            p1 = hand.get_px(p1_id)
            p2 = hand.get_px(p2_id)
            cv2.line(frame, p1, p2, (40, 40, 40), 3, cv2.LINE_AA)
            cv2.line(frame, p1, p2, bone_color, 1, cv2.LINE_AA)

        # 2. Joints & Nodes
        for i in range(21):
            pt = hand.get_px(i)
            if i in FINGER_TIPS:
                color = config.CV_COLOR_INDEX_TIP if i == 8 else (
                    config.CV_COLOR_THUMB_TIP if i == 4 else config.CV_COLOR_TIP
                )
                cv2.circle(frame, pt, 7, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 9, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 3, config.CV_COLOR_JOINT, -1, cv2.LINE_AA)

        # 3. Bounding Box & Sci-Fi Corner Brackets
        x1, y1, x2, y2 = hand.bbox
        bracket_len = 16
        box_col = config.CV_COLOR_BOX if is_right else (255, 120, 220)

        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), box_col, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), box_col, 2, cv2.LINE_AA)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), box_col, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), box_col, 2, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), box_col, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), box_col, 2, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), box_col, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), box_col, 2, cv2.LINE_AA)

        # Badge Label
        label = f"{hand.handedness} ({int(hand.detection_score * 100)}%)"
        cv2.putText(frame, label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

    def close(self):
        """Releases detector resources cleanly."""
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None
