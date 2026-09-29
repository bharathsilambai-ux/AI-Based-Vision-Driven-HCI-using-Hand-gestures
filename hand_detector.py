"""
Hand Detector module using MediaPipe.
Provides dual-mode support:
1. Modern MediaPipe Tasks API (mp.tasks.python.vision.HandLandmarker)
2. Legacy MediaPipe Solutions API (mp.solutions.hands) fallback

Extracts 21 3D hand landmarks, calculates finger extended states,
computes pairwise landmark distances, and renders rich HUD overlays.
"""

import math
import os
import urllib.request
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import config

# Hand skeleton connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                                 # Palm base
]

# Tip landmarks for [Thumb, Index, Middle, Ring, Pinky]
FINGER_TIPS = [4, 8, 12, 16, 20]
# PIP joints (or IP for thumb)
FINGER_PIPS = [3, 6, 10, 14, 18]
# MCP joints (base of fingers)
FINGER_MCPS = [2, 5, 9, 13, 17]


class HandLandmarkData:
    """Convenience container for hand landmarks and metadata."""
    def __init__(self, raw_landmarks, frame_w: int, frame_h: int, handedness: str = "Unknown"):
        self.handedness = handedness
        self.landmarks: List[Tuple[float, float, float, int, int]] = []
        # Store as (norm_x, norm_y, norm_z, px_x, px_y)
        x_coords = []
        y_coords = []

        for lm in raw_landmarks:
            nx = float(lm.x)
            ny = float(lm.y)
            nz = float(lm.z)
            px = int(nx * frame_w)
            py = int(ny * frame_h)
            self.landmarks.append((nx, ny, nz, px, py))
            x_coords.append(px)
            y_coords.append(py)

        # Calculate bounding box with padding
        pad = 20
        self.bbox = (
            max(0, min(x_coords) - pad),
            max(0, min(y_coords) - pad),
            min(frame_w, max(x_coords) + pad),
            min(frame_h, max(y_coords) + pad)
        )

    def get_norm(self, lm_id: int) -> Tuple[float, float, float]:
        """Returns (x, y, z) normalized coordinates (0.0 to 1.0)."""
        lm = self.landmarks[lm_id]
        return lm[0], lm[1], lm[2]

    def get_px(self, lm_id: int) -> Tuple[int, int]:
        """Returns (px, py) pixel coordinates."""
        lm = self.landmarks[lm_id]
        return lm[3], lm[4]


class HandDetector:
    """
    Robust real-time Hand Detector supporting MediaPipe Tasks & Solutions APIs.
    """

    def __init__(
        self,
        max_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE
    ):
        self.max_hands = max_hands
        self.min_detection_conf = min_detection_confidence
        self.min_tracking_conf = min_tracking_confidence
        self.mode = "tasks"  # "tasks" or "solutions"

        self.landmarker = None
        self._init_detector()

    def _ensure_model_file(self) -> str:
        """Ensures the hand_landmarker.task model is available locally."""
        model_path = config.MODEL_PATH
        if not os.path.exists(model_path):
            print(f"[HandDetector] Model file missing: {model_path}")
            print(f"[HandDetector] Downloading from {config.MODEL_URL}...")
            try:
                urllib.request.urlretrieve(config.MODEL_URL, model_path)
                print(f"[HandDetector] Download complete ({os.path.getsize(model_path)} bytes)")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download MediaPipe hand landmark model: {e}\n"
                    f"Please download manually from {config.MODEL_URL} to {model_path}"
                )
        return model_path

    def _init_detector(self):
        """Initializes the detector using the best available MediaPipe API."""
        # 1. Try modern MediaPipe Tasks API (0.10.x+)
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
            self.mode = "tasks"
            print("[HandDetector] Initialized using MediaPipe Tasks Vision API.")
            return
        except Exception as e:
            print(f"[HandDetector] MediaPipe Tasks API init notice: {e}")

        # 2. Fallback to legacy Solutions API if available
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.mp_hands = mp.solutions.hands
            self.landmarker = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.min_detection_conf,
                min_tracking_confidence=self.min_tracking_conf
            )
            self.mode = "solutions"
            print("[HandDetector] Initialized using legacy MediaPipe Solutions API.")
            return

        raise RuntimeError("Could not initialize MediaPipe hand detection backend.")

    def find_hands(
        self,
        frame: np.ndarray,
        draw: bool = True
    ) -> Tuple[Optional[HandLandmarkData], np.ndarray]:
        """
        Processes a BGR OpenCV frame and detects hands.
        Returns:
            (HandLandmarkData or None, annotated_frame)
        """
        frame_h, frame_w = frame.shape[:2]
        annotated = frame.copy() if draw else frame

        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_data: Optional[HandLandmarkData] = None

        if self.mode == "tasks":
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect(mp_image)

                if result and result.hand_landmarks and len(result.hand_landmarks) > 0:
                    raw_lms = result.hand_landmarks[0]
                    handedness = "Right"
                    if result.handedness and len(result.handedness) > 0:
                        handedness = result.handedness[0][0].display_name or result.handedness[0][0].category_name
                    hand_data = HandLandmarkData(raw_lms, frame_w, frame_h, handedness)
            except Exception as e:
                print(f"[HandDetector] Warning during inference: {e}")
        else:
            # Solutions fallback
            try:
                result = self.landmarker.process(rgb)
                if result.multi_hand_landmarks and len(result.multi_hand_landmarks) > 0:
                    raw_lms = result.multi_hand_landmarks[0].landmark
                    handedness = "Right"
                    if result.multi_handedness and len(result.multi_handedness) > 0:
                        handedness = result.multi_handedness[0].classification[0].label
                    hand_data = HandLandmarkData(raw_lms, frame_w, frame_h, handedness)
            except Exception as e:
                print(f"[HandDetector] Solutions inference warning: {e}")

        if hand_data is not None and draw:
            self.draw_skeleton(annotated, hand_data)

        return hand_data, annotated

    def draw_skeleton(self, frame: np.ndarray, hand: HandLandmarkData):
        """Renders modern, sci-fi styled hand skeleton & fingertip nodes."""
        # 1. Draw Bones
        for p1_id, p2_id in HAND_CONNECTIONS:
            p1 = hand.get_px(p1_id)
            p2 = hand.get_px(p2_id)
            # Bone line with anti-aliasing
            cv2.line(frame, p1, p2, (80, 80, 80), 3, cv2.LINE_AA)
            cv2.line(frame, p1, p2, config.CV_COLOR_BONE, 1, cv2.LINE_AA)

        # 2. Draw Joints
        for i in range(21):
            pt = hand.get_px(i)
            if i in FINGER_TIPS:
                # Highlighted Fingertip Node (Outer glow + Inner dot)
                color = config.CV_COLOR_TIP if i != 8 else (0, 230, 255) # Cyan for index
                cv2.circle(frame, pt, 7, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 9, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 3, config.CV_COLOR_JOINT, -1, cv2.LINE_AA)

        # 3. Draw Bounding Box with sci-fi bracket corners
        x1, y1, x2, y2 = hand.bbox
        line_len = 16
        col = (100, 220, 100)
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + line_len, y1), col, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + line_len), col, 2, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - line_len, y1), col, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + line_len), col, 2, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + line_len, y2), col, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - line_len), col, 2, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - line_len, y2), col, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - line_len), col, 2, cv2.LINE_AA)

        # Label
        label = f"{hand.handedness} Hand"
        cv2.putText(frame, label, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    @staticmethod
    def count_fingers_up(hand: HandLandmarkData) -> List[int]:
        """
        Determines which fingers are raised [Thumb, Index, Middle, Ring, Pinky].
        Returns a list of 5 integers (1 if raised, 0 if curled/down).
        """
        fingers = [0, 0, 0, 0, 0]

        # 1. Thumb calculation:
        # Check angle or relative horizontal distance to MCP joint
        # Note: Thumb extends sideways. When hand is upright:
        thumb_tip = hand.get_norm(4)
        thumb_ip = hand.get_norm(3)
        thumb_mcp = hand.get_norm(2)
        wrist = hand.get_norm(0)

        # Thumb vector distance from pinky base (landmark 17)
        pinky_mcp = hand.get_norm(17)
        dist_tip_to_pinky = math.hypot(thumb_tip[0] - pinky_mcp[0], thumb_tip[1] - pinky_mcp[1])
        dist_ip_to_pinky = math.hypot(thumb_ip[0] - pinky_mcp[0], thumb_ip[1] - pinky_mcp[1])
        # If tip is further from pinky MCP than IP joint is, thumb is extended
        if dist_tip_to_pinky > dist_ip_to_pinky * 1.1:
            fingers[0] = 1

        # 2. Four Fingers (Index, Middle, Ring, Pinky):
        # Finger is extended upward if tip Y is above (less than) PIP Y and MCP Y
        for idx, (tip_id, pip_id, mcp_id) in enumerate(zip(FINGER_TIPS[1:], FINGER_PIPS[1:], FINGER_MCPS[1:]), start=1):
            tip = hand.get_norm(tip_id)
            pip = hand.get_norm(pip_id)
            mcp = hand.get_norm(mcp_id)

            # In image space, lower Y means higher vertically
            if tip[1] < pip[1] and pip[1] < mcp[1]:
                fingers[idx] = 1
            else:
                fingers[idx] = 0

        return fingers

    @staticmethod
    def calculate_distance(hand: HandLandmarkData, id1: int, id2: int) -> float:
        """Calculates normalized Euclidean distance between two landmarks."""
        p1 = hand.get_norm(id1)
        p2 = hand.get_norm(id2)
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def close(self):
        """Releases detector resources cleanly."""
        if self.landmarker is not None:
            if hasattr(self.landmarker, "close"):
                self.landmarker.close()
            self.landmarker = None
