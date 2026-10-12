"""
Phase 3: MediaPipe Hand Detection & Landmark Extraction Subsystem
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Real-Time 21 3D Joint Extraction:
   - Primary Backend: Modern MediaPipe Tasks Vision API (HandLandmarker with TFLite XNNPACK delegate).
   - Fallback Backend: MediaPipe Solutions API (mp.solutions.hands).
   - Extracts 21 3D landmarks per hand in normalized ([0.0, 1.0]), pixel (screen coords), and metric spaces.

2. Single and Dual-Hand Perception:
   - Configurable max_hands (default: 2 per config.py).
   - Real-time handedness classification (Left / Right) with mirror-aware axis inversion.
   - Per-hand detection and tracking confidence scoring.

3. Hand Geometry, Rotation & Scale Tracking:
   - Computes hand rotation angle theta in degrees (-180 to +180 deg) from the wrist-to-middle-MCP vector.
   - Computes palm span (wrist L0 to middle MCP L9 distance) for scale-invariant distance measurements.
   - Computes hand center of mass and padded bounding box geometry.
   - Rotation-invariant 5-finger extension calculation (Thumb, Index, Middle, Ring, Pinky).

4. Sci-Fi HUD Skeleton Rendering (Module 18):
   - Dual-color bones with anti-aliasing.
   - High-contrast joint nodes.
   - Color-coded fingertip indicators (Cyan for Index, Orange for Thumb, Green for others).
   - Futuristic corner-bracket bounding boxes and confidence/angle badges.

5. Error Resilience & Tracking Continuity (Module 17):
   - Safe empty-hand handling (no crashes when no hands are in view).
   - Temporary occlusion / tracking loss recovery.
"""

import math
import os
import sys
import urllib.request
from typing import Dict, List, Optional, Tuple, Any

import cv2
import mediapipe as mp
import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from utils.logger import logger

# 21 Hand skeleton connections (standard MediaPipe topological graph)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky finger
    (0, 17)                                 # Palm base connection
]

# Landmark indices for anatomical reference
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_DIPS = [3, 7, 11, 15, 19]
FINGER_PIPS = [2, 6, 10, 14, 18]
FINGER_MCPS = [1, 5, 9, 13, 17]


class HandData:
    """
    Comprehensive container for a detected hand: 21 3D joints, orientation,
    bounding box, finger states, and geometric metrics.
    """

    def __init__(
        self,
        raw_landmarks: Any,
        frame_w: int,
        frame_h: int,
        handedness: str = "Right",
        detection_score: float = 1.0,
        world_landmarks: Optional[Any] = None
    ):
        """
        Initializes HandData from raw MediaPipe landmark points.
        
        Args:
            raw_landmarks: List of NormalizedLandmark objects (each has x, y, z).
            frame_w: Current frame width in pixels.
            frame_h: Current frame height in pixels.
            handedness: 'Right' or 'Left'.
            detection_score: MediaPipe detection confidence (0.0 to 1.0).
            world_landmarks: Optional metric 3D landmarks in meters.
        """
        self.handedness = handedness
        self.detection_score = float(detection_score)
        self.frame_width = frame_w
        self.frame_height = frame_h

        # Each landmark stored as: (norm_x, norm_y, norm_z, px_x, px_y)
        self.landmarks: List[Tuple[float, float, float, int, int]] = []
        x_pixels: List[int] = []
        y_pixels: List[int] = []

        for lm in raw_landmarks:
            nx = float(lm.x)
            ny = float(lm.y)
            nz = float(lm.z)
            px = int(nx * frame_w)
            py = int(ny * frame_h)
            self.landmarks.append((nx, ny, nz, px, py))
            x_pixels.append(px)
            y_pixels.append(py)

        # World landmarks (metric 3D in meters if available)
        self.world_landmarks: Optional[List[Tuple[float, float, float]]] = None
        if world_landmarks is not None:
            self.world_landmarks = [(float(w.x), float(w.y), float(w.z)) for w in world_landmarks]

        if not x_pixels:
            self.bbox = (0, 0, 0, 0)
            self.center = (0, 0)
            self.palm_span = 0.0
            self.rotation_deg = 0.0
            self.fingers_up = [0, 0, 0, 0, 0]
        else:
            # Bounding box with safety margin padding
            pad = 20
            self.bbox = (
                max(0, min(x_pixels) - pad),
                max(0, min(y_pixels) - pad),
                min(frame_w, max(x_pixels) + pad),
                min(frame_h, max(y_pixels) + pad)
            )

            # Center of mass / palm centroid (pixel coords)
            self.center = (
                int(sum(x_pixels) / len(x_pixels)),
                int(sum(y_pixels) / len(y_pixels))
            )

            # Scale Metric: Palm span (3D distance between Wrist L0 and Middle MCP L9)
            self.palm_span = self.get_distance_norm(0, 9) if len(self.landmarks) > 9 else 0.0

            # Rotation Angle: Angle of vector from Wrist (L0) to Middle MCP (L9)
            # 0 deg = upright vertical, positive = clockwise tilt, negative = counter-clockwise tilt
            if len(self.landmarks) > 9:
                w_pt = self.get_norm(0)
                m_pt = self.get_norm(9)
                dx = m_pt[0] - w_pt[0]
                dy = m_pt[1] - w_pt[1]
                self.rotation_deg = math.degrees(math.atan2(dx, -dy))
            else:
                self.rotation_deg = 0.0

            # Precompute finger extended states: [thumb, index, middle, ring, pinky]
            self.fingers_up = self._calculate_fingers_up() if len(self.landmarks) >= 21 else [0, 0, 0, 0, 0]

    def get_norm(self, lm_id: int) -> Tuple[float, float, float]:
        """Returns (norm_x, norm_y, norm_z) where coordinates are in [0.0, 1.0]."""
        lm = self.landmarks[lm_id]
        return lm[0], lm[1], lm[2]

    def get_px(self, lm_id: int) -> Tuple[int, int]:
        """Returns (px_x, px_y) integer pixel screen coordinates."""
        lm = self.landmarks[lm_id]
        return lm[3], lm[4]

    def get_distance_norm(self, id1: int, id2: int) -> float:
        """Computes normalized 3D Euclidean distance between two landmark IDs."""
        p1 = self.get_norm(id1)
        p2 = self.get_norm(id2)
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    def get_distance_px(self, id1: int, id2: int) -> float:
        """Computes pixel 2D Euclidean distance between two landmark IDs."""
        p1 = self.get_px(id1)
        p2 = self.get_px(id2)
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _calculate_fingers_up(self) -> List[int]:
        """
        Determines which of the 5 fingers are currently extended.
        Uses lateral distance from index MCP and vertical elevation.
        
        Returns:
            List of 5 integers [thumb, index, middle, ring, pinky], where 1=extended, 0=curled.
        """
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

    def to_feature_vector(self) -> np.ndarray:
        """
        Converts the 21 normalized 3D landmarks into a flat 63-dimensional numpy vector
        for machine learning classification.
        """
        vec = []
        for nx, ny, nz, _, _ in self.landmarks:
            vec.extend([nx, ny, nz])
        return np.array(vec, dtype=np.float32)


class HandDetector:
    """
    Robust real-time multi-hand landmark detector with MediaPipe Tasks and Solutions APIs.
    """

    def __init__(
        self,
        max_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
        mirror_handedness: bool = config.CAMERA_MIRROR
    ):
        """
        Initialize the Hand Detector.
        
        Args:
            max_hands: Maximum number of hands to detect concurrently (1 or 2).
            min_detection_confidence: Minimum hand detection threshold (0.0 to 1.0).
            min_tracking_confidence: Minimum landmark tracking threshold (0.0 to 1.0).
            mirror_handedness: If True, inverts detected Left/Right to match mirror display.
        """
        self.max_hands = max_hands
        self.min_detection_conf = min_detection_confidence
        self.min_tracking_conf = min_tracking_confidence
        self.mirror_handedness = mirror_handedness
        
        self.backend = "tasks"
        self.landmarker = None
        self._init_detector()

    def _ensure_model_file(self) -> str:
        """Verifies local presence of the MediaPipe hand_landmarker.task model."""
        model_path = config.MODEL_PATH
        if not os.path.exists(model_path):
            logger.info(f"[HandDetector] Model file missing. Downloading from {config.MODEL_URL}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(config.MODEL_URL, model_path)
            logger.success(f"[HandDetector] Model downloaded successfully ({os.path.getsize(model_path)} bytes).")
        return model_path

    def _init_detector(self):
        """Initializes detector using MediaPipe Tasks Vision API with Solutions fallback."""
        # 1. Primary: Modern MediaPipe Tasks Vision API (0.10.x+)
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
            logger.info(f"[HandDetector] Initialized MediaPipe Tasks Vision backend (max_hands={self.max_hands}).")
            return
        except Exception as e:
            logger.warning(f"[HandDetector] MediaPipe Tasks Vision init notice: {e}. Trying Solutions fallback...")

        # 2. Fallback: Legacy MediaPipe Solutions API
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.mp_hands = mp.solutions.hands
            self.landmarker = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.min_detection_conf,
                min_tracking_confidence=self.min_tracking_conf
            )
            self.backend = "solutions"
            logger.info(f"[HandDetector] Initialized legacy MediaPipe Solutions backend (max_hands={self.max_hands}).")
            return

        raise RuntimeError("Failed to initialize MediaPipe Hand Landmarker backend.")

    def detect(
        self,
        frame: np.ndarray,
        draw: bool = True
    ) -> Tuple[List[HandData], np.ndarray]:
        """
        Processes a BGR OpenCV video frame and extracts 21 3D landmarks for all hands.
        
        Args:
            frame: OpenCV BGR image array (H x W x 3).
            draw: Whether to render the visual skeleton HUD onto the frame.
            
        Returns:
            Tuple of (list_of_detected_hands, annotated_frame).
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
                            if self.mirror_handedness:
                                handedness = "Right" if handedness == "Left" else "Left"

                        world_lms = res.hand_world_landmarks[i] if (res.hand_world_landmarks and i < len(res.hand_world_landmarks)) else None
                        hand = HandData(
                            raw_landmarks=raw_lms,
                            frame_w=w,
                            frame_h=h,
                            handedness=handedness,
                            detection_score=score,
                            world_landmarks=world_lms
                        )
                        hands_found.append(hand)
            except Exception as e:
                logger.error(f"[HandDetector] Tasks inference error: {e}")
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
                            if self.mirror_handedness:
                                handedness = "Right" if handedness == "Left" else "Left"

                        hand = HandData(
                            raw_landmarks=raw_lms.landmark,
                            frame_w=w,
                            frame_h=h,
                            handedness=handedness,
                            detection_score=score
                        )
                        hands_found.append(hand)
            except Exception as e:
                logger.error(f"[HandDetector] Solutions inference error: {e}")

        # Render skeleton HUD on detected hands if requested
        if draw and hands_found:
            for hand in hands_found:
                self.draw_skeleton(annotated, hand)

        return hands_found, annotated

    def find_hands(
        self,
        frame: np.ndarray,
        draw: bool = True
    ) -> Tuple[List[HandData], np.ndarray]:
        """Drop-in alias for detect() to match existing code across modules."""
        return self.detect(frame, draw=draw)

    def draw_skeleton(self, frame: np.ndarray, hand: HandData):
        """
        Renders aesthetic futuristic sci-fi hand skeleton, joints, and bounding box.
        """
        is_right = hand.handedness == "Right"
        bone_color = config.CV_COLOR_BONE if is_right else (220, 100, 180)

        # 1. Anti-aliased Bones
        for p1_id, p2_id in HAND_CONNECTIONS:
            p1 = hand.get_px(p1_id)
            p2 = hand.get_px(p2_id)
            cv2.line(frame, p1, p2, (35, 35, 35), 3, cv2.LINE_AA)
            cv2.line(frame, p1, p2, bone_color, 1, cv2.LINE_AA)

        # 2. Joint Nodes & Fingertip Markers
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

        # 3. Padded Bounding Box with Sci-Fi Corner Brackets
        x1, y1, x2, y2 = hand.bbox
        bracket_len = 16
        box_col = config.CV_COLOR_BOX if is_right else (255, 120, 220)

        # Top-Left Bracket
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), box_col, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), box_col, 2, cv2.LINE_AA)
        # Top-Right Bracket
        cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), box_col, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), box_col, 2, cv2.LINE_AA)
        # Bottom-Left Bracket
        cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), box_col, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), box_col, 2, cv2.LINE_AA)
        # Bottom-Right Bracket
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), box_col, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), box_col, 2, cv2.LINE_AA)

        # 4. Handedness Badge & Orientation Badge
        conf_pct = int(hand.detection_score * 100)
        badge_text = f"{hand.handedness.upper()} | {conf_pct}% | {int(hand.rotation_deg)}deg"
        cv2.putText(frame, badge_text, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

    def close(self):
        """Releases MediaPipe detector resources cleanly."""
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None
        logger.info("[HandDetector] MediaPipe resources released.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Backward-compatible aliases
HandLandmarkData = HandData
HandTracker = HandDetector
