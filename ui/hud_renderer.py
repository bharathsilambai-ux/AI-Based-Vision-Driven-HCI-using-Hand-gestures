"""
Phase 16: Professional Real-Time HUD Overlay & Vision Dashboard Renderer (Module 18)
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Glassmorphism HUD Overlay (Module 18):
   - Translucent frosted glass effect status cards with alpha blending (cv2.addWeighted).
   - High-contrast cyber-styled color palette:
     * Background Cards: Deep Midnight (#0B0E17 with 75% opacity)
     * Accent Primary: Electric Cyan (BGR: 255, 230, 0)
     * Accent Success: Emerald Mint (BGR: 120, 230, 60)
     * Accent Warning: Amber Gold (BGR: 0, 180, 255)
     * Accent Danger: Crimson Coral (BGR: 80, 80, 255)
     * Skeleton Bones: Multi-hue neon gradients per finger digit.

2. Comprehensive Telemetry & Heads-Up Display:
   - Header Bar: Operational Mode, Moving Average FPS, End-to-End Latency (ms).
   - Gesture Recognition Card: Identified gesture, confidence gauge meter, stability counter.
   - Dynamic Trajectory Glow Trail: Fading motion polyline tracing fingertips.
   - Active Interaction Box: Calibrated interaction window with corner targeting reticles.
   - Action Execution Card: Last executed action name and execution timestamp.
   - Multimodal Voice Badge: Voice listening state and recognized voice intent.

3. Standalone & Headless Verification Architecture:
   - Works purely on NumPy image arrays without requiring display hardware, enabling 100% automated CI tests.
"""

import collections
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy as np
from gestures.gesture_mapping import OperationalMode
from gestures.realtime_recognizer import GestureRecognitionResult
from utils.logger import logger

# Hand Landmark Connections (MediaPipe Topology)
HAND_CONNECTIONS = [
    # Palm
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm Base
]

# Distinct Neon Colors per Finger (BGR)
FINGER_COLORS = {
    "thumb": (0, 215, 255),    # Amber Gold
    "index": (255, 230, 0),    # Electric Cyan
    "middle": (255, 120, 0),   # Neon Sky Blue
    "ring": (220, 60, 180),    # Magenta
    "pinky": (180, 40, 255),   # Violet
    "palm": (150, 150, 160),   # Muted Silver
}


class HUDRenderer:
    """
    Renders high-framerate visual HUD overlays onto video frames, including
    skeletal landmark networks, trajectory trails, and status telemetry cards.
    """

    def __init__(
        self,
        show_skeleton: bool = True,
        show_interaction_box: bool = True,
        show_telemetry: bool = True,
        show_trajectory: bool = True,
        interaction_box: Tuple[float, float, float, float] = (0.20, 0.20, 0.80, 0.80),
    ):
        self.show_skeleton = show_skeleton
        self.show_interaction_box = show_interaction_box
        self.show_telemetry = show_telemetry
        self.show_trajectory = show_trajectory
        self.interaction_box = interaction_box  # (x_min, y_min, x_max, y_max) normalized

        # Motion trajectory history buffer
        self.trajectory_buffer: collections.deque = collections.deque(maxlen=24)

        # FPS calculation state
        self._fps_history: collections.deque = collections.deque(maxlen=30)
        self._last_frame_time: float = time.perf_counter()

    def update_fps(self) -> float:
        """Calculates moving average FPS."""
        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._fps_history.append(1.0 / dt)
        return float(np.mean(self._fps_history)) if self._fps_history else 30.0

    def render(
        self,
        frame: np.ndarray,
        recognition_result: Optional[GestureRecognitionResult] = None,
        raw_landmarks: Optional[np.ndarray] = None,
        last_action: Optional[str] = None,
        voice_intent_str: Optional[str] = None,
        latency_ms: float = 0.0,
        fps_override: Optional[float] = None,
    ) -> np.ndarray:
        """
        Renders the complete HUD overlay onto a copy of the input frame.

        Args:
            frame: (H, W, 3) BGR image array.
            recognition_result: Structured result from RealtimeGestureRecognizer.
            raw_landmarks: (21, 2) or (21, 3) landmark coordinates.
            last_action: Last dispatched controller action name.
            voice_intent_str: Active spoken intent or confirmation message.
            latency_ms: Processing pipeline latency in milliseconds.
            fps_override: Optional explicit FPS for testing.

        Returns:
            Rendered BGR image array with HUD.
        """
        h, w, c = frame.shape
        out = frame.copy()

        fps = fps_override if fps_override is not None else self.update_fps()

        # 1. Render Interaction Zone Bounding Box
        if self.show_interaction_box:
            self._draw_interaction_box(out, w, h)

        # 2. Render Hand Skeleton & Keypoints
        if self.show_skeleton and raw_landmarks is not None and len(raw_landmarks) == 21:
            self._draw_hand_skeleton(out, raw_landmarks, w, h)

        # 3. Render Dynamic Motion Trajectory Trail
        if self.show_trajectory and raw_landmarks is not None and len(raw_landmarks) > 8:
            idx_tip = raw_landmarks[8]
            norm_x, norm_y = float(idx_tip[0]), float(idx_tip[1])
            px, py = int(norm_x * w), int(norm_y * h)
            self.trajectory_buffer.append((px, py, time.time()))
            self._draw_trajectory_trail(out)

        # 4. Render Telemetry & Status Cards
        if self.show_telemetry:
            self._draw_telemetry_cards(
                out,
                recognition_result=recognition_result,
                last_action=last_action,
                voice_intent_str=voice_intent_str,
                fps=fps,
                latency_ms=latency_ms,
                w=w,
                h=h,
            )

        return out

    def _draw_interaction_box(self, img: np.ndarray, w: int, h: int):
        """Draws normalized active interaction box with corner reticles."""
        x1, y1, x2, y2 = self.interaction_box
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)

        # Subtle semi-transparent box outline
        overlay = img.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (180, 180, 180), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

        # Draw stylish corner brackets
        corner_len = min(24, (px2 - px1) // 6)
        bracket_color = (255, 230, 0)  # Cyan
        thick = 2

        # Top-Left
        cv2.line(img, (px1, py1), (px1 + corner_len, py1), bracket_color, thick, cv2.LINE_AA)
        cv2.line(img, (px1, py1), (px1, py1 + corner_len), bracket_color, thick, cv2.LINE_AA)
        # Top-Right
        cv2.line(img, (px2, py1), (px2 - corner_len, py1), bracket_color, thick, cv2.LINE_AA)
        cv2.line(img, (px2, py1), (px2, py1 + corner_len), bracket_color, thick, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(img, (px1, py2), (px1 + corner_len, py2), bracket_color, thick, cv2.LINE_AA)
        cv2.line(img, (px1, py2), (px1, py2 - corner_len), bracket_color, thick, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(img, (px2, py2), (px2 - corner_len, py2), bracket_color, thick, cv2.LINE_AA)
        cv2.line(img, (px2, py2), (px2, py2 - corner_len), bracket_color, thick, cv2.LINE_AA)

        # Small label
        cv2.putText(img, "ACTIVE INTERACTION ZONE", (px1 + 6, py1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    def _draw_hand_skeleton(self, img: np.ndarray, landmarks: np.ndarray, w: int, h: int):
        """Draws anti-aliased colored bones and glowing keypoints for all 21 landmarks."""
        pts = []
        for lm in landmarks:
            px = int(float(lm[0]) * w)
            py = int(float(lm[1]) * h)
            pts.append((px, py))

        # Draw bones
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = pts[start_idx]
            pt2 = pts[end_idx]

            # Assign color based on finger group
            if end_idx <= 4:
                color = FINGER_COLORS["thumb"]
            elif end_idx <= 8:
                color = FINGER_COLORS["index"]
            elif end_idx <= 12:
                color = FINGER_COLORS["middle"]
            elif end_idx <= 16:
                color = FINGER_COLORS["ring"]
            elif end_idx <= 20:
                color = FINGER_COLORS["pinky"]
            else:
                color = FINGER_COLORS["palm"]

            cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

        # Draw keypoint dots
        for idx, (px, py) in enumerate(pts):
            # Fingertips (4, 8, 12, 16, 20) get larger glowing markers
            if idx in [4, 8, 12, 16, 20]:
                cv2.circle(img, (px, py), 6, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), 2, (0, 215, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (px, py), 3, (255, 230, 0), -1, cv2.LINE_AA)

    def _draw_trajectory_trail(self, img: np.ndarray):
        """Draws fading neon motion polyline representing dynamic hand gestures."""
        now = time.time()
        # Filter points older than 1.0s
        pts = [(x, y, (1.0 - min(1.0, (now - t) / 0.8))) for x, y, t in self.trajectory_buffer if (now - t) <= 0.8]

        for i in range(1, len(pts)):
            pt1 = (pts[i - 1][0], pts[i - 1][1])
            pt2 = (pts[i][0], pts[i][1])
            alpha = pts[i][2]

            # Dynamic thickness & alpha fade
            thickness = max(1, int(3 * alpha))
            color = (int(255 * alpha), int(200 * alpha), int(50 * alpha))
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

    def _draw_telemetry_cards(
        self,
        img: np.ndarray,
        recognition_result: Optional[GestureRecognitionResult],
        last_action: Optional[str],
        voice_intent_str: Optional[str],
        fps: float,
        latency_ms: float,
        w: int,
        h: int,
    ):
        """Draws sleek cyber translucent HUD cards on top and bottom of frame."""
        overlay = img.copy()

        # 1. Top Status Banner (Card height = 48)
        card_h = 48
        cv2.rectangle(overlay, (0, 0), (w, card_h), (15, 17, 24), -1)

        # 2. Bottom Status Bar (Card height = 36)
        cv2.rectangle(overlay, (0, h - 36), (w, h), (15, 17, 24), -1)

        # Blend with 75% opacity
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # Top Header Elements
        # App Title & Active Mode Badge
        mode_str = recognition_result.active_mode if recognition_result else OperationalMode.MOUSE
        mode_text = f"MODE: [{mode_str}]"
        cv2.putText(img, "VISION HCI", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, mode_text, (130, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 230, 0), 1, cv2.LINE_AA)

        # Subtitle
        cv2.putText(img, "AI-Driven Gesture Engine", (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 170), 1, cv2.LINE_AA)

        # FPS & Latency Badges (Top Right)
        fps_str = f"FPS: {fps:4.1f}"
        lat_str = f"LAT: {latency_ms:4.1f} ms"
        cv2.putText(img, fps_str, (w - 190, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 230, 60), 1, cv2.LINE_AA)
        cv2.putText(img, lat_str, (w - 100, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1, cv2.LINE_AA)

        # Gesture Detection Status Badge
        gesture_name = recognition_result.confirmed_gesture if recognition_result else "NONE"
        confidence = recognition_result.confidence if recognition_result else 0.0

        if gesture_name != "NONE":
            # Active Gesture Pill (Left Middle)
            pill_w = 260
            pill_h = 70
            px, py = 14, 60
            cv2.rectangle(overlay, (px, py), (px + pill_w, py + pill_h), (20, 24, 35), -1)
            cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)
            cv2.rectangle(img, (px, py), (px + pill_w, py + pill_h), (255, 230, 0), 1, cv2.LINE_AA)

            # Gesture Label
            cv2.putText(img, f"GESTURE: {gesture_name}", (px + 10, py + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            # Confidence Progress Bar
            bar_x, bar_y, bar_w, bar_h = px + 10, py + 38, 240, 10
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)
            fill_w = int(bar_w * min(1.0, max(0.0, confidence)))
            conf_color = (120, 230, 60) if confidence >= 0.70 else (0, 180, 255)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), conf_color, -1)
            cv2.putText(img, f"{confidence*100:.0f}%", (px + 10, py + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # Bottom Bar: Last Action & Voice Feedback
        action_text = f"ACTION: {last_action or 'IDLE'}"
        cv2.putText(img, action_text, (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if voice_intent_str:
            voice_text = f"VOICE: \"{voice_intent_str}\""
            cv2.putText(img, voice_text, (w - 300, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)
