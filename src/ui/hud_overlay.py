"""
OpenCV Glassmorphism Heads-Up Display (HUD) Overlay Module
==========================================================
Renders real-time telemetry cards, confidence meters, active control
zones, FPS counters, and cybernetic hand markers onto video frames.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import config


class HUDOverlay:
    """Renders modern, translucent UI overlays directly on OpenCV frames."""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_bold = cv2.FONT_HERSHEY_DUPLEX

    def draw_glass_rect(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int] = config.COLOR_BG_DARK,
        alpha: float = 0.65,
        border_color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Draws a translucent frosted-glass rectangle with optional border."""
        x1, y1 = pt1
        x2, y2 = pt2
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return img

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

        if border_color:
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

        return img

    def draw_active_region(self, frame: np.ndarray, is_active: bool = True) -> np.ndarray:
        """Draws bounding box demarcating active cursor control region with corner accents."""
        h, w = frame.shape[:2]
        x1 = config.MARGIN_LEFT
        y1 = config.MARGIN_TOP
        x2 = w - config.MARGIN_RIGHT
        y2 = h - config.MARGIN_BOTTOM

        color = (0, 220, 255) if is_active else (100, 100, 100)
        # Subtle dotted/thin inner frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 70, 80), 1, cv2.LINE_AA)

        # High-tech sci-fi corner brackets
        bracket_len = 20
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, 2, cv2.LINE_AA)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, 2, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, 2, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, 2, cv2.LINE_AA)

        # Label
        cv2.putText(frame, "ACTIVE ZONE", (x1 + 6, y1 - 6), self.font, 0.35, (180, 190, 200), 1, cv2.LINE_AA)
        return frame

    def render_hud(
        self,
        frame: np.ndarray,
        gesture_name: str,
        action_name: str,
        confidence: float,
        fps: float,
        mode: str = "Rule-Based",
        control_enabled: bool = True,
        hand_data: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Renders complete production telemetry overlay onto camera frame.
        """
        h, w = frame.shape[:2]

        # 1. Top Navigation Bar Card
        self.draw_glass_rect(frame, (10, 10), (w - 10, 50), color=(15, 18, 25), alpha=0.8, border_color=(60, 70, 80))

        # Project Title
        cv2.putText(frame, "VISION HCI CONTROLLER", (22, 35), self.font_bold, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Engine Mode Badge
        mode_color = (255, 180, 50) if mode == "Rule-Based" else (100, 220, 100)
        cv2.putText(frame, f"MODE: {mode.upper()}", (w - 330, 35), self.font, 0.45, mode_color, 1, cv2.LINE_AA)

        # Control Status Badge
        status_text = "CONTROL: ON" if control_enabled else "CONTROL: OFF"
        status_color = (80, 220, 80) if control_enabled else (80, 80, 220)
        cv2.putText(frame, status_text, (w - 180, 35), self.font_bold, 0.45, status_color, 1, cv2.LINE_AA)

        # FPS Counter
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 65, 35), self.font_bold, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

        # 2. Draw Active Region Bounding Box
        self.draw_active_region(frame, is_active=control_enabled)

        # 3. Bottom Gesture & Telemetry Card
        card_h = 100
        y_start = h - card_h - 10
        self.draw_glass_rect(frame, (10, y_start), (w - 10, h - 10), color=(15, 18, 25), alpha=0.85, border_color=(70, 80, 95))

        # Detected Gesture Section (Left side)
        cv2.putText(frame, "DETECTED GESTURE", (24, y_start + 24), self.font, 0.40, (160, 170, 180), 1, cv2.LINE_AA)
        disp_gesture = config.GESTURE_DISPLAY_NAMES.get(gesture_name, gesture_name)
        cv2.putText(frame, disp_gesture, (24, y_start + 56), self.font_bold, 0.65, (0, 230, 255), 2, cv2.LINE_AA)

        # Confidence Progress Bar
        bar_x = 24
        bar_y = y_start + 72
        bar_w = 200
        bar_h = 8
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 55, 65), -1)
        fill_w = int(bar_w * min(1.0, max(0.0, confidence)))

        # Dynamic bar color
        if confidence >= 0.80:
            bar_col = (80, 220, 80)
        elif confidence >= 0.60:
            bar_col = (50, 180, 240)
        else:
            bar_col = (80, 80, 220)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
        cv2.putText(frame, f"{int(confidence * 100)}%", (bar_x + bar_w + 8, bar_y + 8), self.font, 0.35, (200, 210, 220), 1, cv2.LINE_AA)

        # Action Dispatch Section (Right side)
        right_x = int(w * 0.52)
        cv2.putText(frame, "MAPPED ACTION", (right_x, y_start + 24), self.font, 0.40, (160, 170, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, action_name, (right_x, y_start + 56), self.font_bold, 0.60, (100, 255, 140), 2, cv2.LINE_AA)

        # 4. Draw Hand Bounding Box and Handedness if detected
        if hand_data:
            bbox = hand_data.get("bbox")
            if bbox:
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (80, 200, 120), 1, cv2.LINE_AA)
                handedness = hand_data.get("handedness", "Hand")
                score = hand_data.get("confidence", 0.9)
                tag = f"{handedness} ({int(score * 100)}%)"
                cv2.putText(frame, tag, (xmin, max(15, ymin - 6)), self.font, 0.38, (120, 240, 160), 1, cv2.LINE_AA)

        return frame
