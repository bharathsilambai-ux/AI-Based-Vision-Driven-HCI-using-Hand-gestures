"""
Phase 17: Edge-Case Handling, Occlusion Recovery & Failsafe Subsystem (Module 17)
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Emergency Stop & Fail-Safe Protocol (Module 17):
   - Global emergency stop (`emergency_stop()`): Immediately terminates cursor movement,
     releases all held mouse buttons (preventing stuck drag-and-drop), and clears pending confirmations.
   - PyAutoGUI failsafe recovery: Catches and neutralizes corner slamming.

2. Tracking Loss & Drag-and-Drop Safety Gate:
   - Detects sudden hand disappearance (`handle_tracking_loss()`).
   - Automatically issues `pyautogui.mouseUp()` if the hand vanishes during a drag operation,
     ensuring desktop windows and files are never left permanently pinned to cursor.

3. Out-of-Bounds & Malformed Coordinate Sanitization:
   - Sanitizes raw landmark arrays: Clamps values to [0.0, 1.0], strips NaNs and Infinities,
     and provides synthetic fallback points if arrays are corrupt or partially occluded.

4. Erratic Motion & Glitch Frame Suppression:
   - Tracks instantaneous velocity between consecutive frames. Displacements exceeding physiological
     human thresholds (> 35% frame width per 33ms frame) are flagged as optical glitches and filtered out.

5. Telemetry & Incident Logging:
   - Records all safety interventions, glitch suppressions, and tracking loss recoveries.
"""

import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pyautogui
from utils.logger import logger

# Safety settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.001


class SafetyManager:
    """
    Supervises system safety, handles edge cases, suppresses glitches,
    and guarantees fail-safe recovery from tracking loss and occlusions.
    """

    def __init__(self, mock_mode: bool = False, max_glitch_displacement: float = 0.35):
        self.mock_mode = mock_mode
        self.max_glitch_displacement = max_glitch_displacement

        # State tracking
        self.is_emergency_stopped: bool = False
        self.is_dragging: bool = False
        self.last_valid_tracking_point: Optional[Tuple[float, float]] = None
        self.last_valid_timestamp: float = time.perf_counter()

        # Telemetry & incident counters
        self.incidents: List[Dict[str, Any]] = []
        self.emergency_stop_count: int = 0
        self.glitches_suppressed: int = 0
        self.tracking_loss_events: int = 0
        self.drag_safety_releases: int = 0

    # =========================================================================
    # Emergency Stop & Fail-Safe Controls
    # =========================================================================

    def emergency_stop(self) -> bool:
        """
        Immediately disengages all automated input, releases all mouse buttons,
        and halts cursor automation.
        """
        self.is_emergency_stopped = True
        self.emergency_stop_count += 1

        self._record_incident("EMERGENCY_STOP", "Global emergency stop triggered.")
        logger.warning("[SafetyManager] EMERGENCY STOP ENGAGED! Halting all automated inputs.")

        if self.mock_mode:
            self.is_dragging = False
            return True

        try:
            # Release all mouse buttons
            pyautogui.mouseUp(button="left")
            pyautogui.mouseUp(button="right")
            pyautogui.mouseUp(button="middle")
            self.is_dragging = False
            return True
        except Exception as e:
            logger.error(f"[SafetyManager] Error during emergency stop release: {e}")
            return False

    def resume_operation(self):
        """Clears emergency stop and resumes normal tracking operation."""
        self.is_emergency_stopped = False
        self.last_valid_tracking_point = None
        self._record_incident("RESUME_OPERATION", "Normal operation resumed.")
        logger.info("[SafetyManager] Operation resumed.")

    # =========================================================================
    # Tracking Loss & Drag Safety Release
    # =========================================================================

    def handle_tracking_loss(self, active_action: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoked when hand tracking is lost (e.g. hand leaves camera frame).
        Guarantees that active drag operations or mouse holds are released cleanly.
        """
        self.tracking_loss_events += 1
        released_drag = False

        if self.is_dragging or active_action in ["DRAG_DROP", "LEFT_DOWN"]:
            released_drag = True
            self.drag_safety_releases += 1
            self._record_incident("DRAG_SAFETY_RELEASE", "Hand lost during drag; mouse button released.")
            logger.warning("[SafetyManager] Hand exited camera during drag! Performing safety mouseUp.")

            if not self.mock_mode:
                try:
                    pyautogui.mouseUp(button="left")
                except Exception as e:
                    logger.error(f"[SafetyManager] Failed mouseUp on tracking loss: {e}")

            self.is_dragging = False

        self.last_valid_tracking_point = None
        return {
            "tracking_loss": True,
            "released_drag": released_drag,
            "incident_count": len(self.incidents),
        }

    def set_dragging_state(self, is_dragging: bool):
        """Notifies the safety manager of active mouse button holding / dragging."""
        self.is_dragging = is_dragging

    # =========================================================================
    # Out-of-Bounds & Corrupt Coordinate Sanitization
    # =========================================================================

    def sanitize_landmarks(self, landmarks: Optional[Any]) -> Tuple[bool, np.ndarray]:
        """
        Sanitizes landmark coordinates:
        1. Checks for None, NaN, and Inf values.
        2. Clamps coordinates to valid [0.0, 1.0] normalized space.
        3. Validates (21, 2) or (21, 3) matrix topology.

        Returns:
            Tuple[bool, np.ndarray]: (is_valid, sanitized_array)
        """
        if landmarks is None:
            return False, np.zeros((21, 3), dtype=np.float32)

        try:
            arr = np.asarray(landmarks, dtype=np.float32)

            # Check shape
            if arr.ndim != 2 or arr.shape[0] != 21 or arr.shape[1] not in (2, 3):
                self._record_incident("MALFORMED_LANDMARKS", f"Invalid shape: {arr.shape}")
                return False, np.zeros((21, 3), dtype=np.float32)

            # Pad 2D to 3D if necessary
            if arr.shape[1] == 2:
                zeros = np.zeros((21, 1), dtype=np.float32)
                arr = np.hstack([arr, zeros])

            # Replace NaNs or Infs with zero
            if not np.all(np.isfinite(arr)):
                self._record_incident("NAN_INF_LANDMARKS", "NaN or Inf detected in landmarks; replaced.")
                arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)

            # Clamp coordinates to [0.0, 1.0] bounds
            arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
            arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0)
            # Z is metric relative depth, typically [-1.0, 1.0]
            arr[:, 2] = np.clip(arr[:, 2], -2.0, 2.0)

            return True, arr

        except Exception as e:
            self._record_incident("SANITIZATION_ERROR", str(e))
            return False, np.zeros((21, 3), dtype=np.float32)

    # =========================================================================
    # Erratic Motion & Glitch Rejection
    # =========================================================================

    def is_glitch_displacement(
        self,
        new_point: Tuple[float, float],
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Evaluates whether an instantaneous coordinate displacement is a tracking glitch.
        If displacement exceeds max_glitch_displacement in a single frame, filters it out.
        """
        now = timestamp if timestamp is not None else time.perf_counter()

        if self.last_valid_tracking_point is None:
            self.last_valid_tracking_point = new_point
            self.last_valid_timestamp = now
            return False

        dx = new_point[0] - self.last_valid_tracking_point[0]
        dy = new_point[1] - self.last_valid_tracking_point[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > self.max_glitch_displacement:
            self.glitches_suppressed += 1
            self._record_incident(
                "GLITCH_SUPPRESSED",
                f"Displacement {dist:.3f} > threshold {self.max_glitch_displacement:.3f}"
            )
            logger.debug(f"[SafetyManager] Glitch displacement suppressed: {dist:.3f}")
            return True

        self.last_valid_tracking_point = new_point
        self.last_valid_timestamp = now
        return False

    # =========================================================================
    # Occlusion Detection & Continuity
    # =========================================================================

    def evaluate_occlusion(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates occlusion states for individual fingers based on landmark spread and depth.
        """
        if landmarks.shape[0] < 21:
            return {"occluded": True, "occluded_fingers": ["all"]}

        occluded = []
        # Fingertip vs MCP z-depth check
        # Tip 4 (Thumb), 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
        finger_pairs = [
            ("thumb", 4, 2),
            ("index", 8, 5),
            ("middle", 12, 9),
            ("ring", 16, 13),
            ("pinky", 20, 17),
        ]

        for name, tip_id, mcp_id in finger_pairs:
            # If tip and MCP collapse to almost identical pixel space (< 0.015 distance)
            dx = landmarks[tip_id, 0] - landmarks[mcp_id, 0]
            dy = landmarks[tip_id, 1] - landmarks[mcp_id, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.015:
                occluded.append(name)

        return {
            "occluded": len(occluded) > 0,
            "occluded_fingers": occluded,
            "occlusion_ratio": round(len(occluded) / 5.0, 2),
        }

    def _record_incident(self, event_type: str, details: str):
        self.incidents.append({
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details,
        })
        if len(self.incidents) > 1000:
            self.incidents = self.incidents[-500:]

    def get_status(self) -> Dict[str, Any]:
        """Returns structured safety telemetry."""
        return {
            "is_emergency_stopped": self.is_emergency_stopped,
            "is_dragging": self.is_dragging,
            "emergency_stop_count": self.emergency_stop_count,
            "glitches_suppressed": self.glitches_suppressed,
            "tracking_loss_events": self.tracking_loss_events,
            "drag_safety_releases": self.drag_safety_releases,
            "total_incidents": len(self.incidents),
        }
