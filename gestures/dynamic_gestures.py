"""
Phase 7: Dynamic Hand Gesture Recognition & Motion Tracking Subsystem
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Temporal Kinematic Trajectory Tracking:
   - Maintains a sliding temporal window of recent landmark positions (Wrist, Index Tip, Palm Center).
   - Computes instantaneous and smoothed velocities (vx, vy in px/sec), scalar speed, and net displacement.

2. Robust Dynamic Gesture Detection:
   - Detects 4 Canonical Swipes: SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN.
   - Detects PINCH_DRAG (continuous movement during active pinch state).
   - Detects CIRCLE_CLOCKWISE and CIRCLE_COUNTER_CLOCKWISE gestures via angular accumulation.
   - Enforces directional dominance ratio (e.g. |dx| > 1.5 * |dy| for horizontal swipes) to prevent diagonal ambiguity.

3. Velocity and Displacement Thresholding:
   - Configurable minimum displacement (in pixels or normalized units) and minimum speed.
   - Prevents accidental micro-movements or jitter from triggering false swipes.

4. Gesture Debouncing & Cooldown (Module 6):
   - Dynamic gestures require a cool-off window before re-triggering the same gesture.
"""

import collections
import math
import os
import sys
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.logger import logger

DYNAMIC_GESTURE_CLASSES = [
    "SWIPE_LEFT",
    "SWIPE_RIGHT",
    "SWIPE_UP",
    "SWIPE_DOWN",
    "PINCH_DRAG",
    "CIRCLE_CW",
    "CIRCLE_CCW",
]


class TrajectoryPoint:
    """Represents a timestamped 2D/3D tracking point in screen or normalized space."""

    def __init__(self, x: float, y: float, z: float = 0.0, timestamp: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.timestamp = float(timestamp) if timestamp > 0 else time.perf_counter()

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


class DynamicGestureResult:
    """Structured output for dynamic motion evaluation."""

    def __init__(
        self,
        gesture: str = "NONE",
        confidence: float = 0.0,
        is_dynamic: bool = False,
        velocity: Tuple[float, float] = (0.0, 0.0),
        speed: float = 0.0,
        displacement: Tuple[float, float] = (0.0, 0.0),
        net_distance: float = 0.0,
        direction_degrees: float = 0.0,
        is_cooldown_active: bool = False,
        latency_ms: float = 0.0,
        status: str = "IDLE",
    ):
        self.gesture = gesture
        self.confidence = float(confidence)
        self.is_dynamic = is_dynamic
        self.velocity = velocity  # (vx, vy) in units/sec
        self.speed = float(speed)
        self.displacement = displacement  # (dx, dy)
        self.net_distance = float(net_distance)
        self.direction_degrees = float(direction_degrees)
        self.is_cooldown_active = is_cooldown_active
        self.latency_ms = float(latency_ms)
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gesture": self.gesture,
            "confidence": round(self.confidence, 4),
            "is_dynamic": self.is_dynamic,
            "velocity": (round(self.velocity[0], 2), round(self.velocity[1], 2)),
            "speed": round(self.speed, 2),
            "displacement": (round(self.displacement[0], 2), round(self.displacement[1], 2)),
            "net_distance": round(self.net_distance, 2),
            "direction_degrees": round(self.direction_degrees, 1),
            "is_cooldown_active": self.is_cooldown_active,
            "latency_ms": round(self.latency_ms, 3),
            "status": self.status,
        }

    def __repr__(self) -> str:
        return f"<DynamicGestureResult gesture='{self.gesture}' speed={self.speed:.1f} status='{self.status}'>"


class DynamicGestureRecognizer:
    """
    Tracks hand landmark trajectories over time to recognize dynamic movement gestures:
    Swipes (Left, Right, Up, Down), Circular patterns, and Drag motions.
    """

    def __init__(
        self,
        history_window_size: int = 15,
        min_swipe_displacement: float = 45.0,  # pixels (or ~0.10 in normalized [0, 1] units)
        min_swipe_velocity: float = 80.0,     # pixels/sec (or ~0.35 in norm/sec)
        dominance_ratio: float = 1.35,        # |dx| / |dy| or vice-versa
        swipe_cooldown_sec: float = 0.40,
        frame_width: int = 640,
        frame_height: int = 480,
    ):
        self.history_window_size = history_window_size
        self.min_swipe_displacement = min_swipe_displacement
        self.min_swipe_velocity = min_swipe_velocity
        self.dominance_ratio = dominance_ratio
        self.swipe_cooldown_sec = swipe_cooldown_sec
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Deque of recent trajectory points (normalized and pixel coordinates)
        self._history: Deque[TrajectoryPoint] = collections.deque(maxlen=history_window_size)
        self._last_trigger_time: float = 0.0
        self._last_triggered_gesture: str = "NONE"

    def reset(self):
        """Clears trajectory history and resets state."""
        self._history.clear()
        self._last_trigger_time = 0.0
        self._last_triggered_gesture = "NONE"

    def update(
        self,
        tracking_point: Optional[Union[Tuple[float, float], np.ndarray, List[float]]],
        is_pinching: bool = False,
        timestamp: Optional[float] = None,
    ) -> DynamicGestureResult:
        """
        Updates the trajectory buffer with the current tracking point and evaluates gestures.

        Args:
            tracking_point: (x, y) coordinates of key landmark (e.g. index tip or palm center).
            is_pinching: True if static recognizer reports active PINCH pose.
            timestamp: Optional monotonic timestamp.

        Returns:
            DynamicGestureResult containing detected motion gesture and kinematics.
        """
        t0 = time.perf_counter()
        now = timestamp if timestamp is not None else t0

        if tracking_point is None:
            # When tracking is lost, flush history
            self._history.clear()
            return DynamicGestureResult(
                gesture="NONE",
                confidence=0.0,
                is_dynamic=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                status="NO_TRACKING_POINT",
            )

        # Handle coordinate normalization
        tx = float(tracking_point[0])
        ty = float(tracking_point[1])

        # If points are normalized in [0, 1], scale to pixel space for uniform thresholding
        if tx <= 1.05 and ty <= 1.05:
            px = tx * self.frame_width
            py = ty * self.frame_height
        else:
            px = tx
            py = ty

        # Append to buffer
        self._history.append(TrajectoryPoint(px, py, timestamp=now))

        # Check if cooldown is currently active
        cooldown_elapsed = now - self._last_trigger_time
        in_cooldown = cooldown_elapsed < self.swipe_cooldown_sec

        # Need at least 4 frames in history for kinematic evaluation
        if len(self._history) < 4:
            return DynamicGestureResult(
                gesture="NONE",
                confidence=0.0,
                is_dynamic=False,
                is_cooldown_active=in_cooldown,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                status="BUFFERING",
            )

        # Compute Kinematics across the window
        p_start = self._history[0]
        p_end = self._history[-1]
        dt = max(p_end.timestamp - p_start.timestamp, 1e-4)

        dx = p_end.x - p_start.x
        dy = p_end.y - p_start.y
        dist = math.hypot(dx, dy)
        vx = dx / dt
        vy = dy / dt
        speed = dist / dt
        angle_deg = math.degrees(math.atan2(dy, dx))  # 0 = right, 90 = down, 180 = left, -90 = up

        # 1. Check for PINCH_DRAG
        if is_pinching and dist >= self.min_swipe_displacement * 0.5:
            latency = (time.perf_counter() - t0) * 1000.0
            return DynamicGestureResult(
                gesture="PINCH_DRAG",
                confidence=0.90,
                is_dynamic=True,
                velocity=(vx, vy),
                speed=speed,
                displacement=(dx, dy),
                net_distance=dist,
                direction_degrees=angle_deg,
                is_cooldown_active=False,
                latency_ms=latency,
                status="PINCH_DRAG_ACTIVE",
            )

        # 2. Check for Swipe Motions
        detected_swipe = "NONE"
        swipe_conf = 0.0

        if dist >= self.min_swipe_displacement and speed >= self.min_swipe_velocity:
            abs_dx = abs(dx)
            abs_dy = abs(dy)

            # Check Horizontal Swipe Dominance
            if abs_dx >= abs_dy * self.dominance_ratio:
                if dx > 0:
                    detected_swipe = "SWIPE_RIGHT"
                else:
                    detected_swipe = "SWIPE_LEFT"
                swipe_conf = min(0.98, 0.70 + (dist / (self.min_swipe_displacement * 2.5)) * 0.28)

            # Check Vertical Swipe Dominance
            elif abs_dy >= abs_dx * self.dominance_ratio:
                if dy > 0:
                    detected_swipe = "SWIPE_DOWN"
                else:
                    detected_swipe = "SWIPE_UP"
                swipe_conf = min(0.98, 0.70 + (dist / (self.min_swipe_displacement * 2.5)) * 0.28)

        # 3. Check for Circular Gestures (Clockwise / Counter-Clockwise)
        if detected_swipe == "NONE" and len(self._history) >= 8:
            circ_gesture, circ_conf = self._check_circular_motion()
            if circ_gesture != "NONE":
                detected_swipe = circ_gesture
                swipe_conf = circ_conf

        # 4. Handle Cooldown and Triggering
        if detected_swipe != "NONE":
            if in_cooldown:
                latency = (time.perf_counter() - t0) * 1000.0
                return DynamicGestureResult(
                    gesture="NONE",
                    confidence=0.0,
                    is_dynamic=False,
                    velocity=(vx, vy),
                    speed=speed,
                    displacement=(dx, dy),
                    net_distance=dist,
                    direction_degrees=angle_deg,
                    is_cooldown_active=True,
                    latency_ms=latency,
                    status=f"COOLDOWN ({cooldown_elapsed:.2f}s < {self.swipe_cooldown_sec:.2f}s)",
                )

            # Trigger dynamic gesture!
            self._last_trigger_time = now
            self._last_triggered_gesture = detected_swipe
            self._history.clear()  # Clear trajectory to prevent immediate re-trigger

            latency = (time.perf_counter() - t0) * 1000.0
            return DynamicGestureResult(
                gesture=detected_swipe,
                confidence=swipe_conf,
                is_dynamic=True,
                velocity=(vx, vy),
                speed=speed,
                displacement=(dx, dy),
                net_distance=dist,
                direction_degrees=angle_deg,
                is_cooldown_active=False,
                latency_ms=latency,
                status=f"TRIGGERED_{detected_swipe}",
            )

        # No dynamic gesture triggered
        latency = (time.perf_counter() - t0) * 1000.0
        return DynamicGestureResult(
            gesture="NONE",
            confidence=0.0,
            is_dynamic=False,
            velocity=(vx, vy),
            speed=speed,
            displacement=(dx, dy),
            net_distance=dist,
            direction_degrees=angle_deg,
            is_cooldown_active=in_cooldown,
            latency_ms=latency,
            status="TRACKING",
        )

    def _check_circular_motion(self) -> Tuple[str, float]:
        """
        Evaluates the accumulated angle changes in the history to detect circular strokes.
        Returns ('CIRCLE_CW' | 'CIRCLE_CCW' | 'NONE', confidence).
        """
        pts = list(self._history)
        if len(pts) < 8:
            return "NONE", 0.0

        # Calculate centroid of points
        cx = sum(p.x for p in pts) / len(pts)
        cy = sum(p.y for p in pts) / len(pts)

        # Verify radius is large enough
        radii = [math.hypot(p.x - cx, p.y - cy) for p in pts]
        mean_radius = sum(radii) / len(radii)
        if mean_radius < self.min_swipe_displacement * 0.4:
            return "NONE", 0.0

        # Check total angle swept around centroid
        total_angle = 0.0
        for i in range(1, len(pts)):
            a1 = math.atan2(pts[i - 1].y - cy, pts[i - 1].x - cx)
            a2 = math.atan2(pts[i].y - cy, pts[i].x - cx)
            da = a2 - a1
            # Wrap to [-pi, pi]
            while da > math.pi:
                da -= 2 * math.pi
            while da < -math.pi:
                da += 2 * math.pi
            total_angle += da

        # 300 degrees in radians ~= 5.23
        if total_angle >= 4.5:
            return "CIRCLE_CW", min(0.95, 0.70 + (total_angle / 6.28) * 0.25)
        elif total_angle <= -4.5:
            return "CIRCLE_CCW", min(0.95, 0.70 + (abs(total_angle) / 6.28) * 0.25)

        return "NONE", 0.0
