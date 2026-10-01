"""
Temporal Motion & Swipe Gesture Detection Module
================================================
Tracks hand centroid trajectories over sliding time windows
to identify horizontal velocity spikes (Swipe Left / Swipe Right).
"""

import time
from collections import deque
from typing import Optional, Tuple
import config


class SwipeDetector:
    """Detects rapid directional hand sweeps using optical trajectory analysis."""

    def __init__(
        self,
        buffer_size: int = config.SWIPE_BUFFER_SIZE,
        velocity_threshold: float = config.SWIPE_VELOCITY_THRESHOLD,
        min_displacement: float = config.SWIPE_MIN_DISPLACEMENT,
        max_vertical_deviation: float = config.SWIPE_MAX_VERTICAL_DEVIATION,
        cooldown_sec: float = config.SWIPE_COOLDOWN_SEC,
    ):
        self.history = deque(maxlen=buffer_size)
        self.velocity_threshold = velocity_threshold
        self.min_displacement = min_displacement
        self.max_vertical_deviation = max_vertical_deviation
        self.cooldown_sec = cooldown_sec
        self.last_swipe_time = 0.0

    def update(self, palm_center_pixel: Tuple[int, int]) -> Optional[str]:
        """
        Updates hand trajectory with current palm pixel coordinate.
        Returns:
            config.GESTURE_SWIPE_LEFT, config.GESTURE_SWIPE_RIGHT, or None
        """
        now = time.time()
        self.history.append((now, palm_center_pixel[0], palm_center_pixel[1]))

        # Check cooldown
        if (now - self.last_swipe_time) < self.cooldown_sec:
            return None

        # Need at least 5 frames to compute velocity trend
        if len(self.history) < 5:
            return None

        t_start, x_start, y_start = self.history[0]
        t_end, x_end, y_end = self.history[-1]

        dt = t_end - t_start
        if dt < 0.08 or dt > 0.45:  # A swipe must occur within 80ms - 450ms
            return None

        dx = x_end - x_start
        dy = abs(y_end - y_start)

        # Check for excessive vertical deviation (reject diagonal/waving motions)
        if dy > self.max_vertical_deviation:
            return None

        velocity_x = dx / dt

        # Swipe Right
        if velocity_x > self.velocity_threshold and dx > self.min_displacement:
            self.last_swipe_time = now
            self.history.clear()
            return config.GESTURE_SWIPE_RIGHT

        # Swipe Left
        if velocity_x < -self.velocity_threshold and dx < -self.min_displacement:
            self.last_swipe_time = now
            self.history.clear()
            return config.GESTURE_SWIPE_LEFT

        return None

    def reset(self):
        """Clears trajectory buffer."""
        self.history.clear()
