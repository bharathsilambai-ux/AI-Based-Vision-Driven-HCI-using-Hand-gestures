"""
Coordinate Smoother & Screen Remapping Module
=============================================
Maps webcam active region to monitor resolution and applies
Exponential Moving Average (EMA) filtering to eliminate jitter.
"""

import numpy as np
import pyautogui
from typing import Tuple
import config


class CoordinateSmoother:
    """Provides jitter reduction and linear interpolation for cursor control."""

    def __init__(
        self,
        frame_width: int = config.FRAME_WIDTH,
        frame_height: int = config.FRAME_HEIGHT,
        alpha: float = config.SMOOTHING_ALPHA,
        deadzone_px: float = 2.0,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.alpha = alpha
        self.deadzone_px = deadzone_px

        # Query OS screen resolution safely
        try:
            self.screen_width, self.screen_height = pyautogui.size()
        except Exception:
            self.screen_width, self.screen_height = 1920, 1080

        # Active boundary box coordinates within camera frame
        self.x_min = config.MARGIN_LEFT
        self.x_max = frame_width - config.MARGIN_RIGHT
        self.y_min = config.MARGIN_TOP
        self.y_max = frame_height - config.MARGIN_BOTTOM

        # Smoothed state
        self.prev_screen_x = self.screen_width / 2.0
        self.prev_screen_y = self.screen_height / 2.0
        self.initialized = False

    def update_frame_size(self, width: int, height: int):
        """Updates frame dimensions if camera resolution changes."""
        self.frame_width = width
        self.frame_height = height
        self.x_min = config.MARGIN_LEFT
        self.x_max = width - config.MARGIN_RIGHT
        self.y_min = config.MARGIN_TOP
        self.y_max = height - config.MARGIN_BOTTOM

    def smooth_and_map(self, raw_x_px: int, raw_y_px: int) -> Tuple[int, int]:
        """
        Maps a point in webcam pixel space to monitor screen coordinates with EMA filter.
        
        Args:
            raw_x_px: Hand keypoint X in camera frame
            raw_y_px: Hand keypoint Y in camera frame
            
        Returns:
            (screen_x, screen_y) as integer coordinates
        """
        # 1. Linear interpolation from active frame margin to monitor bounds
        target_x = np.interp(raw_x_px, (self.x_min, self.x_max), (0, self.screen_width))
        target_y = np.interp(raw_y_px, (self.y_min, self.y_max), (0, self.screen_height))

        # Clamp to screen boundary
        target_x = max(0, min(self.screen_width - 1, target_x))
        target_y = max(0, min(self.screen_height - 1, target_y))

        if not self.initialized:
            self.prev_screen_x = target_x
            self.prev_screen_y = target_y
            self.initialized = True
            return int(target_x), int(target_y)

        # 2. Deadzone check: prevent microscopic sensor jitter
        dist = np.hypot(target_x - self.prev_screen_x, target_y - self.prev_screen_y)
        if dist < self.deadzone_px:
            return int(self.prev_screen_x), int(self.prev_screen_y)

        # 3. Exponential Moving Average (EMA)
        curr_x = self.alpha * target_x + (1.0 - self.alpha) * self.prev_screen_x
        curr_y = self.alpha * target_y + (1.0 - self.alpha) * self.prev_screen_y

        self.prev_screen_x = curr_x
        self.prev_screen_y = curr_y

        return int(curr_x), int(curr_y)

    def reset(self):
        """Resets smoothed coordinates to center screen."""
        self.initialized = False
        self.prev_screen_x = self.screen_width / 2.0
        self.prev_screen_y = self.screen_height / 2.0
