"""
Camera Acquisition Module
=========================
Manages thread-safe video capture, frame preprocessing,
horizontal mirroring, and dynamic FPS tracking.
"""

import cv2
import time
import logging
import numpy as np
from typing import Tuple, Optional
import config

logger = logging.getLogger(__name__)


class CameraStream:
    """Encapsulates OpenCV VideoCapture with FPS tracking and error handling."""

    def __init__(
        self,
        camera_id: int = config.CAMERA_ID,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        flip_h: bool = config.FLIP_HORIZONTAL,
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.flip_h = flip_h

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False

        # FPS Calculation variables
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        self.prev_frame_time = 0.0

    def start(self) -> bool:
        """Initializes the webcam device."""
        logger.info("Initializing webcam on index %d...", self.camera_id)
        # Use DSHOW on Windows if standard backend fails
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap or not self.cap.isOpened():
            logger.warning("DSHOW capture failed, falling back to default CAP_ANY...")
            self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            logger.error("Failed to open webcam at index %d.", self.camera_id)
            self.is_opened = False
            return False

        # Set hardware properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

        # Query actual negotiated resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Webcam started successfully (%dx%d).", actual_w, actual_h)
        self.width = actual_w
        self.height = actual_h
        self.is_opened = True
        self.prev_frame_time = time.time()
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads a frame from the webcam, applies mirror flip, and updates FPS.
        Returns:
            (success, frame_bgr)
        """
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Empty frame received from webcam.")
            return False, None

        # Horizontal flip for intuitive mirror interaction
        if self.flip_h:
            frame = cv2.flip(frame, 1)

        # Update FPS
        curr_time = time.time()
        time_diff = curr_time - self.prev_frame_time
        if time_diff > 0:
            instant_fps = 1.0 / time_diff
            # Smooth FPS using Exponential Moving Average
            self.fps = 0.85 * self.fps + 0.15 * instant_fps if self.fps > 0 else instant_fps
        self.prev_frame_time = curr_time
        self.frame_count += 1

        return True, frame

    def get_fps(self) -> float:
        """Returns the current smoothed frame rate."""
        return round(self.fps, 1)

    def release(self):
        """Releases the camera hardware."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info("Webcam released.")
        self.is_opened = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
