"""
Threaded Camera Capture Module.
Handles video acquisition, backend fallbacks, thread decoupling,
lighting quality evaluation, and horizontal mirroring.
"""

import os
import queue
import sys
import threading
import time
from typing import Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import numpy as np

import config
from utils.logger import logger


class ThreadedCamera:
    """
    Decoupled OpenCV camera capture using a background acquisition thread.
    Prevents OpenCV buffer accumulation lag and ensures fresh frames.
    """

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        fps: int = config.TARGET_FPS,
        mirror: bool = config.CAMERA_MIRROR
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = fps
        self.mirror = mirror

        self.cap: Optional[cv2.VideoCapture] = None
        self.actual_device_index = -1
        self.is_running = False

        self._thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_ready_event = threading.Event()

        # Telemetry
        self.fps_measured = 0.0
        self.brightness = 0.0
        self.is_dark = False

    def start(self) -> bool:
        """Finds and initializes the webcam device and starts capture thread."""
        if self.is_running:
            return True

        logger.info(f"Probing camera index {self.camera_index}...")
        self.cap, self.actual_device_index = config.open_camera_device(self.camera_index)

        if self.cap is None or not self.cap.isOpened():
            logger.error("Unable to open any functional webcam device.")
            return False

        self.is_running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()
        logger.success(f"Threaded camera initialized on device {self.actual_device_index}.")
        return True

    def _capture_worker(self):
        """Continuously pulls latest frames to avoid buffer latency."""
        frame_count = 0
        last_time = time.time()

        while self.is_running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # Mirror horizontally if requested
            if self.mirror:
                frame = cv2.flip(frame, 1)

            # Measure brightness
            b_val = float(np.mean(frame))
            self.brightness = b_val
            self.is_dark = b_val < 35.0

            # Store latest frame safely
            with self._frame_lock:
                self._latest_frame = frame
                self._frame_ready_event.set()

            # Measure FPS
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                self.fps_measured = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            # Sleep slightly to match target frame rate
            time.sleep(1.0 / (self.target_fps * 1.5))

    def read(self, timeout: float = 0.2) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieves the latest available frame.
        Returns: (success, frame_bgr)
        """
        if not self.is_running:
            return False, None

        if self._frame_ready_event.wait(timeout=timeout):
            with self._frame_lock:
                self._frame_ready_event.clear()
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
                    return True, frame
        return False, None

    def stop(self):
        """Cleanly halts thread and releases camera."""
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        logger.info("Camera capture halted and device released.")
