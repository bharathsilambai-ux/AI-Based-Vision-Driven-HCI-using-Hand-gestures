"""
Mouse Control Module using PyAutoGUI.
Maps hand coordinates to OS screen resolution with EMA smoothing,
deadzones, non-linear velocity acceleration, and drag-and-drop state handling.
"""

import math
import os
import sys
import time
from typing import Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pyautogui

import config
from utils.logger import logger

try:
    from .dispatcher import action_dispatcher
except ImportError:
    from dispatcher import action_dispatcher

# High-speed Windows API mouse acceleration
HAVE_WIN32 = False
_user32 = None
if sys.platform == "win32":
    try:
        import ctypes
        _user32 = ctypes.windll.user32
        HAVE_WIN32 = True
    except Exception:
        HAVE_WIN32 = False

# PyAutoGUI configurations
pyautogui.FAILSAFE = config.FAILSAFE_ENABLED
pyautogui.PAUSE = 0.001


class MouseController:
    """Controls OS cursor movement and mouse clicks safely."""

    def __init__(
        self,
        sensitivity: float = config.CURSOR_SENSITIVITY_DEFAULT,
        deadzone_px: float = config.DEADZONE_PIXELS_DEFAULT,
        smoothing_alpha: float = config.SMOOTHING_ALPHA_DEFAULT,
        active_box: list = config.ACTIVE_BOX_DEFAULT
    ):
        self.sensitivity = sensitivity
        self.deadzone_px = deadzone_px
        self.smoothing_alpha = smoothing_alpha
        self.active_box = list(active_box)

        # Get screen geometry
        try:
            self.screen_w, self.screen_h = pyautogui.size()
        except Exception:
            self.screen_w, self.screen_h = 1920, 1080

        # State tracking
        self.prev_cursor_x: Optional[float] = None
        self.prev_cursor_y: Optional[float] = None
        self.is_dragging = False
        self.last_click_time = 0.0

    def set_sensitivity(self, value: float):
        self.sensitivity = max(config.CURSOR_SENSITIVITY_MIN, min(config.CURSOR_SENSITIVITY_MAX, float(value)))

    def set_deadzone(self, value: float):
        self.deadzone_px = max(config.DEADZONE_PIXELS_MIN, min(config.DEADZONE_PIXELS_MAX, float(value)))

    def set_smoothing_alpha(self, value: float):
        self.smoothing_alpha = max(0.1, min(1.0, float(value)))

    def set_active_box(self, box: list):
        if len(box) == 4:
            self.active_box = box

    def move_cursor_to_norm(self, norm_x: float, norm_y: float):
        """
        Maps normalized coordinate within active box to OS screen pixels,
        applies EMA smoothing and deadzone, then moves mouse.
        """
        x_min, y_min, x_max, y_max = self.active_box
        box_w = max(0.1, x_max - x_min)
        box_h = max(0.1, y_max - y_min)

        # 1. Normalize relative to active movement box
        rel_x = (norm_x - x_min) / box_w
        rel_y = (norm_y - y_min) / box_h

        # Clamp between 0.0 and 1.0
        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))

        # 2. Map to screen resolution
        raw_screen_x = rel_x * self.screen_w
        raw_screen_y = rel_y * self.screen_h

        # 3. Apply EMA smoothing
        if self.prev_cursor_x is None or self.prev_cursor_y is None:
            smooth_x = raw_screen_x
            smooth_y = raw_screen_y
        else:
            alpha = self.smoothing_alpha
            smooth_x = alpha * raw_screen_x + (1.0 - alpha) * self.prev_cursor_x
            smooth_y = alpha * raw_screen_y + (1.0 - alpha) * self.prev_cursor_y

            # Check deadzone
            dist = math.hypot(smooth_x - self.prev_cursor_x, smooth_y - self.prev_cursor_y)
            if dist < self.deadzone_px:
                return  # Inside deadzone; suppress micro-tremor

        self.prev_cursor_x = smooth_x
        self.prev_cursor_y = smooth_y

        target_x = int(max(0, min(self.screen_w - 1, smooth_x)))
        target_y = int(max(0, min(self.screen_h - 1, smooth_y)))

        # Direct hardware SetCursorPos on Windows executes in < 0.02ms with zero lag
        if HAVE_WIN32 and _user32 is not None:
            try:
                _user32.SetCursorPos(target_x, target_y)
                return
            except Exception:
                pass

        # Fallback to async dispatcher
        action_dispatcher.submit(pyautogui.moveTo, target_x, target_y)

    def left_click(self):
        """Dispatches a single left click."""
        if HAVE_WIN32 and _user32 is not None:
            def _win_click():
                try:
                    _user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
                    time.sleep(0.015)
                    _user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
                except Exception:
                    pyautogui.click(button="left")
            action_dispatcher.submit(_win_click, priority=True)
        else:
            action_dispatcher.submit(pyautogui.click, button="left", priority=True)
        self.last_click_time = time.time()
        logger.info("[Mouse] Left click dispatched.")

    def double_click(self):
        """Dispatches a double click."""
        if HAVE_WIN32 and _user32 is not None:
            def _win_double_click():
                try:
                    _user32.mouse_event(0x0002, 0, 0, 0, 0)
                    time.sleep(0.012)
                    _user32.mouse_event(0x0004, 0, 0, 0, 0)
                    time.sleep(0.040)
                    _user32.mouse_event(0x0002, 0, 0, 0, 0)
                    time.sleep(0.012)
                    _user32.mouse_event(0x0004, 0, 0, 0, 0)
                except Exception:
                    pyautogui.doubleClick()
            action_dispatcher.submit(_win_double_click, priority=True)
        else:
            action_dispatcher.submit(pyautogui.doubleClick, priority=True)
        self.last_click_time = time.time()
        logger.info("[Mouse] Double click dispatched.")

    def right_click(self):
        """Dispatches a right click."""
        if HAVE_WIN32 and _user32 is not None:
            def _win_rclick():
                try:
                    _user32.mouse_event(0x0008, 0, 0, 0, 0)  # MOUSEEVENTF_RIGHTDOWN
                    time.sleep(0.015)
                    _user32.mouse_event(0x0010, 0, 0, 0, 0)  # MOUSEEVENTF_RIGHTUP
                except Exception:
                    pyautogui.click(button="right")
            action_dispatcher.submit(_win_rclick, priority=True)
        else:
            action_dispatcher.submit(pyautogui.click, button="right", priority=True)
        self.last_click_time = time.time()
        logger.info("[Mouse] Right click dispatched.")

    def start_drag(self):
        """Presses mouse down for drag operation."""
        if not self.is_dragging:
            self.is_dragging = True
            if HAVE_WIN32 and _user32 is not None:
                def _win_down():
                    try:
                        _user32.mouse_event(0x0002, 0, 0, 0, 0)
                    except Exception:
                        pyautogui.mouseDown(button="left")
                action_dispatcher.submit(_win_down, priority=True)
            else:
                action_dispatcher.submit(pyautogui.mouseDown, button="left", priority=True)
            logger.info("[Mouse] Drag started (mouseDown).")

    def stop_drag(self):
        """Releases mouse for drop operation."""
        if self.is_dragging:
            self.is_dragging = False
            if HAVE_WIN32 and _user32 is not None:
                def _win_up():
                    try:
                        _user32.mouse_event(0x0004, 0, 0, 0, 0)
                    except Exception:
                        pyautogui.mouseUp(button="left")
                action_dispatcher.submit(_win_up, priority=True)
            else:
                action_dispatcher.submit(pyautogui.mouseUp, button="left", priority=True)
            logger.info("[Mouse] Drag released (mouseUp).")

    def scroll_vertical(self, amount: int):
        """Dispatches vertical mouse wheel scroll."""
        if amount != 0:
            if HAVE_WIN32 and _user32 is not None:
                def _win_scroll():
                    try:
                        # MOUSEEVENTF_WHEEL = 0x0800
                        _user32.mouse_event(0x0800, 0, 0, int(amount * 35), 0)
                    except Exception:
                        pyautogui.scroll(amount)
                action_dispatcher.submit(_win_scroll)
            else:
                action_dispatcher.submit(pyautogui.scroll, amount)

    def hover(self):
        """Maintains cursor hover state without clicking."""
        logger.info("[Mouse] Hover / select state active.")

    def reset(self):
        """Resets cursor tracking state and releases any active drag."""
        if self.is_dragging:
            self.stop_drag()
        self.prev_cursor_x = None
        self.prev_cursor_y = None
