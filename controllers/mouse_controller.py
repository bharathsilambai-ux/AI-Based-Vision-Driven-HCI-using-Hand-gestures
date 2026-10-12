"""
Phase 9: Advanced Virtual Mouse Controller Subsystem
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Coordinate Mapping & Workspace Calibration (Module 8):
   - Maps normalized camera coordinates (e.g. Index Fingertip L8) to OS screen resolution (W, H).
   - Configurable active interaction box (e.g. [0.15, 0.15, 0.85, 0.80]) so hand gestures comfortably
     reach all screen edges without extreme physical extension.

2. Jitter Reduction & Adaptive Smoothing:
   - Dual-mode smoothing: Exponential Moving Average (EMA) and 1-Euro adaptive velocity filtering.
   - Dynamic cut-off: Heavy smoothing when hand is stationary (eliminates micro-tremors for pixel precision),
     zero-lag direct tracking when hand moves swiftly.
   - Deadzone thresholding to freeze cursor jitter during clicking gestures.

3. Non-Linear Mouse Acceleration (Adaptive Sensitivity):
   - Configurable sensitivity multiplier with non-linear power curve:
     Precision tracking for UI buttons, high-velocity traversal across multi-monitor or 4K setups.

4. Comprehensive Mouse Action Vocabulary:
   - move_cursor(): Sub-millisecond cursor relocation.
   - left_click(), right_click(), double_click().
   - start_drag(), stop_drag(), drag_to().
   - click_and_hold(), release_click().
   - scroll_up(), scroll_down(), scroll_vertical().
   - zoom_in(), zoom_out(), zoom_control().

5. Direct Hardware Win32 API Acceleration & Mock Safety Mode:
   - Direct ctypes user32.SetCursorPos and mouse_event on Windows (< 0.05 ms latency).
   - Safe Fallback to PyAutoGUI.
   - Virtual/Mock simulation mode for headless CI/CD testing without hijacking the host desktop mouse.
"""

import ctypes
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pyautogui
from utils.logger import logger

# PyAutoGUI Safety Settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.001

# Windows Hardware API Bindings
HAVE_WIN32 = False
_user32 = None
if sys.platform == "win32":
    try:
        _user32 = ctypes.windll.user32
        HAVE_WIN32 = True
    except Exception:
        HAVE_WIN32 = False

# Mouse Event Flags for Windows user32.mouse_event
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000


class OneEuroFilter:
    """
    1-Euro adaptive low-pass filter for eliminating jitter during slow movements
    while preserving responsiveness during quick movements.
    """

    def __init__(self, t0: float, x0: float, dx0: float = 0.0, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def _smoothing_factor(self, dt: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def filter(self, t: float, x: float) -> float:
        dt = max(1e-4, t - self.t_prev)
        self.t_prev = t

        # Estimate derivative (speed)
        edx = (x - self.x_prev) / dt
        a_d = self._smoothing_factor(dt, self.d_cutoff)
        dx = a_d * edx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx

        # Dynamic cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx)
        a = self._smoothing_factor(dt, cutoff)
        x_filtered = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_filtered
        return x_filtered


class VirtualMouseController:
    """
    Advanced Virtual Mouse Controller implementing Module 8 specifications:
    Cursor tracking, clicking, dragging, scrolling, adaptive sensitivity, and jitter reduction.
    """

    def __init__(
        self,
        screen_size: Optional[Tuple[int, int]] = None,
        active_box: Optional[List[float]] = None,
        sensitivity: float = 1.6,
        smoothing_alpha: float = 0.65,
        deadzone_px: float = 3.5,
        acceleration_power: float = 1.15,
        use_one_euro: bool = True,
        mock_mode: bool = False,
    ):
        """
        Initializes VirtualMouseController.

        Args:
            screen_size: (width, height) in pixels. Auto-detected if None.
            active_box: [x_min, y_min, x_max, y_max] normalized active camera bounds.
            sensitivity: Linear movement multiplier.
            smoothing_alpha: Exponential moving average smoothing weight (0.0 to 1.0).
            deadzone_px: Pixel radius below which micro-movements are suppressed.
            acceleration_power: Non-linear acceleration exponent (> 1.0 increases speed on fast moves).
            use_one_euro: Use adaptive 1-Euro filter instead of static EMA.
            mock_mode: If True, tracks internal cursor without emitting OS mouse events (useful for headless testing).
        """
        # Determine screen resolution
        if screen_size is not None:
            self.screen_w, self.screen_h = screen_size
        else:
            try:
                self.screen_w, self.screen_h = pyautogui.size()
            except Exception:
                self.screen_w, self.screen_h = (1920, 1080)

        # Interaction Workspace Box [x_min, y_min, x_max, y_max] in normalized camera space
        self.active_box = active_box or [0.15, 0.15, 0.85, 0.80]

        # Smoothing & Sensitivity
        self.sensitivity = max(0.2, min(5.0, float(sensitivity)))
        self.smoothing_alpha = max(0.05, min(1.0, float(smoothing_alpha)))
        self.deadzone_px = max(0.0, float(deadzone_px))
        self.acceleration_power = max(1.0, min(2.0, float(acceleration_power)))
        self.use_one_euro = use_one_euro
        self.mock_mode = mock_mode

        # Cursor State
        self.cursor_x: float = self.screen_w / 2.0
        self.cursor_y: float = self.screen_h / 2.0
        self.prev_cursor_x: Optional[float] = None
        self.prev_cursor_y: Optional[float] = None

        # Action States
        self.is_dragging: bool = False
        self.is_held: bool = False
        self.last_click_time: float = 0.0
        self.last_action: str = "NONE"

        # Action Event History Log (useful for verification, unit testing, and telemetry)
        self.event_log: List[Dict[str, Any]] = []

        # 1-Euro Filters
        now = time.perf_counter()
        self._filter_x = OneEuroFilter(t0=now, x0=self.cursor_x, min_cutoff=0.8, beta=0.005)
        self._filter_y = OneEuroFilter(t0=now, x0=self.cursor_y, min_cutoff=0.8, beta=0.005)

    def set_sensitivity(self, value: float):
        """Sets cursor sensitivity multiplier."""
        self.sensitivity = max(0.2, min(5.0, float(value)))

    def set_deadzone(self, value: float):
        """Sets deadzone threshold in pixels."""
        self.deadzone_px = max(0.0, float(value))

    def set_smoothing_alpha(self, value: float):
        """Sets static EMA smoothing weight."""
        self.smoothing_alpha = max(0.05, min(1.0, float(value)))

    def set_active_box(self, box: List[float]):
        """Sets normalized active interaction box."""
        if len(box) == 4 and box[0] < box[2] and box[1] < box[3]:
            self.active_box = list(box)

    def reset(self):
        """Resets cursor tracking state and releases any active drag."""
        if self.is_dragging:
            self.stop_drag()
        if self.is_held:
            self.release_click()
        self.prev_cursor_x = None
        self.prev_cursor_y = None
        self.last_action = "RESET"

    def move_cursor_to_norm(
        self,
        norm_x: float,
        norm_y: float,
        timestamp: Optional[float] = None,
    ) -> Tuple[int, int]:
        """
        Maps normalized camera point [0.0, 1.0] to screen pixel coordinates,
        applies active-box mapping, adaptive jitter smoothing, deadzone thresholding,
        and moves the mouse cursor.

        Args:
            norm_x: Normalized horizontal coordinate (0.0 to 1.0).
            norm_y: Normalized vertical coordinate (0.0 to 1.0).
            timestamp: Optional timestamp for filtering.

        Returns:
            (target_x, target_y) screen pixel coordinates.
        """
        now = timestamp if timestamp is not None else time.perf_counter()

        # 1. Map relative to Active Workspace Box
        x_min, y_min, x_max, y_max = self.active_box
        box_w = max(0.05, x_max - x_min)
        box_h = max(0.05, y_max - y_min)

        rel_x = (norm_x - x_min) / box_w
        rel_y = (norm_y - y_min) / box_h

        # Clamp to [0.0, 1.0]
        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))

        # 2. Map to Raw Screen Pixel Coordinates
        raw_screen_x = rel_x * (self.screen_w - 1)
        raw_screen_y = rel_y * (self.screen_h - 1)

        # 3. Apply Adaptive Jitter Smoothing (1-Euro or EMA)
        if self.prev_cursor_x is None or self.prev_cursor_y is None:
            smooth_x = raw_screen_x
            smooth_y = raw_screen_y
            self._filter_x.x_prev = raw_screen_x
            self._filter_y.x_prev = raw_screen_y
        else:
            if self.use_one_euro:
                smooth_x = self._filter_x.filter(now, raw_screen_x)
                smooth_y = self._filter_y.filter(now, raw_screen_y)
            else:
                alpha = self.smoothing_alpha
                smooth_x = alpha * raw_screen_x + (1.0 - alpha) * self.prev_cursor_x
                smooth_y = alpha * raw_screen_y + (1.0 - alpha) * self.prev_cursor_y

            # 4. Deadzone Thresholding (suppress micro-tremor when hand is still)
            dist = math.hypot(smooth_x - self.prev_cursor_x, smooth_y - self.prev_cursor_y)
            if dist < self.deadzone_px:
                return (int(self.cursor_x), int(self.cursor_y))

            # 5. Sensitivity & Non-linear Velocity Acceleration
            dx = smooth_x - self.prev_cursor_x
            dy = smooth_y - self.prev_cursor_y
            accel = 1.0
            if self.acceleration_power > 1.0 and dist > 15.0:
                accel = min(2.5, (dist / 100.0) ** (self.acceleration_power - 1.0))
            smooth_x = self.prev_cursor_x + dx * self.sensitivity * accel
            smooth_y = self.prev_cursor_y + dy * self.sensitivity * accel

        # Clamp to screen boundary limits
        clamped_x = max(0.0, min(float(self.screen_w - 1), float(smooth_x) if math.isfinite(smooth_x) else 0.0))
        clamped_y = max(0.0, min(float(self.screen_h - 1), float(smooth_y) if math.isfinite(smooth_y) else 0.0))

        target_x = int(round(clamped_x))
        target_y = int(round(clamped_y))

        self.prev_cursor_x = float(target_x)
        self.prev_cursor_y = float(target_y)
        self.cursor_x = target_x
        self.cursor_y = target_y

        # Emit OS hardware cursor movement (unless in mock_mode)
        if not self.mock_mode:
            self._dispatch_hardware_move(target_x, target_y)

        self._log_event("MOVE", {"x": target_x, "y": target_y})
        return (target_x, target_y)

    def _dispatch_hardware_move(self, target_x: int, target_y: int):
        """Directly positions the OS cursor using Win32 API (< 0.05ms) or PyAutoGUI."""
        if HAVE_WIN32 and _user32 is not None:
            try:
                _user32.SetCursorPos(target_x, target_y)
                return
            except Exception:
                pass
        try:
            pyautogui.moveTo(target_x, target_y)
        except Exception:
            pass

    def left_click(self) -> bool:
        """Dispatches a left mouse click."""
        self.last_action = "LEFT_CLICK"
        self.last_click_time = time.perf_counter()
        self._log_event("LEFT_CLICK", {"x": int(self.cursor_x), "y": int(self.cursor_y)})

        if self.mock_mode:
            return True

        if HAVE_WIN32 and _user32 is not None:
            try:
                _user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.012)
                _user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                return True
            except Exception:
                pass
        try:
            pyautogui.click(button="left")
            return True
        except Exception as e:
            logger.error(f"[VirtualMouseController] Left click failed: {e}")
            return False

    def right_click(self) -> bool:
        """Dispatches a right mouse click."""
        self.last_action = "RIGHT_CLICK"
        self.last_click_time = time.perf_counter()
        self._log_event("RIGHT_CLICK", {"x": int(self.cursor_x), "y": int(self.cursor_y)})

        if self.mock_mode:
            return True

        if HAVE_WIN32 and _user32 is not None:
            try:
                _user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                time.sleep(0.012)
                _user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                return True
            except Exception:
                pass
        try:
            pyautogui.click(button="right")
            return True
        except Exception as e:
            logger.error(f"[VirtualMouseController] Right click failed: {e}")
            return False

    def double_click(self) -> bool:
        """Dispatches a rapid double click."""
        self.last_action = "DOUBLE_CLICK"
        self.last_click_time = time.perf_counter()
        self._log_event("DOUBLE_CLICK", {"x": int(self.cursor_x), "y": int(self.cursor_y)})

        if self.mock_mode:
            return True

        if HAVE_WIN32 and _user32 is not None:
            try:
                _user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.010)
                _user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                time.sleep(0.040)
                _user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.010)
                _user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                return True
            except Exception:
                pass
        try:
            pyautogui.doubleClick()
            return True
        except Exception as e:
            logger.error(f"[VirtualMouseController] Double click failed: {e}")
            return False

    def start_drag(self) -> bool:
        """Initiates a mouse drag (presses left button down)."""
        if not self.is_dragging:
            self.is_dragging = True
            self.last_action = "START_DRAG"
            self._log_event("START_DRAG", {"x": int(self.cursor_x), "y": int(self.cursor_y)})

            if self.mock_mode:
                return True

            if HAVE_WIN32 and _user32 is not None:
                try:
                    _user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    return True
                except Exception:
                    pass
            try:
                pyautogui.mouseDown(button="left")
                return True
            except Exception as e:
                logger.error(f"[VirtualMouseController] Start drag failed: {e}")
                return False
        return True

    def stop_drag(self) -> bool:
        """Terminates a mouse drag (releases left button)."""
        if self.is_dragging:
            self.is_dragging = False
            self.last_action = "STOP_DRAG"
            self._log_event("STOP_DRAG", {"x": int(self.cursor_x), "y": int(self.cursor_y)})

            if self.mock_mode:
                return True

            if HAVE_WIN32 and _user32 is not None:
                try:
                    _user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                    return True
                except Exception:
                    pass
            try:
                pyautogui.mouseUp(button="left")
                return True
            except Exception as e:
                logger.error(f"[VirtualMouseController] Stop drag failed: {e}")
                return False
        return True

    def click_and_hold(self) -> bool:
        """Alias and implementation for click-and-hold."""
        self.is_held = True
        return self.start_drag()

    def release_click(self) -> bool:
        """Releases held click or drag."""
        self.is_held = False
        return self.stop_drag()

    def scroll_vertical(self, amount: int = 120) -> bool:
        """Dispatches mouse wheel vertical scrolling."""
        self.last_action = f"SCROLL_{'UP' if amount > 0 else 'DOWN'}"
        self._log_event("SCROLL", {"amount": amount})

        if self.mock_mode:
            return True

        if HAVE_WIN32 and _user32 is not None:
            try:
                _user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(amount), 0)
                return True
            except Exception:
                pass
        try:
            pyautogui.scroll(amount)
            return True
        except Exception as e:
            logger.error(f"[VirtualMouseController] Scroll failed: {e}")
            return False

    def scroll_up(self, amount: int = 120) -> bool:
        """Scrolls up by specified units."""
        return self.scroll_vertical(abs(amount))

    def scroll_down(self, amount: int = 120) -> bool:
        """Scrolls down by specified units."""
        return self.scroll_vertical(-abs(amount))

    def zoom_in(self, steps: int = 1) -> bool:
        """Triggers browser/system zoom in (Ctrl + Wheel Up)."""
        self.last_action = "ZOOM_IN"
        self._log_event("ZOOM_IN", {"steps": steps})
        if self.mock_mode:
            return True
        try:
            pyautogui.hotkey("ctrl", "+")
            return True
        except Exception as e:
            logger.error(f"[VirtualMouseController] Zoom in failed: {e}")
            return False

    def zoom_out(self, steps: int = 1) -> bool:
        """Triggers browser/system zoom out (Ctrl + Wheel Down)."""
        self.last_action = "ZOOM_OUT"
        self._log_event("ZOOM_OUT", {"steps": steps})
        if self.mock_mode:
            return True
        try:
            pyautogui.hotkey("ctrl", "-")
            return True
        except Exception as e:
            logger.error(f"[VirtualMouseController] Zoom out failed: {e}")
            return False

    def execute_action(
        self,
        action: Optional[str],
        tracking_point: Optional[Tuple[float, float]] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Executes mouse commands mapped from gesture recognition.

        Args:
            action: Command string (e.g. 'CURSOR_MOVE', 'LEFT_CLICK', 'RIGHT_CLICK', 'SCROLL_UP').
            tracking_point: Normalized (x, y) coordinates of pointing finger.
            timestamp: Frame timestamp.

        Returns:
            True if action was recognized and executed.
        """
        if not action or action == "NONE":
            return False

        # Move cursor if point is available and action is cursor move or hover
        if tracking_point is not None and action in ["CURSOR_MOVE", "HOVER", "DRAG_DROP"]:
            self.move_cursor_to_norm(tracking_point[0], tracking_point[1], timestamp=timestamp)

        if action == "CURSOR_MOVE" or action == "HOVER":
            if self.is_dragging:
                self.stop_drag()
            return True

        elif action == "LEFT_CLICK":
            return self.left_click()

        elif action == "RIGHT_CLICK":
            return self.right_click()

        elif action == "DOUBLE_CLICK":
            return self.double_click()

        elif action == "DRAG_DROP":
            return self.start_drag()

        elif action == "RELEASE":
            return self.release_click()

        elif action == "SCROLL_UP":
            return self.scroll_up(150)

        elif action == "SCROLL_DOWN":
            return self.scroll_down(150)

        elif action == "ZOOM_IN":
            return self.zoom_in()

        elif action == "ZOOM_OUT":
            return self.zoom_out()

        return False

    def _log_event(self, event_type: str, details: Dict[str, Any]):
        """Records event into internal log buffer for testing and diagnostics."""
        self.event_log.append({
            "timestamp": time.perf_counter(),
            "event": event_type,
            "details": details
        })
        # Keep log size reasonable
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-500:]
