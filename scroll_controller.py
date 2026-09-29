"""
Scroll Controller Module.
Converts gesture states and finger displacement into smooth PyAutoGUI system actions.
Includes velocity dampening, deadzones, emergency stops, and thread-safe execution.
"""

import queue
import threading
import time
from typing import Optional, Tuple

import pyautogui

import config
from gesture_recognition import GestureResult, GestureType

# Ensure PyAutoGUI failsafe is enabled
pyautogui.FAILSAFE = config.FAILSAFE_ENABLED
pyautogui.PAUSE = 0.001  # Minimize PyAutoGUI internal latency


class ScrollController:
    """
    Translates gesture outcomes into smooth system scrolling and mouse actions.
    Uses an asynchronous command queue to ensure zero UI frame drops.
    """

    def __init__(
        self,
        sensitivity: float = config.SCROLL_SENSITIVITY_DEFAULT,
        max_speed: int = config.MAX_SCROLL_SPEED,
        click_enabled: bool = True
    ):
        self.sensitivity = sensitivity
        self.max_speed = max_speed
        self.click_enabled = click_enabled

        # Master system enable switch
        self.is_active = True
        self.is_emergency_stopped = False

        # Last scroll direction for UI indicator ("UP", "DOWN", "NONE")
        self.current_direction = "NONE"
        self.last_scroll_time = 0.0

        # Asynchronous worker thread for non-blocking OS input
        self._cmd_queue: queue.Queue = queue.Queue(maxsize=10)
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_running = True
        self._worker_thread.start()

    def set_sensitivity(self, value: float):
        """Sets scroll sensitivity multiplier (1.0 to 20.0)."""
        self.sensitivity = max(config.SCROLL_SENSITIVITY_MIN,
                               min(config.SCROLL_SENSITIVITY_MAX, float(value)))

    def set_click_enabled(self, enabled: bool):
        """Enables or disables pinch click actions."""
        self.click_enabled = enabled

    def emergency_stop(self):
        """Immediately halts all scrolling and disarms the controller."""
        self.is_emergency_stopped = True
        self.is_active = False
        self.current_direction = "NONE"
        # Drain any pending commands in queue
        while not self._cmd_queue.empty():
            try:
                self._cmd_queue.get_nowait()
            except queue.Empty:
                break
        print("[ScrollController] EMERGENCY STOP TRIGGERED.")

    def reset_emergency_stop(self):
        """Clears emergency stop status."""
        self.is_emergency_stopped = False
        self.is_active = True
        print("[ScrollController] Emergency stop cleared. Tracking resumed.")

    def toggle_active(self) -> bool:
        """Toggles active state on/off."""
        if self.is_emergency_stopped:
            self.reset_emergency_stop()
            return True

        self.is_active = not self.is_active
        if not self.is_active:
            self.current_direction = "NONE"
        return self.is_active

    def process_gesture(self, result: GestureResult):
        """
        Receives current GestureResult and issues smooth scroll or click commands.
        """
        # Safety guards: controller inactive or emergency stop
        if not self.is_active or self.is_emergency_stopped:
            self.current_direction = "NONE"
            return

        # Stop conditions: Open palm or no hand detected
        if result.gesture_type in (GestureType.OPEN_PALM, GestureType.NO_HAND):
            self.current_direction = "NONE"
            return

        # Pinch click action
        if result.is_click_event and self.click_enabled:
            self._queue_click()
            self.current_direction = "CLICK"
            return

        # Scrolling actions
        scroll_amount = 0

        if result.gesture_type == GestureType.SCROLL_UP:
            # When moving finger up in image, delta_y is negative.
            # PyAutoGUI: positive value scrolls UP.
            if result.delta_y < 0:
                # Dynamic motion-based velocity
                speed = abs(result.delta_y) * (self.sensitivity / 2.5)
                scroll_amount = int(min(self.max_speed, max(2, speed)))
            else:
                # Sustained static pointing up
                scroll_amount = int(self.sensitivity * 1.5)

            self.current_direction = "UP"

        elif result.gesture_type == GestureType.SCROLL_DOWN:
            # When moving finger down in image, delta_y is positive.
            # PyAutoGUI: negative value scrolls DOWN.
            if result.delta_y > 0:
                # Dynamic motion-based velocity
                speed = abs(result.delta_y) * (self.sensitivity / 2.5)
                scroll_amount = -int(min(self.max_speed, max(2, speed)))
            else:
                # Sustained static pointing down
                scroll_amount = -int(self.sensitivity * 1.5)

            self.current_direction = "DOWN"

        else:
            self.current_direction = "NONE"

        # Dispatch scroll command if non-zero
        if scroll_amount != 0:
            self._queue_scroll(scroll_amount)
            self.last_scroll_time = time.time()

    def _queue_scroll(self, amount: int):
        """Submits scroll command to async worker thread."""
        try:
            self._cmd_queue.put_nowait(("scroll", amount))
        except queue.Full:
            pass  # Drop frame if worker is overloaded to prevent backlog lag

    def _queue_click(self):
        """Submits click command to async worker thread."""
        try:
            self._cmd_queue.put_nowait(("click", 1))
        except queue.Full:
            pass

    def _worker_loop(self):
        """Background thread executing PyAutoGUI calls without blocking the UI."""
        while self._worker_running:
            try:
                cmd, arg = self._cmd_queue.get(timeout=0.1)
                if not self.is_active or self.is_emergency_stopped:
                    self._cmd_queue.task_done()
                    continue

                if cmd == "scroll":
                    # PyAutoGUI scroll units
                    pyautogui.scroll(arg)
                elif cmd == "click":
                    pyautogui.click()

                self._cmd_queue.task_done()
            except queue.Empty:
                continue
            except pyautogui.FailSafeException:
                print("[ScrollController] PyAutoGUI FailSafeException! Mouse in screen corner.")
                self.emergency_stop()
            except Exception as e:
                print(f"[ScrollController] Error executing {cmd}: {e}")

    def close(self):
        """Terminates worker thread cleanly."""
        self._worker_running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)
