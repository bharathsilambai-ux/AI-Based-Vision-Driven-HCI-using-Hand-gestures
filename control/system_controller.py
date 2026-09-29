"""
System Control Module.
Handles disk-saving screenshots, lock workstation, and safe window closing.
"""

import ctypes
from datetime import datetime
import os
import sys
from typing import Optional

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


class SystemController:
    """Manages system-level operations such as screenshots, close window, and lock screen."""

    @staticmethod
    def take_screenshot(save_dir: str = config.SCREENSHOTS_DIR) -> Optional[str]:
        """
        Captures full desktop screenshot and saves with timestamp to screenshots directory.
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)

        def _do_screenshot():
            try:
                img = pyautogui.screenshot()
                img.save(filepath)
                logger.success(f"[System] Screenshot saved: {filepath}")
            except Exception as e:
                logger.error(f"[System] Screenshot failed: {e}")

        action_dispatcher.submit(_do_screenshot, priority=True)
        return filepath

    @staticmethod
    def close_window_safe():
        """Safely closes active window via Alt+F4."""
        def _do_close():
            pyautogui.hotkey("alt", "f4")
            logger.warning("[System] Close Window (Alt+F4) dispatched.")

        action_dispatcher.submit(_do_close, priority=True)

    @staticmethod
    def lock_screen_safe():
        """Locks the Windows workstation safely."""
        def _do_lock():
            try:
                ctypes.windll.user32.LockWorkStation()
                logger.warning("[System] Lock Workstation triggered.")
            except Exception:
                pyautogui.hotkey("win", "l")

        action_dispatcher.submit(_do_lock, priority=True)
