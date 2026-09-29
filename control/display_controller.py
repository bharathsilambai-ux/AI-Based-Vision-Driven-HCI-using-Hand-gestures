"""
Display and Zoom Control Module.
Handles zoom in/out (hotkeys and Ctrl+scroll), screen brightness adjustments
(via screen_brightness_control), and document/image rotation.
"""

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

# Attempt screen_brightness_control import
HAVE_SBC = False
try:
    import screen_brightness_control as sbc
    HAVE_SBC = True
except Exception:
    HAVE_SBC = False


class DisplayController:
    """Controls display zoom, screen brightness, and document rotation."""

    def __init__(self, brightness_step: int = config.BRIGHTNESS_STEP):
        self.brightness_step = brightness_step

    def zoom_in(self, steps: int = 1):
        """Dispatches Ctrl + = / Ctrl + Scroll Up to zoom into document or browser."""
        def _do_zoom():
            for _ in range(steps):
                pyautogui.hotkey("ctrl", "=")
            logger.info(f"[Display] Zoom In dispatched ({steps} steps).")

        action_dispatcher.submit(_do_zoom, priority=True)

    def zoom_out(self, steps: int = 1):
        """Dispatches Ctrl + Minus / Ctrl + Scroll Down to zoom out."""
        def _do_zoom():
            for _ in range(steps):
                pyautogui.hotkey("ctrl", "-")
            logger.info(f"[Display] Zoom Out dispatched ({steps} steps).")

        action_dispatcher.submit(_do_zoom, priority=True)

    def brightness_up(self, step: Optional[int] = None):
        """Increases screen brightness."""
        s = step or self.brightness_step
        if HAVE_SBC:
            try:
                cur_list = sbc.get_brightness()
                cur = cur_list[0] if isinstance(cur_list, list) and cur_list else 50
                new_b = min(100, cur + s)
                sbc.set_brightness(new_b)
                logger.info(f"[Display] Brightness increased to {new_b}%")
                return
            except Exception as e:
                logger.warning(f"[Display] SBC brightness up error: {e}")
        logger.info("[Display] Brightness Up simulated.")

    def brightness_down(self, step: Optional[int] = None):
        """Decreases screen brightness."""
        s = step or self.brightness_step
        if HAVE_SBC:
            try:
                cur_list = sbc.get_brightness()
                cur = cur_list[0] if isinstance(cur_list, list) and cur_list else 50
                new_b = max(10, cur - s)
                sbc.set_brightness(new_b)
                logger.info(f"[Display] Brightness decreased to {new_b}%")
                return
            except Exception as e:
                logger.warning(f"[Display] SBC brightness down error: {e}")
        logger.info("[Display] Brightness Down simulated.")

    def rotate_document(self):
        """Rotates active document / image clockwise (Ctrl+R)."""
        action_dispatcher.submit(pyautogui.hotkey, "ctrl", "r", priority=True)
        logger.info("[Display] Rotate Document (Ctrl+R) dispatched.")
