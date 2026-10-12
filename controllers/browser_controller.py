"""
Phase 12: Web Browser Automation & Navigation Subsystem
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Complete Browser Navigation Suite (Module 11):
   - scroll_up(), scroll_down(): Smooth vertical page scrolling.
   - history_back(), history_forward(): History navigation (Alt+Left, Alt+Right).
   - refresh_page(): Page reloading with optional hard reload (F5 / Ctrl+F5).
   - zoom_in(), zoom_out(), zoom_reset(): Viewport zoom control (Ctrl++, Ctrl+-, Ctrl+0).
   - next_tab(), prev_tab(): Multi-tab navigation (Ctrl+Tab, Ctrl+Shift+Tab).
   - new_tab(), close_tab(), reopen_closed_tab(): Tab lifecycle operations (Ctrl+T, Ctrl+W, Ctrl+Shift+T).
   - focus_address_bar(): Address bar focus (Ctrl+L).

2. Browser Profile Compatibility (Module 13):
   - Cross-browser shortcuts supporting Google Chrome, Microsoft Edge, Mozilla Firefox, Brave, and Opera.

3. Mock & Headless Verification Architecture:
   - Built-in mock_mode preserving internal browser state (tab_count, active_tab_index, zoom_level, scroll_offset)
     for testing without sending raw keypresses to host browser windows.
   - Detailed event logging (event_log).
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pyautogui
from utils.logger import logger

# PyAutoGUI Safety Settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.001


class BrowserTarget:
    """Supported browser application profiles."""
    CHROME = "chrome"
    EDGE = "edge"
    FIREFOX = "firefox"
    BRAVE = "brave"
    GENERIC = "generic"

    ALL_TARGETS = [CHROME, EDGE, FIREFOX, BRAVE, GENERIC]


class BrowserController:
    """
    Subsystem for controlling web browser navigation, multi-tab management,
    page scrolling, and zooming (Module 11).
    """

    def __init__(
        self,
        target_browser: str = BrowserTarget.GENERIC,
        mock_mode: bool = False,
    ):
        self.target_browser = (
            target_browser.lower().strip()
            if target_browser.lower().strip() in BrowserTarget.ALL_TARGETS
            else BrowserTarget.GENERIC
        )
        self.mock_mode = mock_mode

        # Simulated Browser State (for headless verification)
        self.tab_count: int = 1
        self.active_tab_index: int = 0
        self.zoom_level_pct: int = 100
        self.scroll_offset_px: int = 0
        self.history_index: int = 0

        self.last_action: str = "NONE"
        self.last_action_time: float = 0.0
        self.event_log: List[Dict[str, Any]] = []

    def set_target(self, target_browser: str) -> bool:
        """Sets the active browser target profile."""
        canonical = target_browser.lower().strip()
        if canonical in BrowserTarget.ALL_TARGETS:
            self.target_browser = canonical
            logger.info(f"[BrowserController] Active browser target profile set to: '{self.target_browser}'")
            return True
        self.target_browser = BrowserTarget.GENERIC
        return False

    # =========================================================================
    # Page Scrolling & Viewport Navigation (Module 11)
    # =========================================================================

    def scroll_up(self, amount: int = 150) -> bool:
        """Scrolls up the active web page."""
        self.scroll_offset_px = max(0, self.scroll_offset_px - amount)
        self.last_action = "SCROLL_UP"
        self._log_event("SCROLL_UP", {"amount": amount, "offset": self.scroll_offset_px})

        if self.mock_mode:
            return True

        try:
            pyautogui.scroll(amount)
            return True
        except Exception as e:
            logger.error(f"[BrowserController] Scroll up failed: {e}")
            return False

    def scroll_down(self, amount: int = 150) -> bool:
        """Scrolls down the active web page."""
        self.scroll_offset_px += amount
        self.last_action = "SCROLL_DOWN"
        self._log_event("SCROLL_DOWN", {"amount": amount, "offset": self.scroll_offset_px})

        if self.mock_mode:
            return True

        try:
            pyautogui.scroll(-amount)
            return True
        except Exception as e:
            logger.error(f"[BrowserController] Scroll down failed: {e}")
            return False

    def history_back(self) -> bool:
        """Navigates back in browser history (Alt + Left)."""
        self.history_index = max(0, self.history_index - 1)
        self.last_action = "HISTORY_BACK"
        self._log_event("HISTORY_BACK", {"history_index": self.history_index})

        if self.mock_mode:
            return True

        try:
            if sys.platform == "darwin":
                pyautogui.hotkey("command", "[")
            else:
                pyautogui.hotkey("alt", "left")
            return True
        except Exception as e:
            logger.error(f"[BrowserController] History back failed: {e}")
            return False

    def history_forward(self) -> bool:
        """Navigates forward in browser history (Alt + Right)."""
        self.history_index += 1
        self.last_action = "HISTORY_FORWARD"
        self._log_event("HISTORY_FORWARD", {"history_index": self.history_index})

        if self.mock_mode:
            return True

        try:
            if sys.platform == "darwin":
                pyautogui.hotkey("command", "]")
            else:
                pyautogui.hotkey("alt", "right")
            return True
        except Exception as e:
            logger.error(f"[BrowserController] History forward failed: {e}")
            return False

    def refresh_page(self, hard_reload: bool = False) -> bool:
        """Reloads the current page (F5 or Ctrl + F5)."""
        self.last_action = "REFRESH_PAGE"
        self._log_event("REFRESH_PAGE", {"hard": hard_reload})

        if self.mock_mode:
            return True

        try:
            if hard_reload:
                pyautogui.hotkey("ctrl", "f5")
            else:
                pyautogui.press("f5")
            return True
        except Exception as e:
            logger.error(f"[BrowserController] Refresh page failed: {e}")
            return False

    # =========================================================================
    # Zoom Controls (Module 11)
    # =========================================================================

    def zoom_in(self, step: int = 10) -> bool:
        """Zooms into the webpage (Ctrl + Plus)."""
        self.zoom_level_pct = min(500, self.zoom_level_pct + step)
        self.last_action = "ZOOM_IN"
        self._log_event("ZOOM_IN", {"zoom": self.zoom_level_pct})

        if self.mock_mode:
            return True

        try:
            pyautogui.hotkey("ctrl", "+")
            return True
        except Exception as e:
            logger.error(f"[BrowserController] Zoom in failed: {e}")
            return False

    def zoom_out(self, step: int = 10) -> bool:
        """Zooms out of the webpage (Ctrl + Minus)."""
        self.zoom_level_pct = max(25, self.zoom_level_pct - step)
        self.last_action = "ZOOM_OUT"
        self._log_event("ZOOM_OUT", {"zoom": self.zoom_level_pct})

        if self.mock_mode:
            return True

        try:
            pyautogui.hotkey("ctrl", "-")
            return True
        except Exception as e:
            logger.error(f"[BrowserController] Zoom out failed: {e}")
            return False

    def zoom_reset(self) -> bool:
        """Resets zoom level to 100% (Ctrl + 0)."""
        self.zoom_level_pct = 100
        self.last_action = "ZOOM_RESET"
        self._log_event("ZOOM_RESET", {"zoom": 100})

        if self.mock_mode:
            return True

        try:
            pyautogui.hotkey("ctrl", "0")
            return True
        except Exception as e:
            logger.error(f"[BrowserController] Zoom reset failed: {e}")
            return False

    # =========================================================================
    # Tab Management (Module 11)
    # =========================================================================

    def next_tab(self) -> bool:
        """Switches to the next open tab (Ctrl + Tab)."""
        if self.tab_count > 1:
            self.active_tab_index = (self.active_tab_index + 1) % self.tab_count
        self.last_action = "TAB_SWITCH_NEXT"
        self._log_event("TAB_SWITCH_NEXT", {"active_tab": self.active_tab_index})

        if self.mock_mode:
            return True

        try:
            pyautogui.hotkey("ctrl", "tab")
            return True
        except Exception as e:
            logger.error(f"[BrowserController] Next tab failed: {e}")
            return False

    def prev_tab(self) -> bool:
        """Switches to the previous open tab (Ctrl + Shift + Tab)."""
        if self.tab_count > 1:
            self.active_tab_index = (self.active_tab_index - 1) % self.tab_count
        self.last_action = "TAB_SWITCH_PREV"
        self._log_event("TAB_SWITCH_PREV", {"active_tab": self.active_tab_index})

        if self.mock_mode:
            return True

        try:
            pyautogui.hotkey("ctrl", "shift", "tab")
            return True
        except Exception as e:
            logger.error(f"[BrowserController] Previous tab failed: {e}")
            return False

    def new_tab(self) -> bool:
        """Opens a new browser tab (Ctrl + T)."""
        self.tab_count += 1
        self.active_tab_index = self.tab_count - 1
        self.last_action = "NEW_TAB"
        self._log_event("NEW_TAB", {"tab_count": self.tab_count})

        if self.mock_mode:
            return True

        try:
            pyautogui.hotkey("ctrl", "t")
            return True
        except Exception as e:
            logger.error(f"[BrowserController] New tab failed: {e}")
            return False

    def close_tab(self) -> bool:
        """Closes the current browser tab (Ctrl + W)."""
        self.tab_count = max(1, self.tab_count - 1)
        self.active_tab_index = min(self.tab_count - 1, self.active_tab_index)
        self.last_action = "CLOSE_TAB"
        self._log_event("CLOSE_TAB", {"tab_count": self.tab_count})

        if self.mock_mode:
            return True

        try:
            pyautogui.hotkey("ctrl", "w")
            return True
        except Exception as e:
            logger.error(f"[BrowserController] Close tab failed: {e}")
            return False

    def reopen_closed_tab(self) -> bool:
        """Reopens the most recently closed tab (Ctrl + Shift + T)."""
        self.tab_count += 1
        self.last_action = "REOPEN_TAB"
        self._log_event("REOPEN_TAB", {"tab_count": self.tab_count})

        if self.mock_mode:
            return True

        try:
            pyautogui.hotkey("ctrl", "shift", "t")
            return True
        except Exception as e:
            logger.error(f"[BrowserController] Reopen closed tab failed: {e}")
            return False

    def focus_address_bar(self) -> bool:
        """Focuses the browser URL address bar (Ctrl + L)."""
        self.last_action = "FOCUS_URL"
        self._log_event("FOCUS_URL", {})

        if self.mock_mode:
            return True

        try:
            pyautogui.hotkey("ctrl", "l")
            return True
        except Exception as e:
            logger.error(f"[BrowserController] Focus address bar failed: {e}")
            return False

    # =========================================================================
    # Unified Action Router
    # =========================================================================

    def execute_action(self, action: Optional[str], **kwargs) -> bool:
        """
        Executes browser commands mapped from real-time gesture recognition in BROWSER mode.

        Supported actions:
        SCROLL_UP, SCROLL_DOWN, HISTORY_BACK, HISTORY_FORWARD, REFRESH_PAGE,
        ZOOM_IN, ZOOM_OUT, ZOOM_RESET, TAB_SWITCH_NEXT, TAB_SWITCH_PREV,
        NEW_TAB, CLOSE_TAB, REOPEN_TAB, FOCUS_URL.
        """
        if not action or action == "NONE":
            return False

        action_upper = action.upper().strip()

        if action_upper in ["SCROLL_UP", "PAGE_SCROLL_UP"]:
            return self.scroll_up()

        elif action_upper in ["SCROLL_DOWN", "PAGE_SCROLL_DOWN"]:
            return self.scroll_down()

        elif action_upper in ["HISTORY_BACK", "BACK"]:
            return self.history_back()

        elif action_upper in ["HISTORY_FORWARD", "FORWARD"]:
            return self.history_forward()

        elif action_upper in ["REFRESH_PAGE", "RELOAD_PAGE", "REFRESH"]:
            return self.refresh_page(kwargs.get("hard", False))

        elif action_upper == "ZOOM_IN":
            return self.zoom_in()

        elif action_upper == "ZOOM_OUT":
            return self.zoom_out()

        elif action_upper == "ZOOM_RESET":
            return self.zoom_reset()

        elif action_upper in ["TAB_SWITCH_NEXT", "NEXT_TAB"]:
            return self.next_tab()

        elif action_upper in ["TAB_SWITCH_PREV", "PREV_TAB"]:
            return self.prev_tab()

        elif action_upper == "NEW_TAB":
            return self.new_tab()

        elif action_upper == "CLOSE_TAB":
            return self.close_tab()

        elif action_upper in ["REOPEN_TAB", "REOPEN_CLOSED_TAB"]:
            return self.reopen_closed_tab()

        elif action_upper in ["FOCUS_URL", "ADDRESS_BAR"]:
            return self.focus_address_bar()

        return False

    def _log_event(self, event_type: str, details: Dict[str, Any]):
        """Records event into internal log buffer for diagnostics and testing."""
        self.event_log.append({
            "timestamp": time.perf_counter(),
            "event": event_type,
            "details": details
        })
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-500:]
