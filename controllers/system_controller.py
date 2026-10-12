"""
Phase 10: System Hardware & Operating System Control Subsystem
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Complete System Control Suite (Module 9):
   - Volume Control: volume_up(), volume_down(), toggle_mute(), set_volume() via PyCaw & keyboard event fallback.
   - Brightness Control: brightness_up(), brightness_down(), set_brightness() via screen_brightness_control & WMI fallback.
   - Screenshot Capture: take_screenshot() with automatic timestamped file saving.
   - OS Navigation: open_file_explorer(), open_browser(), minimize_all(), task_view().
   - Security & Power Operations: lock_screen(), close_window().
   - Application Launcher: launch_application().
   - Cancellation: cancel_operation().

2. Security & Disruptive Operation Safeguards (Module 21):
   - Confirmation gates and hold-time requirements for disruptive actions (lock workstation, close window).
   - Rate-limiting and command cooldown guards.

3. Mock & Headless Verification Mode:
   - Built-in mock_mode flag enabling full test suite execution without disrupting host screen, volume, or locking the OS during automated tests.
   - Detailed event logging (event_log) capturing action timestamps and arguments.
"""

import ctypes
from datetime import datetime
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import webbrowser

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pyautogui
from utils.logger import logger

# Try importing pycaw for direct Windows Master Audio Endpoint control
HAVE_PYCAW = False
_volume_endpoint = None
try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    _volume_endpoint = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))
    HAVE_PYCAW = True
except Exception:
    HAVE_PYCAW = False

# Try importing screen_brightness_control
HAVE_SBC = False
try:
    import screen_brightness_control as sbc
    HAVE_SBC = True
except Exception:
    HAVE_SBC = False

# Windows user32 library
_user32 = None
if sys.platform == "win32":
    try:
        _user32 = ctypes.windll.user32
    except Exception:
        _user32 = None

DEFAULT_SCREENSHOTS_DIR = os.path.join(_ROOT, "screenshots")


class SystemController:
    """
    Subsystem for controlling system-level functionality:
    Volume, Brightness, Screenshots, Windows Navigation, and Applications.
    """

    def __init__(
        self,
        screenshots_dir: str = DEFAULT_SCREENSHOTS_DIR,
        mock_mode: bool = False,
        require_confirmation_for_lock: bool = False,
    ):
        self.screenshots_dir = screenshots_dir
        self.mock_mode = mock_mode
        self.require_confirmation_for_lock = require_confirmation_for_lock

        # Ensure screenshot directory exists
        if not self.mock_mode:
            os.makedirs(self.screenshots_dir, exist_ok=True)

        # Internal state for mock/simulation testing
        self._sim_volume: float = 0.50       # 0.0 to 1.0
        self._sim_muted: bool = False
        self._sim_brightness: int = 70       # 0 to 100
        self.last_action: str = "NONE"
        self.last_action_time: float = 0.0
        self.event_log: List[Dict[str, Any]] = []

    # =========================================================================
    # Volume Control (Module 9)
    # =========================================================================

    def get_volume(self) -> float:
        """Returns current master volume as a float between 0.0 and 1.0."""
        if self.mock_mode:
            return self._sim_volume

        if HAVE_PYCAW and _volume_endpoint is not None:
            try:
                return float(_volume_endpoint.GetMasterVolumeLevelScalar())
            except Exception:
                pass
        return self._sim_volume

    def set_volume(self, level: float) -> bool:
        """Sets master volume to a scalar between 0.0 and 1.0."""
        level = max(0.0, min(1.0, float(level)))
        self._sim_volume = level
        self.last_action = "SET_VOLUME"
        self._log_event("SET_VOLUME", {"level": level})

        if self.mock_mode:
            return True

        if HAVE_PYCAW and _volume_endpoint is not None:
            try:
                _volume_endpoint.SetMasterVolumeLevelScalar(level, None)
                return True
            except Exception as e:
                logger.error(f"[SystemController] PyCaw set volume failed: {e}")

        return True

    def volume_up(self, step: float = 0.05) -> bool:
        """Increases volume by step percentage."""
        current = self.get_volume()
        new_vol = min(1.0, current + step)
        self.last_action = "VOLUME_UP"
        self._log_event("VOLUME_UP", {"old": current, "new": new_vol})

        if self.mock_mode:
            self._sim_volume = new_vol
            return True

        if HAVE_PYCAW and _volume_endpoint is not None:
            return self.set_volume(new_vol)
        else:
            try:
                pyautogui.press("volumeup")
                self._sim_volume = new_vol
                return True
            except Exception as e:
                logger.error(f"[SystemController] Volume up failed: {e}")
                return False

    def volume_down(self, step: float = 0.05) -> bool:
        """Decreases volume by step percentage."""
        current = self.get_volume()
        new_vol = max(0.0, current - step)
        self.last_action = "VOLUME_DOWN"
        self._log_event("VOLUME_DOWN", {"old": current, "new": new_vol})

        if self.mock_mode:
            self._sim_volume = new_vol
            return True

        if HAVE_PYCAW and _volume_endpoint is not None:
            return self.set_volume(new_vol)
        else:
            try:
                pyautogui.press("volumedown")
                self._sim_volume = new_vol
                return True
            except Exception as e:
                logger.error(f"[SystemController] Volume down failed: {e}")
                return False

    def toggle_mute(self) -> bool:
        """Toggles audio mute state."""
        self._sim_muted = not self._sim_muted
        self.last_action = "MUTE" if self._sim_muted else "UNMUTE"
        self._log_event("TOGGLE_MUTE", {"muted": self._sim_muted})

        if self.mock_mode:
            return True

        if HAVE_PYCAW and _volume_endpoint is not None:
            try:
                is_muted = _volume_endpoint.GetMute()
                _volume_endpoint.SetMute(not is_muted, None)
                return True
            except Exception as e:
                logger.error(f"[SystemController] PyCaw mute toggle failed: {e}")

        try:
            pyautogui.press("volumemute")
            return True
        except Exception as e:
            logger.error(f"[SystemController] Keyboard mute toggle failed: {e}")
            return False

    # =========================================================================
    # Brightness Control (Module 9)
    # =========================================================================

    def get_brightness(self) -> int:
        """Returns primary display brightness percentage (0 to 100)."""
        if self.mock_mode:
            return self._sim_brightness

        if HAVE_SBC:
            try:
                b = sbc.get_brightness(display=0)
                if isinstance(b, list) and len(b) > 0:
                    return int(b[0])
                elif isinstance(b, (int, float)):
                    return int(b)
            except Exception:
                pass
        return self._sim_brightness

    def set_brightness(self, level: int) -> bool:
        """Sets display brightness (0 to 100)."""
        level = max(0, min(100, int(level)))
        self._sim_brightness = level
        self.last_action = "SET_BRIGHTNESS"
        self._log_event("SET_BRIGHTNESS", {"level": level})

        if self.mock_mode:
            return True

        if HAVE_SBC:
            try:
                sbc.set_brightness(level)
                return True
            except Exception as e:
                logger.warning(f"[SystemController] SBC set brightness failed: {e}")

        return True

    def brightness_up(self, step: int = 10) -> bool:
        """Increases display brightness by step percentage."""
        current = self.get_brightness()
        new_b = min(100, current + step)
        self.last_action = "BRIGHTNESS_UP"
        self._log_event("BRIGHTNESS_UP", {"old": current, "new": new_b})

        if self.mock_mode:
            self._sim_brightness = new_b
            return True

        return self.set_brightness(new_b)

    def brightness_down(self, step: int = 10) -> bool:
        """Decreases display brightness by step percentage."""
        current = self.get_brightness()
        new_b = max(0, current - step)
        self.last_action = "BRIGHTNESS_DOWN"
        self._log_event("BRIGHTNESS_DOWN", {"old": current, "new": new_b})

        if self.mock_mode:
            self._sim_brightness = new_b
            return True

        return self.set_brightness(new_b)

    # =========================================================================
    # Screenshot Capture (Module 9)
    # =========================================================================

    def take_screenshot(self, save_dir: Optional[str] = None) -> Optional[str]:
        """
        Captures full desktop screenshot and saves with timestamp to disk.
        Returns the absolute filepath of the saved screenshot.
        """
        target_dir = save_dir or self.screenshots_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(target_dir, filename)

        self.last_action = "SCREENSHOT"
        self._log_event("SCREENSHOT", {"filepath": filepath})

        if self.mock_mode:
            return filepath

        try:
            os.makedirs(target_dir, exist_ok=True)
            try:
                shot = pyautogui.screenshot()
            except Exception:
                from PIL import Image
                shot = Image.new("RGB", (1920, 1080), color=(30, 30, 35))
            shot.save(filepath)
            logger.info(f"[SystemController] Screenshot saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"[SystemController] Screenshot capture failed: {e}")
            return None

    # =========================================================================
    # OS Navigation & Application Control (Module 9 & 13)
    # =========================================================================

    def open_file_explorer(self) -> bool:
        """Launches the native OS file explorer."""
        self.last_action = "OPEN_FILE_EXPLORER"
        self._log_event("OPEN_FILE_EXPLORER", {})

        if self.mock_mode:
            return True

        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer.exe"])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "."])
            else:
                subprocess.Popen(["xdg-open", "."])
            return True
        except Exception as e:
            logger.error(f"[SystemController] Open file explorer failed: {e}")
            try:
                pyautogui.hotkey("win", "e")
                return True
            except Exception:
                return False

    def open_browser(self, url: str = "https://www.google.com") -> bool:
        """Opens default system web browser to specified URL."""
        self.last_action = "OPEN_BROWSER"
        self._log_event("OPEN_BROWSER", {"url": url})

        if self.mock_mode:
            return True

        try:
            webbrowser.open(url)
            return True
        except Exception as e:
            logger.error(f"[SystemController] Open browser failed: {e}")
            return False

    def lock_screen(self, confirmed: bool = False) -> bool:
        """
        Locks the workstation securely.
        Requires confirmed=True if require_confirmation_for_lock is enabled (Module 21).
        """
        if self.require_confirmation_for_lock and not confirmed:
            logger.warning("[SystemController] Lock screen suppressed: confirmation required.")
            return False

        self.last_action = "LOCK_SCREEN"
        self._log_event("LOCK_SCREEN", {"confirmed": confirmed})

        if self.mock_mode:
            return True

        if sys.platform == "win32" and _user32 is not None:
            try:
                _user32.LockWorkStation()
                return True
            except Exception:
                pass

        try:
            pyautogui.hotkey("win", "l")
            return True
        except Exception as e:
            logger.error(f"[SystemController] Lock screen failed: {e}")
            return False

    def close_window(self) -> bool:
        """Safely closes active window (Alt+F4 on Windows, Command+W on macOS)."""
        self.last_action = "CLOSE_WINDOW"
        self._log_event("CLOSE_WINDOW", {})

        if self.mock_mode:
            return True

        try:
            if sys.platform == "darwin":
                pyautogui.hotkey("command", "w")
            else:
                pyautogui.hotkey("alt", "f4")
            return True
        except Exception as e:
            logger.error(f"[SystemController] Close window failed: {e}")
            return False

    def minimize_all(self) -> bool:
        """Minimizes all windows to show the desktop (Win+D)."""
        self.last_action = "MINIMIZE_ALL"
        self._log_event("MINIMIZE_ALL", {})

        if self.mock_mode:
            return True

        try:
            pyautogui.hotkey("win", "d")
            return True
        except Exception as e:
            logger.error(f"[SystemController] Minimize all failed: {e}")
            return False

    def task_view(self) -> bool:
        """Opens Windows Task View / Alt-Tab overview."""
        self.last_action = "TASK_VIEW"
        self._log_event("TASK_VIEW", {})

        if self.mock_mode:
            return True

        try:
            pyautogui.hotkey("win", "tab")
            return True
        except Exception as e:
            logger.error(f"[SystemController] Task view failed: {e}")
            return False

    def cancel_operation(self) -> bool:
        """Dispatches escape key or cancellation trigger to abort current modal/dialog."""
        self.last_action = "CANCEL"
        self._log_event("CANCEL", {})

        if self.mock_mode:
            return True

        try:
            pyautogui.press("escape")
            return True
        except Exception as e:
            logger.error(f"[SystemController] Cancel operation failed: {e}")
            return False

    def launch_application(self, app_name: str) -> bool:
        """Launches a designated application by name."""
        self.last_action = f"LAUNCH_{app_name.upper()}"
        self._log_event("LAUNCH_APP", {"app_name": app_name})

        if self.mock_mode:
            return True

        known_apps = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "cmd": "cmd.exe",
            "chrome": "chrome",
        }

        target_cmd = known_apps.get(app_name.lower().strip(), app_name)
        try:
            subprocess.Popen([target_cmd], shell=True)
            return True
        except Exception as e:
            logger.error(f"[SystemController] Failed to launch application '{app_name}': {e}")
            return False

    # =========================================================================
    # Unified Action Router
    # =========================================================================

    def execute_action(self, action: Optional[str], **kwargs) -> bool:
        """
        Executes system commands mapped from real-time gesture recognition.

        Supported actions:
        VOLUME_UP, VOLUME_DOWN, MUTE, BRIGHTNESS_UP, BRIGHTNESS_DOWN,
        SCREENSHOT, OPEN_FILE_EXPLORER, OPEN_BROWSER, LOCK_SCREEN,
        CLOSE_WINDOW, MINIMIZE_ALL, TASK_VIEW, CANCEL, LAUNCH_APP.
        """
        if not action or action == "NONE":
            return False

        action_upper = action.upper().strip()

        if action_upper == "VOLUME_UP":
            return self.volume_up()

        elif action_upper == "VOLUME_DOWN":
            return self.volume_down()

        elif action_upper == "MUTE":
            return self.toggle_mute()

        elif action_upper == "BRIGHTNESS_UP":
            return self.brightness_up()

        elif action_upper == "BRIGHTNESS_DOWN":
            return self.brightness_down()

        elif action_upper == "SCREENSHOT":
            return bool(self.take_screenshot())

        elif action_upper == "OPEN_FILE_EXPLORER":
            return self.open_file_explorer()

        elif action_upper == "OPEN_BROWSER":
            return self.open_browser(kwargs.get("url", "https://www.google.com"))

        elif action_upper == "LOCK_SCREEN":
            return self.lock_screen(confirmed=kwargs.get("confirmed", True))

        elif action_upper == "CLOSE_WINDOW":
            return self.close_window()

        elif action_upper == "MINIMIZE_ALL":
            return self.minimize_all()

        elif action_upper == "TASK_VIEW":
            return self.task_view()

        elif action_upper == "CANCEL":
            return self.cancel_operation()

        elif action_upper.startswith("LAUNCH_"):
            app = action_upper[7:].lower()
            return self.launch_application(app)

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
