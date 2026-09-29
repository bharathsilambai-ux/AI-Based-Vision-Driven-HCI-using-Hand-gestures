"""
Media Control Module.
Handles system volume (via PyCaw or PyAutoGUI keys) and media playback
(Play/Pause, Next Track, Previous Track, Mute/Unmute).
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

# Attempt PyCaw import for direct Windows Audio Endpoint control
HAVE_PYCAW = False
try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    HAVE_PYCAW = True
except Exception:
    HAVE_PYCAW = False


class MediaController:
    """Controls OS media playback and system audio volume."""

    def __init__(self, volume_step: int = config.VOLUME_STEP):
        self.volume_step = volume_step
        self.endpoint_volume = None
        self._init_pycaw()

    def _init_pycaw(self):
        """Initializes PyCaw endpoint volume interface if available."""
        if HAVE_PYCAW:
            try:
                devices = AudioUtilities.GetSpeakers()
                if hasattr(devices, "EndpointVolume"):
                    self.endpoint_volume = devices.EndpointVolume
                elif hasattr(devices, "Activate"):
                    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                    self.endpoint_volume = interface.QueryInterface(IAudioEndpointVolume)
                else:
                    self.endpoint_volume = getattr(devices, "_volume", None)
                if self.endpoint_volume is not None:
                    logger.info("[Media] PyCaw audio endpoint initialized successfully.")
                else:
                    logger.warning("[Media] PyCaw endpoint not found. Falling back to virtual media keys.")
            except Exception as e:
                logger.warning(f"[Media] PyCaw init notice: {e}. Falling back to virtual media keys.")
                self.endpoint_volume = None

    def volume_up(self, step: Optional[int] = None):
        """Increases system volume."""
        s = step or self.volume_step
        if self.endpoint_volume is not None:
            try:
                cur = self.endpoint_volume.GetMasterVolumeLevelScalar()
                new_v = min(1.0, cur + (s / 100.0))
                self.endpoint_volume.SetMasterVolumeLevelScalar(new_v, None)
                logger.info(f"[Media] Volume increased to {int(new_v * 100)}%")
                return
            except Exception:
                pass
        # Fallback to key
        action_dispatcher.submit(pyautogui.press, "volumeup", priority=True)
        logger.info("[Media] Volume Up key pressed.")

    def volume_down(self, step: Optional[int] = None):
        """Decreases system volume."""
        s = step or self.volume_step
        if self.endpoint_volume is not None:
            try:
                cur = self.endpoint_volume.GetMasterVolumeLevelScalar()
                new_v = max(0.0, cur - (s / 100.0))
                self.endpoint_volume.SetMasterVolumeLevelScalar(new_v, None)
                logger.info(f"[Media] Volume decreased to {int(new_v * 100)}%")
                return
            except Exception:
                pass
        # Fallback to key
        action_dispatcher.submit(pyautogui.press, "volumedown", priority=True)
        logger.info("[Media] Volume Down key pressed.")

    def toggle_mute(self):
        """Toggles audio mute."""
        if self.endpoint_volume is not None:
            try:
                muted = self.endpoint_volume.GetMute()
                self.endpoint_volume.SetMute(not muted, None)
                logger.info(f"[Media] Audio {'Muted' if not muted else 'Unmuted'}")
                return
            except Exception:
                pass
        action_dispatcher.submit(pyautogui.press, "volumemute", priority=True)
        logger.info("[Media] Volume Mute key pressed.")

    def play_pause(self):
        """Toggles media play/pause."""
        action_dispatcher.submit(pyautogui.press, "playpause", priority=True)
        logger.info("[Media] Play/Pause key dispatched.")

    def next_track(self):
        """Skips to next media track."""
        action_dispatcher.submit(pyautogui.press, "nexttrack", priority=True)
        logger.info("[Media] Next Track key dispatched.")

    def prev_track(self):
        """Returns to previous media track."""
        action_dispatcher.submit(pyautogui.press, "prevtrack", priority=True)
        logger.info("[Media] Previous Track key dispatched.")
