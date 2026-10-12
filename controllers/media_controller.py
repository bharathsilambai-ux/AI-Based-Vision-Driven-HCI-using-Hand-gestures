"""
Phase 11: Media Player & Streaming Control Subsystem
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Complete Media Control Operations (Module 10):
   - play_pause(): Toggle audio/video playback via media keys, spacebar, or dedicated shortcuts.
   - next_track(), prev_track(): Track and chapter navigation.
   - volume_up(), volume_down(), toggle_mute(): Integrated sound adjustments.
   - stop(): Stops active media playback.
   - seek_forward(), seek_backward(): Fast-forward / rewind scrubbing (e.g. +10s / -10s).
   - toggle_fullscreen(): Fullscreen display toggle ('f' or Alt+Enter).
   - toggle_captions(): Closed captions / subtitle toggle ('c').

2. Application-Specific Media Context Adaptations (Module 13):
   - Adapts keyboard shortcuts dynamically for YouTube, Spotify, VLC Media Player,
     Netflix, and generic OS Media Keys.

3. Mock & Headless Verification Architecture:
   - Full mock_mode simulation preserving internal playback states (is_playing, track_index, volume, position)
     for automated test execution without hijacking foreground player windows.
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


class MediaTarget:
    """Target media application profiles."""
    GENERIC = "generic"
    YOUTUBE = "youtube"
    SPOTIFY = "spotify"
    VLC = "vlc"
    NETFLIX = "netflix"

    ALL_TARGETS = [GENERIC, YOUTUBE, SPOTIFY, VLC, NETFLIX]


class MediaController:
    """
    Subsystem for controlling media playback across desktop players and web streaming services.
    Supports Module 10 (Media Control) and Module 13 (Application-Specific Context).
    """

    def __init__(
        self,
        target_app: str = MediaTarget.GENERIC,
        mock_mode: bool = False,
    ):
        self.target_app = target_app.lower().strip() if target_app.lower().strip() in MediaTarget.ALL_TARGETS else MediaTarget.GENERIC
        self.mock_mode = mock_mode

        # Simulated Player State (for mock verification and state synchronization)
        self.is_playing: bool = True
        self.track_index: int = 1
        self.playback_position_sec: float = 0.0
        self.is_muted: bool = False
        self.is_fullscreen: bool = False
        self.captions_enabled: bool = False
        self.volume: float = 0.70  # 0.0 to 1.0

        self.last_action: str = "NONE"
        self.last_action_time: float = 0.0
        self.event_log: List[Dict[str, Any]] = []

    def set_target(self, target_app: str) -> bool:
        """Sets the active media target application."""
        canonical = target_app.lower().strip()
        if canonical in MediaTarget.ALL_TARGETS:
            self.target_app = canonical
            logger.info(f"[MediaController] Active media target profile set to: '{self.target_app}'")
            return True
        logger.warning(f"[MediaController] Unknown media target '{target_app}', defaulting to generic.")
        self.target_app = MediaTarget.GENERIC
        return False

    # =========================================================================
    # Playback Controls (Module 10)
    # =========================================================================

    def play_pause(self) -> bool:
        """Toggles play/pause state."""
        self.is_playing = not self.is_playing
        self.last_action = "PLAY" if self.is_playing else "PAUSE"
        self._log_event("PLAY_PAUSE", {"is_playing": self.is_playing, "target": self.target_app})

        if self.mock_mode:
            return True

        try:
            if self.target_app == MediaTarget.YOUTUBE:
                pyautogui.press("k")
            elif self.target_app in [MediaTarget.VLC, MediaTarget.NETFLIX]:
                pyautogui.press("space")
            else:
                pyautogui.press("playpause")
            return True
        except Exception as e:
            logger.error(f"[MediaController] Play/Pause failed: {e}")
            return False

    def next_track(self) -> bool:
        """Skips to the next track or video."""
        self.track_index += 1
        self.playback_position_sec = 0.0
        self.last_action = "NEXT_TRACK"
        self._log_event("NEXT_TRACK", {"track_index": self.track_index, "target": self.target_app})

        if self.mock_mode:
            return True

        try:
            if self.target_app == MediaTarget.YOUTUBE:
                pyautogui.hotkey("shift", "n")
            elif self.target_app == MediaTarget.VLC:
                pyautogui.press("n")
            else:
                pyautogui.press("nexttrack")
            return True
        except Exception as e:
            logger.error(f"[MediaController] Next track failed: {e}")
            return False

    def prev_track(self) -> bool:
        """Returns to the previous track or restarts current track."""
        self.track_index = max(1, self.track_index - 1)
        self.playback_position_sec = 0.0
        self.last_action = "PREV_TRACK"
        self._log_event("PREV_TRACK", {"track_index": self.track_index, "target": self.target_app})

        if self.mock_mode:
            return True

        try:
            if self.target_app == MediaTarget.YOUTUBE:
                pyautogui.hotkey("shift", "p")
            elif self.target_app == MediaTarget.VLC:
                pyautogui.press("p")
            else:
                pyautogui.press("prevtrack")
            return True
        except Exception as e:
            logger.error(f"[MediaController] Previous track failed: {e}")
            return False

    def stop(self) -> bool:
        """Stops playback completely."""
        self.is_playing = False
        self.playback_position_sec = 0.0
        self.last_action = "STOP"
        self._log_event("STOP", {"target": self.target_app})

        if self.mock_mode:
            return True

        try:
            if self.target_app == MediaTarget.VLC:
                pyautogui.press("s")
            else:
                pyautogui.press("stop")
            return True
        except Exception as e:
            logger.error(f"[MediaController] Stop playback failed: {e}")
            return False

    # =========================================================================
    # Scrubbing & Seeking (Module 10 & 13)
    # =========================================================================

    def seek_forward(self, seconds: int = 10) -> bool:
        """Seeks forward by specified seconds."""
        self.playback_position_sec += seconds
        self.last_action = f"SEEK_FORWARD_{seconds}S"
        self._log_event("SEEK_FORWARD", {"seconds": seconds, "position": self.playback_position_sec})

        if self.mock_mode:
            return True

        try:
            if self.target_app == MediaTarget.YOUTUBE:
                if seconds == 10:
                    pyautogui.press("l")
                else:
                    pyautogui.press("right")
            elif self.target_app == MediaTarget.VLC:
                pyautogui.hotkey("shift", "right")
            else:
                pyautogui.press("right")
            return True
        except Exception as e:
            logger.error(f"[MediaController] Seek forward failed: {e}")
            return False

    def seek_backward(self, seconds: int = 10) -> bool:
        """Seeks backward by specified seconds."""
        self.playback_position_sec = max(0.0, self.playback_position_sec - seconds)
        self.last_action = f"SEEK_BACKWARD_{seconds}S"
        self._log_event("SEEK_BACKWARD", {"seconds": seconds, "position": self.playback_position_sec})

        if self.mock_mode:
            return True

        try:
            if self.target_app == MediaTarget.YOUTUBE:
                if seconds == 10:
                    pyautogui.press("j")
                else:
                    pyautogui.press("left")
            elif self.target_app == MediaTarget.VLC:
                pyautogui.hotkey("shift", "left")
            else:
                pyautogui.press("left")
            return True
        except Exception as e:
            logger.error(f"[MediaController] Seek backward failed: {e}")
            return False

    # =========================================================================
    # Display & Audio Adjustments (Module 10 & 13)
    # =========================================================================

    def toggle_fullscreen(self) -> bool:
        """Toggles media player full screen mode."""
        self.is_fullscreen = not self.is_fullscreen
        self.last_action = "FULLSCREEN_TOGGLE"
        self._log_event("FULLSCREEN_TOGGLE", {"is_fullscreen": self.is_fullscreen})

        if self.mock_mode:
            return True

        try:
            if self.target_app in [MediaTarget.YOUTUBE, MediaTarget.VLC, MediaTarget.NETFLIX]:
                pyautogui.press("f")
            else:
                pyautogui.hotkey("alt", "enter")
            return True
        except Exception as e:
            logger.error(f"[MediaController] Toggle fullscreen failed: {e}")
            return False

    def toggle_captions(self) -> bool:
        """Toggles closed captions or subtitles."""
        self.captions_enabled = not self.captions_enabled
        self.last_action = "CAPTIONS_TOGGLE"
        self._log_event("CAPTIONS_TOGGLE", {"captions_enabled": self.captions_enabled})

        if self.mock_mode:
            return True

        try:
            if self.target_app in [MediaTarget.YOUTUBE, MediaTarget.NETFLIX]:
                pyautogui.press("c")
            elif self.target_app == MediaTarget.VLC:
                pyautogui.press("v")
            return True
        except Exception as e:
            logger.error(f"[MediaController] Toggle captions failed: {e}")
            return False

    def volume_up(self, step: float = 0.05) -> bool:
        """Increases playback volume."""
        self.volume = min(1.0, self.volume + step)
        self.last_action = "VOLUME_UP"
        self._log_event("VOLUME_UP", {"volume": self.volume})

        if self.mock_mode:
            return True

        try:
            if self.target_app == MediaTarget.YOUTUBE:
                pyautogui.press("up")
            else:
                pyautogui.press("volumeup")
            return True
        except Exception as e:
            logger.error(f"[MediaController] Volume up failed: {e}")
            return False

    def volume_down(self, step: float = 0.05) -> bool:
        """Decreases playback volume."""
        self.volume = max(0.0, self.volume - step)
        self.last_action = "VOLUME_DOWN"
        self._log_event("VOLUME_DOWN", {"volume": self.volume})

        if self.mock_mode:
            return True

        try:
            if self.target_app == MediaTarget.YOUTUBE:
                pyautogui.press("down")
            else:
                pyautogui.press("volumedown")
            return True
        except Exception as e:
            logger.error(f"[MediaController] Volume down failed: {e}")
            return False

    def toggle_mute(self) -> bool:
        """Mutes or unmutes media player sound."""
        self.is_muted = not self.is_muted
        self.last_action = "MUTE" if self.is_muted else "UNMUTE"
        self._log_event("TOGGLE_MUTE", {"is_muted": self.is_muted})

        if self.mock_mode:
            return True

        try:
            if self.target_app in [MediaTarget.YOUTUBE, MediaTarget.VLC, MediaTarget.NETFLIX]:
                pyautogui.press("m")
            else:
                pyautogui.press("volumemute")
            return True
        except Exception as e:
            logger.error(f"[MediaController] Toggle mute failed: {e}")
            return False

    # =========================================================================
    # Unified Action Router
    # =========================================================================

    def execute_action(self, action: Optional[str], **kwargs) -> bool:
        """
        Executes media commands mapped from real-time gesture recognition in MEDIA mode.

        Supported actions:
        PLAY_PAUSE, NEXT_TRACK, PREV_TRACK, STOP, VOLUME_UP, VOLUME_DOWN, MUTE,
        FAST_FORWARD, REWIND, FULLSCREEN_TOGGLE, CAPTIONS_TOGGLE,
        SEEK_FORWARD_10S, SEEK_BACK_10S.
        """
        if not action or action == "NONE":
            return False

        action_upper = action.upper().strip()

        if action_upper == "PLAY_PAUSE":
            return self.play_pause()

        elif action_upper in ["NEXT_TRACK", "NEXT_FILE", "NEXT"]:
            return self.next_track()

        elif action_upper in ["PREV_TRACK", "PREV_FILE", "PREVIOUS"]:
            return self.prev_track()

        elif action_upper == "STOP":
            return self.stop()

        elif action_upper == "VOLUME_UP":
            return self.volume_up()

        elif action_upper == "VOLUME_DOWN":
            return self.volume_down()

        elif action_upper == "MUTE":
            return self.toggle_mute()

        elif action_upper in ["FAST_FORWARD", "SEEK_FORWARD_10S", "FORWARD"]:
            return self.seek_forward(10)

        elif action_upper in ["REWIND", "SEEK_BACK_10S", "BACK"]:
            return self.seek_backward(10)

        elif action_upper == "FULLSCREEN_TOGGLE":
            return self.toggle_fullscreen()

        elif action_upper == "CAPTIONS_TOGGLE":
            return self.toggle_captions()

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
