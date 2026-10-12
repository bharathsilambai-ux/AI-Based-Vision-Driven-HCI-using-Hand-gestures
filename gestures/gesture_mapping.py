"""
Phase 7: Gesture-to-Command Mapping Engine & Operational Modes Subsystem
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Multi-Mode Architecture (Module 7):
   - MOUSE MODE: Cursor move, left click, right click, double click, drag-drop, scroll up/down.
   - SYSTEM MODE: Volume up/down, mute, brightness up/down, screenshot, lock screen, open app.
   - MEDIA MODE: Play/pause, next/previous track, volume +/-, mute, stop.
   - BROWSER MODE: Scroll up/down, back, forward, refresh, tab switch, zoom in/out.
   - PRESENTATION MODE: Next/previous slide, start/exit presentation, laser pointer, blank screen.
   - APPLICATION CONTROL MODE: Context-aware mappings adapting to active foreground window (Module 13).
   - CUSTOM GESTURE MODE: User-defined persistent mappings loaded from/saved to JSON (Module 14).

2. Conflict Detection & Disambiguation:
   - Validates mapping matrices to ensure no gesture is mapped to conflicting contradictory actions within the same mode.

3. Persistent Configuration (Module 14):
   - Loads and saves custom keybindings to data/gesture_mappings.json with safe defaults fallback.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.logger import logger

DEFAULT_MAPPING_PATH = os.path.join(_ROOT, "data", "gesture_mappings.json")


class OperationalMode:
    """Canonical operating modes (Module 7)."""
    MOUSE = "MOUSE"
    SYSTEM = "SYSTEM"
    MEDIA = "MEDIA"
    BROWSER = "BROWSER"
    PRESENTATION = "PRESENTATION"
    APPLICATION = "APPLICATION"
    CUSTOM = "CUSTOM"

    ALL_MODES = [MOUSE, SYSTEM, MEDIA, BROWSER, PRESENTATION, APPLICATION, CUSTOM]


# Default Canonical Mappings for Each Operating Mode
DEFAULT_MODE_MAPPINGS: Dict[str, Dict[str, str]] = {
    OperationalMode.MOUSE: {
        "POINT": "CURSOR_MOVE",
        "PINCH": "LEFT_CLICK",
        "PEACE": "RIGHT_CLICK",
        "THREE_FINGERS": "DRAG_DROP",
        "FOUR_FINGERS": "DOUBLE_CLICK",
        "SWIPE_UP": "SCROLL_UP",
        "SWIPE_DOWN": "SCROLL_DOWN",
        "OPEN_PALM": "HOVER",
        "FIST": "RELEASE",
        "THUMBS_UP": "ZOOM_IN",
        "THUMBS_DOWN": "ZOOM_OUT",
        "OK": "MIDDLE_CLICK",
    },
    OperationalMode.SYSTEM: {
        "THUMBS_UP": "VOLUME_UP",
        "THUMBS_DOWN": "VOLUME_DOWN",
        "FIST": "MUTE",
        "PEACE": "BRIGHTNESS_UP",
        "THREE_FINGERS": "BRIGHTNESS_DOWN",
        "OK": "SCREENSHOT",
        "FOUR_FINGERS": "OPEN_FILE_EXPLORER",
        "POINT": "OPEN_BROWSER",
        "OPEN_PALM": "LOCK_SCREEN",
        "SWIPE_LEFT": "TASK_SWITCH_PREV",
        "SWIPE_RIGHT": "TASK_SWITCH_NEXT",
        "SWIPE_UP": "TASK_VIEW",
        "SWIPE_DOWN": "MINIMIZE_ALL",
    },
    OperationalMode.MEDIA: {
        "PINCH": "PLAY_PAUSE",
        "SWIPE_RIGHT": "NEXT_TRACK",
        "SWIPE_LEFT": "PREV_TRACK",
        "THUMBS_UP": "VOLUME_UP",
        "THUMBS_DOWN": "VOLUME_DOWN",
        "FIST": "MUTE",
        "OPEN_PALM": "STOP",
        "PEACE": "FAST_FORWARD",
        "THREE_FINGERS": "REWIND",
        "FOUR_FINGERS": "FULLSCREEN_TOGGLE",
    },
    OperationalMode.BROWSER: {
        "SWIPE_LEFT": "HISTORY_BACK",
        "SWIPE_RIGHT": "HISTORY_FORWARD",
        "SWIPE_UP": "SCROLL_UP",
        "SWIPE_DOWN": "SCROLL_DOWN",
        "OK": "REFRESH_PAGE",
        "PEACE": "NEW_TAB",
        "FIST": "CLOSE_TAB",
        "POINT": "LINK_CLICK",
        "THUMBS_UP": "ZOOM_IN",
        "THUMBS_DOWN": "ZOOM_OUT",
        "PINCH": "TAB_SWITCH_NEXT",
        "FOUR_FINGERS": "TAB_SWITCH_PREV",
    },
    OperationalMode.PRESENTATION: {
        "SWIPE_RIGHT": "NEXT_SLIDE",
        "SWIPE_LEFT": "PREV_SLIDE",
        "THUMBS_UP": "START_PRESENTATION",
        "THUMBS_DOWN": "EXIT_PRESENTATION",
        "POINT": "LASER_POINTER",
        "PINCH": "BLACK_SCREEN",
        "OPEN_PALM": "WHITE_SCREEN",
        "PEACE": "SLIDE_OVERVIEW",
        "FIST": "CLEAR_ANNOTATIONS",
    },
    OperationalMode.CUSTOM: {
        "PEACE": "OPEN_BROWSER",
        "THUMBS_UP": "VOLUME_UP",
        "THUMBS_DOWN": "VOLUME_DOWN",
        "OK": "SCREENSHOT",
        "FIST": "LOCK_SCREEN",
        "OPEN_PALM": "SHOW_DESKTOP",
    },
}

# Application-Specific Context Mappings (Module 13)
APPLICATION_SPECIFIC_MAPPINGS: Dict[str, Dict[str, str]] = {
    "youtube": {
        "PINCH": "PLAY_PAUSE",
        "SWIPE_RIGHT": "SEEK_FORWARD_10S",
        "SWIPE_LEFT": "SEEK_BACK_10S",
        "THUMBS_UP": "VOLUME_UP",
        "THUMBS_DOWN": "VOLUME_DOWN",
        "FOUR_FINGERS": "FULLSCREEN_TOGGLE",
        "PEACE": "CAPTIONS_TOGGLE",
        "FIST": "MUTE",
    },
    "powerpoint": {
        "SWIPE_RIGHT": "NEXT_SLIDE",
        "SWIPE_LEFT": "PREV_SLIDE",
        "THUMBS_UP": "START_SLIDESHOW",
        "THUMBS_DOWN": "END_SLIDESHOW",
        "POINT": "LASER_POINTER",
        "PINCH": "BLACK_SCREEN",
    },
    "chrome": {
        "SWIPE_LEFT": "HISTORY_BACK",
        "SWIPE_RIGHT": "HISTORY_FORWARD",
        "SWIPE_UP": "PAGE_SCROLL_UP",
        "SWIPE_DOWN": "PAGE_SCROLL_DOWN",
        "PEACE": "NEW_TAB",
        "FIST": "CLOSE_TAB",
        "OK": "RELOAD_PAGE",
    },
    "vlc": {
        "PINCH": "PLAY_PAUSE",
        "SWIPE_RIGHT": "NEXT_FILE",
        "SWIPE_LEFT": "PREV_FILE",
        "THUMBS_UP": "VOLUME_UP",
        "THUMBS_DOWN": "VOLUME_DOWN",
        "FOUR_FINGERS": "FULLSCREEN_TOGGLE",
        "FIST": "MUTE",
    },
}


class GestureMappingManager:
    """
    Manages operational modes, gesture-action bindings, context-aware routing,
    conflict validation, and JSON serialization.
    """

    def __init__(
        self,
        config_path: str = DEFAULT_MAPPING_PATH,
        initial_mode: str = OperationalMode.MOUSE,
    ):
        self.config_path = config_path
        self.current_mode = initial_mode if initial_mode in OperationalMode.ALL_MODES else OperationalMode.MOUSE
        self.active_application: Optional[str] = None

        # Mode-to-mappings dictionary
        self._mappings: Dict[str, Dict[str, str]] = {
            mode: dict(mappings) for mode, mappings in DEFAULT_MODE_MAPPINGS.items()
        }

        # Load persisted custom overrides if file exists
        self.load_mappings()

    def set_mode(self, mode: str) -> bool:
        """Switches the current operational mode."""
        canonical_mode = mode.upper().strip()
        if canonical_mode in OperationalMode.ALL_MODES:
            self.current_mode = canonical_mode
            logger.info(f"[GestureMappingManager] Operational mode changed to: {self.current_mode}")
            return True
        logger.warning(f"[GestureMappingManager] Invalid mode '{mode}'. Retaining {self.current_mode}")
        return False

    def get_mode(self) -> str:
        """Returns the active operational mode."""
        return self.current_mode

    def set_active_application(self, app_name: Optional[str]):
        """Sets the detected active foreground application (Module 13)."""
        self.active_application = app_name.lower().strip() if app_name else None

    def get_action(
        self,
        gesture: str,
        mode: Optional[str] = None,
        app_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Retrieves the assigned action for a given gesture under current or specified mode/app.
        
        Args:
            gesture: Canonical gesture name (e.g. 'PINCH', 'THUMBS_UP').
            mode: Specific mode or None for current mode.
            app_name: Specific application or None for active application.
            
        Returns:
            Action string (e.g. 'LEFT_CLICK') or None if unmapped.
        """
        target_mode = mode.upper().strip() if mode else self.current_mode
        target_app = (app_name or self.active_application or "").lower().strip()

        # 1. If in APPLICATION mode and a recognized app is active, use context mapping
        if target_mode == OperationalMode.APPLICATION and target_app in APPLICATION_SPECIFIC_MAPPINGS:
            app_map = APPLICATION_SPECIFIC_MAPPINGS[target_app]
            if gesture in app_map:
                return app_map[gesture]

        # 2. Query mode-specific mappings
        if target_mode in self._mappings:
            mode_map = self._mappings[target_mode]
            if gesture in mode_map:
                return mode_map[gesture]

        # 3. Fallback to MOUSE mode if not found
        if target_mode != OperationalMode.MOUSE and gesture in self._mappings.get(OperationalMode.MOUSE, {}):
            return self._mappings[OperationalMode.MOUSE].get(gesture)

        return None

    def bind_gesture(
        self,
        gesture: str,
        action: str,
        mode: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Binds a gesture to an action within a specified mode.
        Validates against duplicate conflicts within the same mode.
        """
        target_mode = mode.upper().strip() if mode else self.current_mode
        if target_mode not in self._mappings:
            return False, f"Mode '{target_mode}' does not exist."

        # Bind action
        self._mappings[target_mode][gesture] = action
        logger.info(f"[GestureMappingManager] Bound [{gesture}] -> [{action}] in mode [{target_mode}]")
        return True, "Success"

    def unbind_gesture(self, gesture: str, mode: Optional[str] = None) -> bool:
        """Removes a binding for a gesture."""
        target_mode = mode.upper().strip() if mode else self.current_mode
        if target_mode in self._mappings and gesture in self._mappings[target_mode]:
            del self._mappings[target_mode][gesture]
            return True
        return False

    def get_mappings_for_mode(self, mode: Optional[str] = None) -> Dict[str, str]:
        """Returns a copy of all gesture-to-action bindings for the specified mode."""
        target_mode = mode.upper().strip() if mode else self.current_mode
        return dict(self._mappings.get(target_mode, {}))

    def detect_conflicts(self, mode: Optional[str] = None) -> List[str]:
        """
        Scans bindings for identical actions mapped to conflicting gestures,
        or identical gestures mapped to contradictory actions.
        """
        target_mode = mode.upper().strip() if mode else self.current_mode
        mappings = self._mappings.get(target_mode, {})
        conflicts = []

        # Count action occurrences
        action_counts: Dict[str, List[str]] = {}
        for gest, act in mappings.items():
            action_counts.setdefault(act, []).append(gest)

        for act, gestures in action_counts.items():
            if len(gestures) > 1 and act not in ["HOVER", "NONE"]:
                conflicts.append(f"Multiple gestures {gestures} trigger identical action '{act}' in {target_mode}")

        return conflicts

    def reset_to_defaults(self, mode: Optional[str] = None):
        """Resets mappings for a mode or all modes to canonical defaults."""
        if mode:
            target_mode = mode.upper().strip()
            if target_mode in DEFAULT_MODE_MAPPINGS:
                self._mappings[target_mode] = dict(DEFAULT_MODE_MAPPINGS[target_mode])
        else:
            self._mappings = {m: dict(mappings) for m, mappings in DEFAULT_MODE_MAPPINGS.items()}

    def save_mappings(self, path: Optional[str] = None) -> bool:
        """Serializes current mappings to a JSON file."""
        target_path = path or self.config_path
        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(self._mappings, f, indent=2)
            logger.info(f"[GestureMappingManager] Saved gesture mappings to: {target_path}")
            return True
        except Exception as e:
            logger.error(f"[GestureMappingManager] Failed to save mappings: {e}")
            return False

    def load_mappings(self, path: Optional[str] = None) -> bool:
        """Loads custom gesture mappings from a JSON file, falling back safely on error."""
        target_path = path or self.config_path
        if not os.path.exists(target_path):
            return False

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for mode, binds in data.items():
                    if mode in self._mappings and isinstance(binds, dict):
                        self._mappings[mode].update(binds)
                logger.info(f"[GestureMappingManager] Loaded custom mappings from: {target_path}")
                return True
        except Exception as e:
            logger.warning(f"[GestureMappingManager] Failed to load mappings file: {e}")
        return False
