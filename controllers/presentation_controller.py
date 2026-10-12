"""
Phase 13: Presentation & Application-Specific Context Control Subsystem
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Complete Presentation Operations (Module 12):
   - next_slide(), prev_slide(): Slide advancement and backward navigation.
   - start_presentation(), exit_presentation(): Slide show presentation mode lifecycle (F5, Shift+F5, Esc).
   - toggle_laser_pointer(): Virtual laser pointer / attention highlight mode.
   - toggle_black_screen(), toggle_white_screen(): Presentation screen blanking ('B', 'W').
   - slide_overview(), clear_annotations(): Slide navigation grid and ink annotation cleanup.
   - first_slide(), last_slide(): Boundary slide jumping.

2. Application-Specific Context Routing (Module 13):
   - Supports Microsoft PowerPoint, Google Slides, Apple Keynote, and Acrobat/PDF Slide Deck profiles.
   - Provides active context introspection (active application, current mode, recognized gesture, executed command).

3. Headless Mock & Telemetry Verification Architecture:
   - Built-in mock_mode with full slide counter tracking (current_slide_index, is_presenting, is_laser_active)
     allowing unit testing without launching office suites.
   - Comprehensive event logging (event_log).
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


class PresentationTarget:
    """Target presentation software profiles."""
    POWERPOINT = "powerpoint"
    GOOGLE_SLIDES = "slides"
    KEYNOTE = "keynote"
    PDF = "pdf"
    GENERIC = "generic"

    ALL_TARGETS = [POWERPOINT, GOOGLE_SLIDES, KEYNOTE, PDF, GENERIC]


class PresentationController:
    """
    Subsystem for controlling PowerPoint and equivalent presentation software (Module 12)
    with application-specific context awareness (Module 13).
    """

    def __init__(
        self,
        target_app: str = PresentationTarget.POWERPOINT,
        mock_mode: bool = False,
    ):
        self.target_app = (
            target_app.lower().strip()
            if target_app.lower().strip() in PresentationTarget.ALL_TARGETS
            else PresentationTarget.POWERPOINT
        )
        self.mock_mode = mock_mode

        # Simulated Presentation State
        self.is_presenting: bool = False
        self.current_slide_index: int = 1
        self.total_slides: int = 50
        self.laser_pointer_active: bool = False
        self.blank_screen_mode: str = "NONE"  # 'NONE', 'BLACK', 'WHITE'

        self.last_action: str = "NONE"
        self.last_action_time: float = 0.0
        self.event_log: List[Dict[str, Any]] = []

    def set_target(self, target_app: str) -> bool:
        """Sets the active presentation target profile."""
        canonical = target_app.lower().strip()
        if canonical in PresentationTarget.ALL_TARGETS:
            self.target_app = canonical
            logger.info(f"[PresentationController] Active target profile set to: '{self.target_app}'")
            return True
        self.target_app = PresentationTarget.GENERIC
        return False

    # =========================================================================
    # Slide Navigation (Module 12)
    # =========================================================================

    def next_slide(self) -> bool:
        """Advances to the next slide."""
        self.current_slide_index = min(self.total_slides, self.current_slide_index + 1)
        self.last_action = "NEXT_SLIDE"
        self._log_event("NEXT_SLIDE", {"slide": self.current_slide_index, "target": self.target_app})

        if self.mock_mode:
            return True

        try:
            if self.target_app == PresentationTarget.POWERPOINT:
                pyautogui.press("right")
            elif self.target_app == PresentationTarget.GOOGLE_SLIDES:
                pyautogui.press("right")
            else:
                pyautogui.press("pagedown")
            return True
        except Exception as e:
            logger.error(f"[PresentationController] Next slide failed: {e}")
            return False

    def prev_slide(self) -> bool:
        """Navigates back to the previous slide."""
        self.current_slide_index = max(1, self.current_slide_index - 1)
        self.last_action = "PREV_SLIDE"
        self._log_event("PREV_SLIDE", {"slide": self.current_slide_index, "target": self.target_app})

        if self.mock_mode:
            return True

        try:
            if self.target_app == PresentationTarget.POWERPOINT:
                pyautogui.press("left")
            elif self.target_app == PresentationTarget.GOOGLE_SLIDES:
                pyautogui.press("left")
            else:
                pyautogui.press("pageup")
            return True
        except Exception as e:
            logger.error(f"[PresentationController] Previous slide failed: {e}")
            return False

    def first_slide(self) -> bool:
        """Jumps directly to the first slide."""
        self.current_slide_index = 1
        self.last_action = "FIRST_SLIDE"
        self._log_event("FIRST_SLIDE", {"slide": 1})

        if self.mock_mode:
            return True

        try:
            pyautogui.press("home")
            return True
        except Exception as e:
            logger.error(f"[PresentationController] Jump to first slide failed: {e}")
            return False

    def last_slide(self) -> bool:
        """Jumps directly to the last slide."""
        self.current_slide_index = self.total_slides
        self.last_action = "LAST_SLIDE"
        self._log_event("LAST_SLIDE", {"slide": self.total_slides})

        if self.mock_mode:
            return True

        try:
            pyautogui.press("end")
            return True
        except Exception as e:
            logger.error(f"[PresentationController] Jump to last slide failed: {e}")
            return False

    # =========================================================================
    # Slideshow Lifecycle Operations (Module 12)
    # =========================================================================

    def start_presentation(self, from_beginning: bool = True) -> bool:
        """Starts full-screen slideshow (F5 or Shift+F5)."""
        self.is_presenting = True
        if from_beginning:
            self.current_slide_index = 1
        self.last_action = "START_PRESENTATION"
        self._log_event("START_PRESENTATION", {"from_beginning": from_beginning})

        if self.mock_mode:
            return True

        try:
            if from_beginning:
                pyautogui.press("f5")
            else:
                pyautogui.hotkey("shift", "f5")
            return True
        except Exception as e:
            logger.error(f"[PresentationController] Start presentation failed: {e}")
            return False

    def exit_presentation(self) -> bool:
        """Terminates active slideshow presentation (Escape)."""
        self.is_presenting = False
        self.laser_pointer_active = False
        self.blank_screen_mode = "NONE"
        self.last_action = "EXIT_PRESENTATION"
        self._log_event("EXIT_PRESENTATION", {})

        if self.mock_mode:
            return True

        try:
            pyautogui.press("escape")
            return True
        except Exception as e:
            logger.error(f"[PresentationController] Exit presentation failed: {e}")
            return False

    # =========================================================================
    # Presenter Tools & Focus Modes (Module 12)
    # =========================================================================

    def toggle_laser_pointer(self) -> bool:
        """Toggles presentation laser pointer / attention highlight mode."""
        self.laser_pointer_active = not self.laser_pointer_active
        self.last_action = "LASER_POINTER"
        self._log_event("LASER_POINTER", {"active": self.laser_pointer_active})

        if self.mock_mode:
            return True

        try:
            if self.target_app == PresentationTarget.POWERPOINT:
                pyautogui.hotkey("ctrl", "l")
            elif self.target_app == PresentationTarget.GOOGLE_SLIDES:
                pyautogui.press("l")
            return True
        except Exception as e:
            logger.error(f"[PresentationController] Laser pointer toggle failed: {e}")
            return False

    def toggle_black_screen(self) -> bool:
        """Blanks presentation screen to black ('B')."""
        self.blank_screen_mode = "NONE" if self.blank_screen_mode == "BLACK" else "BLACK"
        self.last_action = "BLACK_SCREEN"
        self._log_event("BLACK_SCREEN", {"mode": self.blank_screen_mode})

        if self.mock_mode:
            return True

        try:
            pyautogui.press("b")
            return True
        except Exception as e:
            logger.error(f"[PresentationController] Black screen failed: {e}")
            return False

    def toggle_white_screen(self) -> bool:
        """Blanks presentation screen to white ('W')."""
        self.blank_screen_mode = "NONE" if self.blank_screen_mode == "WHITE" else "WHITE"
        self.last_action = "WHITE_SCREEN"
        self._log_event("WHITE_SCREEN", {"mode": self.blank_screen_mode})

        if self.mock_mode:
            return True

        try:
            pyautogui.press("w")
            return True
        except Exception as e:
            logger.error(f"[PresentationController] White screen failed: {e}")
            return False

    def slide_overview(self) -> bool:
        """Opens slide thumbnail grid / presentation overview."""
        self.last_action = "SLIDE_OVERVIEW"
        self._log_event("SLIDE_OVERVIEW", {})

        if self.mock_mode:
            return True

        try:
            if self.target_app == PresentationTarget.POWERPOINT:
                pyautogui.press("g")
            else:
                pyautogui.hotkey("ctrl", "s")
            return True
        except Exception as e:
            logger.error(f"[PresentationController] Slide overview failed: {e}")
            return False

    def clear_annotations(self) -> bool:
        """Clears all ink markings and pen annotations on the current slide ('E')."""
        self.last_action = "CLEAR_ANNOTATIONS"
        self._log_event("CLEAR_ANNOTATIONS", {})

        if self.mock_mode:
            return True

        try:
            pyautogui.press("e")
            return True
        except Exception as e:
            logger.error(f"[PresentationController] Clear annotations failed: {e}")
            return False

    def toggle_pen(self) -> bool:
        """Toggles presentation pen pointer tool (Ctrl+P in PowerPoint)."""
        self.last_action = "TOGGLE_PEN"
        self._log_event("TOGGLE_PEN", {})

        if self.mock_mode:
            return True

        try:
            pyautogui.hotkey("ctrl", "p")
            return True
        except Exception as e:
            logger.error(f"[PresentationController] Toggle pen failed: {e}")
            return False

    def goto_slide(self, slide_num: int) -> bool:
        """Jumps directly to a specific slide number."""
        target = max(1, min(self.total_slides, int(slide_num)))
        self.current_slide_index = target
        self.last_action = f"GOTO_SLIDE_{target}"
        self._log_event("GOTO_SLIDE", {"slide": target})

        if self.mock_mode:
            return True

        try:
            pyautogui.write(str(target))
            pyautogui.press("enter")
            return True
        except Exception as e:
            logger.error(f"[PresentationController] Goto slide failed: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Returns the current state and telemetry of the presentation controller."""
        return {
            "target_app": self.target_app,
            "mock_mode": self.mock_mode,
            "is_presenting": self.is_presenting,
            "current_slide_index": self.current_slide_index,
            "total_slides": self.total_slides,
            "laser_pointer_active": self.laser_pointer_active,
            "blank_screen_mode": self.blank_screen_mode,
            "last_action": self.last_action,
            "event_count": len(self.event_log),
        }

    # =========================================================================
    # Unified Action Router
    # =========================================================================

    def execute_action(self, action: Optional[str], **kwargs) -> bool:
        """
        Executes presentation commands mapped from real-time gesture recognition in PRESENTATION mode.

        Supported actions:
        NEXT_SLIDE, PREV_SLIDE, START_PRESENTATION, EXIT_PRESENTATION,
        START_SLIDESHOW, END_SLIDESHOW, LASER_POINTER, BLACK_SCREEN,
        WHITE_SCREEN, SLIDE_OVERVIEW, CLEAR_ANNOTATIONS, FIRST_SLIDE, LAST_SLIDE,
        TOGGLE_PEN, GOTO_SLIDE.
        """
        if not action or action == "NONE":
            return False

        action_upper = action.upper().strip()

        if action_upper in ["NEXT_SLIDE", "NEXT"]:
            return self.next_slide()

        elif action_upper in ["PREV_SLIDE", "PREVIOUS_SLIDE", "PREV"]:
            return self.prev_slide()

        elif action_upper in ["START_PRESENTATION", "START_SLIDESHOW"]:
            return self.start_presentation(kwargs.get("from_beginning", True))

        elif action_upper in ["EXIT_PRESENTATION", "END_SLIDESHOW"]:
            return self.exit_presentation()

        elif action_upper == "LASER_POINTER":
            return self.toggle_laser_pointer()

        elif action_upper == "BLACK_SCREEN":
            return self.toggle_black_screen()

        elif action_upper == "WHITE_SCREEN":
            return self.toggle_white_screen()

        elif action_upper == "SLIDE_OVERVIEW":
            return self.slide_overview()

        elif action_upper == "CLEAR_ANNOTATIONS":
            return self.clear_annotations()

        elif action_upper in ["TOGGLE_PEN", "PEN"]:
            return self.toggle_pen()

        elif action_upper == "GOTO_SLIDE":
            slide_num = kwargs.get("slide_num", 1)
            return self.goto_slide(slide_num)

        elif action_upper == "FIRST_SLIDE":
            return self.first_slide()

        elif action_upper == "LAST_SLIDE":
            return self.last_slide()

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
