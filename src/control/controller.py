"""
Operating System Automation & Computer Controller Module
========================================================
Translates recognized gestures and FSM events into OS input commands
using PyAutoGUI (cursor, clicks, drag, scroll, volume, and media keys).
"""

import time
import logging
from pathlib import Path
from typing import Optional, Tuple
import pyautogui
import config

logger = logging.getLogger(__name__)

# Configure PyAutoGUI for ultra-low latency
pyautogui.PAUSE = 0.001
pyautogui.FAILSAFE = True


class ComputerController:
    """Dispatches synthetic user input actions to the operating system."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.is_mouse_down = False
        self.last_action_desc = "System Ready"

    def set_enabled(self, state: bool):
        """Enables or disables dispatching actual OS inputs."""
        self.enabled = state
        if not state and self.is_mouse_down:
            self.mouse_up()
        self.last_action_desc = "Control Active" if state else "Control Paused"
        logger.info("Computer control state changed to: %s", self.enabled)

    def move_cursor(self, x: int, y: int):
        """Moves cursor to screen coordinate."""
        if not self.enabled:
            return
        try:
            pyautogui.moveTo(x, y)
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI FailSafe triggered at screen corner.")

    def click(self, button: str = "left"):
        """Performs single mouse click."""
        if not self.enabled:
            return
        try:
            pyautogui.click(button=button)
            self.last_action_desc = f"{button.capitalize()} Click"
        except pyautogui.FailSafeException:
            pass

    def double_click(self):
        """Performs double click."""
        if not self.enabled:
            return
        try:
            pyautogui.doubleClick()
            self.last_action_desc = "Double Click"
        except pyautogui.FailSafeException:
            pass

    def mouse_down(self, button: str = "left"):
        """Holds mouse button down for dragging."""
        if not self.enabled or self.is_mouse_down:
            return
        try:
            pyautogui.mouseDown(button=button)
            self.is_mouse_down = True
            self.last_action_desc = "Drag Started"
        except pyautogui.FailSafeException:
            pass

    def mouse_up(self, button: str = "left"):
        """Releases held mouse button."""
        if not self.enabled or not self.is_mouse_down:
            return
        try:
            pyautogui.mouseUp(button=button)
            self.is_mouse_down = False
            self.last_action_desc = "Drag Released"
        except pyautogui.FailSafeException:
            pass

    def scroll(self, direction: str = "up", amount: int = config.SCROLL_SPEED_DELTA):
        """Scrolls mouse wheel up or down."""
        if not self.enabled:
            return
        try:
            delta = amount if direction == "up" else -amount
            pyautogui.scroll(delta)
            self.last_action_desc = f"Scroll {direction.capitalize()}"
        except pyautogui.FailSafeException:
            pass

    def volume_up(self):
        """Increases system volume."""
        if not self.enabled:
            return
        try:
            pyautogui.press("volumeup")
            self.last_action_desc = "Volume Up (+)"
        except Exception as e:
            logger.warning("Volume Up press failed: %s", str(e))

    def volume_down(self):
        """Decreases system volume."""
        if not self.enabled:
            return
        try:
            pyautogui.press("volumedown")
            self.last_action_desc = "Volume Down (-)"
        except Exception as e:
            logger.warning("Volume Down press failed: %s", str(e))

    def play_pause(self):
        """Toggles media play/pause."""
        if not self.enabled:
            return
        try:
            pyautogui.press("playpause")
            self.last_action_desc = "Play / Pause Media"
        except Exception as e:
            logger.warning("Play/Pause press failed: %s", str(e))

    def next_track(self):
        """Triggers next track or forward slide."""
        if not self.enabled:
            return
        try:
            pyautogui.press("nexttrack")
            # Also simulate Right Arrow for presentation slide decks
            pyautogui.press("right")
            self.last_action_desc = "Next Track / Slide"
        except Exception as e:
            logger.warning("Next Track press failed: %s", str(e))

    def prev_track(self):
        """Triggers previous track or backward slide."""
        if not self.enabled:
            return
        try:
            pyautogui.press("prevtrack")
            # Also simulate Left Arrow for presentation slide decks
            pyautogui.press("left")
            self.last_action_desc = "Prev Track / Slide"
        except Exception as e:
            logger.warning("Prev Track press failed: %s", str(e))

    def take_screenshot(self, output_dir: Optional[Path] = None) -> Optional[Path]:
        """Captures and saves a full screen screenshot."""
        if not self.enabled:
            return None
        try:
            save_folder = output_dir or (config.BASE_DIR / "screenshots")
            save_folder.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = save_folder / f"screenshot_{timestamp}.png"
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            self.last_action_desc = f"Screenshot Saved"
            logger.info("Screenshot saved to: %s", filepath)
            return filepath
        except Exception as e:
            logger.error("Screenshot capture failed: %s", str(e))
            return None

    def dispatch(
        self,
        gesture: str,
        event_type: str,
        screen_coords: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        Routes gesture and event from FSM to corresponding OS actions.
        Returns:
            Description string of the action taken.
        """
        if not self.enabled:
            return "Control Disabled"

        # 1. Cursor Navigation
        if event_type == "continuous" and screen_coords:
            self.move_cursor(screen_coords[0], screen_coords[1])
            self.last_action_desc = f"Cursor ({screen_coords[0]}, {screen_coords[1]})"
            return self.last_action_desc

        # 2. Drag Operations
        if event_type == "drag_start":
            self.mouse_down(button="left")
            return "Drag Started"
        elif event_type == "drag_hold" and screen_coords:
            self.move_cursor(screen_coords[0], screen_coords[1])
            return "Dragging Cursor"
        elif event_type == "drag_release":
            self.mouse_up(button="left")
            return "Drag Released"

        # 3. Clicks
        if event_type == "click":
            self.click("left")
            return "Left Click"
        elif event_type == "double_click":
            self.double_click()
            return "Double Click"
        elif event_type == "right_click":
            self.click("right")
            return "Right Click"

        # 4. Scrolling
        if event_type == "scroll":
            if gesture == config.GESTURE_TWO_FINGERS_UP:
                self.scroll(direction="up")
                return "Scroll Up"
            elif gesture == config.GESTURE_TWO_FINGERS_DOWN:
                self.scroll(direction="down")
                return "Scroll Down"

        # 5. Discrete Key Events
        if event_type == "discrete":
            if gesture == config.GESTURE_THUMB_UP:
                self.volume_up()
                return "Volume Up"
            elif gesture == config.GESTURE_THUMB_DOWN:
                self.volume_down()
                return "Volume Down"
            elif gesture == config.GESTURE_THREE_FINGERS:
                self.play_pause()
                return "Play / Pause"
            elif gesture == config.GESTURE_SWIPE_LEFT:
                self.prev_track()
                return "Swipe Left (Prev)"
            elif gesture == config.GESTURE_SWIPE_RIGHT:
                self.next_track()
                return "Swipe Right (Next)"
            elif gesture == config.GESTURE_SCREENSHOT:
                self.take_screenshot()
                return "Screenshot Taken"

        return self.last_action_desc
