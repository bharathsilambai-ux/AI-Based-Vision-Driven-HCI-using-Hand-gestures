"""
Gesture State & Temporal Debouncing Finite State Machine
========================================================
Maintains temporal stability, debounces noisy transitions, enforces
cooldown periods, and tracks multi-phase gestures (e.g. click, drag, double-click).
"""

import time
from typing import Tuple, Optional
import config


class GestureStateManager:
    """Finite state machine ensuring smooth, jitter-free gesture actions."""

    def __init__(
        self,
        stability_frames: int = config.STABILITY_CONSECUTIVE_FRAMES,
        discrete_cooldown: float = config.COOLDOWN_DISCRETE_SEC,
        scroll_cooldown: float = config.COOLDOWN_SCROLL_SEC,
        double_pinch_window: float = config.DOUBLE_PINCH_WINDOW_SEC,
    ):
        self.stability_frames = stability_frames
        self.discrete_cooldown = discrete_cooldown
        self.scroll_cooldown = scroll_cooldown
        self.double_pinch_window = double_pinch_window

        # State tracking
        self.candidate_gesture: str = config.GESTURE_IDLE
        self.candidate_count: int = 0
        self.active_gesture: str = config.GESTURE_IDLE

        # Action Cooldown Timestamps
        self.last_action_times = {
            config.GESTURE_THUMB_UP: 0.0,
            config.GESTURE_THUMB_DOWN: 0.0,
            config.GESTURE_THREE_FINGERS: 0.0,
            config.GESTURE_SCREENSHOT: 0.0,
            config.GESTURE_SWIPE_LEFT: 0.0,
            config.GESTURE_SWIPE_RIGHT: 0.0,
            config.GESTURE_MIDDLE_PINCH: 0.0,
            config.GESTURE_TWO_FINGERS_UP: 0.0,
            config.GESTURE_TWO_FINGERS_DOWN: 0.0,
        }

        # Dragging & Pinch State
        self.is_dragging = False
        self.last_pinch_release_time = 0.0
        self.last_pinch_start_time = 0.0
        self.pinch_count_window = 0

    def process(self, raw_gesture: str, raw_confidence: float) -> Tuple[str, bool, str]:
        """
        Filters raw instantaneous gesture through the FSM.
        
        Returns:
            (stable_gesture, should_trigger_action, action_event_type)
            action_event_type can be:
            - 'continuous': (e.g. cursor movement)
            - 'discrete': (e.g. volume, screenshot, track)
            - 'scroll': (scroll up/down)
            - 'drag_start': (mouse down)
            - 'drag_hold': (mouse held while moving)
            - 'drag_release': (mouse up)
            - 'click': (single left click)
            - 'double_click': (double left click)
            - 'right_click': (right click)
            - 'none': no event triggered
        """
        now = time.time()

        # Update candidate stability counter
        if raw_gesture == self.candidate_gesture:
            self.candidate_count += 1
        else:
            self.candidate_gesture = raw_gesture
            self.candidate_count = 1

        # Check if candidate achieves required stability
        if self.candidate_count >= self.stability_frames:
            self.active_gesture = self.candidate_gesture
        elif raw_gesture == config.GESTURE_PINCH:
            # Pinch gets fast-track stability (only 2 frames) for immediate click responsiveness
            if self.candidate_count >= 2:
                self.active_gesture = config.GESTURE_PINCH

        # 1. Handle Continuous Cursor Movement
        if self.active_gesture == config.GESTURE_OPEN_PALM:
            if self.is_dragging:
                self.is_dragging = False
                return self.active_gesture, True, "drag_release"
            return self.active_gesture, True, "continuous"

        # 2. Handle Pinch (Left Click, Drag, Double Click)
        if self.active_gesture == config.GESTURE_PINCH:
            if not self.is_dragging:
                self.is_dragging = True
                self.last_pinch_start_time = now

                # Check if this pinch occurred within double pinch window
                if (now - self.last_pinch_release_time) <= self.double_pinch_window:
                    self.pinch_count_window = 0
                    return config.GESTURE_DOUBLE_PINCH, True, "double_click"
                else:
                    self.pinch_count_window = 1
                    # Check if hold duration triggers drag or if it's an initial click
                    return self.active_gesture, True, "drag_start"
            else:
                # Still holding pinch
                return self.active_gesture, True, "drag_hold"

        # If we were dragging but active gesture is no longer pinch
        if self.is_dragging and self.active_gesture != config.GESTURE_PINCH:
            self.is_dragging = False
            self.last_pinch_release_time = now
            # If the pinch was short (< 0.25s), it was a clean click
            pinch_duration = now - self.last_pinch_start_time
            if pinch_duration < 0.28:
                return config.GESTURE_PINCH, True, "click"
            return self.active_gesture, True, "drag_release"

        # 3. Handle Right Click (Middle Pinch)
        if self.active_gesture == config.GESTURE_MIDDLE_PINCH:
            last_t = self.last_action_times.get(config.GESTURE_MIDDLE_PINCH, 0.0)
            if (now - last_t) >= self.discrete_cooldown:
                self.last_action_times[config.GESTURE_MIDDLE_PINCH] = now
                return self.active_gesture, True, "right_click"
            return self.active_gesture, False, "none"

        # 4. Handle Continuous Scrolling (Two Fingers Up / Down)
        if self.active_gesture in (config.GESTURE_TWO_FINGERS_UP, config.GESTURE_TWO_FINGERS_DOWN):
            last_t = self.last_action_times.get(self.active_gesture, 0.0)
            if (now - last_t) >= self.scroll_cooldown:
                self.last_action_times[self.active_gesture] = now
                return self.active_gesture, True, "scroll"
            return self.active_gesture, False, "none"

        # 5. Handle Discrete Debounced Actions (Volume, Play/Pause, Screenshot, Swipes)
        if self.active_gesture in self.last_action_times:
            last_t = self.last_action_times[self.active_gesture]
            if (now - last_t) >= self.discrete_cooldown:
                self.last_action_times[self.active_gesture] = now
                return self.active_gesture, True, "discrete"
            return self.active_gesture, False, "none"

        # 6. Idle or Closed Fist
        return self.active_gesture, False, "none"

    def reset(self):
        """Resets all states."""
        self.candidate_gesture = config.GESTURE_IDLE
        self.candidate_count = 0
        self.active_gesture = config.GESTURE_IDLE
        self.is_dragging = False
