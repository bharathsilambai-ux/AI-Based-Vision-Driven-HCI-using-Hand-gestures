"""
7-State Gesture State Machine.
Enforces the formal lifecycle:
IDLE -> DETECTED -> VERIFYING -> CONFIRMED -> EXECUTING -> COOLDOWN -> IDLE
Prevents accidental triggers, click-spamming, and enforces safety hold durations.
"""

import os
import sys
import time
from typing import Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


class StateMachineResult:
    """Output container of the state machine evaluation."""

    def __init__(
        self,
        state: str,
        gesture_name: str,
        can_execute: bool,
        status_text: str,
        status_color: str,
        progress_pct: float = 0.0
    ):
        self.state = state
        self.gesture_name = gesture_name
        self.can_execute = can_execute
        self.status_text = status_text
        self.status_color = status_color
        self.progress_pct = progress_pct


class GestureStateMachine:
    """
    Robust state machine that filters single-frame noise, verifies stability over time,
    dispatches discrete single actions, and enforces cool-down periods.
    """

    def __init__(self):
        self.state = config.GestureState.IDLE
        self.candidate_gesture = "NONE"
        self.confirmed_gesture = "NONE"
        self.active_action_name = "NONE"

        self.candidate_count = 0
        self.hold_start_time = 0.0
        self.cooldown_start_time = 0.0
        self.cooldown_duration = 0.35

        # Safety hold progress
        self.sensitive_target_duration = config.SENSITIVE_HOLD_DURATION

    def reset(self):
        """Resets the state machine back to IDLE."""
        self.state = config.GestureState.IDLE
        self.candidate_gesture = "NONE"
        self.confirmed_gesture = "NONE"
        self.candidate_count = 0
        self.hold_start_time = 0.0

    def process(
        self,
        raw_gesture: str,
        confidence: float,
        min_confidence: float = config.CONFIDENCE_THRESHOLD_DEFAULT,
        debounce_frames: int = config.GESTURE_DEBOUNCE_FRAMES,
        cooldown_sec: float = config.COOLDOWN_CLICK,
        is_continuous: bool = False,
        is_sensitive: bool = False
    ) -> StateMachineResult:
        """
        Processes instantaneous raw detection through the 7-state machine.
        """
        now = time.time()

        # -------------------------------------------------------------
        # 1. COOLDOWN STATE HANDLING
        # -------------------------------------------------------------
        if self.state == config.GestureState.COOLDOWN:
            elapsed = now - self.cooldown_start_time
            if elapsed < self.cooldown_duration:
                rem = self.cooldown_duration - elapsed
                return StateMachineResult(
                    state=config.GestureState.COOLDOWN,
                    gesture_name=self.confirmed_gesture,
                    can_execute=False,
                    status_text=f"✕ COOLDOWN ({rem:.1f}s)",
                    status_color=config.COLOR_TEXT_MUTED,
                    progress_pct=min(1.0, elapsed / self.cooldown_duration)
                )
            else:
                # Cooldown expired -> return to IDLE
                self.state = config.GestureState.IDLE

        # -------------------------------------------------------------
        # 2. NO HAND OR IDLE GESTURE
        # -------------------------------------------------------------
        if raw_gesture in ("NONE", "IDLE"):
            self.reset()
            return StateMachineResult(
                state=config.GestureState.IDLE,
                gesture_name="IDLE",
                can_execute=False,
                status_text="● SEARCHING HAND",
                status_color=config.COLOR_TEXT_MUTED
            )

        # -------------------------------------------------------------
        # 3. LOW CONFIDENCE CHECK
        # -------------------------------------------------------------
        if confidence < min_confidence:
            return StateMachineResult(
                state=config.GestureState.IDLE,
                gesture_name=raw_gesture,
                can_execute=False,
                status_text="⚠ LOW CONFIDENCE – ACTION BLOCKED",
                status_color=config.COLOR_WARNING
            )

        # -------------------------------------------------------------
        # 4. STATE: EXECUTING (Continuous Gesture Ongoing)
        # -------------------------------------------------------------
        if self.state == config.GestureState.EXECUTING:
            if is_continuous and raw_gesture == self.confirmed_gesture:
                return StateMachineResult(
                    state=config.GestureState.EXECUTING,
                    gesture_name=self.confirmed_gesture,
                    can_execute=True,
                    status_text=f"✓ EXECUTING: {self.confirmed_gesture.replace('_', ' ')}",
                    status_color=config.COLOR_SUCCESS,
                    progress_pct=1.0
                )
            elif not is_continuous:
                # Discrete was already dispatched, move to cooldown or idle
                self.state = config.GestureState.COOLDOWN
                self.cooldown_start_time = now
                self.cooldown_duration = cooldown_sec

        # -------------------------------------------------------------
        # 5. STATE: IDLE -> DETECTED
        # -------------------------------------------------------------
        if self.state == config.GestureState.IDLE:
            self.candidate_gesture = raw_gesture
            self.candidate_count = 1
            self.hold_start_time = now

            target_frames = 1 if (debounce_frames <= 1 or raw_gesture == "OPEN_PALM") else debounce_frames
            if target_frames <= 1:
                self.state = config.GestureState.CONFIRMED
            else:
                self.state = config.GestureState.DETECTED
                return StateMachineResult(
                    state=config.GestureState.DETECTED,
                    gesture_name=raw_gesture,
                    can_execute=False,
                    status_text=f"● DETECTED: {raw_gesture.replace('_', ' ')}",
                    status_color=config.COLOR_CYAN
                )

        # -------------------------------------------------------------
        # 6. STATE: DETECTED / VERIFYING
        # -------------------------------------------------------------
        if self.state in (config.GestureState.DETECTED, config.GestureState.VERIFYING):
            if raw_gesture == self.candidate_gesture:
                self.candidate_count += 1
                self.state = config.GestureState.VERIFYING
                hold_time = now - self.hold_start_time
                target_frames = 1 if (debounce_frames <= 1 or raw_gesture == "OPEN_PALM") else debounce_frames

                # Check sensitive action hold requirement (e.g. 1.2 seconds for Close Window)
                if is_sensitive:
                    progress = min(1.0, hold_time / self.sensitive_target_duration)
                    if hold_time >= self.sensitive_target_duration:
                        self.state = config.GestureState.CONFIRMED
                    else:
                        return StateMachineResult(
                            state=config.GestureState.VERIFYING,
                            gesture_name=raw_gesture,
                            can_execute=False,
                            status_text=f"⏳ HOLD TO CONFIRM: {int(progress * 100)}%",
                            status_color=config.COLOR_WARNING,
                            progress_pct=progress
                        )
                elif self.candidate_count >= target_frames:
                    self.state = config.GestureState.CONFIRMED
                else:
                    return StateMachineResult(
                        state=config.GestureState.VERIFYING,
                        gesture_name=raw_gesture,
                        can_execute=False,
                        status_text=f"⏳ VERIFYING ({self.candidate_count}/{target_frames})",
                        status_color=config.COLOR_CYAN,
                        progress_pct=self.candidate_count / target_frames
                    )
            else:
                # Gesture changed before verification -> reset
                self.candidate_gesture = raw_gesture
                self.candidate_count = 1
                self.hold_start_time = now
                self.state = config.GestureState.DETECTED
                return StateMachineResult(
                    state=config.GestureState.DETECTED,
                    gesture_name=raw_gesture,
                    can_execute=False,
                    status_text=f"● DETECTED: {raw_gesture.replace('_', ' ')}",
                    status_color=config.COLOR_CYAN
                )

        # -------------------------------------------------------------
        # 7. STATE: CONFIRMED -> EXECUTING
        # -------------------------------------------------------------
        if self.state == config.GestureState.CONFIRMED:
            self.confirmed_gesture = self.candidate_gesture
            self.state = config.GestureState.EXECUTING

            # For continuous motions (like cursor move or scrolling), don't enter cooldown
            if is_continuous:
                return StateMachineResult(
                    state=config.GestureState.EXECUTING,
                    gesture_name=self.confirmed_gesture,
                    can_execute=True,
                    status_text=f"✓ EXECUTING: {self.confirmed_gesture.replace('_', ' ')}",
                    status_color=config.COLOR_SUCCESS,
                    progress_pct=1.0
                )
            else:
                # Discrete trigger (Single click, shortcut, screenshot)
                self.state = config.GestureState.COOLDOWN
                self.cooldown_start_time = now
                self.cooldown_duration = cooldown_sec

                return StateMachineResult(
                    state=config.GestureState.EXECUTING,
                    gesture_name=self.confirmed_gesture,
                    can_execute=True,
                    status_text=f"✓ EXECUTED: {self.confirmed_gesture.replace('_', ' ')}",
                    status_color=config.COLOR_SUCCESS,
                    progress_pct=1.0
                )

        # Fallback
        return StateMachineResult(
            state=self.state,
            gesture_name=self.confirmed_gesture,
            can_execute=False,
            status_text="● MONITORING",
            status_color=config.COLOR_TEXT_MUTED
        )
