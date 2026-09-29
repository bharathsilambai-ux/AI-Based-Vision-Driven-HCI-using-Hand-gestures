"""
Hybrid Gesture Classifier and State Machine Coordinator.
Combines deterministic geometric rules, Scikit-Learn ML models, the 7-state lifecycle,
confidence gating, and mode-aware action routing.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from utils.logger import logger
from vision.feature_extractor import DualHandFeatures, HandFeatures
from vision.hand_tracker import HandData

try:
    from .custom_gesture import CustomGestureManager
    from .gesture_rules import GestureID, GestureRules
    from .state_machine import GestureStateMachine, StateMachineResult
except ImportError:
    from custom_gesture import CustomGestureManager
    from gesture_rules import GestureID, GestureRules
    from state_machine import GestureStateMachine, StateMachineResult


class GestureResult:
    """Complete structured output of the gesture recognition pipeline for a frame."""

    def __init__(
        self,
        gesture_id: str,
        display_label: str,
        confidence: float,
        active_mode: str,
        assigned_action: Optional[str] = None,
        state: str = config.GestureState.IDLE,
        status_text: str = "● IDLE",
        status_color: str = config.COLOR_TEXT_MUTED,
        can_execute: bool = False,
        is_discrete_trigger: bool = False,
        is_double_pinch: bool = False,
        smoothed_point: Optional[Tuple[int, int]] = None,
        delta_x: float = 0.0,
        delta_y: float = 0.0,
        hand_count: int = 0,
        primary_handedness: str = "None",
        progress_pct: float = 0.0
    ):
        self.gesture_id = gesture_id
        self.display_label = display_label
        self.confidence = confidence
        self.active_mode = active_mode
        self.assigned_action = assigned_action
        self.state = state
        self.status_text = status_text
        self.status_color = status_color
        self.can_execute = can_execute
        self.is_discrete_trigger = is_discrete_trigger
        self.is_double_pinch = is_double_pinch
        self.smoothed_point = smoothed_point
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.hand_count = hand_count
        self.primary_handedness = primary_handedness
        self.progress_pct = progress_pct


class GestureClassifier:
    """Central recognizer coordinating geometric rules, custom ML models, and operational modes."""

    def __init__(
        self,
        initial_mode: str = config.DEFAULT_MODE,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD_DEFAULT
    ):
        self.active_mode = initial_mode
        self.confidence_threshold = confidence_threshold

        self.rules_engine = GestureRules()
        self.custom_manager = CustomGestureManager()
        self.state_machine = GestureStateMachine()

        # Coordinate smoothing EMA state
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None
        self.smoothing_alpha = config.SMOOTHING_ALPHA_DEFAULT

        # Double pinch detection state
        self.last_pinch_time = 0.0

        # Load mode action mappings
        self.mode_mappings = self._load_mappings()

    def _load_mappings(self) -> Dict[str, Dict[str, str]]:
        prof = self.custom_manager.config_data.get("profiles", {}).get("default", {})
        return prof.get("modes", {})

    def set_mode(self, mode: str):
        if mode in config.ALL_MODES:
            self.active_mode = mode
            self.state_machine.reset()
            logger.info(f"[Mode] Switched active mode to: {mode}")

    def reload_mappings(self):
        self.custom_manager._load_config()
        self.mode_mappings = self._load_mappings()

    def process(
        self,
        hands: List[HandData],
        features_list: List[HandFeatures],
        dual_features: Optional[DualHandFeatures]
    ) -> GestureResult:
        """
        Processes frame through the complete recognition and state machine pipeline.
        """
        # Case 0: No hand detected
        if not hands or not features_list:
            self.state_machine.reset()
            self.smooth_x = None
            self.smooth_y = None
            return GestureResult(
                gesture_id=GestureID.NONE,
                display_label="No Hand Detected",
                confidence=0.0,
                active_mode=self.active_mode,
                state=config.GestureState.IDLE,
                status_text="● SEARCHING FOR HANDS",
                status_color=config.COLOR_TEXT_MUTED,
                hand_count=0
            )

        hand_count = len(hands)

        # -------------------------------------------------------------
        # 1. DUAL-HAND GESTURE EVALUATION (Priority if 2 hands present)
        # -------------------------------------------------------------
        if hand_count >= 2 and dual_features is not None:
            dual_id, dual_conf = self.rules_engine.classify_dual_hands(dual_features)
            if dual_id != GestureID.NONE:
                sm_res: StateMachineResult = self.state_machine.process(
                    raw_gesture=dual_id,
                    confidence=dual_conf,
                    min_confidence=self.confidence_threshold,
                    debounce_frames=config.GESTURE_DEBOUNCE_FRAMES,
                    cooldown_sec=config.COOLDOWN_SHORTCUT
                )
                action = self._resolve_action(dual_id)
                label = f"2-Hand: {dual_id.replace('_', ' ')}"
                return GestureResult(
                    gesture_id=dual_id,
                    display_label=label,
                    confidence=dual_conf,
                    active_mode=self.active_mode,
                    assigned_action=action,
                    state=sm_res.state,
                    status_text=sm_res.status_text,
                    status_color=sm_res.status_color,
                    can_execute=sm_res.can_execute,
                    is_discrete_trigger=sm_res.can_execute,
                    hand_count=hand_count,
                    primary_handedness="Dual",
                    progress_pct=sm_res.progress_pct
                )

        # -------------------------------------------------------------
        # 2. SINGLE HAND GESTURE EVALUATION
        # -------------------------------------------------------------
        primary_feat = next((f for f in features_list if f.handedness == "Right"), features_list[0])
        primary_hand = primary_feat.hand

        raw_id = GestureID.IDLE
        conf = 0.50

        # In Custom Mode, check Custom ML Classifier first
        if self.active_mode == config.AppMode.CUSTOM:
            ml_vector = primary_feat.to_ml_vector()
            pred_name, pred_conf = self.custom_manager.predict(ml_vector)
            if pred_name is not None and pred_conf >= self.confidence_threshold:
                raw_id = pred_name
                conf = pred_conf

        # Standard & Universal modes: Geometric Rules Engine first for precision & zero-latency
        if raw_id == GestureID.IDLE:
            raw_id, conf = self.rules_engine.classify_single_hand(primary_feat)

        # In Universal Mode, fallback to Custom ML Classifier if no standard gesture matched
        if raw_id == GestureID.IDLE and self.active_mode == config.AppMode.UNIVERSAL:
            ml_vector = primary_feat.to_ml_vector()
            pred_name, pred_conf = self.custom_manager.predict(ml_vector)
            if pred_name is not None and pred_conf >= max(0.80, self.confidence_threshold):
                raw_id = pred_name
                conf = pred_conf

        # -------------------------------------------------------------
        # 3. COORDINATE EMA SMOOTHING & VELOCITY DELTAS
        # -------------------------------------------------------------
        idx_px = primary_hand.get_px(8)
        if self.smooth_x is None or self.smooth_y is None:
            self.smooth_x = float(idx_px[0])
            self.smooth_y = float(idx_px[1])
            delta_x, delta_y = 0.0, 0.0
        else:
            prev_x, prev_y = self.smooth_x, self.smooth_y
            alpha = self.smoothing_alpha
            self.smooth_x = alpha * float(idx_px[0]) + (1.0 - alpha) * self.smooth_x
            self.smooth_y = alpha * float(idx_px[1]) + (1.0 - alpha) * self.smooth_y
            delta_x = self.smooth_x - prev_x
            delta_y = self.smooth_y - prev_y

        smoothed_pt = (int(self.smooth_x), int(self.smooth_y))

        # Double pinch detection:
        now = time.time()
        if raw_id == GestureID.PINCH_INDEX:
            if 0.06 < (now - self.last_pinch_time) <= config.DOUBLE_PINCH_WINDOW:
                raw_id = GestureID.DOUBLE_PINCH_INDEX
            self.last_pinch_time = now

        # Determine tentative action for sensitivity check
        tentative_action = self._resolve_action(raw_id)
        is_sensitive = tentative_action in ("WINDOW_CLOSE_SAFE", "LOCK_SCREEN_SAFE") or raw_id in (
            GestureID.FIVE_FINGERS_HOLD, GestureID.ROCK_SIGN_HOLD
        )

        # Check if continuous action (cursor move, continuous scroll, drag, volume, brightness)
        is_continuous = raw_id in (
            GestureID.INDEX_POINTING,
            GestureID.TWO_FINGERS_UP,
            GestureID.TWO_FINGERS_DOWN,
            GestureID.CLOSED_FIST,
            GestureID.THUMBS_UP,
            GestureID.THUMBS_DOWN,
            GestureID.HAND_MOVE_UP,
            GestureID.HAND_MOVE_DOWN
        )

        # -------------------------------------------------------------
        # 4. PASS THROUGH 7-STATE GESTURE STATE MACHINE
        # -------------------------------------------------------------
        cooldown_val = config.COOLDOWN_CLICK if "PINCH" in raw_id else config.COOLDOWN_SHORTCUT
        debounce_val = 1 if ("PINCH" in raw_id or raw_id == "OPEN_PALM") else config.GESTURE_DEBOUNCE_FRAMES
        sm_res: StateMachineResult = self.state_machine.process(
            raw_gesture=raw_id,
            confidence=conf,
            min_confidence=self.confidence_threshold,
            debounce_frames=debounce_val,
            cooldown_sec=cooldown_val,
            is_continuous=is_continuous,
            is_sensitive=is_sensitive
        )

        # -------------------------------------------------------------
        # 5. RESOLVE TARGET ACTION ACCORDING TO MODE
        # -------------------------------------------------------------
        assigned_action = self._resolve_action(sm_res.gesture_name)

        label = self._format_display_label(sm_res.gesture_name, assigned_action)

        return GestureResult(
            gesture_id=sm_res.gesture_name,
            display_label=label,
            confidence=conf,
            active_mode=self.active_mode,
            assigned_action=assigned_action,
            state=sm_res.state,
            status_text=sm_res.status_text,
            status_color=sm_res.status_color,
            can_execute=sm_res.can_execute,
            is_discrete_trigger=sm_res.can_execute and not is_continuous,
            smoothed_point=smoothed_pt,
            delta_x=delta_x,
            delta_y=delta_y,
            hand_count=hand_count,
            primary_handedness=primary_hand.handedness,
            progress_pct=sm_res.progress_pct
        )

    def _resolve_action(self, gesture_id: str) -> Optional[str]:
        """Resolves gesture ID into configured system action according to mode."""
        # 1. Custom binding check
        custom_action = self.custom_manager.get_bound_action(gesture_id)
        if custom_action:
            return custom_action

        # 2. Active mode map (try exact name and stripped name)
        mode_map = self.mode_mappings.get(self.active_mode, {})
        if gesture_id in mode_map:
            return mode_map[gesture_id]

        stripped_mode = self.active_mode.replace(" Mode", "").strip()
        stripped_map = self.mode_mappings.get(stripped_mode, {})
        if gesture_id in stripped_map:
            return stripped_map[gesture_id]

        # 3. Universal mode map fallback
        univ_map = self.mode_mappings.get(config.AppMode.UNIVERSAL, {})
        if gesture_id in univ_map:
            return univ_map[gesture_id]

        alt_univ_map = self.mode_mappings.get("Universal (All-in-One)", {})
        if gesture_id in alt_univ_map:
            return alt_univ_map[gesture_id]

        # 4. Built-in Comprehensive Operations Standard Fallback
        builtins = {
            "INDEX_POINTING": "MOVE_CURSOR",
            "PINCH_INDEX": "LEFT_CLICK",
            "DOUBLE_PINCH_INDEX": "DOUBLE_CLICK",
            "PINCH_MIDDLE": "RIGHT_CLICK",
            "CLOSED_FIST": "START_DRAG",
            "OPEN_PALM": "RELEASE_DRAG",
            "INDEX_HOLD": "HOVER",
            "TWO_FINGERS_UP": "SCROLL_UP",
            "TWO_FINGERS_DOWN": "SCROLL_DOWN",
            "SWIPE_UP": "PAGE_UP",
            "SWIPE_DOWN": "PAGE_DOWN",
            "SWIPE_LEFT": "BROWSER_BACK",
            "SWIPE_RIGHT": "BROWSER_FORWARD",
            "THUMBS_UP": "VOLUME_UP",
            "THUMBS_DOWN": "VOLUME_DOWN",
            "THUMB_PINKY": "VOLUME_MUTE",
            "OK_SIGN": "TAKE_SCREENSHOT",
            "PEACE_SIGN": "COPY",
            "FINGER_GUN": "SELECT_ALL",
            "THREE_FINGERS": "APP_SWITCH",
            "FOUR_FINGERS": "TASK_VIEW",
            "FIVE_FINGERS_HOLD": "WINDOW_CLOSE_SAFE",
            "ROCK_SIGN": "LOCK_SCREEN_SAFE",
            "ROCK_SIGN_HOLD": "LOCK_SCREEN_SAFE",
            "HAND_MOVE_UP": "VOLUME_UP",
            "HAND_MOVE_DOWN": "VOLUME_DOWN",
            "TWO_HAND_EXPAND": "ZOOM_IN",
            "TWO_HAND_CONTRACT": "ZOOM_OUT",
            "TWO_HAND_ROTATE": "ROTATE_DOCUMENT",
            "TWO_HAND_PALMS": "SHOW_DESKTOP"
        }
        return builtins.get(gesture_id, None)

    def _format_display_label(self, gesture_id: str, action: Optional[str]) -> str:
        clean_name = gesture_id.replace("_", " ").title()
        if action:
            clean_action = action.replace("_", " ").title()
            return f"{clean_name} -> {clean_action}"
        return clean_name


if __name__ == "__main__":
    print("=" * 65)
    print("  GESTURE CLASSIFIER & 7-STATE MACHINE DIAGNOSTIC CHECK")
    print("=" * 65)
    clf = GestureClassifier()
    print(f"• Active Mode:           {clf.active_mode}")
    print(f"• Configured Modes:      {list(clf.mode_mappings.keys())}")
    print(f"• Custom Gestures Model: {'Loaded' if clf.custom_manager.classifier else 'None (Empty)'}")
    print(f"• Core Rules Engine:     Initialized")
    print(f"• 7-State Machine:       {clf.state_machine.state}")
    print("• Sample Action Bindings:")
    sample_tests = ["INDEX_POINTING", "PINCH_INDEX", "DOUBLE_PINCH_INDEX", "TWO_FINGERS_UP", "OK_SIGN", "PEACE_SIGN", "FIVE_FINGERS_HOLD"]
    for gid in sample_tests:
        act = clf._resolve_action(gid)
        print(f"   - {gid:<20} -> {act}")
    print("=" * 65)
    print("  DIAGNOSTIC PASSED: Classifier is ready for real-time inference.")
    print("=" * 65)

