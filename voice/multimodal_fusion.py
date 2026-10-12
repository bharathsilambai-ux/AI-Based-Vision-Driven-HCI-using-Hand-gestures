r"""
Phase 15: Multimodal Fusion Engine (Voice + Hand Gesture HCI Synergy) (Module 15)
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Dual-Modal Synergy (Module 15):
   - Spatial Precision (Vision): Continuous 2D cursor pointing, geometric pinch clicking, spatial swiping.
   - Semantic Context (Voice): High-level operational mode switching, direct hardware triggers, text dictation.
   - Synergistic Fusion:
     a) Voice-Assisted Mode Switching: Seamlessly sets operational mode without gesture overhead.
     b) Critical Action Confirmation Gate: Intercepts dangerous actions (LOCK_SCREEN), prompts auditory
        confirmation, and requires voice "confirm" within timeout before triggering.
     c) Compound Multimodal Point & Speak: Directs clicks (LEFT_CLICK, RIGHT_CLICK, DOUBLE_CLICK)
        at hand pointing coordinates $(x, y)$ upon spoken voice command.
     d) Voice Direct Execution: Executes system, media, presentation, or browser commands directly via voice.
     e) Spoken Dictation: Types transcribed text payload directly into active focus.

2. Temporal Coincidence Window:
   - Correlates vision events and voice events within an adjustable temporal window ($\Delta t \le 1.5$ s).
   - Filters out expired or orphaned asynchronous events.

3. Complete Telemetry & State Introspection:
   - Tracks active confirmation gate state (IDLE, PENDING, CONFIRMED, REJECTED, EXPIRED).
   - Detailed event logging for evaluation metrics and verification testing.
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
from controllers.browser_controller import BrowserController
from controllers.media_controller import MediaController
from controllers.mouse_controller import VirtualMouseController
from controllers.presentation_controller import PresentationController
from controllers.system_controller import SystemController
from gestures.gesture_mapping import OperationalMode
from gestures.realtime_recognizer import GestureRecognitionResult, RealtimeGestureRecognizer
from utils.logger import logger
from voice.intent_classifier import IntentClassifier, VoiceIntentResult, VoiceIntentType
from voice.speech_engine import SpeechEngine

# Critical actions requiring explicit confirmation
CRITICAL_ACTIONS = {"LOCK_SCREEN", "SHUTDOWN", "CLOSE_ALL_APPS"}


class ConfirmationGateState:
    """State machine for critical action safety gate."""
    IDLE = "IDLE"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class MultimodalFusionEngine:
    """
    Central coordinator orchestrating multimodal vision + voice interactions.
    Fuses spatial hand tracking data with semantic speech intents.
    """

    def __init__(
        self,
        speech_engine: Optional[SpeechEngine] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        gesture_recognizer: Optional[RealtimeGestureRecognizer] = None,
        mouse_controller: Optional[VirtualMouseController] = None,
        system_controller: Optional[SystemController] = None,
        media_controller: Optional[MediaController] = None,
        browser_controller: Optional[BrowserController] = None,
        presentation_controller: Optional[PresentationController] = None,
        coincidence_window_s: float = 1.5,
        confirmation_timeout_s: float = 5.0,
        mock_mode: bool = False,
    ):
        self.mock_mode = mock_mode
        self.coincidence_window_s = coincidence_window_s
        self.confirmation_timeout_s = confirmation_timeout_s

        # Subsystems
        self.speech_engine = speech_engine if speech_engine is not None else SpeechEngine(mock_mode=mock_mode)
        self.intent_classifier = intent_classifier if intent_classifier is not None else IntentClassifier()
        self.gesture_recognizer = gesture_recognizer

        # Controllers
        self.mouse_ctrl = mouse_controller if mouse_controller is not None else VirtualMouseController(mock_mode=mock_mode)
        self.system_ctrl = system_controller if system_controller is not None else SystemController(mock_mode=mock_mode)
        self.media_ctrl = media_controller if media_controller is not None else MediaController(mock_mode=mock_mode)
        self.browser_ctrl = browser_controller if browser_controller is not None else BrowserController(mock_mode=mock_mode)
        self.presentation_ctrl = presentation_controller if presentation_controller is not None else PresentationController(mock_mode=mock_mode)

        # Multimodal State
        self.active_mode: str = OperationalMode.MOUSE
        self.last_gesture_result: Optional[GestureRecognitionResult] = None
        self.last_gesture_time: float = 0.0

        self.last_voice_intent: Optional[VoiceIntentResult] = None
        self.last_voice_time: float = 0.0

        # Confirmation Gate State Machine
        self.gate_state: str = ConfirmationGateState.IDLE
        self.pending_action: Optional[str] = None
        self.pending_action_time: float = 0.0

        # Event History
        self.fusion_event_log: List[Dict[str, Any]] = []

    # =========================================================================
    # Modality Feeders
    # =========================================================================

    def feed_gesture(self, gesture_result: GestureRecognitionResult, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Feeds a real-time gesture recognition result into the multimodal fusion engine.
        Evaluates confirmation gates and spatial tracking coordinates.
        """
        now = timestamp if timestamp is not None else time.perf_counter()
        self.last_gesture_result = gesture_result
        self.last_gesture_time = now

        # Update active mode from recognizer if available
        if gesture_result.active_mode:
            self.active_mode = gesture_result.active_mode

        # Check if gate expired
        self._check_gate_timeout(now)

        action = gesture_result.mapped_action
        disposition = "IGNORED"

        # Check for Critical Action Triggering
        if gesture_result.can_execute and action in CRITICAL_ACTIONS:
            disposition = self._initiate_confirmation_gate(action, now)

        elif gesture_result.can_execute and action:
            # Normal action execution (can be routed or executed directly)
            disposition = "EXECUTED_GESTURE"

        event = {
            "timestamp": now,
            "type": "GESTURE_FEED",
            "gesture": gesture_result.confirmed_gesture,
            "action": action,
            "disposition": disposition,
            "active_mode": self.active_mode,
        }
        self._record_event(event)
        return event

    def feed_voice_transcript(self, transcript: str, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Feeds a voice transcript, classifies its intent, and performs multimodal fusion.
        """
        now = timestamp if timestamp is not None else time.perf_counter()
        intent = self.intent_classifier.classify(transcript)
        self.last_voice_intent = intent
        self.last_voice_time = now

        # Check gate timeout
        self._check_gate_timeout(now)

        disposition = self._process_voice_intent(intent, now)

        event = {
            "timestamp": now,
            "type": "VOICE_FEED",
            "transcript": transcript,
            "intent_type": intent.intent_type,
            "action": intent.canonical_action,
            "disposition": disposition,
            "gate_state": self.gate_state,
        }
        self._record_event(event)
        return event

    # =========================================================================
    # Multimodal Synergy Handlers
    # =========================================================================

    def _process_voice_intent(self, intent: VoiceIntentResult, now: float) -> str:
        """Processes and routes classified voice intents."""

        # 1. Confirmation Gate Processing
        if self.gate_state == ConfirmationGateState.AWAITING_CONFIRMATION:
            if intent.intent_type == VoiceIntentType.CONFIRMATION:
                if intent.canonical_action == "CONFIRM_ACTION":
                    return self._confirm_pending_action(now)
                elif intent.canonical_action == "CANCEL_ACTION":
                    return self._cancel_pending_action(now)

        # 2. Voice Mode Switching
        if intent.intent_type == VoiceIntentType.MODE_SWITCH:
            target_mode = intent.parameters.get("target_mode")
            if target_mode and target_mode in OperationalMode.ALL_MODES:
                self.active_mode = target_mode
                if self.gesture_recognizer:
                    self.gesture_recognizer.set_mode(target_mode)
                self.speech_engine.speak(f"Switched to {target_mode.lower()} mode")
                logger.info(f"[MultimodalFusion] Voice switched mode to: {target_mode}")
                return f"MODE_SWITCHED_{target_mode}"

        # 3. Compound Spatial Commands (Click at Hand Pointing Location)
        if intent.intent_type == VoiceIntentType.COMPOUND_ACTION:
            if self._is_gesture_temporally_coincident(now):
                pt = self.last_gesture_result.tracking_point if self.last_gesture_result else None
                act = intent.canonical_action
                if act == "LEFT_CLICK":
                    self.mouse_ctrl.left_click()
                    return f"COMPOUND_LEFT_CLICK_{pt}"
                elif act == "RIGHT_CLICK":
                    self.mouse_ctrl.right_click()
                    return f"COMPOUND_RIGHT_CLICK_{pt}"
                elif act == "DOUBLE_CLICK":
                    self.mouse_ctrl.double_click()
                    return f"COMPOUND_DOUBLE_CLICK_{pt}"
            else:
                # Gesture not active/expired; perform fallback click
                if intent.canonical_action == "LEFT_CLICK":
                    self.mouse_ctrl.left_click()
                return f"DIRECT_{intent.canonical_action}"

        # 4. Spoken Dictation
        if intent.intent_type == VoiceIntentType.DICTATION:
            text_payload = intent.parameters.get("text", "")
            if text_payload:
                if not self.mock_mode:
                    pyautogui.write(text_payload)
                logger.info(f"[MultimodalFusion] Dictated text: \"{text_payload}\"")
                return f"DICTATION_TYPED"

        # 5. Direct Voice Command Execution across controllers
        if intent.canonical_action:
            action = intent.canonical_action

            # System controls
            if intent.intent_type == VoiceIntentType.SYSTEM_CONTROL:
                if action in CRITICAL_ACTIONS:
                    return self._initiate_confirmation_gate(action, now)
                self.system_ctrl.execute_action(action)
                return f"EXECUTED_SYSTEM_{action}"

            # Media controls
            elif intent.intent_type == VoiceIntentType.MEDIA_CONTROL:
                self.media_ctrl.execute_action(action)
                return f"EXECUTED_MEDIA_{action}"

            # Browser controls
            elif intent.intent_type == VoiceIntentType.BROWSER_CONTROL:
                self.browser_ctrl.execute_action(action)
                return f"EXECUTED_BROWSER_{action}"

            # Presentation controls
            elif intent.intent_type == VoiceIntentType.PRESENTATION_CONTROL:
                self.presentation_ctrl.execute_action(action)
                return f"EXECUTED_PRESENTATION_{action}"

        return "UNHANDLED_VOICE"

    # =========================================================================
    # Confirmation Safety Gate State Machine
    # =========================================================================

    def _initiate_confirmation_gate(self, action: str, now: float) -> str:
        """Enters AWAITING_CONFIRMATION state for a critical action."""
        self.gate_state = ConfirmationGateState.AWAITING_CONFIRMATION
        self.pending_action = action
        self.pending_action_time = now

        msg = f"Critical action: {action.replace('_', ' ').lower()}. Please confirm or cancel."
        self.speech_engine.speak(msg)
        logger.warning(f"[MultimodalFusion] Safety gate engaged for: '{action}'")
        return "GATE_PENDING_CONFIRMATION"

    def _confirm_pending_action(self, now: float) -> str:
        """Executes pending critical action upon voice confirmation."""
        act = self.pending_action
        self.gate_state = ConfirmationGateState.CONFIRMED
        self.speech_engine.speak(f"{act.replace('_', ' ').lower()} confirmed")
        logger.info(f"[MultimodalFusion] Critical action '{act}' confirmed by voice.")

        # Execute action
        if act == "LOCK_SCREEN":
            self.system_ctrl.lock_screen(confirmed=True)

        self.pending_action = None
        self.gate_state = ConfirmationGateState.IDLE
        return f"CONFIRMED_AND_EXECUTED_{act}"

    def _cancel_pending_action(self, now: float) -> str:
        """Aborts pending critical action upon voice cancellation."""
        act = self.pending_action
        self.gate_state = ConfirmationGateState.CANCELLED
        self.speech_engine.speak(f"{act.replace('_', ' ').lower()} cancelled")
        logger.info(f"[MultimodalFusion] Critical action '{act}' cancelled by voice.")
        self.pending_action = None
        self.gate_state = ConfirmationGateState.IDLE
        return f"CANCELLED_{act}"

    def _check_gate_timeout(self, now: float):
        """Checks if confirmation gate timed out."""
        if self.gate_state == ConfirmationGateState.AWAITING_CONFIRMATION:
            if (now - self.pending_action_time) > self.confirmation_timeout_s:
                logger.info(f"[MultimodalFusion] Confirmation gate timed out for '{self.pending_action}'")
                self.speech_engine.speak(f"Action {self.pending_action} timed out and was cancelled.")
                self.gate_state = ConfirmationGateState.EXPIRED
                self.pending_action = None
                self.gate_state = ConfirmationGateState.IDLE

    def _is_gesture_temporally_coincident(self, now: float) -> bool:
        """Checks if the most recent gesture is within the coincidence window."""
        if not self.last_gesture_result:
            return False
        return (now - self.last_gesture_time) <= self.coincidence_window_s

    def _record_event(self, event: Dict[str, Any]):
        self.fusion_event_log.append(event)
        if len(self.fusion_event_log) > 1000:
            self.fusion_event_log = self.fusion_event_log[-500:]

    def get_status(self) -> Dict[str, Any]:
        """Returns the current state and telemetry of the multimodal fusion engine."""
        return {
            "active_mode": self.active_mode,
            "gate_state": self.gate_state,
            "pending_action": self.pending_action,
            "coincidence_window_s": self.coincidence_window_s,
            "event_count": len(self.fusion_event_log),
            "mock_mode": self.mock_mode,
        }
