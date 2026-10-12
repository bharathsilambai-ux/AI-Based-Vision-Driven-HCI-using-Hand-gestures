"""
Phase 15: Voice Intent Classification Subsystem (Module 15)
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Canonical Intent Categories (Module 15):
   - MODE_SWITCH: Mode transitions (e.g., "switch to presentation mode", "media mode").
   - SYSTEM_CONTROL: Hardware controls (e.g., "volume up", "mute", "screenshot", "lock screen").
   - MEDIA_CONTROL: Playback controls (e.g., "play pause", "next track", "fast forward").
   - BROWSER_CONTROL: Web navigation (e.g., "scroll down", "new tab", "reload page", "zoom in").
   - PRESENTATION_CONTROL: Slideshow commands (e.g., "next slide", "laser pointer", "start slideshow").
   - CONFIRMATION: Security confirmation gate responses (e.g., "confirm", "proceed", "cancel", "abort").
   - COMPOUND_ACTION: Voice-directed spatial commands (e.g., "click here", "right click", "double click").
   - DICTATION: Text entry automation (e.g., "type hello world").

2. Robust Intent Parsing:
   - Pattern-based regex extraction, keyword matching, and fuzzy fallback scoring.
   - Extracts parameters such as target operational mode or dictation payload.
   - Returns VoiceIntentResult with intent type, canonical action, confidence, and parameters.
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gestures.gesture_mapping import OperationalMode
from utils.logger import logger


class VoiceIntentType:
    """Canonical categories of recognized voice intents."""
    MODE_SWITCH = "MODE_SWITCH"
    SYSTEM_CONTROL = "SYSTEM_CONTROL"
    MEDIA_CONTROL = "MEDIA_CONTROL"
    BROWSER_CONTROL = "BROWSER_CONTROL"
    PRESENTATION_CONTROL = "PRESENTATION_CONTROL"
    CONFIRMATION = "CONFIRMATION"
    COMPOUND_ACTION = "COMPOUND_ACTION"
    DICTATION = "DICTATION"
    UNKNOWN = "UNKNOWN"

    ALL_TYPES = [
        MODE_SWITCH,
        SYSTEM_CONTROL,
        MEDIA_CONTROL,
        BROWSER_CONTROL,
        PRESENTATION_CONTROL,
        CONFIRMATION,
        COMPOUND_ACTION,
        DICTATION,
        UNKNOWN,
    ]


class VoiceIntentResult:
    """Structured result of spoken intent classification."""

    def __init__(
        self,
        raw_transcript: str,
        intent_type: str,
        canonical_action: Optional[str] = None,
        confidence: float = 1.0,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.raw_transcript = raw_transcript
        self.intent_type = intent_type
        self.canonical_action = canonical_action
        self.confidence = float(confidence)
        self.parameters = parameters if parameters is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_transcript": self.raw_transcript,
            "intent_type": self.intent_type,
            "canonical_action": self.canonical_action,
            "confidence": round(self.confidence, 4),
            "parameters": self.parameters,
        }

    def __repr__(self) -> str:
        act = f" -> {self.canonical_action}" if self.canonical_action else ""
        return f"<VoiceIntent {self.intent_type}{act} ({self.confidence*100:.0f}%) [\"{self.raw_transcript}\"]>"


# Comprehensive Voice Command Rules: (Regex Pattern, Intent Type, Canonical Action, Optional Extractor)
INTENT_RULES: List[Tuple[str, str, str, Optional[str]]] = [
    # 1. Mode Switching
    (r"\b(?:switch to|go to|enter|set mode to)?\s*mouse\s*(?:mode)?\b", VoiceIntentType.MODE_SWITCH, "SWITCH_MODE", OperationalMode.MOUSE),
    (r"\b(?:switch to|go to|enter|set mode to)?\s*system\s*(?:mode)?\b", VoiceIntentType.MODE_SWITCH, "SWITCH_MODE", OperationalMode.SYSTEM),
    (r"\b(?:switch to|go to|enter|set mode to)?\s*media\s*(?:mode)?\b", VoiceIntentType.MODE_SWITCH, "SWITCH_MODE", OperationalMode.MEDIA),
    (r"\b(?:switch to|go to|enter|set mode to)?\s*browser\s*(?:mode)?\b", VoiceIntentType.MODE_SWITCH, "SWITCH_MODE", OperationalMode.BROWSER),
    (r"\b(?:switch to|go to|enter|set mode to)?\s*presentation\s*(?:mode)?\b", VoiceIntentType.MODE_SWITCH, "SWITCH_MODE", OperationalMode.PRESENTATION),
    (r"\b(?:switch to|go to|enter|set mode to)?\s*custom\s*(?:mode)?\b", VoiceIntentType.MODE_SWITCH, "SWITCH_MODE", OperationalMode.CUSTOM),

    # 2. System Hardware Controls
    (r"\b(?:volume up|increase volume|louder|turn up volume)\b", VoiceIntentType.SYSTEM_CONTROL, "VOLUME_UP", None),
    (r"\b(?:volume down|decrease volume|lower volume|softer|turn down volume)\b", VoiceIntentType.SYSTEM_CONTROL, "VOLUME_DOWN", None),
    (r"\b(?:mute|unmute|mute sound|mute volume)\b", VoiceIntentType.SYSTEM_CONTROL, "MUTE", None),
    (r"\b(?:brightness up|increase brightness|brighter)\b", VoiceIntentType.SYSTEM_CONTROL, "BRIGHTNESS_UP", None),
    (r"\b(?:brightness down|decrease brightness|dimmer|dim brightness)\b", VoiceIntentType.SYSTEM_CONTROL, "BRIGHTNESS_DOWN", None),
    (r"\b(?:take screenshot|capture screen|screenshot)\b", VoiceIntentType.SYSTEM_CONTROL, "SCREENSHOT", None),
    (r"\b(?:lock screen|lock computer|lock system)\b", VoiceIntentType.SYSTEM_CONTROL, "LOCK_SCREEN", None),

    # 3. Media Controls
    (r"\b(?:play pause|play music|pause music|toggle play|pause video|resume)\b", VoiceIntentType.MEDIA_CONTROL, "PLAY_PAUSE", None),
    (r"\b(?:next track|next song|skip song|next video)\b", VoiceIntentType.MEDIA_CONTROL, "NEXT_TRACK", None),
    (r"\b(?:previous track|prev track|previous song|prev song)\b", VoiceIntentType.MEDIA_CONTROL, "PREV_TRACK", None),
    (r"\b(?:stop playback|stop music|stop video)\b", VoiceIntentType.MEDIA_CONTROL, "STOP", None),
    (r"\b(?:fast forward|seek forward|skip forward)\b", VoiceIntentType.MEDIA_CONTROL, "FAST_FORWARD", None),
    (r"\b(?:rewind|seek backward|skip back)\b", VoiceIntentType.MEDIA_CONTROL, "REWIND", None),
    (r"\b(?:toggle fullscreen|fullscreen mode|exit fullscreen)\b", VoiceIntentType.MEDIA_CONTROL, "FULLSCREEN_TOGGLE", None),

    # 4. Web Browser Controls
    (r"\b(?:scroll down|page down|scroll down page)\b", VoiceIntentType.BROWSER_CONTROL, "SCROLL_DOWN", None),
    (r"\b(?:scroll up|page up|scroll up page)\b", VoiceIntentType.BROWSER_CONTROL, "SCROLL_UP", None),
    (r"\b(?:go back|history back|browser back)\b", VoiceIntentType.BROWSER_CONTROL, "HISTORY_BACK", None),
    (r"\b(?:go forward|history forward|browser forward)\b", VoiceIntentType.BROWSER_CONTROL, "HISTORY_FORWARD", None),
    (r"\b(?:reload page|refresh page|reload|refresh)\b", VoiceIntentType.BROWSER_CONTROL, "REFRESH_PAGE", None),
    (r"\b(?:new tab|open tab)\b", VoiceIntentType.BROWSER_CONTROL, "NEW_TAB", None),
    (r"\b(?:close tab|exit tab)\b", VoiceIntentType.BROWSER_CONTROL, "CLOSE_TAB", None),
    (r"\b(?:zoom in|enlarge)\b", VoiceIntentType.BROWSER_CONTROL, "ZOOM_IN", None),
    (r"\b(?:zoom out|shrink)\b", VoiceIntentType.BROWSER_CONTROL, "ZOOM_OUT", None),

    # 5. Presentation Controls
    (r"\b(?:next slide|advance slide|forward slide)\b", VoiceIntentType.PRESENTATION_CONTROL, "NEXT_SLIDE", None),
    (r"\b(?:previous slide|prev slide|back slide)\b", VoiceIntentType.PRESENTATION_CONTROL, "PREV_SLIDE", None),
    (r"\b(?:start presentation|start slideshow|begin slideshow)\b", VoiceIntentType.PRESENTATION_CONTROL, "START_PRESENTATION", None),
    (r"\b(?:exit presentation|end slideshow|stop presentation)\b", VoiceIntentType.PRESENTATION_CONTROL, "EXIT_PRESENTATION", None),
    (r"\b(?:laser pointer|toggle laser|turn on laser)\b", VoiceIntentType.PRESENTATION_CONTROL, "LASER_POINTER", None),
    (r"\b(?:black screen|blank black)\b", VoiceIntentType.PRESENTATION_CONTROL, "BLACK_SCREEN", None),
    (r"\b(?:white screen|blank white)\b", VoiceIntentType.PRESENTATION_CONTROL, "WHITE_SCREEN", None),
    (r"\b(?:slide overview|overview grid|slide sorter)\b", VoiceIntentType.PRESENTATION_CONTROL, "SLIDE_OVERVIEW", None),
    (r"\b(?:clear annotations|erase ink|clear pen)\b", VoiceIntentType.PRESENTATION_CONTROL, "CLEAR_ANNOTATIONS", None),

    # 6. Confirmation Responses
    (r"\b(?:confirm|yes confirm|proceed|yes proceed|approve)\b", VoiceIntentType.CONFIRMATION, "CONFIRM_ACTION", None),
    (r"\b(?:cancel|abort|no cancel|stop action|reject)\b", VoiceIntentType.CONFIRMATION, "CANCEL_ACTION", None),

    # 7. Compound Spatial Actions (Combine with hand pointer)
    (r"\b(?:click here|click this|left click)\b", VoiceIntentType.COMPOUND_ACTION, "LEFT_CLICK", None),
    (r"\b(?:double click here|double click this|double click)\b", VoiceIntentType.COMPOUND_ACTION, "DOUBLE_CLICK", None),
    (r"\b(?:right click here|right click this|right click|context menu)\b", VoiceIntentType.COMPOUND_ACTION, "RIGHT_CLICK", None),
]


class IntentClassifier:
    """
    Parses transcribed speech strings into canonical voice intents with
    confidence scoring, parameter extraction, and dictation support.
    """

    def __init__(self):
        # Precompile regex patterns for high-speed sub-millisecond classification
        self.compiled_rules = [
            (re.compile(pat, re.IGNORECASE), itype, action, param)
            for pat, itype, action, param in INTENT_RULES
        ]

    def classify(self, transcript: str) -> VoiceIntentResult:
        """
        Classifies speech transcript text into a structured VoiceIntentResult.

        Args:
            transcript: Transcribed voice text (e.g. "switch to presentation mode")

        Returns:
            VoiceIntentResult with intent type, canonical action, and parameters.
        """
        if not transcript or not transcript.strip():
            return VoiceIntentResult("", VoiceIntentType.UNKNOWN, confidence=0.0)

        clean_text = transcript.strip().lower()

        # 1. Check for Dictation: "type <message>" or "dictate <message>"
        dictate_match = re.match(r"^(?:type|dictate|write)\s+(.+)$", clean_text, re.IGNORECASE)
        if dictate_match:
            dictation_body = dictate_match.group(1).strip()
            return VoiceIntentResult(
                raw_transcript=clean_text,
                intent_type=VoiceIntentType.DICTATION,
                canonical_action="TYPE_TEXT",
                confidence=0.98,
                parameters={"text": dictation_body},
            )

        # 2. Match against precompiled canonical command patterns
        for pattern, itype, action, param in self.compiled_rules:
            if pattern.search(clean_text):
                params = {}
                if itype == VoiceIntentType.MODE_SWITCH and param:
                    params["target_mode"] = param
                return VoiceIntentResult(
                    raw_transcript=clean_text,
                    intent_type=itype,
                    canonical_action=action,
                    confidence=0.95,
                    parameters=params,
                )

        # 3. Fallback: Unknown intent
        logger.debug(f"[IntentClassifier] Unrecognized command transcript: \"{clean_text}\"")
        return VoiceIntentResult(
            raw_transcript=clean_text,
            intent_type=VoiceIntentType.UNKNOWN,
            canonical_action=None,
            confidence=0.10,
        )
