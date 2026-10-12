"""
Voice Package: AI-Based Vision-Driven HCI System.
Provides speech recognition, text-to-speech synthesis, intent classification,
and multimodal vision-voice fusion (Module 15).
"""

from .intent_classifier import IntentClassifier, VoiceIntentResult, VoiceIntentType
from .multimodal_fusion import (
    CRITICAL_ACTIONS,
    ConfirmationGateState,
    MultimodalFusionEngine,
)
from .speech_engine import SpeechEngine

__all__ = [
    "SpeechEngine",
    "VoiceIntentType",
    "VoiceIntentResult",
    "IntentClassifier",
    "MultimodalFusionEngine",
    "ConfirmationGateState",
    "CRITICAL_ACTIONS",
]
