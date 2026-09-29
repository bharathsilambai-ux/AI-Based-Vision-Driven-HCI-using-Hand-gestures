"""Recognition package: Geometric rules, ML classifier, 7-state machine, and temporal filters."""
from .gesture_rules import GestureRules, GestureID, RuleGestureEngine
from .custom_gesture import CustomGestureManager
from .state_machine import GestureStateMachine, StateMachineResult
from .temporal_filter import TemporalFilter, FilteredGesture
from .gesture_classifier import GestureClassifier, GestureResult

__all__ = [
    "GestureRules", "GestureID", "RuleGestureEngine",
    "CustomGestureManager",
    "GestureStateMachine", "StateMachineResult",
    "TemporalFilter", "FilteredGesture",
    "GestureClassifier", "GestureResult"
]
