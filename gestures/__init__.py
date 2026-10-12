"""
Gestures Package: AI-Based Vision-Driven HCI System.
Integrates static ML gesture classification, kinematic dynamic trajectory tracking,
multi-mode command mapping, confidence gating, temporal stabilization, and cooldown management.
"""

from .static_gestures import (
    STATIC_GESTURE_CLASSES,
    StaticGestureRecognizer,
    StaticGestureResult,
)
from .dynamic_gestures import (
    DYNAMIC_GESTURE_CLASSES,
    DynamicGestureRecognizer,
    DynamicGestureResult,
    TrajectoryPoint,
)
from .gesture_mapping import (
    APPLICATION_SPECIFIC_MAPPINGS,
    DEFAULT_MODE_MAPPINGS,
    GestureMappingManager,
    OperationalMode,
)
from .custom_mapping import (
    ALL_KNOWN_GESTURES,
    CustomAction,
    CustomActionExecutor,
    CustomActionType,
    CustomMappingEngine,
    MappingProfile,
)
from .realtime_recognizer import (
    CONTINUOUS_ACTIONS,
    DEFAULT_COOLDOWNS,
    GestureRecognitionResult,
    RealtimeGestureRecognizer,
)

__all__ = [
    "STATIC_GESTURE_CLASSES",
    "StaticGestureRecognizer",
    "StaticGestureResult",
    "DYNAMIC_GESTURE_CLASSES",
    "DynamicGestureRecognizer",
    "DynamicGestureResult",
    "TrajectoryPoint",
    "APPLICATION_SPECIFIC_MAPPINGS",
    "DEFAULT_MODE_MAPPINGS",
    "GestureMappingManager",
    "OperationalMode",
    "CONTINUOUS_ACTIONS",
    "DEFAULT_COOLDOWNS",
    "GestureRecognitionResult",
    "RealtimeGestureRecognizer",
    "ALL_KNOWN_GESTURES",
    "CustomAction",
    "CustomActionExecutor",
    "CustomActionType",
    "CustomMappingEngine",
    "MappingProfile",
]

