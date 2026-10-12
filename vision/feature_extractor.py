"""
Vision Feature Extractor Bridge.
Redirects to the modular `features.landmark_processor` implementation
while preserving full backward compatibility across the entire HCI codebase.
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from features.landmark_processor import (
    CANONICAL_ANGLE_TRIPLETS,
    CANONICAL_DISTANCE_PAIRS,
    EXTENDED_ANGLE_TRIPLETS,
    DualHandFeatures,
    FeatureExtractor,
    HandFeatures,
    LandmarkProcessor,
    compute_angle_3d,
)

__all__ = [
    "FeatureExtractor",
    "LandmarkProcessor",
    "HandFeatures",
    "DualHandFeatures",
    "compute_angle_3d",
    "CANONICAL_DISTANCE_PAIRS",
    "CANONICAL_ANGLE_TRIPLETS",
    "EXTENDED_ANGLE_TRIPLETS",
]
