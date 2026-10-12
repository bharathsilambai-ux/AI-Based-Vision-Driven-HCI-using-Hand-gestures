"""
Features Package: Landmark processing, normalizations, and high-dimensional ML vector representations.
"""

from .landmark_processor import (
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
    "LandmarkProcessor",
    "FeatureExtractor",
    "HandFeatures",
    "DualHandFeatures",
    "compute_angle_3d",
    "CANONICAL_DISTANCE_PAIRS",
    "CANONICAL_ANGLE_TRIPLETS",
    "EXTENDED_ANGLE_TRIPLETS",
]
