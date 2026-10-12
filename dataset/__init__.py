"""
Dataset Package: Custom landmark-based hand gesture dataset collection, augmentation, and management.
"""

from .gesture_collector import (
    DataAugmenter,
    DatasetCollector,
    GESTURE_CLASSES,
    GESTURE_SYNONYMS,
    GestureDataset,
    ID_TO_LABEL,
    LABEL_TO_ID,
    SyntheticDatasetGenerator,
    canonicalize_label,
)

__all__ = [
    "GESTURE_CLASSES",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "GESTURE_SYNONYMS",
    "canonicalize_label",
    "DataAugmenter",
    "GestureDataset",
    "SyntheticDatasetGenerator",
    "DatasetCollector",
]
