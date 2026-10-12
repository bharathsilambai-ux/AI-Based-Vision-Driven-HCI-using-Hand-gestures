"""
Vision Camera Module - Bridge to camera.camera_manager.
Maintains backward compatibility while providing access to CameraManager.
"""

from camera.camera_manager import (
    CameraManager,
    CameraStatus,
    LightingCondition,
    FrameMetadata,
    enumerate_cameras,
    ThreadedCamera
)

__all__ = [
    "CameraManager",
    "CameraStatus",
    "LightingCondition",
    "FrameMetadata",
    "enumerate_cameras",
    "ThreadedCamera"
]
