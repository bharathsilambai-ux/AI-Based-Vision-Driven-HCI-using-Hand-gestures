"""
UI Package: AI-Based Vision-Driven HCI System.
Provides professional real-time HUD overlays, telemetry visualizers,
and the vision dashboard coordinator (Module 18).
"""

from .dashboard import DashboardFrameResult, VisionDashboard
from .hud_renderer import FINGER_COLORS, HAND_CONNECTIONS, HUDRenderer

__all__ = [
    "HUDRenderer",
    "VisionDashboard",
    "DashboardFrameResult",
    "HAND_CONNECTIONS",
    "FINGER_COLORS",
]
