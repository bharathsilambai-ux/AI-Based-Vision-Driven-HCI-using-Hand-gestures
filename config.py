"""
Central Configuration for AI-Based Vision-Driven Human-Computer Interaction System.
Contains operational modes, state machine definitions, thresholds, safety timers,
hardware parameters, and GUI themes.
"""

import os
import sys

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Base Project Paths
BASE_DIR = _ROOT
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
GESTURES_CONFIG_PATH = os.path.join(DATA_DIR, "gestures.json")
CUSTOM_MODEL_PATH = os.path.join(MODELS_DIR, "custom_gesture_model.pkl")
TEST_REPORT_PATH = os.path.join(DATA_DIR, "test_report.json")

# Ensure required directories exist
for d in [DATA_DIR, MODELS_DIR, SCREENSHOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# CAMERA & CAPTURE SETTINGS
# ==========================================
CAMERA_INDEX = 0             # Default webcam index
FRAME_WIDTH = 640            # Capture width in px
FRAME_HEIGHT = 480           # Capture height in px
TARGET_FPS = 30              # Target FPS
CAMERA_MIRROR = True         # Horizontal mirroring for natural interaction

def open_camera_device(preferred_index=CAMERA_INDEX):
    """
    Finds and opens the first functional webcam device on Windows.
    Returns (cv2.VideoCapture, actual_index_used) or (None, -1).
    """
    import cv2

    candidates = [preferred_index]
    for i in [0, 1, 2]:
        if i not in candidates:
            candidates.append(i)

    for idx in candidates:
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        return cap, idx
                    cap.release()
            except Exception:
                continue
    return None, -1


# ==========================================
# MEDIAPIPE HAND TRACKING SETTINGS
# ==========================================
MODEL_FILENAME = "hand_landmarker.task"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

MAX_NUM_HANDS = 2            # Dual-hand tracking
MIN_DETECTION_CONFIDENCE = 0.55
MIN_TRACKING_CONFIDENCE = 0.55


# ==========================================
# OPERATIONAL MODES (8 Dedicated Modes)
# ==========================================
class AppMode:
    MOUSE = "Mouse Mode"
    NAVIGATION = "Navigation Mode"
    MEDIA = "Media Mode"
    SYSTEM = "System Mode"
    KEYBOARD = "Keyboard Mode"
    ZOOM = "Zoom Mode"
    TWO_HAND = "Two-Hand Mode"
    CUSTOM = "Custom Gesture Mode"
    UNIVERSAL = "Universal (All-in-One)"

ALL_MODES = [
    AppMode.MOUSE,
    AppMode.NAVIGATION,
    AppMode.MEDIA,
    AppMode.SYSTEM,
    AppMode.KEYBOARD,
    AppMode.ZOOM,
    AppMode.TWO_HAND,
    AppMode.CUSTOM,
    AppMode.UNIVERSAL
]
DEFAULT_MODE = AppMode.UNIVERSAL


# ==========================================
# GESTURE STATE MACHINE STATES
# ==========================================
class GestureState:
    IDLE = "IDLE"
    DETECTED = "DETECTED"
    VERIFYING = "VERIFYING"
    CONFIRMED = "CONFIRMED"
    EXECUTING = "EXECUTING"
    COOLDOWN = "COOLDOWN"


# ==========================================
# GESTURE THRESHOLDS & STABILITY
# ==========================================
SMOOTHING_ALPHA_DEFAULT = 0.65
GESTURE_DEBOUNCE_FRAMES = 2
CONFIDENCE_THRESHOLD_DEFAULT = 0.65

PINCH_THRESHOLD_DEFAULT = 0.065
PINCH_RELEASE_THRESHOLD = 0.090
DOUBLE_PINCH_WINDOW = 0.40

# Minimum action cooldowns (in seconds)
COOLDOWN_CLICK = 0.25
COOLDOWN_DOUBLE_CLICK = 0.45
COOLDOWN_SHORTCUT = 0.55
COOLDOWN_SWIPE = 0.50
COOLDOWN_SCREENSHOT = 1.20
COOLDOWN_SENSITIVE = 1.80  # For Close Window / Lock Screen

# Sensitive action hold duration (seconds of continuous stable gesture before firing)
SENSITIVE_HOLD_DURATION = 1.20

# Swipe velocity threshold (px/s)
SWIPE_THRESHOLD_PIXELS = 40.0


# ==========================================
# CONTROLLER & SENSITIVITY SETTINGS
# ==========================================
CURSOR_SENSITIVITY_DEFAULT = 1.6
CURSOR_SENSITIVITY_MIN = 0.5
CURSOR_SENSITIVITY_MAX = 4.0

SCROLL_SENSITIVITY_DEFAULT = 6.0
SCROLL_SENSITIVITY_MIN = 1.0
SCROLL_SENSITIVITY_MAX = 20.0
MAX_SCROLL_SPEED = 35

DEADZONE_PIXELS_DEFAULT = 6.0
DEADZONE_PIXELS_MIN = 2.0
DEADZONE_PIXELS_MAX = 25.0

VOLUME_STEP = 2
BRIGHTNESS_STEP = 5
ZOOM_FACTOR = 1.15
FAILSAFE_ENABLED = True


# ==========================================
# USER CALIBRATION DEFAULT BOUNDS
# ==========================================
ACTIVE_BOX_DEFAULT = [0.15, 0.15, 0.85, 0.85]


# ==========================================
# GUI & HUD THEME (Modern Cyber Dark)
# ==========================================
APP_TITLE = "AI Vision-Driven Human-Computer Interaction"
APP_SUBTITLE = "Touchless Real-Time Hand Gesture Navigation System"
WINDOW_WIDTH = 1260
WINDOW_HEIGHT = 800

# HEX Colors for CustomTkinter
COLOR_BG_DARK = "#0E1017"
COLOR_SIDEBAR_BG = "#151722"
COLOR_CARD_BG = "#1C1F2E"
COLOR_CARD_BORDER = "#2B3046"
COLOR_ACCENT = "#6366F1"          # Indigo
COLOR_ACCENT_HOVER = "#4F46E5"
COLOR_SUCCESS = "#10B981"         # Emerald Green
COLOR_WARNING = "#F59E0B"         # Amber
COLOR_DANGER = "#EF4444"          # Crimson Red
COLOR_CYAN = "#06B6D4"           # Cyan
COLOR_PURPLE = "#A855F7"         # Violet
COLOR_TEXT_PRIMARY = "#F9FAFB"
COLOR_TEXT_SECONDARY = "#94A3B8"
COLOR_TEXT_MUTED = "#64748B"

# OpenCV HUD Drawing Colors (BGR Format)
CV_COLOR_BONE = (220, 160, 40)
CV_COLOR_JOINT = (255, 255, 255)
CV_COLOR_TIP = (50, 230, 50)
CV_COLOR_INDEX_TIP = (255, 220, 0)
CV_COLOR_THUMB_TIP = (50, 150, 255)
CV_COLOR_BOX = (0, 220, 120)
CV_COLOR_ACTIVE_ZONE = (70, 70, 120)
