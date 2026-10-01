"""
Configuration Module for AI-Based Vision-Driven HCI System
===========================================================
Defines all hyper-parameters, thresholds, hardware bindings,
and gesture-to-action mapping constants.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR / "ml"
DATASET_DIR = ML_DIR / "dataset"
MODELS_DIR = ML_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

# Ensure directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ML File Paths
DATASET_CSV_PATH = DATASET_DIR / "gesture_landmarks.csv"
MODEL_PATH = MODELS_DIR / "gesture_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
EVALUATION_METRICS_PATH = MODELS_DIR / "evaluation_report.json"
CONFUSION_MATRIX_IMAGE = MODELS_DIR / "confusion_matrix.png"

# Camera Acquisition Settings
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
FLIP_HORIZONTAL = True  # Natural mirror view

# MediaPipe Hand Tracker Settings
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Active Control Region (Screen Remapping Margin in pixels)
# Users only need to move hand inside this bounding box to reach full screen corners
MARGIN_LEFT = 80
MARGIN_RIGHT = 80
MARGIN_TOP = 60
MARGIN_BOTTOM = 80

# Cursor Smoothing (Exponential Moving Average Filter)
# alpha = 1.0 means no smoothing; smaller values mean smoother, less jitter, slight lag
SMOOTHING_ALPHA = 0.35
ENABLE_CURSOR_ACCELERATION = True

# Geometric & Landmark Thresholds (Normalized Units)
PINCH_THRESHOLD = 0.055            # Distance between thumb tip (4) and index tip (8)
PINCH_RELEASE_THRESHOLD = 0.080    # Hysteresis release distance to prevent flickering
MIDDLE_PINCH_THRESHOLD = 0.060     # Distance between thumb tip (4) and middle tip (12)
DOUBLE_PINCH_WINDOW_SEC = 0.40     # Max elapsed time between two pinches for double-click

# Temporal Debouncing & Cooldown (Seconds)
STABILITY_CONSECUTIVE_FRAMES = 3   # Must detect gesture for N consecutive frames
COOLDOWN_DISCRETE_SEC = 0.45       # For single triggers (Volume, Track, Click, Screenshot)
COOLDOWN_SCROLL_SEC = 0.08         # For continuous scrolling
SCROLL_SPEED_DELTA = 120           # Wheel units per scroll tick
VOLUME_STEP = 2                    # Percent or keypresses

# Swipe Motion Tracking Settings
SWIPE_BUFFER_SIZE = 12             # Number of recent palm center frames to track
SWIPE_VELOCITY_THRESHOLD = 300.0   # Horizontal velocity threshold (pixels per second)
SWIPE_MIN_DISPLACEMENT = 70.0      # Minimum net horizontal pixel displacement
SWIPE_MAX_VERTICAL_DEVIATION = 50.0# Max vertical deviation during a horizontal swipe
SWIPE_COOLDOWN_SEC = 0.80          # Cooldown between consecutive swipe recognitions

# Recognized Gesture Identifier Constants
GESTURE_OPEN_PALM = "open_palm"
GESTURE_PINCH = "pinch"
GESTURE_DOUBLE_PINCH = "double_pinch"
GESTURE_MIDDLE_PINCH = "middle_pinch"
GESTURE_TWO_FINGERS_UP = "two_fingers_up"
GESTURE_TWO_FINGERS_DOWN = "two_fingers_down"
GESTURE_THUMB_UP = "thumb_up"
GESTURE_THUMB_DOWN = "thumb_down"
GESTURE_FIST = "fist"
GESTURE_THREE_FINGERS = "three_fingers"
GESTURE_SWIPE_LEFT = "swipe_left"
GESTURE_SWIPE_RIGHT = "swipe_right"
GESTURE_SCREENSHOT = "screenshot"
GESTURE_IDLE = "idle"

# Gesture Display Names
GESTURE_DISPLAY_NAMES = {
    GESTURE_OPEN_PALM: "Open Palm (Cursor)",
    GESTURE_PINCH: "Pinch (Left Click/Drag)",
    GESTURE_DOUBLE_PINCH: "Double Pinch (Double Click)",
    GESTURE_MIDDLE_PINCH: "Middle Pinch (Right Click)",
    GESTURE_TWO_FINGERS_UP: "Two Fingers Up (Scroll Up)",
    GESTURE_TWO_FINGERS_DOWN: "Two Fingers Down (Scroll Down)",
    GESTURE_THUMB_UP: "Thumb Up (Volume +)",
    GESTURE_THUMB_DOWN: "Thumb Down (Volume -)",
    GESTURE_FIST: "Closed Fist (Pause/Standby)",
    GESTURE_THREE_FINGERS: "Three Fingers (Play/Pause)",
    GESTURE_SWIPE_LEFT: "Swipe Left (Previous Track)",
    GESTURE_SWIPE_RIGHT: "Swipe Right (Next Track)",
    GESTURE_SCREENSHOT: "Screenshot Trigger",
    GESTURE_IDLE: "No Gesture Detected"
}

# Gesture-to-Computer Action Descriptions
GESTURE_ACTION_MAPPING = {
    GESTURE_OPEN_PALM: "Cursor Movement",
    GESTURE_PINCH: "Left Click / Drag",
    GESTURE_DOUBLE_PINCH: "Double Click",
    GESTURE_MIDDLE_PINCH: "Right Click",
    GESTURE_TWO_FINGERS_UP: "Scroll Up",
    GESTURE_TWO_FINGERS_DOWN: "Scroll Down",
    GESTURE_THUMB_UP: "Volume Increase",
    GESTURE_THUMB_DOWN: "Volume Decrease",
    GESTURE_FIST: "Freeze Cursor / Standby",
    GESTURE_THREE_FINGERS: "Media Play / Pause",
    GESTURE_SWIPE_LEFT: "Previous Track / Slide",
    GESTURE_SWIPE_RIGHT: "Next Track / Slide",
    GESTURE_SCREENSHOT: "Capture Screenshot",
    GESTURE_IDLE: "None"
}

# UI Styling Constants (Modern Dark Glassmorphism)
COLOR_BG_DARK = (20, 24, 30)
COLOR_ACCENT_CYAN = (235, 175, 45)      # BGR for OpenCV
COLOR_ACCENT_EMERALD = (80, 205, 80)
COLOR_ACCENT_AMBER = (50, 160, 245)
COLOR_ACCENT_ROSE = (75, 75, 235)
COLOR_TEXT_WHITE = (250, 250, 250)
COLOR_TEXT_MUTED = (160, 165, 175)
COLOR_BOX_ACTIVE = (0, 220, 255)
COLOR_BOX_INACTIVE = (80, 80, 80)
