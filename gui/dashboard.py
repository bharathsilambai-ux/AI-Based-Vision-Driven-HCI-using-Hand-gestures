"""
Master GUI Dashboard for AI Vision-Driven HCI System.
Coordinates real-time vision, dual-hand tracking, 7-state gesture machine,
toast notifications, mode switching, and subpanels.
"""

import os
import queue
import sys
import threading
import time
from typing import Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

import config
from control import (
    DisplayController,
    KeyboardController,
    MediaController,
    MouseController,
    SystemController,
    action_dispatcher
)
from recognition import GestureClassifier, GestureResult
from utils.logger import logger
from utils.performance import perf_tracker
from vision import FeatureExtractor, HandTracker, ThreadedCamera

from .calibration import CalibrationPanel
from .gesture_manager import GestureManagerView
from .performance_view import PerformanceView
from .settings import SettingsPanel
from .testing import GestureTestingView

# Set global appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class MainDashboard(ctk.CTk):
    """
    Central desktop dashboard window for the AI Vision-Driven HCI system.
    """

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title(config.APP_TITLE)
        self.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.minsize(1100, 720)
        self.configure(fg_color=config.COLOR_BG_DARK)

        # Core Vision & Recognition Subsystems
        self.camera = ThreadedCamera()
        self.hand_tracker: Optional[HandTracker] = None  # Instantiated on worker thread
        self.feature_extractor = FeatureExtractor()
        self.classifier = GestureClassifier()

        # Input Controllers
        self.mouse_ctrl = MouseController()
        self.keyboard_ctrl = KeyboardController()
        self.media_ctrl = MediaController()
        self.system_ctrl = SystemController()
        self.display_ctrl = DisplayController()

        # UI & State tracking
        self.is_running = True
        self.is_tracking_active = True
        self.show_skeleton = True
        self.show_hud = True
        self.active_box = list(config.ACTIVE_BOX_DEFAULT)

        # Temporary Toast Notification state: (message_text, expire_timestamp, color)
        self.toast_notification: Optional[Tuple[str, float, str]] = None

        # Inter-thread communication
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.cv_thread: Optional[threading.Thread] = None

        # Build UI layout
        self._create_layout()

        # Keybindings
        self.bind("<space>", lambda e: self._toggle_tracking())
        self.bind("<Escape>", lambda e: self._emergency_stop())
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Start Camera & Vision pipeline
        self._start_pipeline()

        # Start GUI update loop
        self.after(20, self._gui_refresh_loop)

    def _create_layout(self):
        """Builds modern 2-column header and workspace layout."""
        # -------------------------------------------------------------
        # TOP HEADER BAR
        # -------------------------------------------------------------
        self.header_frame = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=10, height=65)
        self.header_frame.pack(fill="x", padx=14, pady=(10, 6))
        self.header_frame.pack_propagate(False)

        # Title block
        title_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_box.pack(side="left", padx=14, pady=6)

        self.lbl_title = ctk.CTkLabel(
            title_box,
            text=config.APP_TITLE,
            font=ctk.CTkFont(family="Segoe UI", size=17, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        self.lbl_title.pack(anchor="w")

        self.lbl_sub = ctk.CTkLabel(
            title_box,
            text=config.APP_SUBTITLE,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=config.COLOR_CYAN
        )
        self.lbl_sub.pack(anchor="w")

        # Telemetry & Badges (Right Header)
        tele_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        tele_box.pack(side="right", padx=14, pady=8)

        # 8-Mode Selector Dropdown
        self.mode_selector = ctk.CTkOptionMenu(
            tele_box,
            values=config.ALL_MODES,
            command=self._on_mode_selected,
            width=180,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=config.COLOR_ACCENT,
            button_color=config.COLOR_ACCENT_HOVER
        )
        self.mode_selector.set(self.classifier.active_mode)
        self.mode_selector.pack(side="left", padx=6)

        self.badge_fps = ctk.CTkLabel(
            tele_box,
            text="FPS: --",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            fg_color="#24283A",
            corner_radius=6,
            text_color="#E0E7FF",
            width=80,
            height=30
        )
        self.badge_fps.pack(side="left", padx=4)

        self.badge_lat = ctk.CTkLabel(
            tele_box,
            text="LAT: -- ms",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            fg_color="#24283A",
            corner_radius=6,
            text_color="#E0E7FF",
            width=90,
            height=30
        )
        self.badge_lat.pack(side="left", padx=4)

        self.badge_status = ctk.CTkLabel(
            tele_box,
            text="TRACKING ACTIVE",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=config.COLOR_SUCCESS,
            corner_radius=6,
            text_color="#FFFFFF",
            width=135,
            height=30
        )
        self.badge_status.pack(side="left", padx=4)

        # -------------------------------------------------------------
        # MAIN BODY CONTAINER (2 Columns)
        # -------------------------------------------------------------
        self.body_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.body_frame.pack(fill="both", expand=True, padx=14, pady=(4, 10))

        # Column 1: Live Video Feed & Fast Controls (Left)
        self.col_left = ctk.CTkFrame(self.body_frame, fg_color=config.COLOR_CARD_BG, corner_radius=10)
        self.col_left.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=0)

        # Camera Display Viewport
        self.camera_label = ctk.CTkLabel(
            self.col_left,
            text="Initializing high-performance vision pipeline...",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            fg_color="#08090E",
            corner_radius=8
        )
        self.camera_label.pack(fill="both", expand=True, padx=10, pady=10)

        # HUD Status Banner Ribbon (Live Gesture, State Machine, Confidence)
        self.hud_ribbon = ctk.CTkFrame(self.col_left, fg_color="#151722", corner_radius=8, height=48)
        self.hud_ribbon.pack(fill="x", padx=10, pady=(0, 8))
        self.hud_ribbon.pack_propagate(False)

        self.hud_gesture_lbl = ctk.CTkLabel(
            self.hud_ribbon,
            text="Gesture: Searching for hands...",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        self.hud_gesture_lbl.pack(side="left", padx=12)

        self.hud_state_lbl = ctk.CTkLabel(
            self.hud_ribbon,
            text="● IDLE",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=config.COLOR_TEXT_MUTED
        )
        self.hud_state_lbl.pack(side="left", padx=12)

        self.hud_conf_lbl = ctk.CTkLabel(
            self.hud_ribbon,
            text="Conf: --%",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=config.COLOR_CYAN
        )
        self.hud_conf_lbl.pack(side="right", padx=12)

        # Fast Control Buttons
        btn_bar = ctk.CTkFrame(self.col_left, fg_color="transparent", height=38)
        btn_bar.pack(fill="x", padx=10, pady=(0, 10))

        self.btn_toggle_tracking = ctk.CTkButton(
            btn_bar,
            text="PAUSE TRACKING (Space)",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=config.COLOR_ACCENT,
            hover_color=config.COLOR_ACCENT_HOVER,
            command=self._toggle_tracking
        )
        self.btn_toggle_tracking.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_emergency_stop = ctk.CTkButton(
            btn_bar,
            text="EMERGENCY STOP (Esc)",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=config.COLOR_DANGER,
            hover_color="#DC2626",
            command=self._emergency_stop
        )
        self.btn_emergency_stop.pack(side="left", fill="x", expand=True, padx=4)

        # Column 2: Tabbed Subpanels (Right)
        self.tabview = ctk.CTkTabview(
            self.body_frame,
            fg_color=config.COLOR_CARD_BG,
            segmented_button_selected_color=config.COLOR_ACCENT,
            segmented_button_selected_hover_color=config.COLOR_ACCENT_HOVER,
            width=480
        )
        self.tabview.pack(side="right", fill="both", expand=False, padx=(0, 0), pady=0)

        # Create Tabs
        self.tab_guide = self.tabview.add("Guide")
        self.tab_studio = self.tabview.add("Studio")
        self.tab_test = self.tabview.add("Testing")
        self.tab_settings = self.tabview.add("Settings")
        self.tab_calib = self.tabview.add("Calibration")
        self.tab_perf = self.tabview.add("Analytics")

        self._build_guide_tab()

        # Instantiate subpanels
        self.studio_view = GestureManagerView(self.tab_studio, gesture_manager=self.classifier.custom_manager)
        self.studio_view.pack(fill="both", expand=True)

        self.testing_view = GestureTestingView(self.tab_test)
        self.testing_view.pack(fill="both", expand=True)

        self.settings_view = SettingsPanel(self.tab_settings, on_param_change=self._on_settings_param_changed)
        self.settings_view.pack(fill="both", expand=True)

        self.calib_view = CalibrationPanel(
            self.tab_calib,
            on_box_change=self._on_box_calibration_changed,
            on_param_change=self._on_settings_param_changed
        )
        self.calib_view.pack(fill="both", expand=True)

        self.perf_view = PerformanceView(self.tab_perf)
        self.perf_view.pack(fill="both", expand=True)

    def _build_guide_tab(self):
        """Constructs cheat sheet reference for all HCI operations."""
        scroll = ctk.CTkScrollableFrame(self.tab_guide, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=4, pady=4)

        categories = [
            ("A. Mouse Operations", [
                ("☝️  Index Pointing", "Move Cursor (EMA Smoothed)", config.COLOR_CYAN),
                ("🤏  Index Pinch", "Left Click", config.COLOR_SUCCESS),
                ("🤏🤏 Double Pinch", "Double Click (<0.40s)", config.COLOR_SUCCESS),
                ("👌  Middle Pinch", "Right Click", config.COLOR_CYAN),
                ("✊  Closed Fist", "Start Drag & Drop", config.COLOR_WARNING),
                ("🖐️  Open Palm", "Release Drag / Drop", config.COLOR_DANGER),
                ("☝️  Index Hold", "Hover / Select", config.COLOR_CYAN),
            ]),
            ("B. Scroll & Navigation", [
                ("✌️  Two Fingers Up", "Scroll Up", config.COLOR_SUCCESS),
                ("👇  Two Fingers Down", "Scroll Down", config.COLOR_SUCCESS),
                ("🖐️⬆️ Swipe Up", "Page Up", config.COLOR_CYAN),
                ("🖐️⬇️ Swipe Down", "Page Down", config.COLOR_CYAN),
                ("🖐️⬅️ Swipe Left", "Browser Back (Alt+Left)", config.COLOR_PURPLE),
                ("🖐️➡️ Swipe Right", "Browser Forward (Alt+Right)", config.COLOR_PURPLE),
                ("👍  Thumbs Up", "Show Desktop (Win+D)", config.COLOR_SUCCESS),
                ("✋  Open Palm", "Halt Scroll / Stop", config.COLOR_DANGER),
            ]),
            ("C. Media Control", [
                ("👍/✌️ Thumbs / Peace", "Play / Pause Media", config.COLOR_PURPLE),
                ("👉  Swipe Right", "Next Track", config.COLOR_PURPLE),
                ("👈  Swipe Left", "Previous Track", config.COLOR_PURPLE),
                ("👆  Hand Move Up", "Volume Up (Endpoint/Direct)", config.COLOR_SUCCESS),
                ("👇  Hand Move Down", "Volume Down", config.COLOR_SUCCESS),
                ("🤙/✊ Thumb-Pinky/Fist", "Mute / Unmute Audio", config.COLOR_WARNING),
            ]),
            ("D. System Control", [
                ("👌  OK Sign", "Take Screenshot (Saved to disk)", config.COLOR_PURPLE),
                ("🤟  Three Fingers", "Application Switcher (Alt+Tab)", config.COLOR_ACCENT),
                ("🖐️  Four Fingers", "Task View (Win+Tab)", config.COLOR_ACCENT),
                ("✊  Closed Fist", "Window Minimize (Win+Down)", config.COLOR_ACCENT),
                ("🖐️  Open Palm", "Window Maximize (Win+Up)", config.COLOR_ACCENT),
                ("🖐️⏳ Five Fingers Hold", "Close Window Safe (Alt+F4, 1.2s)", config.COLOR_DANGER),
                ("🤘⏳ Rock Sign Hold", "Lock Workstation Safe (1.2s)", config.COLOR_WARNING),
                ("✋✋ Two Open Palms", "Show Desktop (Win+D)", config.COLOR_ACCENT),
            ]),
            ("E. Keyboard Shortcuts", [
                ("✌️  Peace Sign", "Copy (Ctrl+C)", config.COLOR_SUCCESS),
                ("🖐️  Open Palm", "Paste (Ctrl+V)", config.COLOR_SUCCESS),
                ("👌  Middle Pinch", "Cut (Ctrl+X)", config.COLOR_SUCCESS),
                ("👉  Finger Gun", "Select All (Ctrl+A)", config.COLOR_CYAN),
                ("👈  Swipe Left", "Undo (Ctrl+Z)", config.COLOR_CYAN),
                ("👉  Swipe Right", "Redo (Ctrl+Y)", config.COLOR_CYAN),
                ("👇  Two Fingers Down", "Enter Key", config.COLOR_SUCCESS),
                ("✊  Closed Fist", "Escape Key", config.COLOR_WARNING),
            ]),
            ("F. Zoom & Display", [
                ("👐↔️ Two Hands Expand", "Zoom In (Ctrl + =)", config.COLOR_CYAN),
                ("👐>< Two Hands Contract", "Zoom Out (Ctrl + -)", config.COLOR_CYAN),
                ("🔄  Two Hands Rotate", "Rotate Document (Ctrl+R)", config.COLOR_PURPLE),
                ("☀⬆️ Hand Move Up", "Brightness Increase", config.COLOR_WARNING),
                ("☀⬇️ Hand Move Down", "Brightness Decrease", config.COLOR_WARNING),
            ]),
            ("G. Custom Studio ML Gestures", [
                ("🎯  Custom Point / Pinch", "User-Trained Random Forest", config.COLOR_ACCENT),
                ("✨  Custom Gestures", "Record & Bind in Studio Tab", config.COLOR_ACCENT),
            ])
        ]

        for cat_title, items in categories:
            lbl = ctk.CTkLabel(
                scroll,
                text=cat_title,
                font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                text_color=config.COLOR_TEXT_PRIMARY
            )
            lbl.pack(anchor="w", padx=6, pady=(10, 4))

            for pose, action_desc, col in items:
                row = ctk.CTkFrame(scroll, fg_color="#212435", corner_radius=6)
                row.pack(fill="x", padx=4, pady=2)

                ctk.CTkLabel(
                    row,
                    text=pose,
                    font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                    text_color=config.COLOR_TEXT_PRIMARY
                ).pack(side="left", padx=10, pady=4)

                ctk.CTkLabel(
                    row,
                    text=action_desc,
                    font=ctk.CTkFont(family="Segoe UI", size=11),
                    text_color=col
                ).pack(side="right", padx=10, pady=4)

    def show_toast(self, message: str, color: str = config.COLOR_SUCCESS, duration: float = 1.5):
        """Displays temporary toast overlay on the live preview."""
        self.toast_notification = (message, time.time() + duration, color)

    # -------------------------------------------------------------
    # EVENT HANDLERS & CALLBACKS
    # -------------------------------------------------------------
    def _toggle_tracking(self):
        active = action_dispatcher.toggle_active()
        self.is_tracking_active = active
        if active:
            self.badge_status.configure(text="TRACKING ACTIVE", fg_color=config.COLOR_SUCCESS)
            self.btn_toggle_tracking.configure(text="PAUSE TRACKING (Space)", fg_color=config.COLOR_ACCENT)
        else:
            self.badge_status.configure(text="TRACKING PAUSED", fg_color=config.COLOR_WARNING)
            self.btn_toggle_tracking.configure(text="RESUME TRACKING (Space)", fg_color="#D97706")

    def _emergency_stop(self):
        action_dispatcher.emergency_stop()
        self.is_tracking_active = False
        self.badge_status.configure(text="EMERGENCY STOP", fg_color=config.COLOR_DANGER)
        self.btn_toggle_tracking.configure(text="RESUME TRACKING (Space)", fg_color=config.COLOR_ACCENT)
        self.hud_gesture_lbl.configure(text="Gesture: EMERGENCY STOP TRIGGERED", text_color=config.COLOR_DANGER)
        self.show_toast("EMERGENCY STOP TRIGGERED", color=config.COLOR_DANGER, duration=2.5)

    def _on_mode_selected(self, mode: str):
        self.classifier.set_mode(mode)

    def _on_box_calibration_changed(self, box: list):
        self.active_box = box
        self.mouse_ctrl.set_active_box(box)
        logger.info(f"[Calibration] Updated active box margins: {box}")

    def _on_settings_param_changed(self, name: str, value):
        if name == "cursor_sens":
            self.mouse_ctrl.set_sensitivity(value)
        elif name == "deadzone":
            self.mouse_ctrl.set_deadzone(value)
        elif name == "smoothing_alpha":
            self.mouse_ctrl.set_smoothing_alpha(value)
            self.classifier.smoothing_alpha = value
        elif name == "confidence_threshold":
            self.classifier.confidence_threshold = value
        elif name == "show_skeleton":
            self.show_skeleton = bool(value)
        elif name == "show_hud":
            self.show_hud = bool(value)
        elif name == "camera_mirror":
            self.camera.mirror = bool(value)

    # -------------------------------------------------------------
    # VISION PIPELINE WORKER THREAD
    # -------------------------------------------------------------
    def _start_pipeline(self):
        self.camera_label.configure(
            text="● Initializing camera and AI vision pipeline...\nPlease wait a moment."
        )
        self.badge_status.configure(text="STARTING...", fg_color=config.COLOR_WARNING)
        self.cv_thread = threading.Thread(target=self._cv_worker_loop, daemon=True)
        self.cv_thread.start()

    def _cv_worker_loop(self):
        logger.info("[Vision Worker] Probing camera device...")
        if not self.camera.start():
            def _report_cam_error():
                self.camera_label.configure(
                    text="⚠️ Camera Unavailable\n\nPlease ensure your webcam is connected and not in use by another application."
                )
                self.badge_status.configure(text="CAMERA ERROR", fg_color=config.COLOR_DANGER)
            self.after(0, _report_cam_error)
            return

        logger.info("[Vision Worker] Initializing HandTracker...")
        try:
            self.hand_tracker = HandTracker(max_hands=config.MAX_NUM_HANDS)
            self.after(0, lambda: self.badge_status.configure(text="TRACKING ACTIVE", fg_color=config.COLOR_SUCCESS))
        except Exception as e:
            logger.error(f"HandTracker error: {e}")
            self.after(0, lambda: self.badge_status.configure(text="TRACKER ERROR", fg_color=config.COLOR_DANGER))
            return

        while self.is_running:
            perf_tracker.start_frame()
            ret, frame = self.camera.read(timeout=0.2)
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            h, w = frame.shape[:2]

            # 1. Multi-Hand Detection
            hands, annotated = self.hand_tracker.find_hands(frame, draw=self.show_skeleton)

            # Auto-calibrate live frame processing
            if self.calib_view.is_auto_calibrating and hands:
                self.calib_view.process_auto_calib_frame(hands[0])

            # 2. Feature Extraction
            features_list = [self.feature_extractor.extract_single_hand(h_data) for h_data in hands]
            dual_features = self.feature_extractor.extract_dual_hands(hands) if len(hands) >= 2 else None

            # Custom Gesture Recording session
            if self.classifier.custom_manager.is_recording and features_list:
                self.classifier.custom_manager.record_frame_features(features_list[0].to_ml_vector())

            # 3. Gesture Classification & 7-State Machine Processing
            res: GestureResult = self.classifier.process(hands, features_list, dual_features)

            # Measure performance latency
            lat = perf_tracker.end_frame()
            if res.gesture_id not in ("NONE", "IDLE"):
                perf_tracker.record_gesture_event(res.gesture_id, is_correct=True, latency_ms=lat)

            # 4. Action Dispatching (Only when state machine confirms execution)
            if self.is_tracking_active:
                self._dispatch_action(res, lat)

            # Empirical testing frame evaluation
            self.testing_view.evaluate_live_frame(res.gesture_id, res.confidence, lat)

            # 5. Render Sci-Fi HUD & Toast Overlays
            if self.show_hud:
                self._draw_hud_overlays(annotated, res)

            # Send frame to UI Queue
            display_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            try:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait((display_rgb, res))
            except queue.Full:
                pass

        if self.hand_tracker is not None:
            self.hand_tracker.close()
        self.camera.stop()
        logger.info("[Vision Worker] Shutdown complete.")

    def _dispatch_action(self, res: GestureResult, lat: float):
        """Routes confirmed gestures into OS actions."""
        action = res.assigned_action
        if not action:
            return

        now = time.time()

        # ---------------- MOUSE ACTIONS ----------------
        if action == "MOVE_CURSOR":
            if res.smoothed_point is not None:
                norm_x = res.smoothed_point[0] / config.FRAME_WIDTH
                norm_y = res.smoothed_point[1] / config.FRAME_HEIGHT
                self.mouse_ctrl.move_cursor_to_norm(norm_x, norm_y)

        elif action == "HOVER":
            self.mouse_ctrl.hover()

        elif action == "LEFT_CLICK" and (res.is_discrete_trigger or res.can_execute):
            self.mouse_ctrl.left_click()
            self.show_toast("✓ LEFT CLICK EXECUTED", config.COLOR_SUCCESS)

        elif action == "DOUBLE_CLICK" and (res.is_discrete_trigger or res.can_execute):
            self.mouse_ctrl.double_click()
            self.show_toast("✓ DOUBLE CLICK EXECUTED", config.COLOR_SUCCESS)

        elif action == "RIGHT_CLICK" and (res.is_discrete_trigger or res.can_execute):
            self.mouse_ctrl.right_click()
            self.show_toast("✓ RIGHT CLICK EXECUTED", config.COLOR_CYAN)

        elif action == "START_DRAG":
            self.mouse_ctrl.start_drag()
            if res.smoothed_point is not None:
                norm_x = res.smoothed_point[0] / config.FRAME_WIDTH
                norm_y = res.smoothed_point[1] / config.FRAME_HEIGHT
                self.mouse_ctrl.move_cursor_to_norm(norm_x, norm_y)

        elif action in ("RELEASE_DRAG", "RELEASE_DRAG_OR_STOP", "HALT_SCROLL"):
            self.mouse_ctrl.stop_drag()
            if action == "HALT_SCROLL":
                self.show_toast("✋ SCROLL HALTED", config.COLOR_WARNING)

        # ---------------- SCROLL ACTIONS ----------------
        elif action == "SCROLL_UP":
            speed = int(min(config.MAX_SCROLL_SPEED, max(4, abs(res.delta_y) * 2.8)))
            self.mouse_ctrl.scroll_vertical(speed)

        elif action == "SCROLL_DOWN":
            speed = int(min(config.MAX_SCROLL_SPEED, max(4, abs(res.delta_y) * 2.8)))
            self.mouse_ctrl.scroll_vertical(-speed)

        elif action == "SCROLL_VERTICAL":
            if res.delta_y < -1.0:
                self.mouse_ctrl.scroll_vertical(int(min(config.MAX_SCROLL_SPEED, abs(res.delta_y) * 2.5)))
            elif res.delta_y > 1.0:
                self.mouse_ctrl.scroll_vertical(-int(min(config.MAX_SCROLL_SPEED, abs(res.delta_y) * 2.5)))

        elif action == "PAGE_UP" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.page_up()
            self.show_toast("✓ PAGE UP", config.COLOR_CYAN)

        elif action == "PAGE_DOWN" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.page_down()
            self.show_toast("✓ PAGE DOWN", config.COLOR_CYAN)

        elif action == "BROWSER_BACK" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.browser_back()
            self.show_toast("✓ BROWSER BACK", config.COLOR_PURPLE)

        elif action == "BROWSER_FORWARD" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.browser_forward()
            self.show_toast("✓ BROWSER FORWARD", config.COLOR_PURPLE)

        # ---------------- MEDIA ACTIONS ----------------
        elif action == "MEDIA_PLAY_PAUSE" and (res.is_discrete_trigger or res.can_execute):
            self.media_ctrl.play_pause()
            self.show_toast("✓ PLAY / PAUSE", config.COLOR_PURPLE)

        elif action == "MEDIA_NEXT_TRACK" and (res.is_discrete_trigger or res.can_execute):
            self.media_ctrl.next_track()
            self.show_toast("✓ NEXT TRACK", config.COLOR_PURPLE)

        elif action == "MEDIA_PREV_TRACK" and (res.is_discrete_trigger or res.can_execute):
            self.media_ctrl.prev_track()
            self.show_toast("✓ PREVIOUS TRACK", config.COLOR_PURPLE)

        elif action == "VOLUME_UP":
            if now - getattr(self, "_last_vol_time", 0.0) >= 0.14:
                self._last_vol_time = now
                self.media_ctrl.volume_up()
                self.show_toast("🔊 VOLUME UP", config.COLOR_SUCCESS, duration=0.8)

        elif action == "VOLUME_DOWN":
            if now - getattr(self, "_last_vol_time", 0.0) >= 0.14:
                self._last_vol_time = now
                self.media_ctrl.volume_down()
                self.show_toast("🔉 VOLUME DOWN", config.COLOR_SUCCESS, duration=0.8)

        elif action == "VOLUME_MUTE" and (res.is_discrete_trigger or res.can_execute):
            self.media_ctrl.toggle_mute()
            self.show_toast("✓ MUTE TOGGLE", config.COLOR_WARNING)

        # ---------------- SYSTEM ACTIONS ----------------
        elif action == "TAKE_SCREENSHOT" and (res.is_discrete_trigger or res.can_execute):
            self.system_ctrl.take_screenshot()
            self.show_toast("✓ SCREENSHOT SAVED", config.COLOR_SUCCESS, duration=2.0)

        elif action == "APP_SWITCH" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.app_switch()
            self.show_toast("✓ APP SWITCHER (Alt+Tab)", config.COLOR_ACCENT)

        elif action == "TASK_VIEW" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.task_view()
            self.show_toast("✓ TASK VIEW (Win+Tab)", config.COLOR_ACCENT)

        elif action == "SHOW_DESKTOP" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.show_desktop()
            self.show_toast("✓ SHOW DESKTOP (Win+D)", config.COLOR_ACCENT)

        elif action == "WINDOW_MINIMIZE" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.window_minimize()
            self.show_toast("✓ WINDOW MINIMIZE", config.COLOR_ACCENT)

        elif action == "WINDOW_MAXIMIZE" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.window_maximize()
            self.show_toast("✓ WINDOW MAXIMIZE", config.COLOR_ACCENT)

        elif action == "WINDOW_CLOSE_SAFE" and (res.is_discrete_trigger or res.can_execute):
            self.system_ctrl.close_window_safe()
            self.show_toast("✓ WINDOW CLOSED (Alt+F4)", config.COLOR_DANGER, duration=2.0)

        elif action == "LOCK_SCREEN_SAFE" and (res.is_discrete_trigger or res.can_execute):
            self.system_ctrl.lock_screen_safe()
            self.show_toast("✓ WORKSTATION LOCKED", config.COLOR_WARNING, duration=2.0)

        # ---------------- KEYBOARD ACTIONS ----------------
        elif action == "COPY" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.copy()
            self.show_toast("✓ COPIED (Ctrl+C)", config.COLOR_SUCCESS)

        elif action == "PASTE" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.paste()
            self.show_toast("✓ PASTED (Ctrl+V)", config.COLOR_SUCCESS)

        elif action == "CUT" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.cut()
            self.show_toast("✓ CUT (Ctrl+X)", config.COLOR_SUCCESS)

        elif action == "SELECT_ALL" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.select_all()
            self.show_toast("✓ SELECT ALL (Ctrl+A)", config.COLOR_CYAN)

        elif action == "UNDO" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.undo()
            self.show_toast("✓ UNDO (Ctrl+Z)", config.COLOR_CYAN)

        elif action == "REDO" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.redo()
            self.show_toast("✓ REDO (Ctrl+Y)", config.COLOR_CYAN)

        elif action == "KEY_ENTER" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.enter()
            self.show_toast("✓ ENTER KEY", config.COLOR_SUCCESS)

        elif action == "KEY_ESCAPE" and (res.is_discrete_trigger or res.can_execute):
            self.keyboard_ctrl.escape()
            self.show_toast("✓ ESCAPE KEY", config.COLOR_WARNING)

        # ---------------- ZOOM & DISPLAY ----------------
        elif action == "ZOOM_IN":
            if now - getattr(self, "_last_zoom_time", 0.0) >= 0.25:
                self._last_zoom_time = now
                self.display_ctrl.zoom_in()
                self.show_toast("✓ ZOOM IN", config.COLOR_CYAN, duration=0.8)

        elif action == "ZOOM_OUT":
            if now - getattr(self, "_last_zoom_time", 0.0) >= 0.25:
                self._last_zoom_time = now
                self.display_ctrl.zoom_out()
                self.show_toast("✓ ZOOM OUT", config.COLOR_CYAN, duration=0.8)

        elif action == "ROTATE_DOCUMENT" and (res.is_discrete_trigger or res.can_execute):
            self.display_ctrl.rotate_document()
            self.show_toast("✓ ROTATE DOCUMENT", config.COLOR_PURPLE)

        elif action == "BRIGHTNESS_INCREASE":
            if now - getattr(self, "_last_bright_time", 0.0) >= 0.15:
                self._last_bright_time = now
                self.display_ctrl.brightness_up()
                self.show_toast("☀ BRIGHTNESS UP", config.COLOR_WARNING, duration=0.8)

        elif action == "BRIGHTNESS_DECREASE":
            if now - getattr(self, "_last_bright_time", 0.0) >= 0.15:
                self._last_bright_time = now
                self.display_ctrl.brightness_down()
                self.show_toast("☀ BRIGHTNESS DOWN", config.COLOR_WARNING, duration=0.8)

    def _draw_hud_overlays(self, frame: np.ndarray, res: GestureResult):
        """Renders target crosshairs, active box, and live toast notification banners."""
        h, w = frame.shape[:2]

        # 1. Active Movement Area Box
        x_min, y_min, x_max, y_max = self.active_box
        bx1, by1 = int(x_min * w), int(y_min * h)
        bx2, by2 = int(x_max * w), int(y_max * h)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), config.CV_COLOR_ACTIVE_ZONE, 1, cv2.LINE_AA)

        # 2. Target Cursor Crosshair on Smoothed Index Tip
        if res.smoothed_point is not None:
            sx, sy = res.smoothed_point
            cv2.circle(frame, (sx, sy), 14, config.CV_COLOR_INDEX_TIP, 2, cv2.LINE_AA)
            cv2.drawMarker(frame, (sx, sy), (255, 255, 255), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)

        # 3. Live Action Toast Notification Overlay Banner
        if self.toast_notification is not None:
            msg, expire_time, hex_col = self.toast_notification
            if time.time() < expire_time:
                # Draw semi-transparent pill banner at top
                overlay = frame.copy()
                cv2.rectangle(overlay, (w // 2 - 160, 16), (w // 2 + 160, 56), (15, 20, 30), -1)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                cv2.rectangle(frame, (w // 2 - 160, 16), (w // 2 + 160, 56), (0, 220, 120), 2)
                cv2.putText(frame, msg, (w // 2 - 145, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                self.toast_notification = None

    # -------------------------------------------------------------
    # GUI REFRESH LOOP (RUNS ON MAIN THREAD)
    # -------------------------------------------------------------
    def _gui_refresh_loop(self):
        if not self.is_running:
            return

        try:
            display_rgb, res = self.frame_queue.get_nowait()

            pil_img = Image.fromarray(display_rgb)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(pil_img.width, pil_img.height))
            self.camera_label.configure(image=ctk_img, text="")
            self.camera_label.image = ctk_img

            # Update Telemetry Badges
            self.badge_fps.configure(text=f"FPS: {perf_tracker.fps:.1f}")
            self.badge_lat.configure(text=f"LAT: {perf_tracker.current_latency_ms:.0f} ms")

            # Update HUD Ribbon
            self.hud_gesture_lbl.configure(text=f"Gesture: {res.display_label}")
            self.hud_state_lbl.configure(text=res.status_text, text_color=res.status_color)
            self.hud_conf_lbl.configure(text=f"Conf: {int(res.confidence * 100)}%")

            # Update Studio progress
            self.studio_view.update_recording_progress()

            # Update Performance numbers
            self.perf_view.update_metrics()

        except queue.Empty:
            pass

        self.after(25, self._gui_refresh_loop)

    def _on_closing(self):
        logger.info("[App] Window close requested. Releasing resources...")
        self.is_running = False
        action_dispatcher.stop()
        self.destroy()
