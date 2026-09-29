"""
Desktop Graphical User Interface for Touchless Screen Scrolling System.
Built using CustomTkinter with modern dark-mode aesthetic, real-time webcam preview,
live telemetry HUD, parameter adjustment sliders, and safety overrides.
"""

import math
import queue
import threading
import time
from typing import Optional

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

import config
from gesture_recognition import GestureRecognizer, GestureResult, GestureType
from hand_detector import HandDetector, HandLandmarkData
from scroll_controller import ScrollController

# Set global appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class TouchlessScrollApp(ctk.CTk):
    """
    Main Desktop Application Window.
    Coordinates Camera capture thread, CV pipeline, and responsive UI dashboard.
    """

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title(config.APP_TITLE)
        self.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.minsize(980, 680)
        self.configure(fg_color=config.COLOR_BG_DARK)

        # Core CV and Controller instances
        self.hand_detector: Optional[HandDetector] = None  # Initialized in camera worker thread
        self.gesture_recognizer = GestureRecognizer()
        self.scroll_controller = ScrollController()

        # Camera & State variables
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera_running = False
        self.is_tracking_active = True
        self.show_skeleton = True
        self.show_hud = True

        # Performance metrics
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()

        # Threading communication
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.capture_thread: Optional[threading.Thread] = None
        self.app_running = True

        # Build UI layout
        self._create_layout()

        # Keybindings
        self.bind("<space>", lambda event: self._toggle_tracking())
        self.bind("<Escape>", lambda event: self._emergency_stop())

        # Protocol for clean window exit
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Start camera capture
        self._start_camera()

        # Start GUI update cycle
        self.after(20, self._gui_update_loop)

    def _create_layout(self):
        """Constructs the modern 2-column dashboard layout."""
        # Top Header Bar
        self.header_frame = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=10, height=65)
        self.header_frame.pack(fill="x", padx=16, pady=(12, 8))
        self.header_frame.pack_propagate(False)

        # Title & Subtitle
        title_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_box.pack(side="left", padx=16, pady=8)

        self.title_label = ctk.CTkLabel(
            title_box,
            text="Touchless Screen Scrolling",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        self.title_label.pack(anchor="w")

        self.subtitle_label = ctk.CTkLabel(
            title_box,
            text="AI Real-Time Hand Gesture Navigation System",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=config.COLOR_CYAN
        )
        self.subtitle_label.pack(anchor="w")

        # Top Right Telemetry Badges
        telemetry_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        telemetry_box.pack(side="right", padx=16, pady=10)

        self.fps_badge = ctk.CTkLabel(
            telemetry_box,
            text="FPS: --",
            font=ctk.CTkFont(family="Consolas", size=13, weight="bold"),
            fg_color="#262A3B",
            corner_radius=6,
            text_color="#E0E7FF",
            width=85,
            height=32
        )
        self.fps_badge.pack(side="right", padx=6)

        self.lighting_badge = ctk.CTkLabel(
            telemetry_box,
            text="Light: --",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            fg_color="#262A3B",
            corner_radius=6,
            text_color="#E0E7FF",
            width=110,
            height=32
        )
        self.lighting_badge.pack(side="right", padx=6)

        self.status_badge = ctk.CTkLabel(
            telemetry_box,
            text="TRACKING ACTIVE",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=config.COLOR_SUCCESS,
            corner_radius=6,
            text_color="#FFFFFF",
            width=140,
            height=32
        )
        self.status_badge.pack(side="right", padx=6)

        # Main Body Container (2 Columns)
        self.body_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.body_frame.pack(fill="both", expand=True, padx=16, pady=8)

        # Column 1: Camera Preview & Dynamic HUD (Left, takes ~60% width)
        self.viewport_frame = ctk.CTkFrame(self.body_frame, fg_color=config.COLOR_CARD_BG, corner_radius=12)
        self.viewport_frame.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=0)

        # Camera Display Canvas/Label
        self.camera_label = ctk.CTkLabel(
            self.viewport_frame,
            text="Initializing camera...",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            fg_color="#0A0B10",
            corner_radius=10
        )
        self.camera_label.pack(fill="both", expand=True, padx=12, pady=12)

        # Live Gesture & Direction HUD Ribbon
        self.hud_ribbon = ctk.CTkFrame(self.viewport_frame, fg_color="#171824", corner_radius=10, height=54)
        self.hud_ribbon.pack(fill="x", padx=12, pady=(0, 12))
        self.hud_ribbon.pack_propagate(False)

        self.hud_gesture_label = ctk.CTkLabel(
            self.hud_ribbon,
            text="Gesture: Detecting...",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        self.hud_gesture_label.pack(side="left", padx=16)

        self.hud_direction_label = ctk.CTkLabel(
            self.hud_ribbon,
            text="Scroll: --",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=config.COLOR_TEXT_SECONDARY
        )
        self.hud_direction_label.pack(side="right", padx=16)

        # Column 2: Controls & Guide Panel (Right, takes ~40% width)
        self.panel_frame = ctk.CTkScrollableFrame(
            self.body_frame,
            fg_color=config.COLOR_CARD_BG,
            corner_radius=12,
            width=360
        )
        self.panel_frame.pack(side="right", fill="both", expand=False, padx=(0, 0), pady=0)

        self._build_control_cards()

    def _build_control_cards(self):
        """Builds interactive cards inside the right control panel."""
        # -------------------------------------------------------------
        # CARD 1: PRIMARY ACTION BUTTONS
        # -------------------------------------------------------------
        card1 = ctk.CTkFrame(self.panel_frame, fg_color="#242738", corner_radius=10)
        card1.pack(fill="x", padx=4, pady=(4, 10))

        c1_title = ctk.CTkLabel(
            card1,
            text="System Controls",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        c1_title.pack(anchor="w", padx=12, pady=(10, 8))

        # Start / Stop Tracking Toggle
        self.btn_toggle_tracking = ctk.CTkButton(
            card1,
            text="PAUSE TRACKING (Space)",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=config.COLOR_ACCENT,
            hover_color=config.COLOR_ACCENT_HOVER,
            height=38,
            command=self._toggle_tracking
        )
        self.btn_toggle_tracking.pack(fill="x", padx=12, pady=4)

        # Emergency Stop Button
        self.btn_emergency_stop = ctk.CTkButton(
            card1,
            text="EMERGENCY STOP (Esc)",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=config.COLOR_DANGER,
            hover_color="#DC2626",
            height=38,
            command=self._emergency_stop
        )
        self.btn_emergency_stop.pack(fill="x", padx=12, pady=4)

        # Camera On/Off Toggle
        self.btn_toggle_camera = ctk.CTkButton(
            card1,
            text="Disable Camera",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#374151",
            hover_color="#4B5563",
            height=32,
            command=self._toggle_camera
        )
        self.btn_toggle_camera.pack(fill="x", padx=12, pady=(4, 12))

        # -------------------------------------------------------------
        # CARD 2: REAL-TIME SENSITIVITY & TUNING
        # -------------------------------------------------------------
        card2 = ctk.CTkFrame(self.panel_frame, fg_color="#242738", corner_radius=10)
        card2.pack(fill="x", padx=4, pady=8)

        c2_title = ctk.CTkLabel(
            card2,
            text="Sensitivity & Thresholds",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        c2_title.pack(anchor="w", padx=12, pady=(10, 6))

        # 1. Scroll Sensitivity Slider
        self.lbl_sens_title = ctk.CTkLabel(
            card2,
            text=f"Scroll Sensitivity: {config.SCROLL_SENSITIVITY_DEFAULT:.1f}",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        self.lbl_sens_title.pack(anchor="w", padx=12, pady=(4, 0))

        self.slider_sens = ctk.CTkSlider(
            card2,
            from_=config.SCROLL_SENSITIVITY_MIN,
            to=config.SCROLL_SENSITIVITY_MAX,
            number_of_steps=38,
            command=self._on_sensitivity_change
        )
        self.slider_sens.set(config.SCROLL_SENSITIVITY_DEFAULT)
        self.slider_sens.pack(fill="x", padx=12, pady=(2, 8))

        # 2. Deadzone Threshold Slider
        self.lbl_deadzone_title = ctk.CTkLabel(
            card2,
            text=f"Jitter Deadzone: {config.DEADZONE_PIXELS_DEFAULT:.0f} px",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        self.lbl_deadzone_title.pack(anchor="w", padx=12, pady=(4, 0))

        self.slider_deadzone = ctk.CTkSlider(
            card2,
            from_=config.DEADZONE_PIXELS_MIN,
            to=config.DEADZONE_PIXELS_MAX,
            number_of_steps=23,
            command=self._on_deadzone_change
        )
        self.slider_deadzone.set(config.DEADZONE_PIXELS_DEFAULT)
        self.slider_deadzone.pack(fill="x", padx=12, pady=(2, 8))

        # 3. Landmark Smoothing Factor Slider (EMA)
        self.lbl_smoothing_title = ctk.CTkLabel(
            card2,
            text=f"EMA Smoothing Alpha: {config.SMOOTHING_ALPHA_DEFAULT:.2f}",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        self.lbl_smoothing_title.pack(anchor="w", padx=12, pady=(4, 0))

        self.slider_smoothing = ctk.CTkSlider(
            card2,
            from_=0.10,
            to=0.95,
            number_of_steps=17,
            command=self._on_smoothing_change
        )
        self.slider_smoothing.set(config.SMOOTHING_ALPHA_DEFAULT)
        self.slider_smoothing.pack(fill="x", padx=12, pady=(2, 8))

        # 4. Feature Toggles
        self.switch_click = ctk.CTkSwitch(
            card2,
            text="Pinch-to-Click Enabled",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=self._on_click_toggle
        )
        self.switch_click.select()
        self.switch_click.pack(anchor="w", padx=12, pady=6)

        self.switch_skeleton = ctk.CTkSwitch(
            card2,
            text="Draw Landmark Skeleton",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=self._on_skeleton_toggle
        )
        self.switch_skeleton.select()
        self.switch_skeleton.pack(anchor="w", padx=12, pady=(6, 12))

        # -------------------------------------------------------------
        # CARD 3: GESTURE CHEAT SHEET
        # -------------------------------------------------------------
        card3 = ctk.CTkFrame(self.panel_frame, fg_color="#242738", corner_radius=10)
        card3.pack(fill="x", padx=4, pady=(8, 12))

        c3_title = ctk.CTkLabel(
            card3,
            text="Gesture Reference Guide",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        c3_title.pack(anchor="w", padx=12, pady=(10, 6))

        gestures = [
            ("☝️  1 Finger Up + Move Up", "Scroll Screen UP", config.COLOR_SUCCESS),
            ("👇  1 Finger Down + Move Down", "Scroll Screen DOWN", config.COLOR_SUCCESS),
            ("✌️  2 Fingers Raised", "Arm Scrolling Mode", config.COLOR_CYAN),
            ("✋  Open Palm (5 Fingers)", "Instantly Halt Scrolling", config.COLOR_WARNING),
            ("🤏  Pinch (Thumb + Index)", "Single Mouse Click", config.COLOR_ACCENT),
        ]

        for gesture_txt, action_txt, col in gestures:
            g_row = ctk.CTkFrame(card3, fg_color="#1A1C29", corner_radius=6)
            g_row.pack(fill="x", padx=12, pady=3)

            g_lbl = ctk.CTkLabel(
                g_row,
                text=gesture_txt,
                font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                text_color=config.COLOR_TEXT_PRIMARY
            )
            g_lbl.pack(side="left", padx=8, pady=4)

            a_lbl = ctk.CTkLabel(
                g_row,
                text=action_txt,
                font=ctk.CTkFont(family="Segoe UI", size=11),
                text_color=col
            )
            a_lbl.pack(side="right", padx=8, pady=4)

        # Padding at bottom of card
        ctk.CTkFrame(card3, fg_color="transparent", height=6).pack()

    # -------------------------------------------------------------
    # EVENT HANDLERS & CALLBACKS
    # -------------------------------------------------------------
    def _toggle_tracking(self):
        """Toggles gesture tracking and scrolling active state."""
        active = self.scroll_controller.toggle_active()
        self.is_tracking_active = active
        if active:
            self.status_badge.configure(text="TRACKING ACTIVE", fg_color=config.COLOR_SUCCESS)
            self.btn_toggle_tracking.configure(text="PAUSE TRACKING (Space)", fg_color=config.COLOR_ACCENT)
        else:
            self.status_badge.configure(text="TRACKING PAUSED", fg_color=config.COLOR_WARNING)
            self.btn_toggle_tracking.configure(text="RESUME TRACKING (Space)", fg_color="#D97706")

    def _emergency_stop(self):
        """Immediately halts all actions via emergency override."""
        self.scroll_controller.emergency_stop()
        self.is_tracking_active = False
        self.status_badge.configure(text="EMERGENCY STOP", fg_color=config.COLOR_DANGER)
        self.btn_toggle_tracking.configure(text="RESUME TRACKING (Space)", fg_color=config.COLOR_ACCENT)
        self.hud_direction_label.configure(text="Scroll: HALTED (Emergency)", text_color=config.COLOR_DANGER)

    def _toggle_camera(self):
        """Enables or disables camera capture."""
        if self.is_camera_running:
            self.is_camera_running = False
            self.btn_toggle_camera.configure(text="Enable Camera")
            self.camera_label.configure(image="", text="Camera Disabled\nClick 'Enable Camera' to resume")
        else:
            self._start_camera()
            self.btn_toggle_camera.configure(text="Disable Camera")

    def _on_sensitivity_change(self, value):
        self.scroll_controller.set_sensitivity(value)
        self.lbl_sens_title.configure(text=f"Scroll Sensitivity: {value:.1f}")

    def _on_deadzone_change(self, value):
        self.deadzone_value = value
        self.lbl_deadzone_title.configure(text=f"Jitter Deadzone: {value:.0f} px")

    def _on_smoothing_change(self, value):
        self.gesture_recognizer.update_parameters(smoothing_alpha=value)
        self.lbl_smoothing_title.configure(text=f"EMA Smoothing Alpha: {value:.2f}")

    def _on_click_toggle(self):
        enabled = bool(self.switch_click.get())
        self.scroll_controller.set_click_enabled(enabled)

    def _on_skeleton_toggle(self):
        self.show_skeleton = bool(self.switch_skeleton.get())

    # -------------------------------------------------------------
    # CAMERA CAPTURE WORKER & PIPELINE
    # -------------------------------------------------------------
    def _start_camera(self):
        """Initializes OpenCV video capture in a background thread."""
        if self.is_camera_running:
            return

        self.is_camera_running = True
        self.capture_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.capture_thread.start()

    def _camera_worker(self):
        """Worker thread dedicated to acquiring frames, running CV, and updating state."""
        # Initialize HandDetector directly on the worker thread for C++ thread-safety
        print("[Camera Worker] Initializing HandDetector on camera thread...")
        try:
            self.hand_detector = HandDetector()
        except Exception as e:
            print(f"[Camera Worker] HandDetector initialization error: {e}")
            self.camera_label.configure(
                text=f"⚠️ Hand Detector Error:\n{e}\n\nPlease check your internet connection to download the model.",
                image=""
            )
            self.status_badge.configure(text="DETECTOR ERROR", fg_color=config.COLOR_DANGER)
            self.is_camera_running = False
            return

        print("[Camera Worker] Acquiring functional webcam device...")
        self.cap, actual_idx = config.open_camera_device(config.CAMERA_INDEX)

        if self.cap is None or not self.cap.isOpened():
            print("[GUI] Error: Unable to open webcam.")
            self.camera_label.configure(
                text="⚠️ Camera Unavailable\n\n"
                     "• Check if another app (Zoom, Teams, Camera) is using the webcam.\n"
                     "• Verify Camera permissions in Windows Settings > Privacy > Camera.\n"
                     "• If using an external webcam, change CAMERA_INDEX in config.py.",
                image=""
            )
            self.status_badge.configure(text="CAMERA ERROR", fg_color=config.COLOR_DANGER)
            self.is_camera_running = False
            return

        deadzone = config.DEADZONE_PIXELS_DEFAULT

        while self.app_running and self.is_camera_running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Flip horizontally for natural mirror preview
            frame = cv2.flip(frame, 1)

            # Measure FPS
            self.frame_count += 1
            now = time.time()
            elapsed = now - self.last_fps_time
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.last_fps_time = now

            # Check lighting condition & physical shutter status
            mean_brightness = float(np.mean(frame))
            is_dark = mean_brightness < 40.0

            # Step 1: Hand Landmark Detection
            hand_data, annotated_frame = self.hand_detector.find_hands(frame, draw=self.show_skeleton)

            # If camera view is pitch black / shutter is closed, display prominent advisory overlay
            if is_dark:
                h, w = annotated_frame.shape[:2]
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (20, 20), (w - 20, 155), (10, 10, 30), -1)
                cv2.addWeighted(overlay, 0.85, annotated_frame, 0.15, 0, annotated_frame)
                cv2.rectangle(annotated_frame, (20, 20), (w - 20, 155), (0, 165, 255), 2)

                cv2.putText(annotated_frame, f"WARNING: CAMERA FEED IS DARK ({int(mean_brightness)}/255)",
                            (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, "1. Slide OPEN laptop webcam physical privacy shutter/slider",
                            (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, "2. Check laptop camera privacy hotkey (e.g. F8 / F10 / Fn+F10)",
                            (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, "3. Turn on room lighting and hold hand 1-2 feet from camera",
                            (35, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 240, 180), 1, cv2.LINE_AA)

            # Step 2: Gesture Recognition & Coordinate Smoothing
            current_deadzone = getattr(self, "deadzone_value", deadzone)
            gesture_res = self.gesture_recognizer.process(hand_data, deadzone_pixels=current_deadzone)

            # Step 3: Draw smoothed fingertip target on annotated frame
            if gesture_res.smoothed_tip is not None and self.show_skeleton:
                sx, sy = gesture_res.smoothed_tip
                cv2.circle(annotated_frame, (sx, sy), 12, (255, 230, 0), 2, cv2.LINE_AA)
                cv2.drawMarker(annotated_frame, (sx, sy), (255, 255, 255),
                               cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)

            # Step 4: Dispatch to Scroll Controller
            self.scroll_controller.process_gesture(gesture_res)

            # Push processed frame & telemetry to UI Queue
            display_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            try:
                # Discard stale frames if UI hasn't picked them up yet
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait((display_rgb, gesture_res, self.fps, hand_data is not None, mean_brightness))
            except queue.Full:
                pass

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.hand_detector is not None:
            try:
                self.hand_detector.close()
            except Exception:
                pass
            self.hand_detector = None
        print("[Camera Worker] Camera and detector released cleanly.")

    # -------------------------------------------------------------
    # GUI REFRESH CYCLE (RUNS ON MAIN THREAD)
    # -------------------------------------------------------------
    def _gui_update_loop(self):
        """Pulls latest processed frame from queue and updates GUI components."""
        if not self.app_running:
            return

        try:
            display_rgb, gesture_res, current_fps, hand_detected, mean_brightness = self.frame_queue.get_nowait()

            # Update FPS badge
            self.fps_badge.configure(text=f"FPS: {current_fps:.1f}")

            # Update Lighting badge
            if mean_brightness < 40:
                self.lighting_badge.configure(text=f"Light: LOW ({int(mean_brightness)})", text_color="#F87171")
            else:
                self.lighting_badge.configure(text=f"Light: OK ({int(mean_brightness)})", text_color="#34D399")

            # Update Status badge
            if self.scroll_controller.is_emergency_stopped:
                self.status_badge.configure(text="EMERGENCY STOP", fg_color=config.COLOR_DANGER)
            elif not self.is_tracking_active:
                self.status_badge.configure(text="TRACKING PAUSED", fg_color=config.COLOR_WARNING)
            elif mean_brightness < 40:
                self.status_badge.configure(text="SHUTTER CLOSED?", fg_color=config.COLOR_DANGER)
            elif hand_detected:
                self.status_badge.configure(text="HAND DETECTED", fg_color=config.COLOR_SUCCESS)
            else:
                self.status_badge.configure(text="SEARCHING HAND", fg_color=config.COLOR_CYAN)

            # Update HUD Gesture Banner
            self.hud_gesture_label.configure(
                text=f"Gesture: {gesture_res.display_label}",
                text_color=gesture_res.status_color
            )

            # Update Direction Label
            direction = self.scroll_controller.current_direction
            if direction == "UP":
                self.hud_direction_label.configure(text="Scroll: ▲ UP", text_color=config.COLOR_SUCCESS)
            elif direction == "DOWN":
                self.hud_direction_label.configure(text="Scroll: ▼ DOWN", text_color=config.COLOR_SUCCESS)
            elif direction == "CLICK":
                self.hud_direction_label.configure(text="Scroll: ● CLICK", text_color=config.COLOR_ACCENT)
            elif not self.is_tracking_active:
                self.hud_direction_label.configure(text="Scroll: PAUSED", text_color=config.COLOR_WARNING)
            else:
                self.hud_direction_label.configure(text="Scroll: -- IDLE", text_color=config.COLOR_TEXT_SECONDARY)

            # Convert numpy array to Tkinter-compatible CTkImage
            h, w = display_rgb.shape[:2]
            # Responsive scaling for viewport
            viewport_w = self.viewport_frame.winfo_width()
            viewport_h = self.viewport_frame.winfo_height()
            target_w = max(420, viewport_w - 24) if viewport_w > 100 else 640
            target_h = int(target_w * (h / w))
            max_h = max(320, viewport_h - 80) if viewport_h > 100 else 480
            if target_h > max_h:
                target_h = max_h
                target_w = int(target_h * (w / h))

            target_w = max(320, int(target_w))
            target_h = max(240, int(target_h))

            pil_img = Image.fromarray(display_rgb)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
            self.camera_label.configure(image=ctk_img, text="")
            self.camera_label.image = ctk_img

        except queue.Empty:
            pass
        except Exception as e:
            # Print any unexpected rendering exception
            print(f"[GUI Frame Loop] Render notice: {e}")

        # Schedule next cycle
        if self.app_running:
            self.after(20, self._gui_update_loop)

    def _on_closing(self):
        """Performs clean shutdown of threads, hardware, and windows."""
        print("[GUI] Shutting down application...")
        self.app_running = False
        self.is_camera_running = False

        # Wait for camera thread to exit cleanly before destroying Tkinter
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

        if self.scroll_controller:
            self.scroll_controller.close()

        self.destroy()
        print("[GUI] Shutdown complete.")


def run_app():
    """Starts the application."""
    app = TouchlessScrollApp()
    app.mainloop()


if __name__ == "__main__":
    run_app()
