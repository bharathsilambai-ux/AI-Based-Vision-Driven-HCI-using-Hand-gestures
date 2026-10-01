"""
Desktop GUI Dashboard Module (Tkinter)
======================================
Provides an interactive desktop interface with live camera viewport,
runtime parameter tuning sliders, engine mode toggles, and event logging.
"""

import sys
import time
import queue
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import config

logger = logging.getLogger(__name__)


class ModernGUIDashboard:
    """Tkinter-based management dashboard for the Hand Gesture HCI system."""

    def __init__(self, root: tk.Tk, controller_pipeline):
        self.root = root
        self.pipeline = controller_pipeline
        self.root.title("AI-Based Vision HCI Controller — Dashboard")
        self.root.geometry("1100x720")
        self.root.minsize(980, 640)
        self.root.configure(bg="#12161f")

        # Runtime states
        self.is_running = True
        self.log_queue = queue.Queue()

        self._configure_styles()
        self._build_layout()
        self._bind_events()

    def _configure_styles(self):
        """Sets up dark-mode ttk styling."""
        style = ttk.Style()
        style.theme_use("clam")

        # General colors
        bg_dark = "#12161f"
        bg_card = "#1a2130"
        fg_white = "#f0f4fc"
        accent_blue = "#00b4d8"

        style.configure("TFrame", background=bg_dark)
        style.configure("Card.TFrame", background=bg_card, relief="flat")
        style.configure("TLabel", background=bg_dark, foreground=fg_white, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=bg_card, foreground=fg_white, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=bg_card, foreground="#00e5ff", font=("Segoe UI", 13, "bold"))
        style.configure("Header.TLabel", background=bg_dark, foreground="#ffffff", font=("Segoe UI", 16, "bold"))
        style.configure("SubHeader.TLabel", background=bg_dark, foreground="#8892b0", font=("Segoe UI", 10))

        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
            background="#00b4d8",
            foreground="#ffffff",
            borderwidth=0,
            padding=6,
        )
        style.map("Accent.TButton", background=[("active", "#0096c7"), ("disabled", "#334155")])

        style.configure(
            "Danger.TButton",
            font=("Segoe UI", 10, "bold"),
            background="#e63946",
            foreground="#ffffff",
            borderwidth=0,
            padding=6,
        )
        style.map("Danger.TButton", background=[("active", "#d90429"), ("disabled", "#334155")])

        style.configure("TScale", background=bg_card, troughcolor="#0f172a")

    def _build_layout(self):
        """Constructs layout split into Left (Camera) and Right (Controls & Stats)."""
        # Header banner
        header_frame = ttk.Frame(self.root, padding="15 12 15 8")
        header_frame.pack(fill=tk.X)

        title_lbl = ttk.Label(header_frame, text="AI VISION-DRIVEN HCI SYSTEM", style="Header.TLabel")
        title_lbl.pack(side=tk.LEFT)

        sub_lbl = ttk.Label(header_frame, text="Human-Computer Interaction via Hand Gesture Recognition", style="SubHeader.TLabel")
        sub_lbl.pack(side=tk.LEFT, padx=15, pady=4)

        # Main horizontal container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left Column: Video Viewport
        left_col = ttk.Frame(main_container)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_canvas = tk.Canvas(left_col, bg="#0a0d14", highlightthickness=1, highlightbackground="#2d3748")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom Bar under video: Quick Toggles
        btn_bar = ttk.Frame(left_col, padding="8 8 0 0")
        btn_bar.pack(fill=tk.X)

        self.btn_toggle_cam = ttk.Button(btn_bar, text="Pause System", style="Danger.TButton", command=self._toggle_running)
        self.btn_toggle_cam.pack(side=tk.LEFT, padx=(0, 8))

        self.var_control_enabled = tk.BooleanVar(value=True)
        self.chk_control = tk.Checkbutton(
            btn_bar,
            text="Enable OS Mouse/Key Control",
            variable=self.var_control_enabled,
            bg="#12161f",
            fg="#64dfdf",
            selectcolor="#1a2130",
            activebackground="#12161f",
            activeforeground="#64dfdf",
            font=("Segoe UI", 10, "bold"),
            command=self._on_control_toggled,
        )
        self.chk_control.pack(side=tk.LEFT, padx=10)

        # Right Column: Telemetry & Controls
        right_col = ttk.Frame(main_container, width=380)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right_col.pack_propagate(False)

        # 1. Telemetry Card
        telemetry_card = ttk.Frame(right_col, style="Card.TFrame", padding="14")
        telemetry_card.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(telemetry_card, text="LIVE RECOGNITION", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 8))

        # Gesture Display Badge
        self.lbl_gesture = tk.Label(
            telemetry_card,
            text="Waiting for hand...",
            font=("Segoe UI", 13, "bold"),
            bg="#242c3d",
            fg="#00e5ff",
            pady=8,
        )
        self.lbl_gesture.pack(fill=tk.X, pady=(0, 6))

        # Action Display
        act_row = ttk.Frame(telemetry_card, style="Card.TFrame")
        act_row.pack(fill=tk.X, pady=2)
        ttk.Label(act_row, text="Active Action:", style="Card.TLabel", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.lbl_action = ttk.Label(act_row, text="Ready", style="Card.TLabel", foreground="#52b788")
        self.lbl_action.pack(side=tk.RIGHT)

        # Confidence Bar
        conf_row = ttk.Frame(telemetry_card, style="Card.TFrame")
        conf_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(conf_row, text="Confidence:", style="Card.TLabel").pack(side=tk.LEFT)
        self.lbl_conf = ttk.Label(conf_row, text="0%", style="Card.TLabel", font=("Segoe UI", 9, "bold"))
        self.lbl_conf.pack(side=tk.RIGHT)

        self.prog_conf = ttk.Progressbar(telemetry_card, orient=tk.HORIZONTAL, mode="determinate")
        self.prog_conf.pack(fill=tk.X, pady=(2, 6))

        # FPS
        fps_row = ttk.Frame(telemetry_card, style="Card.TFrame")
        fps_row.pack(fill=tk.X, pady=2)
        ttk.Label(fps_row, text="Frame Rate:", style="Card.TLabel").pack(side=tk.LEFT)
        self.lbl_fps = ttk.Label(fps_row, text="0 FPS", style="Card.TLabel", foreground="#ffd166")
        self.lbl_fps.pack(side=tk.RIGHT)

        # 2. Engine Mode Selector Card
        mode_card = ttk.Frame(right_col, style="Card.TFrame", padding="14")
        mode_card.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(mode_card, text="CLASSIFIER ENGINE", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 8))

        self.var_engine_mode = tk.StringVar(value="rule")
        rb_rule = tk.Radiobutton(
            mode_card,
            text="Rule-Based Engine (Deterministic)",
            variable=self.var_engine_mode,
            value="rule",
            bg="#1a2130",
            fg="#f0f4fc",
            selectcolor="#0f172a",
            activebackground="#1a2130",
            activeforeground="#00b4d8",
            command=self._on_engine_toggled,
        )
        rb_rule.pack(anchor=tk.W, pady=2)

        rb_ml = tk.Radiobutton(
            mode_card,
            text="Machine Learning Model (Random Forest)",
            variable=self.var_engine_mode,
            value="ml",
            bg="#1a2130",
            fg="#f0f4fc",
            selectcolor="#0f172a",
            activebackground="#1a2130",
            activeforeground="#00b4d8",
            command=self._on_engine_toggled,
        )
        rb_ml.pack(anchor=tk.W, pady=2)

        # 3. Parameter Tuning Sliders
        tuning_card = ttk.Frame(right_col, style="Card.TFrame", padding="14")
        tuning_card.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(tuning_card, text="TUNING PARAMETERS", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 6))

        # Smoothing Alpha Slider
        ttk.Label(tuning_card, text="Cursor Smoothing Factor (Alpha):", style="Card.TLabel", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.scale_alpha = ttk.Scale(tuning_card, from_=0.10, to=0.85, value=config.SMOOTHING_ALPHA, command=self._on_alpha_changed)
        self.scale_alpha.pack(fill=tk.X, pady=(2, 6))

        # Pinch Threshold Slider
        ttk.Label(tuning_card, text="Pinch Sensitivity Distance:", style="Card.TLabel", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.scale_pinch = ttk.Scale(tuning_card, from_=0.03, to=0.10, value=config.PINCH_THRESHOLD, command=self._on_pinch_changed)
        self.scale_pinch.pack(fill=tk.X, pady=(2, 6))

        # 4. Event Log Console
        log_card = ttk.Frame(right_col, style="Card.TFrame", padding="12")
        log_card.pack(fill=tk.BOTH, expand=True)

        ttk.Label(log_card, text="ACTION EVENT LOG", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 4))

        self.txt_log = tk.Text(
            log_card,
            height=6,
            bg="#0d1117",
            fg="#94a3b8",
            insertbackground="#00e5ff",
            font=("Consolas", 8),
            relief="flat",
            wrap=tk.WORD,
        )
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        self.append_log("System dashboard initialized.")

    def _bind_events(self):
        """Binds close event."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _toggle_running(self):
        """Toggles video processing."""
        self.is_running = not self.is_running
        if self.is_running:
            self.btn_toggle_cam.configure(text="Pause System", style="Danger.TButton")
            self.append_log("Video tracking resumed.")
        else:
            self.btn_toggle_cam.configure(text="Resume System", style="Accent.TButton")
            self.append_log("Video tracking paused.")

    def _on_control_toggled(self):
        """Enables/disables mouse control."""
        state = self.var_control_enabled.get()
        self.pipeline.controller.set_enabled(state)
        self.append_log(f"OS Control: {'ENABLED' if state else 'DISABLED'}")

    def _on_engine_toggled(self):
        """Switches between Rule-based and ML classifier."""
        mode = self.var_engine_mode.get()
        if mode == "ml":
            if not self.pipeline.ml_classifier.is_ready:
                messagebox.showwarning(
                    "Model Not Found",
                    "ML model artifacts not found!\nPlease train the model first by running:\npython ml/train_model.py\n\nFalling back to Rule-Based engine."
                )
                self.var_engine_mode.set("rule")
                return
        self.pipeline.mode = "ml" if mode == "ml" else "rule"
        self.append_log(f"Classifier Engine set to: {self.pipeline.mode.upper()}")

    def _on_alpha_changed(self, val):
        alpha = float(val)
        self.pipeline.smoother.alpha = alpha

    def _on_pinch_changed(self, val):
        pinch = float(val)
        self.pipeline.rule_engine.pinch_threshold = pinch

    def append_log(self, message: str):
        """Thread-safe logging to GUI text console."""
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        self.txt_log.insert(tk.END, line)
        self.txt_log.see(tk.END)

    def update_frame(self, bgr_frame):
        """Updates canvas with latest processed frame."""
        if not self.is_running:
            return

        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        if canvas_w <= 10 or canvas_h <= 10:
            return

        # Resize keeping aspect ratio
        fh, fw = bgr_frame.shape[:2]
        ratio = min(canvas_w / fw, canvas_h / fh)
        nw, nh = int(fw * ratio), int(fh * ratio)

        resized = cv2.resize(bgr_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tk_img = ImageTk.PhotoImage(image=pil_img)

        self.video_canvas.delete("all")
        # Center image in canvas
        cx = (canvas_w - nw) // 2
        cy = (canvas_h - nh) // 2
        self.video_canvas.create_image(cx, cy, anchor=tk.NW, image=tk_img)
        self.video_canvas.image = tk_img  # Keep reference

    def update_telemetry(self, gesture: str, action: str, conf: float, fps: float):
        """Updates live labels and progress bars."""
        disp_gesture = config.GESTURE_DISPLAY_NAMES.get(gesture, gesture)
        self.lbl_gesture.config(text=disp_gesture)
        self.lbl_action.config(text=action)
        self.lbl_conf.config(text=f"{int(conf * 100)}%")
        self.prog_conf["value"] = int(conf * 100)
        self.lbl_fps.config(text=f"{fps:.0f} FPS")

    def on_close(self):
        """Handles application shutdown."""
        self.is_running = False
        self.pipeline.stop()
        self.root.destroy()
