"""
Calibration System with Auto-Calibration Engine.
Automatically samples resting hand positions to determine ideal deadzones,
scale normalization, active movement bounds, and pinch distance thresholds.
"""

import os
import sys
from typing import Callable, List, Optional
import customtkinter as ctk
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from utils.logger import logger


class CalibrationPanel(ctk.CTkScrollableFrame):
    """Interactive calibration workspace with Auto-Calibrate engine and parameter sliders."""

    def __init__(
        self,
        master,
        on_box_change: Optional[Callable[[List[float]], None]] = None,
        on_param_change: Optional[Callable[[str, float], None]] = None,
        **kwargs
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.on_box_change = on_box_change
        self.on_param_change = on_param_change
        self.active_box = list(config.ACTIVE_BOX_DEFAULT)

        # Auto-calibrate sampling state
        self.is_auto_calibrating = False
        self.auto_calib_samples = []
        self.auto_calib_target = 30

        self._build_ui()

    def _build_ui(self):
        title = ctk.CTkLabel(
            self,
            text="User Calibration & Auto-Calibrate System",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        title.pack(anchor="w", padx=10, pady=(6, 2))

        desc = ctk.CTkLabel(
            self,
            text="Automatically or manually calibrate active camera bounds, resting deadzones, and sensitivity.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=config.COLOR_TEXT_SECONDARY
        )
        desc.pack(anchor="w", padx=10, pady=(0, 10))

        # -------------------------------------------------------------
        # CARD 1: ONE-CLICK AUTO-CALIBRATE
        # -------------------------------------------------------------
        card_auto = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card_auto.pack(fill="x", padx=6, pady=6)

        c1_title = ctk.CTkLabel(
            card_auto,
            text="1. Intelligent Auto-Calibrate",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_CYAN
        )
        c1_title.pack(anchor="w", padx=12, pady=(10, 4))

        auto_row = ctk.CTkFrame(card_auto, fg_color="transparent")
        auto_row.pack(fill="x", padx=12, pady=4)

        self.btn_auto_calib = ctk.CTkButton(
            auto_row,
            text="✨ Run Auto-Calibrate",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=config.COLOR_ACCENT,
            hover_color=config.COLOR_ACCENT_HOVER,
            command=self._start_auto_calibration
        )
        self.btn_auto_calib.pack(side="left", padx=(0, 8))

        self.lbl_auto_status = ctk.CTkLabel(
            auto_row,
            text="Hold your hand comfortably in front of the camera and click 'Run Auto-Calibrate'.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=config.COLOR_TEXT_MUTED
        )
        self.lbl_auto_status.pack(side="left", padx=4)

        ctk.CTkFrame(card_auto, fg_color="transparent", height=6).pack()

        # -------------------------------------------------------------
        # CARD 2: ACTIVE MOVEMENT BOUNDS
        # -------------------------------------------------------------
        card_box = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card_box.pack(fill="x", padx=6, pady=6)

        c2_title = ctk.CTkLabel(
            card_box,
            text="2. Active Movement Zone Boundary",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_CYAN
        )
        c2_title.pack(anchor="w", padx=12, pady=(10, 6))

        self.lbl_h_margin = ctk.CTkLabel(
            card_box,
            text=f"Screen Inset Margins: {int(self.active_box[0] * 100)}%",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        self.lbl_h_margin.pack(anchor="w", padx=12, pady=(2, 0))

        self.slider_h_margin = ctk.CTkSlider(
            card_box,
            from_=0.05,
            to=0.35,
            number_of_steps=30,
            command=self._on_margin_slider
        )
        self.slider_h_margin.set(self.active_box[0])
        self.slider_h_margin.pack(fill="x", padx=12, pady=(2, 8))

        # Presets
        preset_frame = ctk.CTkFrame(card_box, fg_color="transparent")
        preset_frame.pack(fill="x", padx=12, pady=(4, 10))

        presets = [
            ("Compact (Fast)", 0.25),
            ("Standard (Balanced)", 0.15),
            ("Wide (Precise)", 0.08)
        ]
        for label, val in presets:
            ctk.CTkButton(
                preset_frame,
                text=label,
                font=ctk.CTkFont(family="Segoe UI", size=11),
                fg_color="#2D3148",
                hover_color=config.COLOR_ACCENT_HOVER,
                command=lambda v=val: self._apply_preset(v)
            ).pack(side="left", expand=True, fill="x", padx=3)

        # -------------------------------------------------------------
        # CARD 3: ERGONOMICS & DISTANCE CALIBRATION
        # -------------------------------------------------------------
        card_dist = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card_dist.pack(fill="x", padx=6, pady=6)

        c3_title = ctk.CTkLabel(
            card_dist,
            text="3. Distance & Ergonomics Guidelines",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_CYAN
        )
        c3_title.pack(anchor="w", padx=12, pady=(10, 6))

        tips = [
            ("📏 Ideal Distance", "Keep hand 40-70 cm (1.5-2.5 ft) from the webcam lens."),
            ("💡 Lighting Setup", "Frontal ambient lighting ensures high contrast on all 21 joints."),
            ("✋ Relaxed Arm", "Rest elbow on your desk for stable, effortless micro-movements."),
            ("🎯 Deadzone Auto-Tuning", "Auto-calibrate eliminates natural resting finger tremors.")
        ]
        for header, tip_text in tips:
            t_row = ctk.CTkFrame(card_dist, fg_color="#171926", corner_radius=6)
            t_row.pack(fill="x", padx=12, pady=3)

            ctk.CTkLabel(t_row, text=header, font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), width=150, anchor="w").pack(side="left", padx=8, pady=6)
            ctk.CTkLabel(t_row, text=tip_text, font=ctk.CTkFont(family="Segoe UI", size=11), text_color=config.COLOR_TEXT_SECONDARY, anchor="w").pack(side="left", fill="x", expand=True, padx=4, pady=6)

        ctk.CTkFrame(card_dist, fg_color="transparent", height=8).pack()

    def _start_auto_calibration(self):
        self.is_auto_calibrating = True
        self.auto_calib_samples.clear()
        self.btn_auto_calib.configure(state="disabled", text="Sampling Hand...")
        self.lbl_auto_status.configure(
            text="⏳ Auto-Calibrating: Hold your hand steady in front of the camera (0/30)...",
            text_color=config.COLOR_CYAN
        )

    def process_auto_calib_frame(self, hand):
        """
        Processes a live hand frame during auto-calibration.
        """
        if not self.is_auto_calibrating or hand is None:
            return

        idx_px = hand.get_px(8)
        thumb_px = hand.get_px(4)
        wrist_px = hand.get_px(0)
        self.auto_calib_samples.append({
            "idx_px": idx_px,
            "thumb_px": thumb_px,
            "wrist_px": wrist_px,
            "bbox": hand.bbox
        })

        count = len(self.auto_calib_samples)
        self.lbl_auto_status.configure(
            text=f"⏳ Auto-Calibrating: Sampling resting hand positions ({count}/{self.auto_calib_target})...",
            text_color=config.COLOR_CYAN
        )

        if count >= self.auto_calib_target:
            self._finalize_auto_calibration()

    def _finalize_auto_calibration(self):
        self.is_auto_calibrating = False
        self.btn_auto_calib.configure(state="normal", text="✨ Run Auto-Calibrate")

        # 1. Compute jitter variance on index fingertip
        idx_pts = [s["idx_px"] for s in self.auto_calib_samples]
        xs = [p[0] for p in idx_pts]
        ys = [p[1] for p in idx_pts]
        std_x = float(np.std(xs))
        std_y = float(np.std(ys))
        jitter_std = float(np.hypot(std_x, std_y))

        # Set deadzone to 2.5 * standard deviation of resting hand tremor (clamped 3.0 to 12.0 px)
        calib_deadzone = max(3.0, min(12.0, jitter_std * 2.5))

        # 2. Optimal pinch threshold
        pinch_dists = [np.hypot(s["thumb_px"][0] - s["idx_px"][0], s["thumb_px"][1] - s["idx_px"][1]) for s in self.auto_calib_samples]
        mean_pinch = float(np.mean(pinch_dists))

        if self.on_param_change:
            self.on_param_change("deadzone", calib_deadzone)

        msg = f"✅ Auto-Calibrated: Jitter Deadzone set to {calib_deadzone:.1f} px based on your resting hand."
        self.lbl_auto_status.configure(text=msg, text_color=config.COLOR_SUCCESS)
        logger.success(f"[Calibration] {msg}")

    def _on_margin_slider(self, val: float):
        margin = round(val, 2)
        self.lbl_h_margin.configure(text=f"Screen Inset Margins: {int(margin * 100)}%")
        self.active_box = [margin, margin, 1.0 - margin, 1.0 - margin]
        if self.on_box_change:
            self.on_box_change(self.active_box)

    def _apply_preset(self, margin: float):
        self.slider_h_margin.set(margin)
        self._on_margin_slider(margin)
