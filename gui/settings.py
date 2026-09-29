"""
Settings & Thresholds Panel.
Provides live sliders and toggles to dynamically tune sensitivity,
filtering parameters, confidence gates, and visual overlays.
"""

import os
import sys
from typing import Callable, Optional
import customtkinter as ctk

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


class SettingsPanel(ctk.CTkScrollableFrame):
    """Configuration sliders and parameter tuning panel."""

    def __init__(
        self,
        master,
        on_param_change: Optional[Callable[[str, float], None]] = None,
        **kwargs
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.on_param_change = on_param_change

        self._build_ui()

    def _build_ui(self):
        title = ctk.CTkLabel(
            self,
            text="System Parameters & Tuning",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        title.pack(anchor="w", padx=10, pady=(6, 2))

        desc = ctk.CTkLabel(
            self,
            text="Adjust real-time tracking sensitivity, deadzones, and temporal filters.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=config.COLOR_TEXT_SECONDARY
        )
        desc.pack(anchor="w", padx=10, pady=(0, 10))

        # CARD 1: SENSITIVITY & MOTION
        card1 = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card1.pack(fill="x", padx=6, pady=6)

        c1_title = ctk.CTkLabel(
            card1,
            text="Motion & Cursor Sensitivity",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_CYAN
        )
        c1_title.pack(anchor="w", padx=12, pady=(10, 6))

        # 1. Cursor Sensitivity
        self.lbl_cursor_sens = ctk.CTkLabel(
            card1,
            text=f"Cursor Speed Multiplier: {config.CURSOR_SENSITIVITY_DEFAULT:.1f}x",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        self.lbl_cursor_sens.pack(anchor="w", padx=12, pady=(2, 0))

        self.slider_cursor_sens = ctk.CTkSlider(
            card1,
            from_=config.CURSOR_SENSITIVITY_MIN,
            to=config.CURSOR_SENSITIVITY_MAX,
            number_of_steps=35,
            command=lambda v: self._handle_change("cursor_sens", v, self.lbl_cursor_sens, "Cursor Speed Multiplier", "{:.1f}x")
        )
        self.slider_cursor_sens.set(config.CURSOR_SENSITIVITY_DEFAULT)
        self.slider_cursor_sens.pack(fill="x", padx=12, pady=(2, 8))

        # 2. Scroll Sensitivity
        self.lbl_scroll_sens = ctk.CTkLabel(
            card1,
            text=f"Scroll Sensitivity: {config.SCROLL_SENSITIVITY_DEFAULT:.1f}",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        self.lbl_scroll_sens.pack(anchor="w", padx=12, pady=(4, 0))

        self.slider_scroll_sens = ctk.CTkSlider(
            card1,
            from_=config.SCROLL_SENSITIVITY_MIN,
            to=config.SCROLL_SENSITIVITY_MAX,
            number_of_steps=38,
            command=lambda v: self._handle_change("scroll_sens", v, self.lbl_scroll_sens, "Scroll Sensitivity", "{:.1f}")
        )
        self.slider_scroll_sens.set(config.SCROLL_SENSITIVITY_DEFAULT)
        self.slider_scroll_sens.pack(fill="x", padx=12, pady=(2, 8))

        # 3. Jitter Deadzone
        self.lbl_deadzone = ctk.CTkLabel(
            card1,
            text=f"Jitter Deadzone: {config.DEADZONE_PIXELS_DEFAULT:.0f} px",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        self.lbl_deadzone.pack(anchor="w", padx=12, pady=(4, 0))

        self.slider_deadzone = ctk.CTkSlider(
            card1,
            from_=config.DEADZONE_PIXELS_MIN,
            to=config.DEADZONE_PIXELS_MAX,
            number_of_steps=23,
            command=lambda v: self._handle_change("deadzone", v, self.lbl_deadzone, "Jitter Deadzone", "{:.0f} px")
        )
        self.slider_deadzone.set(config.DEADZONE_PIXELS_DEFAULT)
        self.slider_deadzone.pack(fill="x", padx=12, pady=(2, 10))

        # CARD 2: STABILITY & FILTERS
        card2 = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card2.pack(fill="x", padx=6, pady=6)

        c2_title = ctk.CTkLabel(
            card2,
            text="Stability & Confidence Filtering",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_CYAN
        )
        c2_title.pack(anchor="w", padx=12, pady=(10, 6))

        # 4. EMA Smoothing Alpha
        self.lbl_smoothing = ctk.CTkLabel(
            card2,
            text=f"EMA Smoothing Alpha: {config.SMOOTHING_ALPHA_DEFAULT:.2f}",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        self.lbl_smoothing.pack(anchor="w", padx=12, pady=(2, 0))

        self.slider_smoothing = ctk.CTkSlider(
            card2,
            from_=0.10,
            to=0.95,
            number_of_steps=17,
            command=lambda v: self._handle_change("smoothing_alpha", v, self.lbl_smoothing, "EMA Smoothing Alpha", "{:.2f}")
        )
        self.slider_smoothing.set(config.SMOOTHING_ALPHA_DEFAULT)
        self.slider_smoothing.pack(fill="x", padx=12, pady=(2, 8))

        # 5. Confidence Threshold
        self.lbl_conf = ctk.CTkLabel(
            card2,
            text=f"Confidence Gate Threshold: {int(config.CONFIDENCE_THRESHOLD_DEFAULT * 100)}%",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        self.lbl_conf.pack(anchor="w", padx=12, pady=(4, 0))

        self.slider_conf = ctk.CTkSlider(
            card2,
            from_=0.40,
            to=0.95,
            number_of_steps=11,
            command=lambda v: self._handle_change("confidence_threshold", v, self.lbl_conf, "Confidence Gate Threshold", lambda val: f"{int(val * 100)}%")
        )
        self.slider_conf.set(config.CONFIDENCE_THRESHOLD_DEFAULT)
        self.slider_conf.pack(fill="x", padx=12, pady=(2, 8))

        # 6. Debounce Frames
        self.lbl_debounce = ctk.CTkLabel(
            card2,
            text=f"Debounce Persistence: {config.GESTURE_DEBOUNCE_FRAMES} frames",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        self.lbl_debounce.pack(anchor="w", padx=12, pady=(4, 0))

        self.slider_debounce = ctk.CTkSlider(
            card2,
            from_=1,
            to=8,
            number_of_steps=7,
            command=lambda v: self._handle_change("debounce_frames", int(v), self.lbl_debounce, "Debounce Persistence", "{:.0f} frames")
        )
        self.slider_debounce.set(config.GESTURE_DEBOUNCE_FRAMES)
        self.slider_debounce.pack(fill="x", padx=12, pady=(2, 10))

        # CARD 3: FEATURE TOGGLES
        card3 = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card3.pack(fill="x", padx=6, pady=6)

        c3_title = ctk.CTkLabel(
            card3,
            text="Visual Overlays & Toggles",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_CYAN
        )
        c3_title.pack(anchor="w", padx=12, pady=(10, 6))

        self.switch_skeleton = ctk.CTkSwitch(
            card3,
            text="Render Sci-Fi Hand Skeleton & Nodes",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=lambda: self._handle_toggle("show_skeleton", bool(self.switch_skeleton.get()))
        )
        self.switch_skeleton.select()
        self.switch_skeleton.pack(anchor="w", padx=12, pady=6)

        self.switch_hud = ctk.CTkSwitch(
            card3,
            text="Render Live Telemetry & Target Crosshairs",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=lambda: self._handle_toggle("show_hud", bool(self.switch_hud.get()))
        )
        self.switch_hud.select()
        self.switch_hud.pack(anchor="w", padx=12, pady=6)

        self.switch_mirror = ctk.CTkSwitch(
            card3,
            text="Mirror Camera Horizontally",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=lambda: self._handle_toggle("camera_mirror", bool(self.switch_mirror.get()))
        )
        self.switch_mirror.select()
        self.switch_mirror.pack(anchor="w", padx=12, pady=(6, 12))

    def _handle_change(self, param_name: str, value: float, label_widget: ctk.CTkLabel, prefix: str, fmt):
        if callable(fmt):
            formatted_val = fmt(value)
        else:
            formatted_val = fmt.format(value)
        label_widget.configure(text=f"{prefix}: {formatted_val}")

        if self.on_param_change:
            self.on_param_change(param_name, value)

    def _handle_toggle(self, toggle_name: str, state: bool):
        if self.on_param_change:
            self.on_param_change(toggle_name, state)
