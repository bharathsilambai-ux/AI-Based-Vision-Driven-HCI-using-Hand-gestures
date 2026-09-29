"""
Gesture Manager & Action Mapping View.
Provides an interface for custom gesture creation, sample recording,
Random Forest model training, and dynamic gesture-to-action rebinding.
"""

import os
import sys
from typing import Callable, List, Optional
import customtkinter as ctk

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from recognition.custom_gesture import CustomGestureManager
from utils.logger import logger

AVAILABLE_ACTIONS = [
    "MOVE_CURSOR",
    "LEFT_CLICK",
    "DOUBLE_CLICK",
    "RIGHT_CLICK",
    "START_DRAG",
    "RELEASE_DRAG",
    "HOVER",
    "SCROLL_UP",
    "SCROLL_DOWN",
    "PAGE_UP",
    "PAGE_DOWN",
    "BROWSER_BACK",
    "BROWSER_FORWARD",
    "MEDIA_PLAY_PAUSE",
    "MEDIA_NEXT_TRACK",
    "MEDIA_PREV_TRACK",
    "VOLUME_UP",
    "VOLUME_DOWN",
    "VOLUME_MUTE",
    "TAKE_SCREENSHOT",
    "APP_SWITCH",
    "TASK_VIEW",
    "SHOW_DESKTOP",
    "WINDOW_MINIMIZE",
    "WINDOW_MAXIMIZE",
    "WINDOW_CLOSE_SAFE",
    "LOCK_SCREEN_SAFE",
    "COPY",
    "PASTE",
    "CUT",
    "SELECT_ALL",
    "UNDO",
    "REDO",
    "KEY_ENTER",
    "KEY_ESCAPE",
    "ZOOM_IN",
    "ZOOM_OUT",
    "BRIGHTNESS_INCREASE",
    "BRIGHTNESS_DECREASE",
    "ROTATE_DOCUMENT"
]


class GestureManagerView(ctk.CTkScrollableFrame):
    """Custom gesture recording, ML model training, and mapping manager."""

    def __init__(self, master, gesture_manager: CustomGestureManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.manager = gesture_manager

        self._build_ui()
        self.refresh_table()

    def _build_ui(self):
        title = ctk.CTkLabel(
            self,
            text="Custom Gesture Studio & Action Rebinding",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        title.pack(anchor="w", padx=10, pady=(6, 2))

        desc = ctk.CTkLabel(
            self,
            text="Train personalized gestures using Random Forest and bind them to any computer operation.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=config.COLOR_TEXT_SECONDARY
        )
        desc.pack(anchor="w", padx=10, pady=(0, 10))

        # CARD 1: RECORD GESTURE SAMPLES
        card_rec = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card_rec.pack(fill="x", padx=6, pady=6)

        c1_title = ctk.CTkLabel(
            card_rec,
            text="1. Record Training Samples",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_CYAN
        )
        c1_title.pack(anchor="w", padx=12, pady=(10, 4))

        form_frame = ctk.CTkFrame(card_rec, fg_color="transparent")
        form_frame.pack(fill="x", padx=12, pady=4)

        self.txt_gesture_name = ctk.CTkEntry(
            form_frame,
            placeholder_text="Enter Gesture Name (e.g. THREE_FINGER_TAP)",
            width=230,
            font=ctk.CTkFont(family="Segoe UI", size=12)
        )
        self.txt_gesture_name.pack(side="left", padx=(0, 8))

        self.btn_record = ctk.CTkButton(
            form_frame,
            text="Record 40 Frames",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=config.COLOR_ACCENT,
            hover_color=config.COLOR_ACCENT_HOVER,
            command=self._start_recording_clicked
        )
        self.btn_record.pack(side="left", padx=4)

        self.lbl_rec_status = ctk.CTkLabel(
            card_rec,
            text="Status: Ready to record",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=config.COLOR_TEXT_MUTED
        )
        self.lbl_rec_status.pack(anchor="w", padx=12, pady=(4, 10))

        # CARD 2: TRAIN CLASSIFIER & BIND ACTION
        card_train = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card_train.pack(fill="x", padx=6, pady=6)

        c2_title = ctk.CTkLabel(
            card_train,
            text="2. Train Classifier & Assign Action",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_CYAN
        )
        c2_title.pack(anchor="w", padx=12, pady=(10, 4))

        t_row = ctk.CTkFrame(card_train, fg_color="transparent")
        t_row.pack(fill="x", padx=12, pady=4)

        self.btn_train = ctk.CTkButton(
            t_row,
            text="Train Random Forest",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=config.COLOR_SUCCESS,
            hover_color="#059669",
            command=self._train_model_clicked
        )
        self.btn_train.pack(side="left", padx=(0, 8))

        self.action_dropdown = ctk.CTkComboBox(
            t_row,
            values=AVAILABLE_ACTIONS,
            width=210,
            font=ctk.CTkFont(family="Segoe UI", size=11)
        )
        self.action_dropdown.set(AVAILABLE_ACTIONS[0])
        self.action_dropdown.pack(side="left", padx=4)

        self.btn_bind = ctk.CTkButton(
            t_row,
            text="Bind Action",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color="#374151",
            hover_color="#4B5563",
            width=100,
            command=self._bind_action_clicked
        )
        self.btn_bind.pack(side="left", padx=4)

        self.lbl_train_status = ctk.CTkLabel(
            card_train,
            text="Model: Loaded" if self.manager.classifier else "Model: Not trained yet",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=config.COLOR_TEXT_MUTED
        )
        self.lbl_train_status.pack(anchor="w", padx=12, pady=(4, 10))

        # CARD 3: REGISTERED GESTURES
        card_table = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card_table.pack(fill="both", expand=True, padx=6, pady=6)

        c3_title = ctk.CTkLabel(
            card_table,
            text="Registered Custom Gestures & Bindings",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        c3_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.table_scroll = ctk.CTkScrollableFrame(card_table, fg_color="#151722", height=150)
        self.table_scroll.pack(fill="both", expand=True, padx=12, pady=(4, 12))

    def _start_recording_clicked(self):
        name = self.txt_gesture_name.get().strip()
        if not name:
            self.lbl_rec_status.configure(text="⚠️ Please enter a gesture name.", text_color=config.COLOR_WARNING)
            return

        success = self.manager.start_recording(name, target_samples=40)
        if success:
            self.lbl_rec_status.configure(
                text=f"🔴 RECORDING '{name.upper()}': Hold your hand gesture in front of the camera...",
                text_color=config.COLOR_DANGER
            )
            self.btn_record.configure(state="disabled")

    def update_recording_progress(self):
        if self.manager.is_recording:
            count = self.manager.recorded_samples_count
            total = self.manager.target_samples
            self.lbl_rec_status.configure(
                text=f"🔴 RECORDING '{self.manager.current_recording_name}': {count}/{total} frames captured",
                text_color=config.COLOR_DANGER
            )
        else:
            if self.btn_record.cget("state") == "disabled":
                self.btn_record.configure(state="normal")
                self.lbl_rec_status.configure(
                    text=f"✅ Recording finished for '{self.manager.current_recording_name}'! Click 'Train Random Forest'.",
                    text_color=config.COLOR_SUCCESS
                )
                self.refresh_table()

    def _train_model_clicked(self):
        self.lbl_train_status.configure(text="⏳ Training Random Forest classifier...", text_color=config.COLOR_CYAN)
        self.update()
        success, msg = self.manager.train_model()
        if success:
            self.lbl_train_status.configure(text=f"✅ {msg}", text_color=config.COLOR_SUCCESS)
        else:
            self.lbl_train_status.configure(text=f"⚠️ {msg}", text_color=config.COLOR_WARNING)
        self.refresh_table()

    def _bind_action_clicked(self):
        name = self.txt_gesture_name.get().strip()
        action = self.action_dropdown.get()
        if not name:
            self.lbl_train_status.configure(text="⚠️ Enter a gesture name above to assign an action.", text_color=config.COLOR_WARNING)
            return
        self.manager.bind_action(name, action)
        self.lbl_train_status.configure(text=f"✅ Assigned '{name.upper()}' -> '{action}'.", text_color=config.COLOR_SUCCESS)
        self.refresh_table()

    def refresh_table(self):
        for child in self.table_scroll.winfo_children():
            child.destroy()

        registered = self.manager.get_registered_gestures()
        if not registered:
            ctk.CTkLabel(
                self.table_scroll,
                text="No custom gestures trained yet.",
                font=ctk.CTkFont(family="Segoe UI", size=11),
                text_color=config.COLOR_TEXT_MUTED
            ).pack(pady=12)
            return

        for name, action, count in registered:
            row = ctk.CTkFrame(self.table_scroll, fg_color="#212435", corner_radius=6)
            row.pack(fill="x", padx=4, pady=3)

            ctk.CTkLabel(
                row,
                text=f"✋ {name} ({count} frames)",
                font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                text_color=config.COLOR_TEXT_PRIMARY
            ).pack(side="left", padx=10, pady=4)

            ctk.CTkLabel(
                row,
                text=f"⚡ {action}",
                font=ctk.CTkFont(family="Segoe UI", size=11),
                text_color=config.COLOR_CYAN
            ).pack(side="right", padx=10, pady=4)
