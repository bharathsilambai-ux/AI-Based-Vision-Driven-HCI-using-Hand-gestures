"""
Custom Gesture Training Studio View.
Interactive GUI component for recording new gestures, training Random Forest models,
and binding gestures to arbitrary computer actions.
"""

from typing import Callable, List, Optional
import customtkinter as ctk
import config
from recognition.custom_gesture import CustomGestureManager
from utils.logger import logger


AVAILABLE_ACTIONS = [
    "TAKE_SCREENSHOT",
    "LEFT_CLICK",
    "DOUBLE_CLICK",
    "RIGHT_CLICK",
    "START_DRAG",
    "RELEASE_DRAG",
    "MEDIA_PLAY_PAUSE",
    "MEDIA_NEXT_TRACK",
    "MEDIA_PREV_TRACK",
    "VOLUME_UP",
    "VOLUME_DOWN",
    "VOLUME_MUTE",
    "APP_SWITCH",
    "TASK_VIEW",
    "SHOW_DESKTOP",
    "WINDOW_MINIMIZE",
    "WINDOW_MAXIMIZE",
    "COPY",
    "PASTE",
    "KEY_ENTER",
    "KEY_ESCAPE",
    "ZOOM_IN",
    "ZOOM_OUT",
    "ROTATE_DOCUMENT",
    "BROWSER_BACK",
    "BROWSER_FORWARD"
]


class CustomGestureStudio(ctk.CTkFrame):
    """Studio interface for recording, training, and binding custom gestures."""

    def __init__(self, master, gesture_manager: CustomGestureManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.manager = gesture_manager

        self._build_ui()
        self.refresh_table()

    def _build_ui(self):
        # Header / Title
        title_lbl = ctk.CTkLabel(
            self,
            text="Personalized Gesture Training Studio",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        title_lbl.pack(anchor="w", padx=10, pady=(6, 2))

        desc_lbl = ctk.CTkLabel(
            self,
            text="Record novel hand poses and train a lightweight Random Forest classifier.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=config.COLOR_TEXT_SECONDARY
        )
        desc_lbl.pack(anchor="w", padx=10, pady=(0, 10))

        # Card 1: Record New Gesture
        card_rec = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card_rec.pack(fill="x", padx=6, pady=6)

        c1_title = ctk.CTkLabel(
            card_rec,
            text="1. Record Gesture Samples",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_CYAN
        )
        c1_title.pack(anchor="w", padx=12, pady=(10, 4))

        form_frame = ctk.CTkFrame(card_rec, fg_color="transparent")
        form_frame.pack(fill="x", padx=12, pady=4)

        self.txt_gesture_name = ctk.CTkEntry(
            form_frame,
            placeholder_text="Enter Gesture Name (e.g. THREE_FINGER_TAP)",
            width=220,
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

        # Card 2: Train Model & Bind Action
        card_train = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card_train.pack(fill="x", padx=6, pady=6)

        c2_title = ctk.CTkLabel(
            card_train,
            text="2. Train Classifier & Bind Action",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_CYAN
        )
        c2_title.pack(anchor="w", padx=12, pady=(10, 4))

        t_row = ctk.CTkFrame(card_train, fg_color="transparent")
        t_row.pack(fill="x", padx=12, pady=4)

        self.btn_train = ctk.CTkButton(
            t_row,
            text="Train Random Forest Model",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=config.COLOR_SUCCESS,
            hover_color="#059669",
            command=self._train_model_clicked
        )
        self.btn_train.pack(side="left", padx=(0, 8))

        self.action_dropdown = ctk.CTkComboBox(
            t_row,
            values=AVAILABLE_ACTIONS,
            width=200,
            font=ctk.CTkFont(family="Segoe UI", size=11)
        )
        self.action_dropdown.set(AVAILABLE_ACTIONS[0])
        self.action_dropdown.pack(side="left", padx=4)

        self.btn_bind = ctk.CTkButton(
            t_row,
            text="Assign Action",
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

        # Card 3: Registered Gestures Table
        card_table = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card_table.pack(fill="both", expand=True, padx=6, pady=6)

        c3_title = ctk.CTkLabel(
            card_table,
            text="Registered Custom Gestures",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        c3_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.table_scroll = ctk.CTkScrollableFrame(card_table, fg_color="#171824", height=140)
        self.table_scroll.pack(fill="both", expand=True, padx=12, pady=(4, 12))

    def _start_recording_clicked(self):
        name = self.txt_gesture_name.get().strip()
        if not name:
            self.lbl_rec_status.configure(text="⚠️ Please enter a valid gesture name.", text_color=config.COLOR_WARNING)
            return

        success = self.manager.start_recording(name, target_samples=40)
        if success:
            self.lbl_rec_status.configure(
                text=f"🔴 RECORDING '{name.upper()}': Hold gesture in front of camera...",
                text_color=config.COLOR_DANGER
            )
            self.btn_record.configure(state="disabled")

    def update_recording_progress(self):
        """Called periodically by dashboard loop to update sample counter."""
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
                    text=f"✅ Recording complete for '{self.manager.current_recording_name}'! Click 'Train Model'.",
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
            self.lbl_train_status.configure(text="⚠️ Enter or select a gesture name above to bind.", text_color=config.COLOR_WARNING)
            return
        self.manager.bind_action(name, action)
        self.lbl_train_status.configure(text=f"✅ Bound '{name.upper()}' -> '{action}'.", text_color=config.COLOR_SUCCESS)
        self.refresh_table()

    def refresh_table(self):
        """Rebuilds rows in the registered gestures scroll view."""
        for child in self.table_scroll.winfo_children():
            child.destroy()

        registered = self.manager.get_registered_gestures()
        if not registered:
            empty_lbl = ctk.CTkLabel(
                self.table_scroll,
                text="No custom gestures recorded yet.",
                font=ctk.CTkFont(family="Segoe UI", size=11),
                text_color=config.COLOR_TEXT_MUTED
            )
            empty_lbl.pack(pady=10)
            return

        for name, action, count in registered:
            row = ctk.CTkFrame(self.table_scroll, fg_color="#212435", corner_radius=6)
            row.pack(fill="x", padx=4, pady=3)

            n_lbl = ctk.CTkLabel(
                row,
                text=f"✋ {name} ({count} samples)",
                font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                text_color=config.COLOR_TEXT_PRIMARY
            )
            n_lbl.pack(side="left", padx=10, pady=4)

            a_lbl = ctk.CTkLabel(
                row,
                text=f"⚡ {action}",
                font=ctk.CTkFont(family="Segoe UI", size=11),
                text_color=config.COLOR_CYAN
            )
            a_lbl.pack(side="right", padx=10, pady=4)
