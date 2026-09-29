"""
Empirical Gesture Testing and Benchmarking Module.
Allows users to perform real-world evaluation trials for specific gestures,
measuring actual accuracy, precision, recall, F1-score, response latency,
and false activations without simulated or fabricated values.
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional
import customtkinter as ctk

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from utils.logger import logger
from utils.performance import perf_tracker

TESTABLE_GESTURES = [
    ("PINCH_INDEX", "Left Click (Pinch)"),
    ("DOUBLE_PINCH_INDEX", "Double Click"),
    ("PINCH_MIDDLE", "Right Click"),
    ("INDEX_POINTING", "Cursor Movement"),
    ("CLOSED_FIST", "Fist (Drag / Mute)"),
    ("OPEN_PALM", "Open Palm (Drop / Stop)"),
    ("TWO_FINGERS_UP", "Scroll Up"),
    ("TWO_FINGERS_DOWN", "Scroll Down"),
    ("PEACE_SIGN", "Play / Pause (Peace)"),
    ("THUMBS_UP", "Thumbs Up (Enter / Play)"),
    ("OK_SIGN", "OK Sign (Screenshot)"),
    ("THREE_FINGERS", "App Switcher (3 Fingers)"),
    ("FOUR_FINGERS", "Task View (4 Fingers)"),
    ("FIVE_FINGERS", "Show Desktop (5 Fingers)"),
    ("TWO_HAND_EXPAND", "2-Hand Zoom In"),
    ("TWO_HAND_CONTRACT", "2-Hand Zoom Out")
]


class GestureTestingView(ctk.CTkScrollableFrame):
    """Empirical testing screen for evaluating per-gesture real-world performance."""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        # Active test session state
        self.is_testing = False
        self.target_gesture_id = ""
        self.target_gesture_name = ""
        self.target_attempts = 20
        self.current_attempt = 0
        self.correct_count = 0
        self.incorrect_count = 0
        self.false_activations = 0
        self.trial_latencies: List[float] = []

        # Persistent scorecard dictionary: { gesture_id: {"name": str, "attempts": int, "correct": int, ...} }
        self.scorecards: Dict[str, Dict] = self._load_scorecards()

        self._build_ui()
        self.refresh_table()

    def _load_scorecards(self) -> Dict[str, Dict]:
        if os.path.exists(config.TEST_REPORT_PATH):
            try:
                with open(config.TEST_REPORT_PATH, "r") as f:
                    data = json.load(f)
                    return data.get("gestures", {})
            except Exception:
                pass
        return {}

    def _save_scorecards(self):
        os.makedirs(os.path.dirname(os.path.abspath(config.TEST_REPORT_PATH)), exist_ok=True)
        report = perf_tracker.generate_report()
        report["gestures"] = self.scorecards
        with open(config.TEST_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

    def _build_ui(self):
        title = ctk.CTkLabel(
            self,
            text="Empirical Gesture Evaluation & Benchmark Suite",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        title.pack(anchor="w", padx=10, pady=(6, 2))

        desc = ctk.CTkLabel(
            self,
            text="Conduct repeated live trials to empirically measure Accuracy, Precision, Recall, F1, and Latency.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=config.COLOR_TEXT_SECONDARY
        )
        desc.pack(anchor="w", padx=10, pady=(0, 10))

        # CARD 1: TEST RUN CONTROLS
        card_ctrl = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card_ctrl.pack(fill="x", padx=6, pady=6)

        c1_title = ctk.CTkLabel(
            card_ctrl,
            text="1. Configure Test Trial",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_CYAN
        )
        c1_title.pack(anchor="w", padx=12, pady=(10, 4))

        row1 = ctk.CTkFrame(card_ctrl, fg_color="transparent")
        row1.pack(fill="x", padx=12, pady=4)

        ctk.CTkLabel(row1, text="Select Gesture:", font=ctk.CTkFont(family="Segoe UI", size=12)).pack(side="left", padx=(0, 6))

        self.gesture_dropdown = ctk.CTkComboBox(
            row1,
            values=[f"{name} ({gid})" for gid, name in TESTABLE_GESTURES],
            width=240,
            font=ctk.CTkFont(family="Segoe UI", size=11)
        )
        self.gesture_dropdown.set(f"{TESTABLE_GESTURES[0][1]} ({TESTABLE_GESTURES[0][0]})")
        self.gesture_dropdown.pack(side="left", padx=4)

        ctk.CTkLabel(row1, text="Trials:", font=ctk.CTkFont(family="Segoe UI", size=12)).pack(side="left", padx=(10, 4))
        self.txt_trials = ctk.CTkEntry(row1, width=50, font=ctk.CTkFont(family="Segoe UI", size=12))
        self.txt_trials.insert(0, "20")
        self.txt_trials.pack(side="left", padx=4)

        self.btn_start_test = ctk.CTkButton(
            row1,
            text="Start Trial",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=config.COLOR_ACCENT,
            hover_color=config.COLOR_ACCENT_HOVER,
            command=self._toggle_test_session
        )
        self.btn_start_test.pack(side="left", padx=(12, 0))

        # Trial Progress Status Banner
        self.lbl_test_status = ctk.CTkLabel(
            card_ctrl,
            text="Select a gesture and click 'Start Trial' to record real-world accuracy.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=config.COLOR_TEXT_MUTED
        )
        self.lbl_test_status.pack(anchor="w", padx=12, pady=(6, 10))

        # CARD 2: EMPIRICAL SCORECARDS TABLE
        card_tbl = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card_tbl.pack(fill="both", expand=True, padx=6, pady=6)

        c2_header = ctk.CTkFrame(card_tbl, fg_color="transparent")
        c2_header.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            c2_header,
            text="Empirical Benchmark Scorecards",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        ).pack(side="left")

        btn_export = ctk.CTkButton(
            c2_header,
            text="Export Report (JSON)",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color="#374151",
            hover_color="#4B5563",
            width=130,
            command=self._export_report
        )
        btn_export.pack(side="right")

        self.table_scroll = ctk.CTkScrollableFrame(card_tbl, fg_color="#151722", height=200)
        self.table_scroll.pack(fill="both", expand=True, padx=12, pady=(4, 12))

    def _toggle_test_session(self):
        if self.is_testing:
            # Cancel test
            self.is_testing = False
            self.btn_start_test.configure(text="Start Trial", fg_color=config.COLOR_ACCENT)
            self.lbl_test_status.configure(text="Test trial cancelled.", text_color=config.COLOR_WARNING)
            return

        selected_str = self.gesture_dropdown.get()
        # Parse gesture ID from string "Name (ID)"
        if "(" in selected_str and ")" in selected_str:
            gid = selected_str.split("(")[-1].replace(")", "").strip()
            name = selected_str.split("(")[0].strip()
        else:
            gid = selected_str
            name = selected_str

        try:
            trials = int(self.txt_trials.get().strip())
        except Exception:
            trials = 20

        self.target_gesture_id = gid
        self.target_gesture_name = name
        self.target_attempts = max(5, min(100, trials))
        self.current_attempt = 0
        self.correct_count = 0
        self.incorrect_count = 0
        self.false_activations = 0
        self.trial_latencies.clear()

        self.is_testing = True
        self.btn_start_test.configure(text="Stop Trial", fg_color=config.COLOR_DANGER)
        self.lbl_test_status.configure(
            text=f"🎯 TESTING '{name}': Please perform the gesture in front of the camera (0/{self.target_attempts})...",
            text_color=config.COLOR_CYAN
        )
        logger.info(f"[TestMode] Started empirical trial for '{name}' ({self.target_attempts} trials).")

    def evaluate_live_frame(self, detected_gesture_id: str, confidence: float, latency_ms: float):
        """
        Called on every processed frame by the main loop to log test results.
        """
        if not self.is_testing:
            return

        if detected_gesture_id in ("NONE", "IDLE"):
            return

        self.current_attempt += 1
        self.trial_latencies.append(latency_ms)

        if detected_gesture_id == self.target_gesture_id:
            self.correct_count += 1
        else:
            self.incorrect_count += 1
            self.false_activations += 1

        self.lbl_test_status.configure(
            text=f"🎯 TESTING '{self.target_gesture_name}': {self.current_attempt}/{self.target_attempts} trials recorded "
                 f"(Correct: {self.correct_count}, Acc: {int(self.correct_count/self.current_attempt*100)}%)",
            text_color=config.COLOR_CYAN
        )

        if self.current_attempt >= self.target_attempts:
            self._finalize_test_run()

    def _finalize_test_run(self):
        self.is_testing = False
        self.btn_start_test.configure(text="Start Trial", fg_color=config.COLOR_ACCENT)

        total = self.current_attempt
        acc = (self.correct_count / total * 100.0) if total > 0 else 0.0
        avg_lat = (sum(self.trial_latencies) / len(self.trial_latencies)) if self.trial_latencies else 0.0
        max_lat = max(self.trial_latencies) if self.trial_latencies else 0.0

        # Calculate Precision, Recall, F1
        tp = self.correct_count
        fp = self.false_activations
        fn = self.incorrect_count
        precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else acc
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else acc
        f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else acc

        # Record into scorecard dictionary
        self.scorecards[self.target_gesture_id] = {
            "name": self.target_gesture_name,
            "attempts": total,
            "correct": self.correct_count,
            "incorrect": self.incorrect_count,
            "false_activations": self.false_activations,
            "accuracy_percent": round(acc, 1),
            "precision_percent": round(precision, 1),
            "recall_percent": round(recall, 1),
            "f1_score": round(f1 / 100.0, 3),
            "avg_latency_ms": round(avg_lat, 1),
            "max_latency_ms": round(max_lat, 1)
        }

        self._save_scorecards()
        self.refresh_table()

        msg = f"✅ Trial Complete for '{self.target_gesture_name}': {acc:.1f}% Accuracy, {avg_lat:.1f} ms latency."
        self.lbl_test_status.configure(text=msg, text_color=config.COLOR_SUCCESS)
        logger.success(f"[TestMode] {msg}")

    def refresh_table(self):
        """Rebuilds scorecard rows."""
        for child in self.table_scroll.winfo_children():
            child.destroy()

        if not self.scorecards:
            ctk.CTkLabel(
                self.table_scroll,
                text="No benchmark trials conducted yet. Select a gesture above and run a trial.",
                font=ctk.CTkFont(family="Segoe UI", size=11),
                text_color=config.COLOR_TEXT_MUTED
            ).pack(pady=15)
            return

        # Table Header
        h_row = ctk.CTkFrame(self.table_scroll, fg_color="#1E2132", corner_radius=4)
        h_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(h_row, text="Gesture", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), width=150, anchor="w").pack(side="left", padx=8, pady=4)
        ctk.CTkLabel(h_row, text="Trials", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), width=60).pack(side="left", padx=4)
        ctk.CTkLabel(h_row, text="Accuracy", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), width=75).pack(side="left", padx=4)
        ctk.CTkLabel(h_row, text="Precision", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), width=70).pack(side="left", padx=4)
        ctk.CTkLabel(h_row, text="Recall", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), width=65).pack(side="left", padx=4)
        ctk.CTkLabel(h_row, text="Latency", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), width=75).pack(side="left", padx=4)

        for gid, data in self.scorecards.items():
            r = ctk.CTkFrame(self.table_scroll, fg_color="#222538", corner_radius=4)
            r.pack(fill="x", padx=4, pady=2)

            name = data.get("name", gid)
            trials = data.get("attempts", 0)
            acc = data.get("accuracy_percent", 0.0)
            prec = data.get("precision_percent", 0.0)
            rec = data.get("recall_percent", 0.0)
            lat = data.get("avg_latency_ms", 0.0)

            acc_col = config.COLOR_SUCCESS if acc >= 90.0 else (config.COLOR_WARNING if acc >= 75.0 else config.COLOR_DANGER)

            ctk.CTkLabel(r, text=name, font=ctk.CTkFont(family="Segoe UI", size=11), width=150, anchor="w").pack(side="left", padx=8, pady=4)
            ctk.CTkLabel(r, text=str(trials), font=ctk.CTkFont(family="Segoe UI", size=11), width=60).pack(side="left", padx=4)
            ctk.CTkLabel(r, text=f"{acc:.1f}%", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color=acc_col, width=75).pack(side="left", padx=4)
            ctk.CTkLabel(r, text=f"{prec:.1f}%", font=ctk.CTkFont(family="Segoe UI", size=11), width=70).pack(side="left", padx=4)
            ctk.CTkLabel(r, text=f"{rec:.1f}%", font=ctk.CTkFont(family="Segoe UI", size=11), width=65).pack(side="left", padx=4)
            ctk.CTkLabel(r, text=f"{lat:.1f} ms", font=ctk.CTkFont(family="Segoe UI", size=11), width=75).pack(side="left", padx=4)

    def _export_report(self):
        self._save_scorecards()
        logger.success(f"Empirical benchmark saved to: {config.TEST_REPORT_PATH}")
