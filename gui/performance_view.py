"""
Performance Evaluation and Benchmark Analytics View.
Displays live telemetry (FPS, Latency, Accuracy, False Trigger counts),
interactive gesture testing evaluation mode, and system logs.
"""

import os
import sys
from typing import Optional
import customtkinter as ctk

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from utils.logger import logger
from utils.performance import perf_tracker


class PerformanceView(ctk.CTkScrollableFrame):
    """Real-time performance metrics, evaluation testing suite, and system log feed."""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self._build_ui()
        logger.register_callback(self._on_new_log)

    def _build_ui(self):
        title = ctk.CTkLabel(
            self,
            text="Performance Analytics & Gesture Evaluation",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        title.pack(anchor="w", padx=10, pady=(6, 2))

        desc = ctk.CTkLabel(
            self,
            text="Real-time telemetry, accuracy scorecards, processing latency, and benchmark exports.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=config.COLOR_TEXT_SECONDARY
        )
        desc.pack(anchor="w", padx=10, pady=(0, 10))

        # -------------------------------------------------------------
        # METRIC TILES (4-Card Grid)
        # -------------------------------------------------------------
        tiles_frame = ctk.CTkFrame(self, fg_color="transparent")
        tiles_frame.pack(fill="x", padx=4, pady=4)
        tiles_frame.columnconfigure((0, 1, 2, 3), weight=1)

        # Card 1: FPS
        c1 = ctk.CTkFrame(tiles_frame, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        c1.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(c1, text="Frame Rate", font=ctk.CTkFont(family="Segoe UI", size=11), text_color=config.COLOR_TEXT_MUTED).pack(pady=(8, 0))
        self.val_fps = ctk.CTkLabel(c1, text="-- FPS", font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"), text_color=config.COLOR_CYAN)
        self.val_fps.pack(pady=(0, 8))

        # Card 2: Latency
        c2 = ctk.CTkFrame(tiles_frame, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        c2.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(c2, text="Processing Latency", font=ctk.CTkFont(family="Segoe UI", size=11), text_color=config.COLOR_TEXT_MUTED).pack(pady=(8, 0))
        self.val_lat = ctk.CTkLabel(c2, text="-- ms", font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"), text_color=config.COLOR_WARNING)
        self.val_lat.pack(pady=(0, 8))

        # Card 3: Accuracy
        c3 = ctk.CTkFrame(tiles_frame, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        c3.grid(row=0, column=2, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(c3, text="Recognition Accuracy", font=ctk.CTkFont(family="Segoe UI", size=11), text_color=config.COLOR_TEXT_MUTED).pack(pady=(8, 0))
        self.val_acc = ctk.CTkLabel(c3, text="100.0%", font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"), text_color=config.COLOR_SUCCESS)
        self.val_acc.pack(pady=(0, 8))

        # Card 4: Recognitions
        c4 = ctk.CTkFrame(tiles_frame, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        c4.grid(row=0, column=3, padx=4, pady=4, sticky="nsew")
        ctk.CTkLabel(c4, text="Total Recognitions", font=ctk.CTkFont(family="Segoe UI", size=11), text_color=config.COLOR_TEXT_MUTED).pack(pady=(8, 0))
        self.val_count = ctk.CTkLabel(c4, text="0", font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"), text_color=config.COLOR_TEXT_PRIMARY)
        self.val_count.pack(pady=(0, 8))

        # -------------------------------------------------------------
        # BENCHMARK EXPORT & TESTING CONTROLS
        # -------------------------------------------------------------
        card_bench = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card_bench.pack(fill="x", padx=6, pady=6)

        b_title = ctk.CTkLabel(
            card_bench,
            text="Benchmark Reporting & Evaluation Export",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_CYAN
        )
        b_title.pack(anchor="w", padx=12, pady=(10, 4))

        b_row = ctk.CTkFrame(card_bench, fg_color="transparent")
        b_row.pack(fill="x", padx=12, pady=4)

        self.btn_export = ctk.CTkButton(
            b_row,
            text="Export Benchmark Report (JSON)",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=config.COLOR_ACCENT,
            hover_color=config.COLOR_ACCENT_HOVER,
            command=self._export_benchmark_report
        )
        self.btn_export.pack(side="left", padx=(0, 8))

        self.lbl_export_status = ctk.CTkLabel(
            b_row,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=config.COLOR_SUCCESS
        )
        self.lbl_export_status.pack(side="left", padx=4)

        ctk.CTkFrame(card_bench, fg_color="transparent", height=6).pack()

        # -------------------------------------------------------------
        # LIVE LOG FEED
        # -------------------------------------------------------------
        card_log = ctk.CTkFrame(self, fg_color=config.COLOR_CARD_BG, corner_radius=8)
        card_log.pack(fill="both", expand=True, padx=6, pady=6)

        l_title = ctk.CTkLabel(
            card_log,
            text="System Event & Telemetry Logs",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=config.COLOR_TEXT_PRIMARY
        )
        l_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.log_textbox = ctk.CTkTextbox(
            card_log,
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color="#141622",
            text_color="#CBD5E1",
            height=140
        )
        self.log_textbox.pack(fill="both", expand=True, padx=12, pady=(4, 12))

        # Populate initial logs
        for entry in logger.get_recent_logs()[-15:]:
            self.log_textbox.insert("end", entry + "\n")
        self.log_textbox.see("end")

    def update_metrics(self):
        """Refreshes live performance numbers from perf_tracker."""
        fps_val = perf_tracker.fps
        lat_val = perf_tracker.avg_latency_ms
        acc_val = perf_tracker.accuracy
        tot_val = perf_tracker.total_recognitions

        self.val_fps.configure(text=f"{fps_val:.1f} FPS")
        self.val_lat.configure(text=f"{lat_val:.1f} ms")
        self.val_acc.configure(text=f"{acc_val:.1f}%")
        self.val_count.configure(text=str(tot_val))

    def _on_new_log(self, formatted_log: str):
        try:
            self.log_textbox.insert("end", formatted_log + "\n")
            self.log_textbox.see("end")
        except Exception:
            pass

    def _export_benchmark_report(self):
        filepath = "data/benchmark_report.json"
        rep = perf_tracker.save_report_to_disk(filepath)
        self.lbl_export_status.configure(text=f"✅ Saved to {filepath}")
        logger.success(f"Benchmark report exported to {filepath}")
