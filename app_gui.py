"""
Desktop GUI Application Launcher
================================
1-click runner for the Tkinter Desktop Dashboard.
Usage:
    python app_gui.py
"""

import sys
import tkinter as tk
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.pipeline import HCIPipeline
from src.ui.gui_dashboard import ModernGUIDashboard


def launch_gui():
    pipeline = HCIPipeline(mode="rule", enable_control=True)

    if not pipeline.start():
        print("[ERROR] Failed to start camera for GUI. Please check if camera is connected and free.")
        sys.exit(1)

    root = tk.Tk()
    dashboard = ModernGUIDashboard(root, pipeline)

    def camera_loop():
        if dashboard.is_running:
            success, frame, telemetry = pipeline.process_one_frame()
            if success and frame is not None:
                dashboard.update_frame(frame)
                dashboard.update_telemetry(
                    gesture=telemetry.get("gesture", "idle"),
                    action=telemetry.get("action", "None"),
                    conf=telemetry.get("confidence", 0.0),
                    fps=telemetry.get("fps", 0.0),
                )
        root.after(15, camera_loop)

    root.after(50, camera_loop)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
