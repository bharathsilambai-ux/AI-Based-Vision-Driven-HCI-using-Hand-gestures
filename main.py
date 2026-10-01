"""
AI-Based Vision-Driven Human Computer Interaction (HCI) System
==============================================================
Main Entry Point
----------------
Usage:
    python main.py                  # Runs with OpenCV HUD (Rule-Based default)
    python main.py --mode ml        # Runs with Trained ML Classifier
    python main.py --gui            # Runs Desktop GUI Dashboard (Tkinter)
    python main.py --no-control     # Runs without triggering OS mouse/keyboard actions (Safe Mode)
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.pipeline import HCIPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Based Vision-Driven HCI System Using Hand Gesture Recognition"
    )
    parser.add_argument(
        "--mode",
        choices=["rule", "ml"],
        default="rule",
        help="Gesture classification mode: 'rule' (default) or 'ml' (trained model)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Desktop Tkinter Dashboard instead of OpenCV window",
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Disable OS mouse/keyboard actions (visual detection only / safe mode)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_ID,
        help=f"Webcam index (default: {config.CAMERA_ID})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    enable_control = not args.no_control

    pipeline = HCIPipeline(
        mode=args.mode,
        enable_control=enable_control,
        camera_id=args.camera,
    )

    if args.gui:
        import tkinter as tk
        from src.ui.gui_dashboard import ModernGUIDashboard

        if not pipeline.start():
            print("[ERROR] Failed to initialize camera. Exiting.")
            sys.exit(1)

        root = tk.Tk()
        dashboard = ModernGUIDashboard(root, pipeline)

        def poll_camera():
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
            root.after(15, poll_camera)

        root.after(50, poll_camera)
        root.mainloop()
    else:
        # Standard OpenCV window mode
        pipeline.run_cv_window()


if __name__ == "__main__":
    main()
