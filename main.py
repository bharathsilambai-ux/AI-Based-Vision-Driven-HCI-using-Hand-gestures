"""
Main Entry Point for AI-Based Vision-Driven Human-Computer Interaction System.

Usage:
    python main.py              # Launch the full modern CustomTkinter GUI Desktop Dashboard
    python main.py --cli        # Launch in lightweight OpenCV CV-only window mode
    python main.py --test       # Run automated unit and integration tests
    python main.py --help       # Display help and command options
"""

import argparse
import sys
import time

import cv2
import numpy as np

import config
from control import (
    DisplayController,
    KeyboardController,
    MediaController,
    MouseController,
    SystemController,
    action_dispatcher
)
from gui.dashboard import MainDashboard
from recognition import GestureClassifier, GestureResult
from utils.logger import logger
from utils.performance import perf_tracker
from vision import FeatureExtractor, HandTracker, ThreadedCamera


def run_gui_mode():
    """Launches the modern CustomTkinter Desktop Application."""
    logger.info("Starting AI-Based Vision-Driven HCI Desktop Application...")
    app = MainDashboard()
    app.mainloop()


def run_cli_mode():
    """
    Runs lightweight OpenCV window mode (useful for debugging, low-resource systems,
    or headless validation).
    """
    print("=" * 70)
    print("  AI Vision-Driven HCI - OpenCV Real-Time Preview Mode")
    print("=" * 70)
    print("Controls:")
    print("  [Space]  - Toggle Tracking Active / Paused")
    print("  [Esc]    - Emergency Stop")
    print("  [1-7]    - Switch Operational Mode")
    print("  [q]      - Quit Application")
    print("=" * 70)

    cam = ThreadedCamera()
    if not cam.start():
        print("[Error] Could not initialize webcam.")
        sys.exit(1)

    tracker = HandTracker(max_hands=config.MAX_NUM_HANDS)
    extractor = FeatureExtractor()
    classifier = GestureClassifier()
    mouse_ctrl = MouseController()
    keyboard_ctrl = KeyboardController()
    media_ctrl = MediaController()
    system_ctrl = SystemController()
    display_ctrl = DisplayController()

    try:
        while True:
            perf_tracker.start_frame()
            ret, frame = cam.read(timeout=0.2)
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]

            # Hand tracking & feature extraction
            hands, annotated = tracker.find_hands(frame, draw=True)
            feats = [extractor.extract_single_hand(h_data) for h_data in hands]
            dual_feats = extractor.extract_dual_hands(hands) if len(hands) >= 2 else None

            # Classification
            res: GestureResult = classifier.process(hands, feats, dual_feats)

            # Action dispatching
            if action_dispatcher.is_active and res.assigned_action:
                action = res.assigned_action
                if action == "MOVE_CURSOR" and res.smoothed_point is not None:
                    mouse_ctrl.move_cursor_to_norm(res.smoothed_point[0] / w, res.smoothed_point[1] / h)
                elif action == "HOVER":
                    mouse_ctrl.hover()
                elif action == "LEFT_CLICK" and (res.is_discrete_trigger or res.can_execute):
                    mouse_ctrl.left_click()
                elif action == "DOUBLE_CLICK" and (res.is_discrete_trigger or res.can_execute):
                    mouse_ctrl.double_click()
                elif action == "RIGHT_CLICK" and (res.is_discrete_trigger or res.can_execute):
                    mouse_ctrl.right_click()
                elif action == "START_DRAG":
                    mouse_ctrl.start_drag()
                    if res.smoothed_point is not None:
                        mouse_ctrl.move_cursor_to_norm(res.smoothed_point[0] / w, res.smoothed_point[1] / h)
                elif action in ("RELEASE_DRAG", "RELEASE_DRAG_OR_STOP", "HALT_SCROLL"):
                    mouse_ctrl.stop_drag()
                elif action == "SCROLL_UP":
                    mouse_ctrl.scroll_vertical(int(min(config.MAX_SCROLL_SPEED, max(4, abs(res.delta_y) * 2.8))))
                elif action == "SCROLL_DOWN":
                    mouse_ctrl.scroll_vertical(-int(min(config.MAX_SCROLL_SPEED, max(4, abs(res.delta_y) * 2.8))))
                elif action == "SCROLL_VERTICAL" and abs(res.delta_y) > 1.0:
                    mouse_ctrl.scroll_vertical(int(-np.sign(res.delta_y) * min(config.MAX_SCROLL_SPEED, abs(res.delta_y) * 2.5)))
                elif action == "PAGE_UP" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.page_up()
                elif action == "PAGE_DOWN" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.page_down()
                elif action == "BROWSER_BACK" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.browser_back()
                elif action == "BROWSER_FORWARD" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.browser_forward()
                elif action == "TAKE_SCREENSHOT" and (res.is_discrete_trigger or res.can_execute):
                    system_ctrl.take_screenshot()
                elif action == "MEDIA_PLAY_PAUSE" and (res.is_discrete_trigger or res.can_execute):
                    media_ctrl.play_pause()
                elif action == "MEDIA_NEXT_TRACK" and (res.is_discrete_trigger or res.can_execute):
                    media_ctrl.next_track()
                elif action == "MEDIA_PREV_TRACK" and (res.is_discrete_trigger or res.can_execute):
                    media_ctrl.prev_track()
                elif action == "VOLUME_UP":
                    media_ctrl.volume_up()
                elif action == "VOLUME_DOWN":
                    media_ctrl.volume_down()
                elif action == "VOLUME_MUTE" and (res.is_discrete_trigger or res.can_execute):
                    media_ctrl.toggle_mute()
                elif action == "APP_SWITCH" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.app_switch()
                elif action == "TASK_VIEW" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.task_view()
                elif action == "SHOW_DESKTOP" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.show_desktop()
                elif action == "WINDOW_MINIMIZE" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.window_minimize()
                elif action == "WINDOW_MAXIMIZE" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.window_maximize()
                elif action == "WINDOW_CLOSE_SAFE" and (res.is_discrete_trigger or res.can_execute):
                    system_ctrl.close_window_safe()
                elif action == "LOCK_SCREEN_SAFE" and (res.is_discrete_trigger or res.can_execute):
                    system_ctrl.lock_screen_safe()
                elif action == "COPY" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.copy()
                elif action == "PASTE" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.paste()
                elif action == "CUT" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.cut()
                elif action == "SELECT_ALL" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.select_all()
                elif action == "UNDO" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.undo()
                elif action == "REDO" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.redo()
                elif action == "KEY_ENTER" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.enter()
                elif action == "KEY_ESCAPE" and (res.is_discrete_trigger or res.can_execute):
                    keyboard_ctrl.escape()
                elif action == "ZOOM_IN":
                    display_ctrl.zoom_in()
                elif action == "ZOOM_OUT":
                    display_ctrl.zoom_out()
                elif action == "ROTATE_DOCUMENT" and (res.is_discrete_trigger or res.can_execute):
                    display_ctrl.rotate_document()
                elif action == "BRIGHTNESS_INCREASE":
                    display_ctrl.brightness_up()
                elif action == "BRIGHTNESS_DECREASE":
                    display_ctrl.brightness_down()

            # Measure performance
            lat = perf_tracker.end_frame()
            fps = perf_tracker.fps

            # Render HUD bar on OpenCV window
            cv2.rectangle(annotated, (0, 0), (w, 44), (20, 22, 34), -1)
            status_str = "ACTIVE" if action_dispatcher.is_active else "PAUSED"
            status_col = (50, 220, 50) if action_dispatcher.is_active else (50, 180, 255)
            cv2.putText(annotated, f"[{classifier.active_mode}] Status: {status_str}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Gesture: {res.display_label}", (240, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (w - 100, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 240, 255), 1, cv2.LINE_AA)

            cv2.imshow("AI Vision-Driven HCI (Press 'q' to Quit)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                action_dispatcher.toggle_active()
            elif key == 27:  # Esc
                action_dispatcher.emergency_stop()
            elif ord('1') <= key <= ord('7'):
                mode_idx = key - ord('1')
                if mode_idx < len(config.ALL_MODES):
                    classifier.set_mode(config.ALL_MODES[mode_idx])

    finally:
        cam.stop()
        tracker.close()
        action_dispatcher.stop()
        cv2.destroyAllWindows()
        print("[CLI] Exited cleanly.")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Based Vision-Driven Human-Computer Interaction Using Hand Gesture Recognition"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in lightweight OpenCV preview mode instead of full CustomTkinter GUI"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the comprehensive test suite"
    )

    args = parser.parse_args()

    if args.test:
        import test_system
        test_system.run_all_tests()
    elif args.cli:
        run_cli_mode()
    else:
        run_gui_mode()


if __name__ == "__main__":
    main()
