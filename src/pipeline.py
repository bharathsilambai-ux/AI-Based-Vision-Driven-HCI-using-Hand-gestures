"""
Core HCI Controller Pipeline
============================
Integrates Camera, MediaPipe Hand Tracking, Invariant Normalization,
Dual-Classifier Engines (Rule/ML), Cursor Smoother, FSM Debouncer,
PyAutoGUI Action Dispatcher, and HUD Visualizer.
"""

import cv2
import time
import logging
from typing import Tuple, Optional, Dict, Any
import config
from src.vision.camera import CameraStream
from src.vision.hand_detector import HandDetector
from src.vision.landmark_utils import LandmarkProcessor
from src.gestures.rule_engine import RuleGestureEngine
from src.gestures.ml_classifier import MLGestureClassifier
from src.gestures.swipe_detector import SwipeDetector
from src.gestures.gesture_state import GestureStateManager
from src.control.smoother import CoordinateSmoother
from src.control.controller import ComputerController
from src.ui.hud_overlay import HUDOverlay

logger = logging.getLogger(__name__)


class HCIPipeline:
    """Master controller coordinating real-time vision, gesture recognition, and OS control."""

    def __init__(
        self,
        mode: str = "rule",
        enable_control: bool = True,
        camera_id: int = config.CAMERA_ID,
    ):
        self.mode = mode  # 'rule' or 'ml'
        self.running = False

        # Instantiate sub-modules
        self.camera = CameraStream(camera_id=camera_id)
        self.detector = HandDetector()
        self.rule_engine = RuleGestureEngine()
        self.ml_classifier = MLGestureClassifier()
        self.swipe_detector = SwipeDetector()
        self.state_manager = GestureStateManager()
        self.smoother = CoordinateSmoother()
        self.controller = ComputerController(enabled=enable_control)
        self.hud = HUDOverlay()

        # Telemetry cache
        self.current_gesture = config.GESTURE_IDLE
        self.current_confidence = 0.0
        self.current_action = "Ready"

    def start(self) -> bool:
        """Initializes camera and sets running flag."""
        success = self.camera.start()
        if success:
            self.running = True
            self.smoother.update_frame_size(self.camera.width, self.camera.height)
            logger.info("HCI Pipeline successfully started.")
        return success

    def process_one_frame(self) -> Tuple[bool, Optional[Any], Dict[str, Any]]:
        """
        Executes a single end-to-end iteration of the HCI pipeline.
        
        Returns:
            (success, processed_bgr_frame, telemetry_dict)
        """
        success, frame = self.camera.read_frame()
        if not success or frame is None:
            return False, None, {}

        fps = self.camera.get_fps()
        h, w = frame.shape[:2]

        # 1. Detect Hand Landmarks
        self.detector.find_hands(frame)
        hand_data = self.detector.get_hand_data((h, w), hand_index=0)

        raw_gesture = config.GESTURE_IDLE
        raw_confidence = 0.0
        screen_coords = None

        if hand_data:
            landmarks_norm = hand_data["landmarks_norm"]
            landmarks_pixel = hand_data["landmarks_pixel"]
            handedness = hand_data["handedness"]

            # Calculate palm center for swipe detection and cursor reference
            proc = LandmarkProcessor
            palm_center = proc.get_palm_center(landmarks_norm)
            palm_center_px = (int(palm_center[0] * w), int(palm_center[1] * h))

            # 2. Check for Swipe Motions First (velocity spikes)
            swipe_gesture = self.swipe_detector.update(palm_center_px)
            if swipe_gesture:
                raw_gesture = swipe_gesture
                raw_confidence = 0.98
            else:
                # 3. Static Posture Classification
                if self.mode == "ml" and self.ml_classifier.is_ready:
                    raw_gesture, raw_confidence = self.ml_classifier.predict(landmarks_norm)
                else:
                    raw_gesture, raw_confidence = self.rule_engine.classify(landmarks_norm, handedness)

            # 4. Compute Smoothed Screen Coordinates (using Index fingertip or palm center)
            # Index fingertip is landmark 8
            idx_tip_px = landmarks_pixel[8]
            screen_coords = self.smoother.smooth_and_map(idx_tip_px[0], idx_tip_px[1])

            # Draw aesthetic hand skeleton
            frame = self.detector.draw_hand_mesh(frame, hand_index=0, modern_style=True)
        else:
            self.swipe_detector.reset()
            self.state_manager.reset()

        # 5. Process through Temporal FSM (Debounce & Cooldown)
        stable_gesture, trigger_action, event_type = self.state_manager.process(raw_gesture, raw_confidence)
        self.current_gesture = stable_gesture
        self.current_confidence = raw_confidence

        # 6. Dispatch Computer Action via PyAutoGUI
        if trigger_action:
            action_desc = self.controller.dispatch(
                gesture=stable_gesture,
                event_type=event_type,
                screen_coords=screen_coords,
            )
            self.current_action = action_desc
        elif stable_gesture == config.GESTURE_FIST:
            self.current_action = "Standby (Frozen)"
        elif stable_gesture == config.GESTURE_IDLE:
            self.current_action = "Idle"

        # 7. Render Modern HUD Telemetry
        frame = self.hud.render_hud(
            frame=frame,
            gesture_name=self.current_gesture,
            action_name=self.current_action,
            confidence=self.current_confidence,
            fps=fps,
            mode="Machine Learning" if (self.mode == "ml" and self.ml_classifier.is_ready) else "Rule-Based",
            control_enabled=self.controller.enabled,
            hand_data=hand_data,
        )

        telemetry = {
            "gesture": self.current_gesture,
            "action": self.current_action,
            "confidence": self.current_confidence,
            "fps": fps,
            "screen_coords": screen_coords,
            "mode": self.mode,
            "control_enabled": self.controller.enabled,
        }

        return True, frame, telemetry

    def run_cv_window(self):
        """Standalone OpenCV interactive loop."""
        if not self.start():
            print("[ERROR] Could not start camera stream.")
            return

        window_name = "AI Hand Gesture HCI Controller"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

        print("\n" + "=" * 60)
        print("  AI VISION-DRIVEN HCI CONTROLLER — ACTIVE")
        print("=" * 60)
        print("  Controls:")
        print("    [Q]       - Quit application")
        print("    [C]       - Toggle OS mouse/keyboard control ON/OFF")
        print("    [M]       - Toggle Classifier Mode (Rule-Based <-> ML)")
        print("    [S]       - Save instant screenshot")
        print("=" * 60 + "\n")

        try:
            while self.running:
                success, frame, telemetry = self.process_one_frame()
                if not success or frame is None:
                    continue

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or Esc
                    break
                elif key == ord("c"):
                    new_state = not self.controller.enabled
                    self.controller.set_enabled(new_state)
                    print(f"[*] OS Control: {'ENABLED' if new_state else 'DISABLED'}")
                elif key == ord("m"):
                    if self.mode == "rule":
                        if self.ml_classifier.is_ready:
                            self.mode = "ml"
                            print("[*] Switched to ML Classifier Mode.")
                        else:
                            print("[!] ML model not found. Train first: python ml/train_model.py")
                    else:
                        self.mode = "rule"
                        print("[*] Switched to Rule-Based Mode.")
                elif key == ord("s"):
                    path = self.controller.take_screenshot()
                    if path:
                        print(f"[*] Screenshot saved: {path}")

        finally:
            self.stop()
            cv2.destroyAllWindows()

    def stop(self):
        """Releases camera and cleanup."""
        self.running = False
        self.camera.release()
        self.detector.close()
        if self.controller.is_mouse_down:
            self.controller.mouse_up()
        logger.info("Pipeline stopped cleanly.")
