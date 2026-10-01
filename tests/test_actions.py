"""
Unit Tests: Coordinate Smoothing, FSM Debouncing, and Action Dispatcher
=======================================================================
"""

import unittest
import time
import config
from src.control.smoother import CoordinateSmoother
from src.gestures.gesture_state import GestureStateManager
from src.control.controller import ComputerController


class TestControlAndFSM(unittest.TestCase):

    def setUp(self):
        self.smoother = CoordinateSmoother(frame_width=640, frame_height=480, alpha=0.35)
        self.fsm = GestureStateManager(stability_frames=3, discrete_cooldown=0.3)
        self.controller = ComputerController(enabled=False)  # Mock / safe mode

    def test_coordinate_smoother_bounds(self):
        """Ensure smoothed coordinates stay within screen dimensions and converge properly."""
        sw, sh = self.smoother.screen_width, self.smoother.screen_height

        # Test bounds always within screen limits
        cx, cy = self.smoother.smooth_and_map(320, 240)
        self.assertTrue(0 <= cx < sw)
        self.assertTrue(0 <= cy < sh)

        # Iterate towards (0, 0) to verify convergence to top-left corner
        for _ in range(30):
            x_min, y_min = self.smoother.smooth_and_map(0, 0)
        self.assertLessEqual(x_min, 2)
        self.assertLessEqual(y_min, 2)

        # Iterate towards (640, 480) to verify convergence to bottom-right corner
        for _ in range(30):
            x_max, y_max = self.smoother.smooth_and_map(640, 480)
        self.assertGreaterEqual(x_max, sw - 3)
        self.assertGreaterEqual(y_max, sh - 3)

    def test_fsm_stability_threshold(self):
        """Verify gesture requires consecutive frames before activating."""
        self.fsm.reset()
        # Frame 1: candidate detected but not active yet
        g, trigger, evt = self.fsm.process(config.GESTURE_THUMB_UP, 0.9)
        self.assertFalse(trigger)

        # Frame 2
        g, trigger, evt = self.fsm.process(config.GESTURE_THUMB_UP, 0.9)
        self.assertFalse(trigger)

        # Frame 3: reaches stability of 3 frames, triggers action
        g, trigger, evt = self.fsm.process(config.GESTURE_THUMB_UP, 0.9)
        self.assertEqual(g, config.GESTURE_THUMB_UP)
        self.assertTrue(trigger)
        self.assertEqual(evt, "discrete")

    def test_fsm_cooldown_rejection(self):
        """Verify immediate consecutive frame does not re-trigger during cooldown."""
        self.fsm.reset()
        # Trigger it
        self.fsm.process(config.GESTURE_THUMB_UP, 0.9)
        self.fsm.process(config.GESTURE_THUMB_UP, 0.9)
        g, trigger, _ = self.fsm.process(config.GESTURE_THUMB_UP, 0.9)
        self.assertTrue(trigger)

        # Immediate next frame should be rejected by cooldown
        g, trigger2, evt2 = self.fsm.process(config.GESTURE_THUMB_UP, 0.9)
        self.assertFalse(trigger2)
        self.assertEqual(evt2, "none")

    def test_controller_mock_dispatch(self):
        """Verify dispatch handles gestures cleanly when control disabled."""
        self.controller.set_enabled(False)
        desc = self.controller.dispatch(config.GESTURE_OPEN_PALM, "continuous", (500, 300))
        self.assertEqual(desc, "Control Disabled")

        self.controller.set_enabled(True)
        # Check safe execution
        desc = self.controller.dispatch(config.GESTURE_OPEN_PALM, "continuous", (500, 300))
        self.assertIn("Cursor", desc)


if __name__ == "__main__":
    unittest.main()
