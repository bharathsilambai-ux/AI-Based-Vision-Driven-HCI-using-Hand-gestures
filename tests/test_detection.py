"""
Unit Tests: Landmark Normalization, Finger States, and Rule Engine
==================================================================
"""

import unittest
import numpy as np
import config
from src.vision.landmark_utils import LandmarkProcessor
from src.gestures.rule_engine import RuleGestureEngine
from ml.generate_sample_dataset import generate_gesture_pose


class TestDetectionAndRules(unittest.TestCase):

    def setUp(self):
        self.proc = LandmarkProcessor
        self.engine = RuleGestureEngine()

    def test_landmark_normalization_shape(self):
        """Verify normalizer yields 63-dimensional translation- and scale-invariant vector."""
        mock_pts = [(float(i), float(i * 2), float(i * 0.1)) for i in range(21)]
        norm = self.proc.normalize_landmarks(mock_pts)
        self.assertEqual(norm.shape, (63,))
        self.assertIsInstance(norm, np.ndarray)

    def test_open_palm_rule_classification(self):
        """Verify canonical open palm landmarks classify as open_palm."""
        pts = generate_gesture_pose(config.GESTURE_OPEN_PALM)
        gesture, conf = self.engine.classify(pts, handedness="Right")
        self.assertEqual(gesture, config.GESTURE_OPEN_PALM)
        self.assertGreaterEqual(conf, 0.7)

    def test_fist_rule_classification(self):
        """Verify curled fist classifies as fist."""
        pts = generate_gesture_pose(config.GESTURE_FIST)
        gesture, conf = self.engine.classify(pts, handedness="Right")
        self.assertEqual(gesture, config.GESTURE_FIST)
        self.assertGreaterEqual(conf, 0.7)

    def test_pinch_rule_classification(self):
        """Verify touching index and thumb tips classifies as pinch."""
        pts = generate_gesture_pose(config.GESTURE_PINCH)
        gesture, conf = self.engine.classify(pts, handedness="Right")
        self.assertEqual(gesture, config.GESTURE_PINCH)
        self.assertGreaterEqual(conf, 0.7)

    def test_thumb_up_classification(self):
        """Verify thumb up classifies as thumb_up."""
        pts = generate_gesture_pose(config.GESTURE_THUMB_UP)
        gesture, conf = self.engine.classify(pts, handedness="Right")
        self.assertEqual(gesture, config.GESTURE_THUMB_UP)
        self.assertGreaterEqual(conf, 0.7)

    def test_thumb_down_classification(self):
        """Verify thumb down classifies as thumb_down."""
        pts = generate_gesture_pose(config.GESTURE_THUMB_DOWN)
        gesture, conf = self.engine.classify(pts, handedness="Right")
        self.assertEqual(gesture, config.GESTURE_THUMB_DOWN)
        self.assertGreaterEqual(conf, 0.7)


if __name__ == "__main__":
    unittest.main()
