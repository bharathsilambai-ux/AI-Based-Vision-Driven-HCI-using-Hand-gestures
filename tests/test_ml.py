"""
Unit Tests: Machine Learning Inference & Artifact Validation
============================================================
"""

import unittest
import config
from src.gestures.ml_classifier import MLGestureClassifier
from ml.generate_sample_dataset import generate_gesture_pose


class TestMLClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = MLGestureClassifier()

    def test_model_loaded(self):
        """Verify model files exist and classifier is ready."""
        self.assertTrue(self.classifier.is_ready, "Trained ML model should be loaded and ready.")

    def test_ml_prediction_open_palm(self):
        """Verify ML model correctly classifies canonical open palm landmarks."""
        pts = generate_gesture_pose(config.GESTURE_OPEN_PALM)
        gesture, conf = self.classifier.predict(pts)
        self.assertEqual(gesture, config.GESTURE_OPEN_PALM)
        self.assertGreater(conf, 0.70)

    def test_ml_prediction_pinch(self):
        """Verify ML model correctly classifies canonical pinch landmarks."""
        pts = generate_gesture_pose(config.GESTURE_PINCH)
        gesture, conf = self.classifier.predict(pts)
        self.assertEqual(gesture, config.GESTURE_PINCH)
        self.assertGreater(conf, 0.70)

    def test_ml_prediction_fist(self):
        """Verify ML model correctly classifies canonical fist landmarks."""
        pts = generate_gesture_pose(config.GESTURE_FIST)
        gesture, conf = self.classifier.predict(pts)
        self.assertEqual(gesture, config.GESTURE_FIST)
        self.assertGreater(conf, 0.70)


if __name__ == "__main__":
    unittest.main()
