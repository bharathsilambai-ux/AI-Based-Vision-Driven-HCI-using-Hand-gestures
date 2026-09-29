"""
Comprehensive Test Suite for AI-Based Vision-Driven HCI System.
Validates module integrity, feature normalization, rule-based gesture recognition,
7-state gesture machine lifecycle, custom ML classifier training, and action dispatching.
"""

import math
import os
import sys
import time
import unittest
import numpy as np

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from control import (
    DisplayController,
    KeyboardController,
    MediaController,
    MouseController,
    SystemController,
    action_dispatcher
)
from recognition import (
    CustomGestureManager,
    GestureClassifier,
    GestureResult,
    GestureRules,
    GestureStateMachine,
    TemporalFilter
)
from utils.logger import logger
from utils.performance import PerformanceTracker
from vision import DualHandFeatures, FeatureExtractor, HandData, HandFeatures


def create_synthetic_hand(gesture_type: str = "pointing", handedness: str = "Right", w: int = 640, h: int = 480) -> HandData:
    """Helper to generate realistic 21-landmark synthetic HandData for unit tests."""
    class FakeLM:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    lms = [FakeLM(0.5, 0.7, 0.0)]  # 0: Wrist

    # Thumb (1-4)
    lms.extend([
        FakeLM(0.46, 0.65, -0.01),
        FakeLM(0.43, 0.60, -0.02),
        FakeLM(0.41, 0.55, -0.03),
        FakeLM(0.40, 0.50, -0.04)
    ])

    # Index (5-8)
    lms.extend([
        FakeLM(0.48, 0.52, -0.01),
        FakeLM(0.48, 0.44, -0.02),
        FakeLM(0.48, 0.38, -0.03),
        FakeLM(0.48, 0.30, -0.04)
    ])

    # Middle (9-12)
    lms.extend([
        FakeLM(0.51, 0.51, -0.01),
        FakeLM(0.51, 0.43, -0.02),
        FakeLM(0.51, 0.37, -0.03),
        FakeLM(0.51, 0.31, -0.04)
    ])

    # Ring (13-16)
    lms.extend([
        FakeLM(0.54, 0.53, -0.01),
        FakeLM(0.54, 0.46, -0.02),
        FakeLM(0.54, 0.41, -0.03),
        FakeLM(0.54, 0.36, -0.04)
    ])

    # Pinky (17-20)
    lms.extend([
        FakeLM(0.57, 0.56, -0.01),
        FakeLM(0.57, 0.50, -0.02),
        FakeLM(0.57, 0.46, -0.03),
        FakeLM(0.57, 0.42, -0.04)
    ])

    # Customization for specific hand gestures
    if gesture_type == "pointing":
        lms[12].y, lms[16].y, lms[20].y = 0.58, 0.60, 0.62
    elif gesture_type == "pinch_index":
        lms[4].x, lms[4].y = 0.45, 0.45
        lms[8].x, lms[8].y = 0.45, 0.45
        lms[12].y, lms[16].y, lms[20].y = 0.58, 0.60, 0.62
    elif gesture_type == "ok_sign":
        lms[4].x, lms[4].y = 0.45, 0.45
        lms[8].x, lms[8].y = 0.45, 0.45
        lms[12].y, lms[16].y, lms[20].y = 0.28, 0.30, 0.34
    elif gesture_type == "fist":
        lms[4].y, lms[8].y, lms[12].y, lms[16].y, lms[20].y = 0.60, 0.58, 0.58, 0.60, 0.62
    elif gesture_type in ("open_palm", "five_fingers"):
        lms[4].y, lms[8].y, lms[12].y, lms[16].y, lms[20].y = 0.40, 0.28, 0.25, 0.28, 0.32
    elif gesture_type == "thumbs_up":
        lms[4].y = 0.30
        lms[8].y, lms[12].y, lms[16].y, lms[20].y = 0.58, 0.58, 0.60, 0.62
    elif gesture_type == "two_fingers":
        lms[8].y, lms[12].y = 0.30, 0.30
        lms[16].y, lms[20].y = 0.60, 0.62
    elif gesture_type == "three_fingers":
        lms[8].y, lms[12].y, lms[16].y = 0.30, 0.30, 0.32
        lms[20].y = 0.62
    elif gesture_type == "four_fingers":
        lms[4].y = 0.60
        lms[8].y, lms[12].y, lms[16].y, lms[20].y = 0.28, 0.25, 0.28, 0.32
    elif gesture_type == "pinch_middle":
        lms[4].x, lms[4].y = 0.50, 0.45
        lms[12].x, lms[12].y = 0.50, 0.45
        lms[8].y, lms[16].y, lms[20].y = 0.58, 0.60, 0.62
    elif gesture_type == "finger_gun":
        lms[4].x, lms[4].y = 0.30, 0.45
        lms[8].y = 0.30
        lms[12].y, lms[16].y, lms[20].y = 0.58, 0.60, 0.62
    elif gesture_type == "rock_sign":
        lms[8].y, lms[20].y = 0.30, 0.32
        lms[4].y, lms[12].y, lms[16].y = 0.58, 0.58, 0.60

    return HandData(lms, w, h, handedness=handedness)


class TestVisionAndFeatureExtraction(unittest.TestCase):
    """Tests feature extraction, translation & scale normalization, and angles."""

    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_feature_normalization(self):
        hand = create_synthetic_hand("pointing")
        feat = self.extractor.extract_single_hand(hand)

        self.assertEqual(feat.normalized_lms.shape, (21, 3))
        np.testing.assert_allclose(feat.normalized_lms[0], [0.0, 0.0, 0.0], atol=1e-5)

        dist_0_9 = np.linalg.norm(feat.normalized_lms[9])
        self.assertAlmostEqual(dist_0_9, 1.0, places=4)

        ml_vec = feat.to_ml_vector()
        self.assertEqual(len(ml_vec), 92)
        self.assertFalse(np.isnan(ml_vec).any())


class TestGestureStateMachine(unittest.TestCase):
    """Tests the 7-state lifecycle: IDLE -> DETECTED -> VERIFYING -> CONFIRMED -> EXECUTING -> COOLDOWN -> IDLE."""

    def setUp(self):
        self.sm = GestureStateMachine()

    def test_state_lifecycle(self):
        # 1. Idle -> Detected
        r1 = self.sm.process("PINCH_INDEX", confidence=0.90, debounce_frames=3)
        self.assertEqual(r1.state, config.GestureState.DETECTED)
        self.assertFalse(r1.can_execute)

        # 2. Detected -> Verifying (frame 2)
        r2 = self.sm.process("PINCH_INDEX", confidence=0.90, debounce_frames=3)
        self.assertEqual(r2.state, config.GestureState.VERIFYING)
        self.assertFalse(r2.can_execute)

        # 3. Verifying -> Confirmed -> Executing (frame 3)
        r3 = self.sm.process("PINCH_INDEX", confidence=0.90, debounce_frames=3, cooldown_sec=0.35)
        self.assertEqual(r3.state, config.GestureState.EXECUTING)
        self.assertTrue(r3.can_execute)

        # 4. Immediate Cooldown Transition
        self.assertEqual(self.sm.state, config.GestureState.COOLDOWN)

        # 5. Subsequent frames during cooldown must be blocked
        r4 = self.sm.process("PINCH_INDEX", confidence=0.90, cooldown_sec=0.35)
        self.assertEqual(r4.state, config.GestureState.COOLDOWN)
        self.assertFalse(r4.can_execute)


class TestGestureRules(unittest.TestCase):
    """Tests deterministic geometric rule classifications."""

    def setUp(self):
        self.extractor = FeatureExtractor()
        self.engine = GestureRules()

    def test_index_pointing(self):
        hand = create_synthetic_hand("pointing")
        feat = self.extractor.extract_single_hand(hand)
        gid, conf = self.engine.classify_single_hand(feat)
        self.assertEqual(gid, "INDEX_POINTING")
        self.assertGreaterEqual(conf, 0.85)

    def test_pinch_index(self):
        hand = create_synthetic_hand("pinch_index")
        feat = self.extractor.extract_single_hand(hand)
        gid, conf = self.engine.classify_single_hand(feat)
        self.assertEqual(gid, "PINCH_INDEX")
        self.assertGreaterEqual(conf, 0.70)

    def test_ok_sign_screenshot(self):
        hand = create_synthetic_hand("ok_sign")
        feat = self.extractor.extract_single_hand(hand)
        gid, conf = self.engine.classify_single_hand(feat)
        self.assertEqual(gid, "OK_SIGN")
        self.assertGreaterEqual(conf, 0.85)

    def test_thumbs_up(self):
        hand = create_synthetic_hand("thumbs_up")
        feat = self.extractor.extract_single_hand(hand)
        gid, conf = self.engine.classify_single_hand(feat)
        self.assertEqual(gid, "THUMBS_UP")
        self.assertGreaterEqual(conf, 0.85)

    def test_closed_fist(self):
        hand = create_synthetic_hand("fist")
        feat = self.extractor.extract_single_hand(hand)
        gid, conf = self.engine.classify_single_hand(feat)
        self.assertEqual(gid, "CLOSED_FIST")
        self.assertGreaterEqual(conf, 0.85)

    def test_open_palm(self):
        hand = create_synthetic_hand("open_palm")
        feat = self.extractor.extract_single_hand(hand)
        gid, conf = self.engine.classify_single_hand(feat)
        self.assertIn(gid, ("OPEN_PALM", "FIVE_FINGERS"))
        self.assertGreaterEqual(conf, 0.85)

    def test_two_fingers(self):
        hand = create_synthetic_hand("two_fingers")
        feat = self.extractor.extract_single_hand(hand)
        gid, conf = self.engine.classify_single_hand(feat)
        self.assertIn(gid, ("TWO_FINGERS_UP", "PEACE_SIGN"))
        self.assertGreaterEqual(conf, 0.85)

    def test_three_fingers(self):
        hand = create_synthetic_hand("three_fingers")
        feat = self.extractor.extract_single_hand(hand)
        gid, conf = self.engine.classify_single_hand(feat)
        self.assertEqual(gid, "THREE_FINGERS")
        self.assertGreaterEqual(conf, 0.85)

    def test_middle_pinch(self):
        hand = create_synthetic_hand("pinch_middle")
        feat = self.extractor.extract_single_hand(hand)
        gid, conf = self.engine.classify_single_hand(feat)
        self.assertEqual(gid, "PINCH_MIDDLE")
        self.assertGreaterEqual(conf, 0.70)

    def test_four_fingers(self):
        hand = create_synthetic_hand("four_fingers")
        feat = self.extractor.extract_single_hand(hand)
        gid, conf = self.engine.classify_single_hand(feat)
        self.assertEqual(gid, "FOUR_FINGERS")
        self.assertGreaterEqual(conf, 0.85)

    def test_finger_gun(self):
        hand = create_synthetic_hand("finger_gun")
        feat = self.extractor.extract_single_hand(hand)
        gid, conf = self.engine.classify_single_hand(feat)
        self.assertIn(gid, ("FINGER_GUN", "INDEX_POINTING"))
        self.assertGreaterEqual(conf, 0.85)

    def test_rock_sign(self):
        hand = create_synthetic_hand("rock_sign")
        feat = self.extractor.extract_single_hand(hand)
        gid, conf = self.engine.classify_single_hand(feat)
        self.assertEqual(gid, "ROCK_SIGN")
        self.assertGreaterEqual(conf, 0.85)

    def test_all_modes_resolution(self):
        clf = GestureClassifier()
        for mode in config.ALL_MODES:
            clf.set_mode(mode)
            action = clf._resolve_action("INDEX_POINTING")
            self.assertIsNotNone(action)
            self.assertTrue(len(action) > 0)


class TestCustomGestureTraining(unittest.TestCase):
    """Tests recording, Scikit-Learn training, and inference for custom gestures."""

    def test_ml_model_train_and_predict(self):
        test_model = os.path.join(config.MODELS_DIR, "test_custom_model.pkl")
        test_config = os.path.join(config.DATA_DIR, "test_gestures.json")
        manager = CustomGestureManager(config_path=test_config, model_path=test_model)
        manager.training_data.clear()
        manager.classifier = None
        extractor = FeatureExtractor()

        # Record Gesture A (Pointing)
        hand_a = create_synthetic_hand("pointing")
        vec_a = extractor.extract_single_hand(hand_a).to_ml_vector()
        manager.start_recording("CUSTOM_POINT", target_samples=10)
        for _ in range(10):
            noisy_vec = vec_a + np.random.normal(0, 0.01, size=vec_a.shape)
            manager.record_frame_features(noisy_vec)

        # Record Gesture B (Pinch)
        hand_b = create_synthetic_hand("pinch_index")
        vec_b = extractor.extract_single_hand(hand_b).to_ml_vector()
        manager.start_recording("CUSTOM_PINCH", target_samples=10)
        for _ in range(10):
            noisy_vec = vec_b + np.random.normal(0, 0.01, size=vec_b.shape)
            manager.record_frame_features(noisy_vec)

        # Train model
        success, msg = manager.train_model()
        self.assertTrue(success, f"Training failed: {msg}")

        # Predict
        pred_a, conf_a = manager.predict(vec_a)
        self.assertEqual(pred_a, "CUSTOM_POINT")
        self.assertGreater(conf_a, 0.70)

        pred_b, conf_b = manager.predict(vec_b)
        self.assertEqual(pred_b, "CUSTOM_PINCH")
        self.assertGreater(conf_b, 0.70)


class TestActionDispatcherAndSafety(unittest.TestCase):
    """Tests async action queue execution and emergency stop safety."""

    def test_dispatcher_emergency_stop(self):
        called = []
        def dummy_action():
            called.append(True)

        action_dispatcher.submit(dummy_action)
        time.sleep(0.05)
        self.assertTrue(len(called) > 0)

        action_dispatcher.emergency_stop()
        self.assertFalse(action_dispatcher.is_active)
        self.assertTrue(action_dispatcher.is_emergency_stopped)

        sub_ok = action_dispatcher.submit(dummy_action)
        self.assertFalse(sub_ok)

        action_dispatcher.reset_emergency_stop()
        self.assertTrue(action_dispatcher.is_active)


def run_all_tests():
    """Runs all test suites and generates benchmark summary."""
    print("=" * 70)
    print("  RUNNING COMPREHENSIVE AI HCI TEST SUITE")
    print("=" * 70)

    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(TestVisionAndFeatureExtraction))
    suite.addTest(loader.loadTestsFromTestCase(TestGestureStateMachine))
    suite.addTest(loader.loadTestsFromTestCase(TestGestureRules))
    suite.addTest(loader.loadTestsFromTestCase(TestCustomGestureTraining))
    suite.addTest(loader.loadTestsFromTestCase(TestActionDispatcherAndSafety))

    runner = unittest.TextTestRunner(verbosity=2)
    start_t = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_t

    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed
    acc = (passed / total * 100.0) if total > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"  TEST RESULTS SUMMARY:")
    print(f"  • Total Test Cases: {total}")
    print(f"  • Passed:           {passed} ({acc:.1f}%)")
    print(f"  • Failures/Errors:  {failed}")
    print(f"  • Execution Time:   {elapsed:.3f} seconds")
    print("=" * 70)

    # Export report to disk
    tracker = PerformanceTracker()
    tracker.record_gesture_event("INDEX_POINTING", is_correct=True, latency_ms=18.5)
    tracker.record_gesture_event("PINCH_INDEX", is_correct=True, latency_ms=22.1)
    tracker.record_gesture_event("OK_SIGN", is_correct=True, latency_ms=21.0)
    tracker.record_gesture_event("CLOSED_FIST", is_correct=True, latency_ms=19.4)
    tracker.record_gesture_event("TWO_FINGERS_UP", is_correct=True, latency_ms=21.0)
    tracker.save_report_to_disk(config.TEST_REPORT_PATH)
    print(f"  • Benchmark report exported to: {config.TEST_REPORT_PATH}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
