"""
Phase 7 Verification Test Suite: Real-Time Gesture Recognition, Motion Tracking & Multi-Mode Mapping
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding & Package Exports (gestures package)
2. Static Gesture Classification with Trained Production ML Model (SVM_RBF)
3. Biomechanical & Geometric Cross-Validation
4. Dynamic Kinematic Trajectory Tracking (Swipes, Pinch-Drag, Circular Strokes)
5. Multi-Mode Command Mapping & Context-Aware App Routing (Modules 7, 13, 14)
6. Mapping Conflict Detection & JSON Persistence
7. Confidence-Based Recognition & Gating (Module 4)
8. Temporal Stabilization, Majority Voting & Spike Filtering (Module 5)
9. Command Cooldown & Execution Gating (Module 6)
10. Input Validation & Error Resilience (Module 17: None, Empty, NaN, Dimension Mismatches)
11. End-to-End Pipeline Latency Microbenchmark (< 5.0 ms / > 200 FPS)
"""

import json
import math
import os
import platform
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import dataset
from dataset import GESTURE_CLASSES, GestureDataset
from detection.hand_detector import HandData
from features.landmark_processor import LandmarkProcessor
import gestures
from gestures import (
    APPLICATION_SPECIFIC_MAPPINGS,
    CONTINUOUS_ACTIONS,
    DEFAULT_COOLDOWNS,
    DEFAULT_MODE_MAPPINGS,
    DYNAMIC_GESTURE_CLASSES,
    DynamicGestureRecognizer,
    DynamicGestureResult,
    GestureMappingManager,
    GestureRecognitionResult,
    OperationalMode,
    RealtimeGestureRecognizer,
    STATIC_GESTURE_CLASSES,
    StaticGestureRecognizer,
    StaticGestureResult,
    TrajectoryPoint,
)
from models import DEFAULT_MODEL_PATH, GesturePredictor


class TestResultCollector:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def record(self, test_name: str, passed: bool, details: str = "", latency_ms: float = 0.0):
        status = "PASS" if passed else "FAIL"
        self.results.append({
            "test": test_name,
            "status": status,
            "details": details,
            "latency_ms": round(latency_ms, 3)
        })
        badge = "[PASS]" if passed else "[FAIL]"
        print(f"  {badge} {test_name:<46} {details}")


collector = TestResultCollector()


def run_tests() -> bool:
    print("=" * 75)
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 7 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     Real-Time Recognition & Stabilization Integration")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Section 1: Scaffolding, Package Exports & Vocabulary
    # -----------------------------------------------------------------
    print("\n1. Module Scaffolding & Package Exports")
    exports_ok = (
        hasattr(gestures, "StaticGestureRecognizer") and
        hasattr(gestures, "DynamicGestureRecognizer") and
        hasattr(gestures, "GestureMappingManager") and
        hasattr(gestures, "RealtimeGestureRecognizer") and
        hasattr(gestures, "OperationalMode")
    )
    collector.record("Gestures Package Exports", exports_ok, "All recognizers and mapping managers exported")

    vocab_ok = (
        len(STATIC_GESTURE_CLASSES) == 10 and
        len(DYNAMIC_GESTURE_CLASSES) >= 4 and
        set(STATIC_GESTURE_CLASSES).issubset(set(GESTURE_CLASSES))
    )
    collector.record("Gesture Vocabulary Division", vocab_ok, f"Static: {len(STATIC_GESTURE_CLASSES)}, Dynamic: {len(DYNAMIC_GESTURE_CLASSES)}")

    # -----------------------------------------------------------------
    # Section 2: Static Gesture Recognition with Production ML Model
    # -----------------------------------------------------------------
    print("\n2. Static Gesture Recognition via ML Classifier (SVM_RBF)")
    dataset_path = os.path.join(_ROOT, "dataset", "gesture_dataset.npz")
    dataset_obj = GestureDataset.load_from_npz(dataset_path)

    static_rec = StaticGestureRecognizer(model_path=DEFAULT_MODEL_PATH)
    collector.record("Static Recognizer Initialization", static_rec.is_ready, f"ML Model: '{static_rec.predictor.model_name}'")

    # Test static classification on a known dataset sample
    test_idx = 0
    test_landmarks = dataset_obj.landmarks[test_idx]
    expected_gesture = dataset_obj.y[test_idx]

    s_res = static_rec.classify_static(test_landmarks)
    static_pred_ok = (
        isinstance(s_res, StaticGestureResult) and
        s_res.gesture in GESTURE_CLASSES and
        0.0 <= s_res.confidence <= 1.0 and
        len(s_res.top_k) >= 3
    )
    collector.record("Static Gesture Classification", static_pred_ok, f"Predicted: '{s_res.gesture}' (Conf: {s_res.confidence*100:.1f}%)")

    # -----------------------------------------------------------------
    # Section 3: Biomechanical & Geometric Cross-Validation
    # -----------------------------------------------------------------
    print("\n3. Biomechanical & Geometric Cross-Validation")
    # Biomechanical test: FIST with extended fingers should fail validation
    invalid_fist_validity = static_rec._validate_biomechanics(
        gesture="FIST",
        finger_states=[1, 1, 1, 1, 1],  # All extended -> impossible for fist
        pinch_dist=1.0,
        landmarks=test_landmarks
    )
    collector.record("Biomechanical Mismatch Guard (Fist)", not invalid_fist_validity, "Correctly rejected FIST when fingers are extended")

    valid_fist_validity = static_rec._validate_biomechanics(
        gesture="FIST",
        finger_states=[0, 0, 0, 0, 0],  # All curled -> true fist
        pinch_dist=1.0,
        landmarks=test_landmarks
    )
    collector.record("Biomechanical Consistency (Fist)", valid_fist_validity, "Correctly validated biomechanically sound fist")

    # -----------------------------------------------------------------
    # Section 4: Dynamic Trajectory Tracking & Swipes
    # -----------------------------------------------------------------
    print("\n4. Dynamic Kinematic Trajectory Tracking (Swipes & Motions)")
    dyn_rec = DynamicGestureRecognizer(swipe_cooldown_sec=0.1)

    # Simulate horizontal swipe right trajectory: X moves from 100 to 300, Y remains flat
    now = time.perf_counter()
    dyn_res = None
    for step in range(8):
        x = 100.0 + step * 25.0  # +200 px net displacement
        y = 200.0 + (step % 2) * 2.0  # negligible vertical drift
        t = now + step * 0.03  # 30 ms per step -> 240 ms total
        dyn_res = dyn_rec.update(tracking_point=(x, y), is_pinching=False, timestamp=t)

    swipe_right_ok = (dyn_res is not None and dyn_res.gesture == "SWIPE_RIGHT" and dyn_res.is_dynamic)
    collector.record("Kinematic Swipe Right Detection", swipe_right_ok, f"Result: '{dyn_res.gesture}' (Speed: {dyn_res.speed:.1f} px/s)")

    # Simulate vertical swipe up trajectory: Y moves from 350 to 150 (-200 px), X remains flat
    dyn_rec.reset()
    now = time.perf_counter() + 1.0
    for step in range(8):
        x = 250.0 + (step % 2) * 2.0
        y = 350.0 - step * 25.0  # -200 px net displacement (Up in screen coords)
        t = now + step * 0.03
        dyn_res = dyn_rec.update(tracking_point=(x, y), is_pinching=False, timestamp=t)

    swipe_up_ok = (dyn_res is not None and dyn_res.gesture == "SWIPE_UP" and dyn_res.is_dynamic)
    collector.record("Kinematic Swipe Up Detection", swipe_up_ok, f"Result: '{dyn_res.gesture}' (Speed: {dyn_res.speed:.1f} px/s)")

    # Test PINCH_DRAG detection
    dyn_rec.reset()
    now = time.perf_counter() + 2.0
    for step in range(5):
        x = 200.0 + step * 10.0
        y = 200.0 + step * 10.0
        t = now + step * 0.03
        dyn_res = dyn_rec.update(tracking_point=(x, y), is_pinching=True, timestamp=t)

    pinch_drag_ok = (dyn_res is not None and dyn_res.gesture == "PINCH_DRAG")
    collector.record("Pinch Drag Kinematic Tracking", pinch_drag_ok, f"Result: '{dyn_res.gesture}'")

    # -----------------------------------------------------------------
    # Section 5: Multi-Mode Mapping & Context-Aware Routing (Modules 7 & 13)
    # -----------------------------------------------------------------
    print("\n5. Multi-Mode Command Mapping & Context Routing")
    mapper = GestureMappingManager()

    modes_covered = [
        OperationalMode.MOUSE,
        OperationalMode.SYSTEM,
        OperationalMode.MEDIA,
        OperationalMode.BROWSER,
        OperationalMode.PRESENTATION,
        OperationalMode.APPLICATION,
        OperationalMode.CUSTOM,
    ]
    all_modes_ok = all(m in OperationalMode.ALL_MODES for m in modes_covered)
    collector.record("7 Operational Modes Supported", all_modes_ok, f"Modes: {len(modes_covered)}")

    # Verify Mode Actions
    mapper.set_mode(OperationalMode.MOUSE)
    act_mouse = mapper.get_action("PINCH")
    collector.record("Mouse Mode Action Resolution", act_mouse == "LEFT_CLICK", f"PINCH in MOUSE -> '{act_mouse}'")

    mapper.set_mode(OperationalMode.MEDIA)
    act_media = mapper.get_action("PINCH")
    collector.record("Media Mode Action Resolution", act_media == "PLAY_PAUSE", f"PINCH in MEDIA -> '{act_media}'")

    mapper.set_mode(OperationalMode.SYSTEM)
    act_sys = mapper.get_action("THUMBS_UP")
    collector.record("System Mode Action Resolution", act_sys == "VOLUME_UP", f"THUMBS_UP in SYSTEM -> '{act_sys}'")

    # Context-Aware Application Routing (Module 13)
    mapper.set_mode(OperationalMode.APPLICATION)
    mapper.set_active_application("youtube")
    act_yt = mapper.get_action("PINCH")
    act_yt_swipe = mapper.get_action("SWIPE_RIGHT")
    yt_ok = (act_yt == "PLAY_PAUSE" and act_yt_swipe == "SEEK_FORWARD_10S")
    collector.record("Context-Aware Application Control (YouTube)", yt_ok, f"PINCH: '{act_yt}', SWIPE_RIGHT: '{act_yt_swipe}'")

    # Conflict Detection & JSON Persistence
    conflicts = mapper.detect_conflicts(OperationalMode.MOUSE)
    collector.record("Mapping Conflict Check (Mouse Mode)", isinstance(conflicts, list), f"Found {len(conflicts)} conflict(s)")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_f:
        tmp_json = tmp_f.name

    try:
        mapper.bind_gesture("PEACE", "CUSTOM_TEST_ACTION", OperationalMode.CUSTOM)
        mapper.save_mappings(tmp_json)
        mapper2 = GestureMappingManager(config_path=tmp_json, initial_mode=OperationalMode.CUSTOM)
        loaded_act = mapper2.get_action("PEACE", OperationalMode.CUSTOM)
        persist_ok = (loaded_act == "CUSTOM_TEST_ACTION")
        collector.record("JSON Mapping Persistence (Module 14)", persist_ok, f"Loaded custom action: '{loaded_act}'")
    finally:
        if os.path.exists(tmp_json):
            os.remove(tmp_json)

    # -----------------------------------------------------------------
    # Section 6: Real-Time Engine, Temporal Stabilization & Cooldown (Modules 4, 5, 6)
    # -----------------------------------------------------------------
    print("\n6. Real-Time Recognition Engine, Confidence & Stabilization")
    recognizer = RealtimeGestureRecognizer(
        confidence_threshold=0.70,
        temporal_window_size=5,
        min_stable_frames=3,
        initial_mode=OperationalMode.MOUSE,
    )

    class MockLandmark:
        def __init__(self, x, y, z=0.0):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    def create_mock_hand_data(landmarks_array, handedness="Right", score=0.98):
        raw_lms = [MockLandmark(pt[0], pt[1], pt[2] if len(pt) > 2 else 0.0) for pt in landmarks_array]
        return HandData(raw_lms, frame_w=640, frame_h=480, handedness=handedness, detection_score=score)

    # Create dummy HandData for point gesture
    point_idx = GESTURE_CLASSES.index("POINT") if "POINT" in GESTURE_CLASSES else 4
    # Find a sample of POINT in dataset
    sample_indices = np.where(dataset_obj.y == "POINT")[0]
    if len(sample_indices) > 0:
        point_sample = dataset_obj.landmarks[sample_indices[0]]
    else:
        point_sample = test_landmarks

    hand_input = create_mock_hand_data(point_sample, handedness="Right", score=0.98)

    # Frame 1: First observation (buffering)
    t_now = time.perf_counter()
    res1 = recognizer.process_frame(hand_input, timestamp=t_now)
    f1_ok = (res1.raw_gesture in GESTURE_CLASSES and not res1.is_stabilized)
    collector.record("Temporal Stabilization - Buffering Phase", f1_ok, f"Raw: '{res1.raw_gesture}', Stabilized: {res1.is_stabilized}")

    # Feed 3 more consistent frames to reach min_stable_frames=3
    res_stable = None
    for i in range(1, 4):
        res_stable = recognizer.process_frame(hand_input, timestamp=t_now + i * 0.033)

    stabilized_ok = (res_stable is not None and res_stable.is_stabilized and res_stable.confirmed_gesture != "NONE")
    collector.record("Temporal Stabilization - Majority Convergence", stabilized_ok, f"Confirmed: '{res_stable.confirmed_gesture}' ({res_stable.confidence*100:.1f}%)")

    # Single-Frame Spike Filter Immunity (Module 5):
    # Inject a single anomalous frame (e.g. FIST sample) into a stable stream
    fist_indices = np.where(dataset_obj.y == "FIST")[0]
    fist_lms = dataset_obj.landmarks[fist_indices[0]] if len(fist_indices) > 0 else test_landmarks
    fist_input = create_mock_hand_data(fist_lms, handedness="Right", score=0.95)

    res_spike = recognizer.process_frame(fist_input, timestamp=t_now + 0.20)
    # The confirmed gesture should STILL be the previous stable gesture (or NONE), NOT immediately the single spike
    spike_filtered = (res_spike.confirmed_gesture != "FIST")
    collector.record("Single-Frame Misclassification Immunity", spike_filtered, f"Raw Spike: '{res_spike.raw_gesture}', Confirmed: '{res_spike.confirmed_gesture}'")

    # Confidence Threshold Gating (Module 4)
    recognizer.set_confidence_threshold(0.99)  # Ultra-high threshold
    res_low_conf = recognizer.process_frame(hand_input, timestamp=t_now + 0.30)
    conf_gated = (not res_low_conf.is_confident and not res_low_conf.can_execute)
    collector.record("Confidence Threshold Gating (Module 4)", conf_gated, f"Status: '{res_low_conf.status_message}'")

    recognizer.set_confidence_threshold(0.70)  # Reset to standard

    # Command Cooldown Gating (Module 6)
    recognizer.reset()
    pinch_indices = np.where(dataset_obj.y == "PINCH")[0]
    pinch_lms = dataset_obj.landmarks[pinch_indices[0]] if len(pinch_indices) > 0 else test_landmarks
    pinch_input = create_mock_hand_data(pinch_lms, handedness="Right", score=0.98)

    t_cool = time.perf_counter() + 10.0
    executed_frames = []
    for i in range(6):
        res = recognizer.process_frame(pinch_input, timestamp=t_cool + i * 0.033)
        if res.can_execute:
            executed_frames.append(i)

    # In the first 6 frames (165 ms span < 250 ms cooldown), exactly 1 frame executes (at frame index 2)
    single_burst_ok = (len(executed_frames) == 1 and executed_frames[0] == 2)

    # After cooldown period expires (+0.35s > 0.25s cooldown), re-trigger must be allowed
    res_after_cooldown = recognizer.process_frame(pinch_input, timestamp=t_cool + 0.35)
    retrigger_ok = res_after_cooldown.can_execute

    cooldown_ok = single_burst_ok and retrigger_ok
    collector.record("Command Cooldown Enforcement (Module 6)", cooldown_ok, f"Executed at frame {executed_frames}, rapid frames blocked, retrigger after cooldown: {retrigger_ok}")

    # -----------------------------------------------------------------
    # Section 7: Error Handling & Input Resilience (Module 17)
    # -----------------------------------------------------------------
    print("\n7. Error Handling & Input Validation (Module 17)")
    # None input
    r_none = recognizer.process_frame(None)
    collector.record("None Frame Input Resilience", r_none.status_message == "NO HAND DETECTED", "Safely handled None input")

    # Empty array
    r_empty = recognizer.process_frame(np.array([]))
    collector.record("Empty Landmark Array Resilience", r_empty.status_message == "NO HAND DETECTED", "Safely handled empty array")

    # NaN / Inf array
    nan_arr = np.full((21, 3), np.nan)
    r_nan = recognizer.process_frame(nan_arr)
    collector.record("NaN / Inf Landmark Array Resilience", r_nan.confirmed_gesture == "NONE", "Safely rejected NaN landmarks")

    # -----------------------------------------------------------------
    # Section 8: End-to-End Pipeline Latency Microbenchmark
    # -----------------------------------------------------------------
    print("\n8. End-to-End Processing Latency Benchmark")
    n_iters = 500
    t_start = time.perf_counter()
    for i in range(n_iters):
        _ = recognizer.process_frame(hand_input, timestamp=t_now + i * 0.033)
    t_total = time.perf_counter() - t_start

    latency_per_frame_ms = (t_total / n_iters) * 1000.0
    throughput_fps = n_iters / t_total

    collector.record(
        "Real-Time Pipeline Latency (< 5.0 ms)",
        latency_per_frame_ms < 5.0,
        f"Mean: {latency_per_frame_ms:.3f} ms | Throughput: {throughput_fps:,.0f} FPS"
    )

    # -----------------------------------------------------------------
    # Summary & Export
    # -----------------------------------------------------------------
    total = len(collector.results)
    passed = sum(1 for r in collector.results if r["status"] == "PASS")
    failed = total - passed
    pass_rate = (passed / total) * 100.0 if total > 0 else 0.0

    print("\n" + "=" * 75)
    print(" PHASE 7 TEST EXECUTION SUMMARY")
    print("=" * 75)
    print(f" Total Tests Run:   {total}")
    print(f" Passed:            {passed}")
    print(f" Failed:            {failed}")
    print(f" Pass Rate:         {pass_rate:.1f}%")
    print("=" * 75)

    report_path = os.path.join(_ROOT, "data", "phase7_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "phase": "Phase 7 - Real-Time Gesture Recognition, Stabilization & Multi-Mode Mapping",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.platform(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": pass_rate,
            "mean_pipeline_latency_ms": latency_per_frame_ms,
            "pipeline_throughput_fps": throughput_fps,
            "results": collector.results
        }, f, indent=2)

    print(f" Benchmark report exported to: {report_path}\n")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
