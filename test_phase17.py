"""
Phase 17 Verification Test Suite: Edge-Case Handling, Occlusion Recovery & Safety (Module 17)
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding & Package Exports (utils.SafetyManager)
2. Emergency Stop Failsafe: Instant input halting, mouse button release, and tracking resumption
3. Tracking Loss Drag Safety Release: Auto-release of held mouse during drag if hand drops out of frame
4. Coordinate Sanitization & Bounds Clamping:
   - Clamping negative coordinates (< 0.0) to 0.0
   - Clamping overflow coordinates (> 1.0) to 1.0
   - NaN and Infinity eradication & finite replacement
   - 2D (21, 2) to 3D (21, 3) matrix normalization
   - Malformed shape error handling without crashes
5. Erratic Motion & Glitch Frame Suppression:
   - Physiological human motion acceptance (< threshold)
   - Impossible optical jump rejection (> 35% frame width jump)
6. Anatomical Finger Occlusion Evaluation
7. Real-Time Classifier Pipeline Defense: Feeding hostile/glitched coordinates through recognizer
8. Sub-Millisecond Safety Checking Latency Benchmark (< 0.05 ms per frame)
"""

import json
import os
import platform
import sys
import time
from typing import Any, Dict, List

import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset import GestureDataset
from detection.hand_detector import HandData
from gestures import GestureRecognitionResult, OperationalMode, RealtimeGestureRecognizer
import utils
from utils import SafetyManager


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
        print(f"  {badge} {test_name:<48} {details}")


collector = TestResultCollector()


class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


def create_mock_hand_data(landmarks_array, handedness="Right", score=0.98):
    raw_lms = [MockLandmark(pt[0], pt[1], pt[2] if len(pt) > 2 else 0.0) for pt in landmarks_array]
    return HandData(raw_lms, frame_w=640, frame_h=480, handedness=handedness, detection_score=score)


def run_tests() -> bool:
    print("=" * 75)
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 17 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     Edge-Case Handling & Safety Management (Module 17)")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Section 1: Scaffolding & Package Exports
    # -----------------------------------------------------------------
    print("\n1. Module Scaffolding & Package Exports")
    exports_ok = hasattr(utils, "SafetyManager")
    collector.record("SafetyManager Package Export", exports_ok, "SafetyManager exported from utils package")

    safety = SafetyManager(mock_mode=True)
    init_ok = (
        safety.mock_mode
        and not safety.is_emergency_stopped
        and not safety.is_dragging
        and safety.emergency_stop_count == 0
    )
    collector.record("SafetyManager Initialization", init_ok, f"Emergency: {safety.is_emergency_stopped} | Dragging: {safety.is_dragging}")

    # -----------------------------------------------------------------
    # Section 2: Emergency Stop & Recovery
    # -----------------------------------------------------------------
    print("\n2. Emergency Stop & Fail-Safe Recovery")
    safety.set_dragging_state(True)
    stop_ok = safety.emergency_stop()
    state_after_stop = safety.is_emergency_stopped and not safety.is_dragging and safety.emergency_stop_count == 1
    collector.record("Emergency Stop Execution", stop_ok and state_after_stop, f"Stopped: {safety.is_emergency_stopped} | Released Drag: True")

    safety.resume_operation()
    resume_ok = not safety.is_emergency_stopped
    collector.record("Safety Resumption", resume_ok, f"Emergency cleared: {resume_ok}")

    # -----------------------------------------------------------------
    # Section 3: Tracking Loss Drag Safety Release
    # -----------------------------------------------------------------
    print("\n3. Tracking Loss Drag Safety Release")
    safety.set_dragging_state(True)
    res_loss = safety.handle_tracking_loss(active_action="DRAG_DROP")
    loss_ok = res_loss["released_drag"] is True and not safety.is_dragging and safety.drag_safety_releases == 1
    collector.record("Tracking Loss Drag Auto-Release", loss_ok, f"Released: {res_loss['released_drag']} | Incidents: {res_loss['incident_count']}")

    # -----------------------------------------------------------------
    # Section 4: Out-of-Bounds Coordinate Sanitization
    # -----------------------------------------------------------------
    print("\n4. Landmark Coordinate Sanitization & Guard Rails")
    # A. Out-of-bounds coordinates (drifting outside camera view)
    hostile_coords = np.ones((21, 3), dtype=np.float32) * 0.5
    hostile_coords[0, 0] = -0.35  # Below 0.0
    hostile_coords[4, 1] = 1.45   # Above 1.0

    valid, clean_arr = safety.sanitize_landmarks(hostile_coords)
    clamping_ok = (
        valid
        and clean_arr[0, 0] == 0.0
        and clean_arr[4, 1] == 1.0
        and np.all(clean_arr[:, :2] >= 0.0)
        and np.all(clean_arr[:, :2] <= 1.0)
    )
    collector.record("Out-of-Bounds Clamping [0.0, 1.0]", clamping_ok, f"Negative -0.35 -> {clean_arr[0,0]} | Overflow 1.45 -> {clean_arr[4,1]}")

    # B. NaN and Infinity Infestation
    nan_coords = np.ones((21, 3), dtype=np.float32) * 0.4
    nan_coords[8, 0] = np.nan
    nan_coords[12, 1] = np.inf
    nan_coords[16, 2] = -np.inf

    valid_nan, clean_nan = safety.sanitize_landmarks(nan_coords)
    nan_ok = valid_nan and np.all(np.isfinite(clean_nan))
    collector.record("NaN & Infinity Eradication", nan_ok, f"All finite: {np.all(np.isfinite(clean_nan))}")

    # C. 2D to 3D matrix padding
    coords_2d = np.ones((21, 2), dtype=np.float32) * 0.6
    valid_2d, clean_3d = safety.sanitize_landmarks(coords_2d)
    pad_ok = valid_2d and clean_3d.shape == (21, 3) and clean_3d[0, 2] == 0.0
    collector.record("2D to 3D Matrix Normalization", pad_ok, f"Shape: (21, 2) -> {clean_3d.shape}")

    # D. Malformed array shape error handling
    malformed = np.ones((10, 5))
    valid_malformed, _ = safety.sanitize_landmarks(malformed)
    malformed_ok = not valid_malformed
    collector.record("Malformed Array Rejection", malformed_ok, f"Rejected invalid (10, 5) shape without crash")

    # -----------------------------------------------------------------
    # Section 5: Erratic Motion & Optical Glitch Suppression
    # -----------------------------------------------------------------
    print("\n5. Erratic Motion & Glitch Rejection")
    safety.last_valid_tracking_point = None

    # Step 1: Normal initial position
    p1 = (0.50, 0.50)
    g1 = safety.is_glitch_displacement(p1)

    # Step 2: Normal physiological step
    p2 = (0.52, 0.51)  # small delta = ~0.022
    g2 = safety.is_glitch_displacement(p2)

    # Step 3: Impossible telekinetic optical glitch jump
    p3 = (0.95, 0.95)  # huge delta = ~0.62
    g3 = safety.is_glitch_displacement(p3)

    glitch_ok = (not g1 and not g2 and g3 is True and safety.glitches_suppressed == 1)
    collector.record("Optical Glitch Frame Rejection", glitch_ok, f"Normal accepted (False, False) | Glitch suppressed (True, count={safety.glitches_suppressed})")

    # -----------------------------------------------------------------
    # Section 6: Finger Occlusion Evaluation
    # -----------------------------------------------------------------
    print("\n6. Finger Occlusion Evaluation")
    dataset_path = os.path.join(_ROOT, "dataset", "gesture_dataset.npz")
    dataset_obj = GestureDataset.load_from_npz(dataset_path)
    peace_idx = np.where(dataset_obj.y == "PEACE")[0][0]
    peace_landmarks = dataset_obj.landmarks[peace_idx].copy()

    # Normal open peace gesture
    occ_normal = safety.evaluate_occlusion(peace_landmarks)

    # Artificially collapse pinky tip into pinky MCP to simulate heavy occlusion
    occluded_landmarks = peace_landmarks.copy()
    occluded_landmarks[20] = occluded_landmarks[17]  # tip 20 = mcp 17

    occ_test = safety.evaluate_occlusion(occluded_landmarks)
    occ_ok = "pinky" in occ_test["occluded_fingers"] and occ_test["occluded"] is True
    collector.record("Anatomical Occlusion Detection", occ_ok, f"Detected occluded fingers: {occ_test['occluded_fingers']}")

    # -----------------------------------------------------------------
    # Section 7: Pipeline Defense Against Hostile Inputs
    # -----------------------------------------------------------------
    print("\n7. Real-Time Pipeline Defense Against Corrupt Inputs")
    recognizer = RealtimeGestureRecognizer(
        initial_mode=OperationalMode.MOUSE,
        confidence_threshold=0.60,
    )

    # Feed completely zeroed / hostile hand data through the recognizer
    hostile_hand = HandData([], 640, 480)
    res_hostile = recognizer.process_frame(hostile_hand)
    pipeline_safe = res_hostile.confirmed_gesture == "NONE" and res_hostile.status_message in ["NO HAND DETECTED", "IDLE"]
    collector.record("Corrupt Hand Data Pipeline Defense", pipeline_safe, f"Safely returned: {res_hostile.status_message}")

    # -----------------------------------------------------------------
    # Section 8: Safety Checking Latency Microbenchmark
    # -----------------------------------------------------------------
    print("\n8. Safety Checking Latency Microbenchmark")
    test_arr = np.random.uniform(0.0, 1.0, size=(21, 3)).astype(np.float32)
    iterations = 10000

    t0 = time.perf_counter()
    for i in range(iterations):
        _, _ = safety.sanitize_landmarks(test_arr)
        _ = safety.is_glitch_displacement((0.5, 0.5))
    t1 = time.perf_counter()

    total_time = t1 - t0
    avg_latency_ms = (total_time / iterations) * 1000.0
    throughput = iterations / total_time
    benchmark_ok = avg_latency_ms < 0.05

    collector.record(
        "Safety Checking Latency Benchmark (< 0.05 ms)",
        benchmark_ok,
        f"Mean latency: {avg_latency_ms:.4f} ms | Throughput: {throughput:,.0f} checks/sec",
        latency_ms=avg_latency_ms,
    )

    # -----------------------------------------------------------------
    # Summary & Report Generation
    # -----------------------------------------------------------------
    print("\n" + "=" * 75)
    print(" VERIFICATION SUMMARY")
    print("=" * 75)
    total_tests = len(collector.results)
    passed_tests = sum(1 for r in collector.results if r["status"] == "PASS")
    pass_rate = (passed_tests / total_tests) * 100.0 if total_tests > 0 else 0.0

    print(f" Total Tests Run:   {total_tests}")
    print(f" Total Passed:     {passed_tests}")
    print(f" Total Failed:     {total_tests - passed_tests}")
    print(f" Pass Rate:        {pass_rate:.1f}%")
    print("=" * 75)

    report_path = os.path.join(_ROOT, "data", "phase17_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_data = {
        "phase": 17,
        "name": "Edge-Case Handling, Occlusion Recovery & Failsafe",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "pass_rate_pct": pass_rate,
            "mean_safety_latency_ms": round(avg_latency_ms, 5),
            "throughput_checks_sec": round(throughput, 1),
            "telemetry": safety.get_status(),
        },
        "results": collector.results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nVerification report generated: {report_path}")
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
