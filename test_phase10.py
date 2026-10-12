"""
Phase 10 Verification Test Suite: System Hardware & OS Controls
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding & Package Exports (controllers.SystemController)
2. Audio Volume Control (volume_up, volume_down, toggle_mute, scalar bounds clamping)
3. Display Brightness Control (brightness_up, brightness_down, set_brightness, 0-100 clamping)
4. Screenshot Capture & File Saving (timestamped PNG path formatting)
5. OS Navigation & Windows Shell Controls (File Explorer, Browser, Task View, Minimize All)
6. Application Launching & Operation Cancellation (launch_application, cancel_operation)
7. Security & Disruptive Operation Safeguards (Module 21: confirmation gating on lock workstation)
8. Unified Action Router (execute_action) for all System Mode Commands
9. End-to-End Gesture-to-System Action Integration (Gesture -> Stabilizer -> SystemController)
10. Sub-Millisecond Dispatch Latency Benchmark (< 0.05 ms per action)
"""

import json
import math
import os
import platform
import sys
import tempfile
import time
from typing import Any, Dict, List

import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import controllers
from controllers import SystemController
from dataset import GestureDataset
from detection.hand_detector import HandData
from gestures import OperationalMode, RealtimeGestureRecognizer


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
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 10 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     System Hardware & OS Automation Controls (Module 9)")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Section 1: Scaffolding & Package Exports
    # -----------------------------------------------------------------
    print("\n1. Module Scaffolding & Package Exports")
    exports_ok = hasattr(controllers, "SystemController")
    collector.record("SystemController Package Export", exports_ok, "controllers.SystemController accessible")

    sys_ctrl = SystemController(mock_mode=True, require_confirmation_for_lock=True)
    collector.record("SystemController Mock Mode Init", sys_ctrl.mock_mode, "Initialized in safe simulation mode")

    # -----------------------------------------------------------------
    # Section 2: Audio Volume Control (Module 9)
    # -----------------------------------------------------------------
    print("\n2. Audio Volume Control & Clamping Bounds")
    sys_ctrl.set_volume(0.50)
    v0 = sys_ctrl.get_volume()

    # Volume Up
    sys_ctrl.volume_up(step=0.10)
    v_up = sys_ctrl.get_volume()
    vol_up_ok = (abs(v_up - 0.60) < 1e-4 and sys_ctrl.last_action == "VOLUME_UP")
    collector.record("Volume Increase (Step: +10%)", vol_up_ok, f"Old: {v0*100:.0f}% -> New: {v_up*100:.0f}%")

    # Volume Down
    sys_ctrl.volume_down(step=0.15)
    v_down = sys_ctrl.get_volume()
    vol_down_ok = (abs(v_down - 0.45) < 1e-4 and sys_ctrl.last_action == "VOLUME_DOWN")
    collector.record("Volume Decrease (Step: -15%)", vol_down_ok, f"Old: {v_up*100:.0f}% -> New: {v_down*100:.0f}%")

    # Volume Clamping: Cannot exceed 1.0 or drop below 0.0
    sys_ctrl.set_volume(1.50)
    v_max = sys_ctrl.get_volume()
    sys_ctrl.set_volume(-0.50)
    v_min = sys_ctrl.get_volume()
    vol_clamp_ok = (v_max == 1.0 and v_min == 0.0)
    collector.record("Volume Boundary Clamping ([0.0, 1.0])", vol_clamp_ok, f"Max clamped to: {v_max}, Min clamped to: {v_min}")

    # Mute Toggle
    sys_ctrl.toggle_mute()
    muted_state = sys_ctrl._sim_muted
    sys_ctrl.toggle_mute()
    unmuted_state = not sys_ctrl._sim_muted
    mute_ok = (muted_state and unmuted_state)
    collector.record("Mute / Unmute State Toggle", mute_ok, f"Toggled to Muted: {muted_state}, back to Unmuted: {unmuted_state}")

    # -----------------------------------------------------------------
    # Section 3: Display Brightness Control (Module 9)
    # -----------------------------------------------------------------
    print("\n3. Display Brightness Control & Clamping Bounds")
    sys_ctrl.set_brightness(70)
    b0 = sys_ctrl.get_brightness()

    # Brightness Up
    sys_ctrl.brightness_up(step=10)
    b_up = sys_ctrl.get_brightness()
    b_up_ok = (b_up == 80 and sys_ctrl.last_action == "BRIGHTNESS_UP")
    collector.record("Brightness Increase (Step: +10%)", b_up_ok, f"Old: {b0}% -> New: {b_up}%")

    # Brightness Down
    sys_ctrl.brightness_down(step=20)
    b_down = sys_ctrl.get_brightness()
    b_down_ok = (b_down == 60 and sys_ctrl.last_action == "BRIGHTNESS_DOWN")
    collector.record("Brightness Decrease (Step: -20%)", b_down_ok, f"Old: {b_up}% -> New: {b_down}%")

    # Brightness Clamping: 0 to 100
    sys_ctrl.set_brightness(150)
    b_max = sys_ctrl.get_brightness()
    sys_ctrl.set_brightness(-50)
    b_min = sys_ctrl.get_brightness()
    b_clamp_ok = (b_max == 100 and b_min == 0)
    collector.record("Brightness Boundary Clamping ([0, 100])", b_clamp_ok, f"Max clamped to: {b_max}%, Min clamped to: {b_min}%")

    # -----------------------------------------------------------------
    # Section 4: Screenshot Capture & File Generation (Module 9)
    # -----------------------------------------------------------------
    print("\n4. Screenshot Capture & File Saving")
    with tempfile.TemporaryDirectory() as tmp_dir:
        real_shot_ctrl = SystemController(screenshots_dir=tmp_dir, mock_mode=False)
        shot_path = real_shot_ctrl.take_screenshot()
        shot_ok = (shot_path is not None and os.path.exists(shot_path) and shot_path.endswith(".png"))
        collector.record("Screenshot Capture & Disk Serialization", shot_ok, f"Saved file: {shot_path}")

    # -----------------------------------------------------------------
    # Section 5: Windows Navigation & Shell Controls (Module 9)
    # -----------------------------------------------------------------
    print("\n5. Operating System Navigation & Window Control")
    exp_ok = sys_ctrl.open_file_explorer() and sys_ctrl.last_action == "OPEN_FILE_EXPLORER"
    collector.record("Open File Explorer Action", exp_ok, f"Action: '{sys_ctrl.last_action}'")

    browser_ok = sys_ctrl.open_browser("https://www.google.com") and sys_ctrl.last_action == "OPEN_BROWSER"
    collector.record("Open Browser Action", browser_ok, f"Action: '{sys_ctrl.last_action}'")

    min_ok = sys_ctrl.minimize_all() and sys_ctrl.last_action == "MINIMIZE_ALL"
    collector.record("Minimize All Windows Action", min_ok, f"Action: '{sys_ctrl.last_action}'")

    task_ok = sys_ctrl.task_view() and sys_ctrl.last_action == "TASK_VIEW"
    collector.record("Task View Overview Action", task_ok, f"Action: '{sys_ctrl.last_action}'")

    close_ok = sys_ctrl.close_window() and sys_ctrl.last_action == "CLOSE_WINDOW"
    collector.record("Close Active Window Action", close_ok, f"Action: '{sys_ctrl.last_action}'")

    cancel_ok = sys_ctrl.cancel_operation() and sys_ctrl.last_action == "CANCEL"
    collector.record("Cancel / Abort Operation Action", cancel_ok, f"Action: '{sys_ctrl.last_action}'")

    app_ok = sys_ctrl.launch_application("notepad") and sys_ctrl.last_action == "LAUNCH_NOTEPAD"
    collector.record("Application Launcher Action", app_ok, f"Action: '{sys_ctrl.last_action}'")

    # -----------------------------------------------------------------
    # Section 6: Security & Safety Confirmation Safeguards (Module 21)
    # -----------------------------------------------------------------
    print("\n6. Security & Safety Safeguards (Module 21)")
    # When require_confirmation_for_lock=True, unconfirmed lock attempt must be suppressed
    sys_ctrl.require_confirmation_for_lock = True
    lock_blocked = not sys_ctrl.lock_screen(confirmed=False)
    collector.record("Lock Screen Safety Confirmation Gate", lock_blocked, "Unconfirmed lock screen safely suppressed")

    lock_allowed = sys_ctrl.lock_screen(confirmed=True) and sys_ctrl.last_action == "LOCK_SCREEN"
    collector.record("Confirmed Lock Screen Execution", lock_allowed, f"Action: '{sys_ctrl.last_action}'")

    # -----------------------------------------------------------------
    # Section 7: Unified Action Router (execute_action())
    # -----------------------------------------------------------------
    print("\n7. Unified Action Router (execute_action)")
    test_actions = [
        ("VOLUME_UP", "VOLUME_UP"),
        ("VOLUME_DOWN", "VOLUME_DOWN"),
        ("MUTE", "MUTE"),
        ("BRIGHTNESS_UP", "BRIGHTNESS_UP"),
        ("BRIGHTNESS_DOWN", "BRIGHTNESS_DOWN"),
        ("SCREENSHOT", "SCREENSHOT"),
        ("OPEN_FILE_EXPLORER", "OPEN_FILE_EXPLORER"),
        ("OPEN_BROWSER", "OPEN_BROWSER"),
        ("LOCK_SCREEN", "LOCK_SCREEN"),
        ("MINIMIZE_ALL", "MINIMIZE_ALL"),
        ("TASK_VIEW", "TASK_VIEW"),
        ("CANCEL", "CANCEL"),
        ("CLOSE_WINDOW", "CLOSE_WINDOW"),
    ]

    all_routed = True
    for cmd, expected_act in test_actions:
        res = sys_ctrl.execute_action(cmd, confirmed=True)
        if not res or sys_ctrl.last_action != expected_act:
            all_routed = False
            break

    collector.record("Unified Action Router Execution", all_routed, f"Successfully routed {len(test_actions)} commands")

    # -----------------------------------------------------------------
    # Section 8: End-to-End Pipeline Integration (Real-Time Gesture -> System Action)
    # -----------------------------------------------------------------
    print("\n8. End-to-End Real-Time Gesture to System Control Integration")
    dataset_path = os.path.join(_ROOT, "dataset", "gesture_dataset.npz")
    dataset_obj = GestureDataset.load_from_npz(dataset_path)

    recognizer = RealtimeGestureRecognizer(
        initial_mode=OperationalMode.SYSTEM,
        confidence_threshold=0.60,
        temporal_window_size=3,
        min_stable_frames=2
    )

    # In SYSTEM mode: THUMBS_UP triggers VOLUME_UP
    t_idx = np.where(dataset_obj.y == "THUMBS_UP")[0][0]
    hand = create_mock_hand_data(dataset_obj.landmarks[t_idx])

    t_now = time.perf_counter()
    recognizer.process_frame(hand, timestamp=t_now)
    r_sys = recognizer.process_frame(hand, timestamp=t_now + 0.033)

    gesture_routed = False
    if r_sys.can_execute and r_sys.mapped_action is not None:
        gesture_routed = sys_ctrl.execute_action(r_sys.mapped_action)

    e2e_ok = (r_sys.confirmed_gesture == "THUMBS_UP" and r_sys.mapped_action == "VOLUME_UP" and gesture_routed)
    collector.record("E2E Gesture Recognition -> System Execution", e2e_ok, f"Gesture: '{r_sys.confirmed_gesture}' -> Action: '{r_sys.mapped_action}' (Executed: {gesture_routed})")

    # -----------------------------------------------------------------
    # Section 9: Sub-Millisecond Action Routing Latency Benchmark
    # -----------------------------------------------------------------
    print("\n9. Action Routing Latency Microbenchmark")
    n_iters = 1000
    t_start = time.perf_counter()
    for i in range(n_iters):
        _ = sys_ctrl.execute_action("VOLUME_UP")
    t_total = time.perf_counter() - t_start

    latency_per_call_ms = (t_total / n_iters) * 1000.0
    throughput_ops = n_iters / t_total

    collector.record(
        "System Dispatch Latency (< 0.05 ms)",
        latency_per_call_ms < 0.05,
        f"Mean: {latency_per_call_ms*1000:.1f} microseconds ({latency_per_call_ms:.4f} ms) | {throughput_ops:,.0f} actions/sec"
    )

    # -----------------------------------------------------------------
    # Summary & Export
    # -----------------------------------------------------------------
    total = len(collector.results)
    passed = sum(1 for r in collector.results if r["status"] == "PASS")
    failed = total - passed
    pass_rate = (passed / total) * 100.0 if total > 0 else 0.0

    print("\n" + "=" * 75)
    print(" PHASE 10 TEST EXECUTION SUMMARY")
    print("=" * 75)
    print(f" Total Tests Run:   {total}")
    print(f" Passed:            {passed}")
    print(f" Failed:            {failed}")
    print(f" Pass Rate:         {pass_rate:.1f}%")
    print("=" * 75)

    report_path = os.path.join(_ROOT, "data", "phase10_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "phase": "Phase 10 - System Hardware & OS Controls",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.platform(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": pass_rate,
            "mean_dispatch_latency_ms": latency_per_call_ms,
            "dispatch_throughput_ops": throughput_ops,
            "results": collector.results
        }, f, indent=2)

    print(f" Benchmark report exported to: {report_path}\n")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
