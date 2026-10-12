"""
Phase 13 Verification Test Suite: Presentation & Application-Specific Controls
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding & Package Exports (controllers.PresentationController, PresentationTarget)
2. Controller Initialization & Default State (slide index 1/50, is_presenting False, mock mode)
3. Slide Navigation: next_slide, prev_slide with state increments
4. Slide Boundary Clamping: Lower bound (slide 1) and Upper bound (slide 50)
5. Jump Navigation: first_slide, last_slide, goto_slide(n)
6. Slideshow Lifecycle: start_presentation (F5 / Shift+F5) and exit_presentation (Esc)
7. Presenter Focus Tools: toggle_laser_pointer, toggle_pen, clear_annotations
8. Presentation Screen Blanking: toggle_black_screen ('B'), toggle_white_screen ('W')
9. Presentation Overview / Slide Sorter: slide_overview ('G' / Ctrl+S)
10. Application-Specific Target Profiles: PowerPoint, Google Slides, Keynote, PDF, Generic
11. Telemetry & State Introspection: get_status, event_log tracking
12. Unified Action Router: execute_action across all canonical presentation actions
13. End-to-End Gesture Recognition Integration in PRESENTATION mode (SWIPE_RIGHT -> NEXT_SLIDE, SWIPE_LEFT -> PREV_SLIDE, etc.)
14. Dispatch Latency & Throughput Microbenchmark (< 0.05 ms/op)
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

import controllers
from controllers import PresentationController, PresentationTarget
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
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 13 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     Presentation & Application-Specific Controls (Module 12 & 13)")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Section 1: Scaffolding & Package Exports
    # -----------------------------------------------------------------
    print("\n1. Module Scaffolding & Package Exports")
    exports_ok = hasattr(controllers, "PresentationController") and hasattr(controllers, "PresentationTarget")
    collector.record("PresentationController Package Exports", exports_ok, "PresentationController and PresentationTarget exported")

    pres = PresentationController(mock_mode=True)
    init_ok = (
        pres.mock_mode
        and pres.target_app == PresentationTarget.POWERPOINT
        and pres.current_slide_index == 1
        and not pres.is_presenting
        and not pres.laser_pointer_active
        and pres.blank_screen_mode == "NONE"
    )
    collector.record("Presentation Controller Initialization", init_ok, f"Mode: Mock | App: {pres.target_app} | Slide: {pres.current_slide_index}/{pres.total_slides}")

    # -----------------------------------------------------------------
    # Section 2: Slide Navigation & Clamping
    # -----------------------------------------------------------------
    print("\n2. Slide Navigation Operations")
    # Next slide
    pres.current_slide_index = 1
    pres.next_slide()
    pres.next_slide()
    next_ok = (pres.current_slide_index == 3 and pres.last_action == "NEXT_SLIDE")
    collector.record("Next Slide Advancement", next_ok, f"Slide index advanced: 1 -> {pres.current_slide_index}")

    # Prev slide
    pres.prev_slide()
    prev_ok = (pres.current_slide_index == 2 and pres.last_action == "PREV_SLIDE")
    collector.record("Previous Slide Navigation", prev_ok, f"Slide index reverted: 3 -> {pres.current_slide_index}")

    # Lower boundary clamping (cannot go below slide 1)
    pres.first_slide()
    pres.prev_slide()
    pres.prev_slide()
    clamp_lower_ok = (pres.current_slide_index == 1)
    collector.record("Lower Bound Slide Clamping", clamp_lower_ok, f"Slide stays at minimum 1 (index={pres.current_slide_index})")

    # Upper boundary clamping (cannot go above total_slides)
    pres.last_slide()
    upper_start = pres.current_slide_index
    pres.next_slide()
    pres.next_slide()
    clamp_upper_ok = (pres.current_slide_index == pres.total_slides == upper_start)
    collector.record("Upper Bound Slide Clamping", clamp_upper_ok, f"Slide clamped at total {pres.total_slides} (index={pres.current_slide_index})")

    # Jump to first and last slides
    pres.first_slide()
    first_ok = (pres.current_slide_index == 1 and pres.last_action == "FIRST_SLIDE")
    pres.last_slide()
    last_ok = (pres.current_slide_index == pres.total_slides and pres.last_action == "LAST_SLIDE")
    collector.record("First & Last Slide Jumping", first_ok and last_ok, f"First: 1, Last: {pres.total_slides}")

    # Goto specific slide
    pres.goto_slide(18)
    goto_ok = (pres.current_slide_index == 18 and "18" in pres.last_action)
    collector.record("Direct Slide Number Jump", goto_ok, f"Directly jumped to slide {pres.current_slide_index}")

    # -----------------------------------------------------------------
    # Section 3: Slideshow Lifecycle Operations
    # -----------------------------------------------------------------
    print("\n3. Slideshow Lifecycle Operations")
    # Start presentation from beginning
    pres.start_presentation(from_beginning=True)
    start_ok = (pres.is_presenting is True and pres.current_slide_index == 1 and pres.last_action == "START_PRESENTATION")
    collector.record("Start Presentation (Beginning)", start_ok, f"Presenting: {pres.is_presenting} | Slide: {pres.current_slide_index}")

    # Advance during presentation
    pres.next_slide()
    pres.next_slide()
    adv_ok = (pres.current_slide_index == 3)
    collector.record("Slideshow Active Advancement", adv_ok, f"Active slide: {pres.current_slide_index}")

    # Exit presentation
    pres.exit_presentation()
    exit_ok = (pres.is_presenting is False and pres.last_action == "EXIT_PRESENTATION")
    collector.record("Exit Presentation Mode", exit_ok, f"Presenting: {pres.is_presenting}")

    # -----------------------------------------------------------------
    # Section 4: Presenter Tools & Focus Modes
    # -----------------------------------------------------------------
    print("\n4. Presenter Focus & Annotation Tools")
    # Laser pointer toggle
    pres.toggle_laser_pointer()
    laser_on = (pres.laser_pointer_active is True and pres.last_action == "LASER_POINTER")
    pres.toggle_laser_pointer()
    laser_off = (pres.laser_pointer_active is False)
    collector.record("Laser Pointer Toggle Mode", laser_on and laser_off, f"Laser On: {laser_on} -> Laser Off: {laser_off}")

    # Pen tool toggle
    pen_ok = pres.toggle_pen() and pres.last_action == "TOGGLE_PEN"
    collector.record("Pen Annotation Tool Toggle", pen_ok, f"Action: {pres.last_action}")

    # Clear annotations
    clear_ok = pres.clear_annotations() and pres.last_action == "CLEAR_ANNOTATIONS"
    collector.record("Clear Annotations ('E')", clear_ok, f"Action: {pres.last_action}")

    # Black screen blanking toggle
    pres.toggle_black_screen()
    black_on = (pres.blank_screen_mode == "BLACK")
    pres.toggle_black_screen()
    black_off = (pres.blank_screen_mode == "NONE")
    collector.record("Black Screen Blanking Toggle", black_on and black_off, f"Mode: NONE -> {black_on} -> {pres.blank_screen_mode}")

    # White screen blanking toggle
    pres.toggle_white_screen()
    white_on = (pres.blank_screen_mode == "WHITE")
    pres.toggle_white_screen()
    white_off = (pres.blank_screen_mode == "NONE")
    collector.record("White Screen Blanking Toggle", white_on and white_off, f"Mode: NONE -> {white_on} -> {pres.blank_screen_mode}")

    # Slide overview / grid sorter
    overview_ok = pres.slide_overview() and pres.last_action == "SLIDE_OVERVIEW"
    collector.record("Slide Overview Grid Sorter", overview_ok, f"Action: {pres.last_action}")

    # -----------------------------------------------------------------
    # Section 5: Target Profiles & Context Awareness (Module 13)
    # -----------------------------------------------------------------
    print("\n5. Target Profiles & Application Context (Module 13)")
    targets_tested = []
    for tgt in [
        PresentationTarget.POWERPOINT,
        PresentationTarget.GOOGLE_SLIDES,
        PresentationTarget.KEYNOTE,
        PresentationTarget.PDF,
        PresentationTarget.GENERIC,
    ]:
        success = pres.set_target(tgt)
        if success and pres.target_app == tgt:
            targets_tested.append(tgt)

    profiles_ok = len(targets_tested) == 5
    collector.record("Presentation Target Profiles", profiles_ok, f"Configured profiles: {', '.join(targets_tested)}")

    # Telemetry introspection
    status = pres.get_status()
    telemetry_ok = (
        isinstance(status, dict)
        and "target_app" in status
        and "current_slide_index" in status
        and "is_presenting" in status
        and status["event_count"] > 10
    )
    collector.record("Presenter Telemetry Introspection", telemetry_ok, f"Logged {status['event_count']} presenter events")

    # -----------------------------------------------------------------
    # Section 6: Unified Action Router
    # -----------------------------------------------------------------
    print("\n6. Unified Action Router Execution")
    actions_to_test = [
        ("NEXT_SLIDE", "NEXT_SLIDE"),
        ("PREV_SLIDE", "PREV_SLIDE"),
        ("START_PRESENTATION", "START_PRESENTATION"),
        ("EXIT_PRESENTATION", "EXIT_PRESENTATION"),
        ("LASER_POINTER", "LASER_POINTER"),
        ("BLACK_SCREEN", "BLACK_SCREEN"),
        ("WHITE_SCREEN", "WHITE_SCREEN"),
        ("SLIDE_OVERVIEW", "SLIDE_OVERVIEW"),
        ("CLEAR_ANNOTATIONS", "CLEAR_ANNOTATIONS"),
        ("TOGGLE_PEN", "TOGGLE_PEN"),
        ("FIRST_SLIDE", "FIRST_SLIDE"),
        ("LAST_SLIDE", "LAST_SLIDE"),
    ]

    all_router_ok = True
    for act_name, expected_last in actions_to_test:
        ok = pres.execute_action(act_name)
        if not ok or pres.last_action != expected_last:
            all_router_ok = False
            break

    collector.record("Unified Action Router Dispatch", all_router_ok, f"Tested {len(actions_to_test)} canonical commands")

    # -----------------------------------------------------------------
    # Section 7: End-to-End Gesture Recognition Integration
    # -----------------------------------------------------------------
    print("\n7. End-to-End Real-Time Gesture Recognition Integration")
    dataset_path = os.path.join(_ROOT, "dataset", "gesture_dataset.npz")
    dataset_obj = GestureDataset.load_from_npz(dataset_path)

    recognizer = RealtimeGestureRecognizer(
        initial_mode=OperationalMode.PRESENTATION,
        confidence_threshold=0.60,
        temporal_window_size=3,
        min_stable_frames=2,
    )

    # In PRESENTATION mode:
    # PEACE gesture -> SLIDE_OVERVIEW
    peace_idx = np.where(dataset_obj.y == "PEACE")[0][0]
    peace_hand = create_mock_hand_data(dataset_obj.landmarks[peace_idx])

    t_now = time.perf_counter()
    recognizer.process_frame(peace_hand, timestamp=t_now)
    r_pres = recognizer.process_frame(peace_hand, timestamp=t_now + 0.033)

    action_executed = False
    if r_pres.can_execute and r_pres.mapped_action is not None:
        action_executed = pres.execute_action(r_pres.mapped_action)

    static_e2e_ok = (
        r_pres.confirmed_gesture == "PEACE"
        and r_pres.mapped_action == "SLIDE_OVERVIEW"
        and action_executed
    )
    collector.record(
        "E2E Static Gesture -> Slide Overview Execution",
        static_e2e_ok,
        f"Gesture: '{r_pres.confirmed_gesture}' -> Action: '{r_pres.mapped_action}' (Executed: {action_executed})"
    )

    # Dynamic gesture routing test in PRESENTATION mode:
    # SWIPE_RIGHT -> NEXT_SLIDE
    act_swipe_r = recognizer.mapping_manager.get_action("SWIPE_RIGHT", OperationalMode.PRESENTATION)
    res_swipe_r = pres.execute_action(act_swipe_r)

    # SWIPE_LEFT -> PREV_SLIDE
    act_swipe_l = recognizer.mapping_manager.get_action("SWIPE_LEFT", OperationalMode.PRESENTATION)
    res_swipe_l = pres.execute_action(act_swipe_l)

    # THUMBS_UP -> START_PRESENTATION
    act_t_up = recognizer.mapping_manager.get_action("THUMBS_UP", OperationalMode.PRESENTATION)
    res_t_up = pres.execute_action(act_t_up)

    dynamic_e2e_ok = (
        act_swipe_r == "NEXT_SLIDE" and res_swipe_r and
        act_swipe_l == "PREV_SLIDE" and res_swipe_l and
        act_t_up == "START_PRESENTATION" and res_t_up
    )
    collector.record(
        "Dynamic Presentation Gesture Mappings",
        dynamic_e2e_ok,
        "SWIPE_RIGHT->NEXT_SLIDE, SWIPE_LEFT->PREV_SLIDE, THUMBS_UP->START_PRESENTATION"
    )

    # -----------------------------------------------------------------
    # Section 8: Latency & Throughput Microbenchmark
    # -----------------------------------------------------------------
    print("\n8. Latency & Throughput Microbenchmark")
    iterations = 5000
    t0 = time.perf_counter()
    for _ in range(iterations):
        pres.next_slide()
    t1 = time.perf_counter()

    total_time = t1 - t0
    avg_latency_ms = (total_time / iterations) * 1000.0
    throughput = iterations / total_time
    benchmark_ok = avg_latency_ms < 0.05

    collector.record(
        "Presenter Command Dispatch Benchmark",
        benchmark_ok,
        f"Mean latency: {avg_latency_ms:.4f} ms | Throughput: {throughput:,.0f} ops/sec",
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

    report_path = os.path.join(_ROOT, "data", "phase13_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_data = {
        "phase": 13,
        "name": "Presentation & Application-Specific Controls",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "pass_rate_pct": pass_rate,
            "mean_dispatch_latency_ms": round(avg_latency_ms, 5),
            "throughput_ops_sec": round(throughput, 1),
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
