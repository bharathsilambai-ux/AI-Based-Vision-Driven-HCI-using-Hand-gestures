"""
Phase 12 Verification Test Suite: Web Browser Automation & Navigation Controls
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding & Package Exports (controllers.BrowserController, BrowserTarget)
2. Page Scrolling Operations & Coordinate Offsets (scroll_up, scroll_down, [>=0] bounds)
3. History Navigation & Page Reloading (history_back, history_forward, refresh_page)
4. Viewport Zoom Operations (zoom_in, zoom_out, zoom_reset, [25%, 500%] clamping)
5. Multi-Tab Lifecycle Management (new_tab, close_tab, next_tab, prev_tab, reopen_closed_tab)
6. Address Bar Navigation (focus_address_bar)
7. Cross-Browser Profile Support (Chrome, Edge, Firefox, Brave, Generic - Module 13)
8. Unified Action Router (execute_action) for All Browser Mode Commands
9. End-to-End Gesture Recognition to Browser Execution (SWIPE_LEFT -> BACK, OK -> REFRESH)
10. Microbenchmark: Sub-Millisecond Dispatch Latency (< 0.05 ms per command)
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
from controllers import BrowserController, BrowserTarget
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
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 12 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     Browser Navigation & Tab Management (Module 11)")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Section 1: Scaffolding & Package Exports
    # -----------------------------------------------------------------
    print("\n1. Module Scaffolding & Package Exports")
    exports_ok = hasattr(controllers, "BrowserController") and hasattr(controllers, "BrowserTarget")
    collector.record("BrowserController Package Exports", exports_ok, "BrowserController and BrowserTarget accessible")

    browser = BrowserController(mock_mode=True)
    init_ok = (browser.mock_mode and browser.target_browser == BrowserTarget.GENERIC and browser.tab_count == 1)
    collector.record("Browser Controller Initialization", init_ok, f"Mode: Mock | Target: {browser.target_browser} | Tabs: {browser.tab_count}")

    # -----------------------------------------------------------------
    # Section 2: Page Scrolling & History Operations
    # -----------------------------------------------------------------
    print("\n2. Page Scrolling & History Operations")
    browser.scroll_offset_px = 0

    # Scroll Down
    browser.scroll_down(150)
    s_down_ok = (browser.scroll_offset_px == 150 and browser.last_action == "SCROLL_DOWN")
    collector.record("Web Page Scroll Down", s_down_ok, f"Offset: 0px -> {browser.scroll_offset_px}px")

    # Scroll Up
    browser.scroll_up(100)
    s_up_ok = (browser.scroll_offset_px == 50 and browser.last_action == "SCROLL_UP")
    collector.record("Web Page Scroll Up", s_up_ok, f"Offset: 150px -> {browser.scroll_offset_px}px")

    # Scroll Floor Clamping (>= 0px)
    browser.scroll_up(200)
    s_floor_ok = (browser.scroll_offset_px == 0)
    collector.record("Scroll Offset Floor Clamping (>= 0px)", s_floor_ok, f"Offset clamped to: {browser.scroll_offset_px}px")

    # History Forward & Back
    h0 = browser.history_index
    browser.history_forward()
    fwd_ok = (browser.history_index == h0 + 1 and browser.last_action == "HISTORY_FORWARD")
    browser.history_back()
    back_ok = (browser.history_index == h0 and browser.last_action == "HISTORY_BACK")
    collector.record("History Forward & Back Navigation", fwd_ok and back_ok, f"History: {h0} -> {h0+1} -> {browser.history_index}")

    # Page Reload / Refresh
    browser.refresh_page(hard_reload=False)
    ref_ok = (browser.last_action == "REFRESH_PAGE")
    collector.record("Page Refresh / Reload Action", ref_ok, f"Action: '{browser.last_action}'")

    # -----------------------------------------------------------------
    # Section 3: Viewport Zoom Controls
    # -----------------------------------------------------------------
    print("\n3. Viewport Zoom Operations")
    browser.zoom_level_pct = 100

    # Zoom In
    browser.zoom_in(step=20)
    z_in_ok = (browser.zoom_level_pct == 120 and browser.last_action == "ZOOM_IN")
    collector.record("Viewport Zoom In (+20%)", z_in_ok, f"Zoom: 100% -> {browser.zoom_level_pct}%")

    # Zoom Out
    browser.zoom_out(step=30)
    z_out_ok = (browser.zoom_level_pct == 90 and browser.last_action == "ZOOM_OUT")
    collector.record("Viewport Zoom Out (-30%)", z_out_ok, f"Zoom: 120% -> {browser.zoom_level_pct}%")

    # Zoom Reset
    browser.zoom_reset()
    z_reset_ok = (browser.zoom_level_pct == 100 and browser.last_action == "ZOOM_RESET")
    collector.record("Viewport Zoom Reset (to 100%)", z_reset_ok, f"Zoom restored to: {browser.zoom_level_pct}%")

    # Zoom Clamping: 25% min to 500% max
    browser.zoom_in(step=1000)
    z_max = browser.zoom_level_pct
    browser.zoom_out(step=2000)
    z_min = browser.zoom_level_pct
    z_clamp_ok = (z_max == 500 and z_min == 25)
    collector.record("Zoom Boundary Clamping ([25%, 500%])", z_clamp_ok, f"Max: {z_max}%, Min: {z_min}%")
    browser.zoom_reset()

    # -----------------------------------------------------------------
    # Section 4: Multi-Tab Lifecycle Management
    # -----------------------------------------------------------------
    print("\n4. Multi-Tab Management & Navigation")
    # New Tab
    browser.tab_count = 1
    browser.active_tab_index = 0
    browser.new_tab()
    new_tab_ok = (browser.tab_count == 2 and browser.active_tab_index == 1 and browser.last_action == "NEW_TAB")
    collector.record("Open New Tab Operation", new_tab_ok, f"Tabs: 1 -> {browser.tab_count} (Active: {browser.active_tab_index})")

    # Next Tab (Cyclical)
    browser.next_tab()
    next_tab_ok = (browser.active_tab_index == 0 and browser.last_action == "TAB_SWITCH_NEXT")
    collector.record("Switch Next Tab (Cyclical)", next_tab_ok, f"Active Tab Index: {browser.active_tab_index}")

    # Previous Tab (Cyclical)
    browser.prev_tab()
    prev_tab_ok = (browser.active_tab_index == 1 and browser.last_action == "TAB_SWITCH_PREV")
    collector.record("Switch Previous Tab (Cyclical)", prev_tab_ok, f"Active Tab Index: {browser.active_tab_index}")

    # Close Tab
    browser.close_tab()
    close_tab_ok = (browser.tab_count == 1 and browser.last_action == "CLOSE_TAB")
    collector.record("Close Tab Operation", close_tab_ok, f"Tabs remaining: {browser.tab_count}")

    # Reopen Closed Tab
    browser.reopen_closed_tab()
    reopen_ok = (browser.tab_count == 2 and browser.last_action == "REOPEN_TAB")
    collector.record("Reopen Closed Tab Operation", reopen_ok, f"Tabs restored: {browser.tab_count}")

    # Focus Address Bar
    browser.focus_address_bar()
    url_ok = (browser.last_action == "FOCUS_URL")
    collector.record("Focus Address Bar Operation", url_ok, f"Action: '{browser.last_action}'")

    # -----------------------------------------------------------------
    # Section 5: Cross-Browser Profile Support (Module 13)
    # -----------------------------------------------------------------
    print("\n5. Cross-Browser Profile Switching (Module 13)")
    profiles = [BrowserTarget.CHROME, BrowserTarget.EDGE, BrowserTarget.FIREFOX, BrowserTarget.BRAVE, BrowserTarget.GENERIC]
    profiles_ok = all(browser.set_target(p) and browser.target_browser == p for p in profiles)
    collector.record("Cross-Browser Target Profiles (Chrome, Edge, Firefox)", profiles_ok, f"Supported: {profiles}")

    # -----------------------------------------------------------------
    # Section 6: Unified Action Router (execute_action())
    # -----------------------------------------------------------------
    print("\n6. Unified Browser Action Router (execute_action)")
    test_commands = [
        ("SCROLL_UP", "SCROLL_UP"),
        ("SCROLL_DOWN", "SCROLL_DOWN"),
        ("HISTORY_BACK", "HISTORY_BACK"),
        ("HISTORY_FORWARD", "HISTORY_FORWARD"),
        ("REFRESH_PAGE", "REFRESH_PAGE"),
        ("ZOOM_IN", "ZOOM_IN"),
        ("ZOOM_OUT", "ZOOM_OUT"),
        ("ZOOM_RESET", "ZOOM_RESET"),
        ("TAB_SWITCH_NEXT", "TAB_SWITCH_NEXT"),
        ("TAB_SWITCH_PREV", "TAB_SWITCH_PREV"),
        ("NEW_TAB", "NEW_TAB"),
        ("CLOSE_TAB", "CLOSE_TAB"),
        ("REOPEN_TAB", "REOPEN_TAB"),
        ("FOCUS_URL", "FOCUS_URL"),
    ]

    all_routed = True
    for cmd, expected_act in test_commands:
        res = browser.execute_action(cmd)
        if not res or browser.last_action != expected_act:
            all_routed = False
            break

    collector.record("Unified Browser Action Routing", all_routed, f"Successfully routed {len(test_commands)} browser commands")

    # -----------------------------------------------------------------
    # Section 7: End-to-End Real-Time Gesture Integration
    # -----------------------------------------------------------------
    print("\n7. End-to-End Real-Time Gesture to Browser Execution")
    dataset_path = os.path.join(_ROOT, "dataset", "gesture_dataset.npz")
    dataset_obj = GestureDataset.load_from_npz(dataset_path)

    recognizer = RealtimeGestureRecognizer(
        initial_mode=OperationalMode.BROWSER,
        confidence_threshold=0.60,
        temporal_window_size=3,
        min_stable_frames=2
    )

    # In BROWSER mode: OK gesture triggers REFRESH_PAGE
    ok_idx = np.where(dataset_obj.y == "OK")[0][0]
    ok_hand = create_mock_hand_data(dataset_obj.landmarks[ok_idx])

    t_now = time.perf_counter()
    recognizer.process_frame(ok_hand, timestamp=t_now)
    r_browser = recognizer.process_frame(ok_hand, timestamp=t_now + 0.033)

    action_executed = False
    if r_browser.can_execute and r_browser.mapped_action is not None:
        action_executed = browser.execute_action(r_browser.mapped_action)

    e2e_ok = (r_browser.confirmed_gesture == "OK" and r_browser.mapped_action == "REFRESH_PAGE" and action_executed)
    collector.record("E2E Gesture Recognition -> Browser Execution", e2e_ok, f"Gesture: '{r_browser.confirmed_gesture}' -> Action: '{r_browser.mapped_action}' (Executed: {action_executed})")

    # -----------------------------------------------------------------
    # Section 8: Sub-Millisecond Action Routing Latency Benchmark
    # -----------------------------------------------------------------
    print("\n8. Action Routing Latency Microbenchmark")
    n_iters = 1000
    t_start = time.perf_counter()
    for i in range(n_iters):
        _ = browser.execute_action("SCROLL_DOWN")
    t_total = time.perf_counter() - t_start

    latency_per_call_ms = (t_total / n_iters) * 1000.0
    throughput_ops = n_iters / t_total

    collector.record(
        "Browser Dispatch Latency (< 0.05 ms)",
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
    print(" PHASE 12 TEST EXECUTION SUMMARY")
    print("=" * 75)
    print(f" Total Tests Run:   {total}")
    print(f" Passed:            {passed}")
    print(f" Failed:            {failed}")
    print(f" Pass Rate:         {pass_rate:.1f}%")
    print("=" * 75)

    report_path = os.path.join(_ROOT, "data", "phase12_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "phase": "Phase 12 - Web Browser Automation & Navigation Controls",
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
