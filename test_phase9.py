"""
Phase 9 Verification Test Suite: Advanced Virtual Mouse Controller & Jitter Filtering
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding & Package Exports (controllers package)
2. Screen Resolution & Active Workspace Coordinate Mapping (Module 8)
3. Boundary Clamping & Active Interaction Box Calibration
4. 1-Euro Adaptive Filter Jitter Reduction & Velocity Tracking
5. Exponential Moving Average (EMA) Smoothing & Deadzone Suppression
6. Adaptive Sensitivity Multiplier & Non-Linear Power Curve
7. Complete Mouse Action Vocabulary (Left, Right, Double Click, Drag, Drop, Hold)
8. Vertical Scrolling & Zoom Control
9. End-to-End Gesture-to-Mouse Dispatching (execute_action())
10. State Cleanup & Safe Drag Release (reset())
11. Microbenchmark: Sub-Millisecond Cursor Movement Latency (< 0.10 ms / > 10,000 moves/sec)
"""

import json
import math
import os
import platform
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import controllers
from controllers import OneEuroFilter, VirtualMouseController


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
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 9 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     Advanced Virtual Mouse Controller (Module 8)")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Section 1: Scaffolding & Package Exports
    # -----------------------------------------------------------------
    print("\n1. Module Scaffolding & Package Exports")
    exports_ok = (
        hasattr(controllers, "VirtualMouseController") and
        hasattr(controllers, "OneEuroFilter")
    )
    collector.record("Controllers Package Exports", exports_ok, "VirtualMouseController and OneEuroFilter accessible")

    # Use simulated 1920x1080 screen in mock mode to avoid interfering with user's desktop cursor
    sim_screen = (1920, 1080)
    mouse = VirtualMouseController(
        screen_size=sim_screen,
        active_box=[0.10, 0.10, 0.90, 0.80],
        sensitivity=1.5,
        smoothing_alpha=0.60,
        deadzone_px=4.0,
        mock_mode=True
    )
    init_ok = (mouse.screen_w == 1920 and mouse.screen_h == 1080 and mouse.mock_mode)
    collector.record("Virtual Mouse Initialization", init_ok, f"Screen: {mouse.screen_w}x{mouse.screen_h} | Mock Mode: {mouse.mock_mode}")

    # -----------------------------------------------------------------
    # Section 2: Active Box Workspace Mapping & Clamping
    # -----------------------------------------------------------------
    print("\n2. Active Workspace Mapping & Edge Calibration")
    # Top-Left corner of active box: (0.10, 0.10) should map to (0, 0)
    mouse.reset()
    tl_x, tl_y = mouse.move_cursor_to_norm(0.10, 0.10)
    tl_ok = (tl_x == 0 and tl_y == 0)
    collector.record("Active Box Top-Left Mapping", tl_ok, f"(0.10, 0.10) -> ({tl_x}, {tl_y}) [Expected: (0, 0)]")

    # Bottom-Right corner of active box: (0.90, 0.80) should map to (1919, 1079)
    mouse.reset()
    br_x, br_y = mouse.move_cursor_to_norm(0.90, 0.80)
    br_ok = (br_x == 1919 and br_y == 1079)
    collector.record("Active Box Bottom-Right Mapping", br_ok, f"(0.90, 0.80) -> ({br_x}, {br_y}) [Expected: (1919, 1079)]")

    # Center of active box: (0.50, 0.45) should map to approximately center screen (960, 540)
    mouse.reset()
    c_x, c_y = mouse.move_cursor_to_norm(0.50, 0.45)
    c_ok = (abs(c_x - 960) <= 5 and abs(c_y - 540) <= 5)
    collector.record("Active Box Center Mapping", c_ok, f"(0.50, 0.45) -> ({c_x}, {c_y}) [Expected: (960, 540)]")

    # Boundary Clamping: coordinates far outside [0.0, 1.0] must safely clamp without error
    mouse.reset()
    clamp_min = mouse.move_cursor_to_norm(-0.50, -0.50)
    mouse.reset()
    clamp_max = mouse.move_cursor_to_norm(1.50, 1.50)
    clamp_ok = (clamp_min == (0, 0) and clamp_max == (1919, 1079))
    collector.record("Workspace Clamping Guard", clamp_ok, f"Min: {clamp_min}, Max: {clamp_max}")

    # -----------------------------------------------------------------
    # Section 3: 1-Euro Adaptive Filter & Jitter Reduction
    # -----------------------------------------------------------------
    print("\n3. 1-Euro Adaptive Filter & Jitter Suppression")
    filter_1euro = OneEuroFilter(t0=0.0, x0=500.0, min_cutoff=0.8, beta=0.005)

    # Stationary jitter: small high-frequency oscillations (+- 2 pixels)
    t = 0.0
    raw_jitter_vals = [500.0 + (2.0 if i % 2 == 0 else -2.0) for i in range(10)]
    filtered_vals = []
    for val in raw_jitter_vals:
        t += 0.033
        filtered_vals.append(filter_1euro.filter(t, val))

    # Calculate variance of raw vs filtered
    raw_var = float(np.var(raw_jitter_vals))
    filtered_var = float(np.var(filtered_vals))
    jitter_attenuated = filtered_var < raw_var * 0.35
    collector.record("1-Euro Low-Speed Jitter Attenuation", jitter_attenuated, f"Raw Var: {raw_var:.2f} -> Filtered Var: {filtered_var:.2f} ({(1 - filtered_var/raw_var)*100:.1f}% reduction)")

    # High-speed movement responsiveness: fast movement from 500 to 1200
    fast_t = t
    for step in range(5):
        fast_t += 0.033
        target = 500.0 + (step + 1) * 140.0  # +700 pixels total
        fast_out = filter_1euro.filter(fast_t, target)

    speed_responsiveness = abs(fast_out - 1200.0) < 150.0  # Adapts cutoff to closely follow swift movements
    collector.record("1-Euro High-Speed Responsiveness", speed_responsiveness, f"Fast Target: 1200.0 -> Followed to: {fast_out:.1f}")

    # Deadzone Thresholding
    mouse.reset()
    mouse.set_deadzone(10.0)  # 10px deadzone
    init_pos = mouse.move_cursor_to_norm(0.50, 0.45)
    # Move by 0.001 (less than 2 pixels on screen)
    micro_pos = mouse.move_cursor_to_norm(0.501, 0.451)
    deadzone_held = (micro_pos == init_pos)
    collector.record("Deadzone Micro-Tremor Suppression", deadzone_held, f"Initial: {init_pos}, Micro-move: {micro_pos} (Suppressed: {deadzone_held})")

    # -----------------------------------------------------------------
    # Section 4: Adaptive Sensitivity & Acceleration Curves
    # -----------------------------------------------------------------
    print("\n4. Adaptive Sensitivity & Non-Linear Acceleration")
    t_sens = time.perf_counter()
    mouse.reset()
    mouse.set_deadzone(0.0)
    mouse.set_sensitivity(1.0)
    p0 = mouse.move_cursor_to_norm(0.40, 0.40, timestamp=t_sens)
    p_low = mouse.move_cursor_to_norm(0.45, 0.40, timestamp=t_sens + 0.033)
    dist_low = abs(p_low[0] - p0[0])

    mouse.reset()
    mouse.set_sensitivity(2.5)
    p0 = mouse.move_cursor_to_norm(0.40, 0.40, timestamp=t_sens + 1.0)
    p_high = mouse.move_cursor_to_norm(0.45, 0.40, timestamp=t_sens + 1.033)
    dist_high = abs(p_high[0] - p0[0])

    sens_scaling_ok = dist_high > dist_low
    collector.record("Configurable Sensitivity Scaling", sens_scaling_ok, f"Low (1.0x): {dist_low}px traversal, High (2.5x): {dist_high}px traversal")

    # -----------------------------------------------------------------
    # Section 5: Complete Mouse Action Vocabulary
    # -----------------------------------------------------------------
    print("\n5. Mouse Action Vocabulary (Click, Drag, Drop, Scroll, Zoom)")
    mouse.reset()

    # Left Click
    click_ok = mouse.left_click() and mouse.last_action == "LEFT_CLICK"
    collector.record("Left Click Dispatch", click_ok, f"Action: '{mouse.last_action}'")

    # Right Click
    rclick_ok = mouse.right_click() and mouse.last_action == "RIGHT_CLICK"
    collector.record("Right Click Dispatch", rclick_ok, f"Action: '{mouse.last_action}'")

    # Double Click
    dclick_ok = mouse.double_click() and mouse.last_action == "DOUBLE_CLICK"
    collector.record("Double Click Dispatch", dclick_ok, f"Action: '{mouse.last_action}'")

    # Drag and Drop Lifecycle
    mouse.start_drag()
    drag_active = mouse.is_dragging
    mouse.move_cursor_to_norm(0.60, 0.60)
    mouse.stop_drag()
    drag_finished = not mouse.is_dragging
    drag_lifecycle_ok = drag_active and drag_finished
    collector.record("Drag and Drop State Lifecycle", drag_lifecycle_ok, f"Started: {drag_active} -> Stopped: {not drag_finished}")

    # Click-and-Hold & Release
    mouse.click_and_hold()
    hold_active = mouse.is_held and mouse.is_dragging
    mouse.release_click()
    hold_released = not mouse.is_held and not mouse.is_dragging
    hold_ok = hold_active and hold_released
    collector.record("Click-and-Hold / Release Lifecycle", hold_ok, f"Held: {hold_active} -> Released: {hold_released}")

    # Vertical Scroll Up and Down
    mouse.scroll_up(120)
    s_up_ok = mouse.last_action == "SCROLL_UP"
    mouse.scroll_down(120)
    s_down_ok = mouse.last_action == "SCROLL_DOWN"
    collector.record("Vertical Scrolling Up & Down", s_up_ok and s_down_ok, f"Up: '{s_up_ok}', Down: '{s_down_ok}'")

    # Zoom In and Zoom Out
    mouse.zoom_in()
    z_in_ok = mouse.last_action == "ZOOM_IN"
    mouse.zoom_out()
    z_out_ok = mouse.last_action == "ZOOM_OUT"
    collector.record("Zoom In / Zoom Out Control", z_in_ok and z_out_ok, f"Zoom In: {z_in_ok}, Zoom Out: {z_out_ok}")

    # -----------------------------------------------------------------
    # Section 6: High-Level Gesture-to-Mouse Action Routing
    # -----------------------------------------------------------------
    print("\n6. Gesture-to-Mouse Dispatching Integration (execute_action())")
    mouse.reset()

    # Route 'CURSOR_MOVE' with tracking point
    moved = mouse.execute_action("CURSOR_MOVE", tracking_point=(0.40, 0.40))
    # Route 'LEFT_CLICK'
    clicked = mouse.execute_action("LEFT_CLICK")
    # Route 'DRAG_DROP'
    dragged = mouse.execute_action("DRAG_DROP")
    # Route 'SCROLL_UP'
    scrolled = mouse.execute_action("SCROLL_UP")
    # Route 'RELEASE'
    released = mouse.execute_action("RELEASE")

    dispatch_all_ok = moved and clicked and dragged and scrolled and released
    collector.record("Unified Action Router (execute_action)", dispatch_all_ok, "CURSOR_MOVE, LEFT_CLICK, DRAG_DROP, SCROLL_UP, RELEASE dispatched")

    # -----------------------------------------------------------------
    # Section 7: State Recovery & Reset Integrity
    # -----------------------------------------------------------------
    print("\n7. Controller State Cleanup & Reset Safety")
    mouse.start_drag()
    mouse.reset()
    clean_reset = (not mouse.is_dragging and not mouse.is_held and mouse.prev_cursor_x is None)
    collector.record("Clean State Recovery on Reset", clean_reset, "Released active drag and flushed coordinates")

    # -----------------------------------------------------------------
    # Section 8: Sub-Millisecond Cursor Latency Microbenchmark
    # -----------------------------------------------------------------
    print("\n8. Sub-Millisecond Cursor Latency Benchmark")
    n_iters = 1000
    t_start = time.perf_counter()
    for i in range(n_iters):
        norm = 0.20 + (i % 50) * 0.01
        _ = mouse.move_cursor_to_norm(norm, norm)
    t_total = time.perf_counter() - t_start

    latency_per_move_ms = (t_total / n_iters) * 1000.0
    throughput_moves_sec = n_iters / t_total

    collector.record(
        "Cursor Movement Latency (< 0.10 ms)",
        latency_per_move_ms < 0.10,
        f"Mean: {latency_per_move_ms*1000:.1f} microseconds ({latency_per_move_ms:.4f} ms) | {throughput_moves_sec:,.0f} moves/sec"
    )

    # -----------------------------------------------------------------
    # Summary & Export
    # -----------------------------------------------------------------
    total = len(collector.results)
    passed = sum(1 for r in collector.results if r["status"] == "PASS")
    failed = total - passed
    pass_rate = (passed / total) * 100.0 if total > 0 else 0.0

    print("\n" + "=" * 75)
    print(" PHASE 9 TEST EXECUTION SUMMARY")
    print("=" * 75)
    print(f" Total Tests Run:   {total}")
    print(f" Passed:            {passed}")
    print(f" Failed:            {failed}")
    print(f" Pass Rate:         {pass_rate:.1f}%")
    print("=" * 75)

    report_path = os.path.join(_ROOT, "data", "phase9_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "phase": "Phase 9 - Virtual Mouse Controller & Jitter Filtering",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.platform(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": pass_rate,
            "mean_cursor_latency_ms": latency_per_move_ms,
            "cursor_move_throughput": throughput_moves_sec,
            "results": collector.results
        }, f, indent=2)

    print(f" Benchmark report exported to: {report_path}\n")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
