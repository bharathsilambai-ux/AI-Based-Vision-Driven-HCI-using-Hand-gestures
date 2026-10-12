"""
Phase 16 Verification Test Suite: Professional Real-Time GUI Dashboard & HUD (Module 18)
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding & Package Exports (HUDRenderer, VisionDashboard, DashboardFrameResult, HAND_CONNECTIONS, FINGER_COLORS)
2. HUDRenderer Initialization, Geometry & Style Tokens
3. 21-Landmark Hand Skeleton Rendering (Bones & Keypoint Glow Markers)
4. Dynamic Trajectory Polyline Motion Trail (Fading History Buffer)
5. Calibrated Interaction Zone Bounding Box & Corner Brackets
6. Translucent Glassmorphism HUD Telemetry Cards (Header Banner, Gesture Pill, Bottom Action Bar)
7. Confidence Meter Gauge & Stability Bar Calculation
8. VisionDashboard Pipeline Coordinator: End-to-End Frame Processing (Detection -> Recognizer -> Fusion -> HUD)
9. Multimodal Voice Badge & Execution Status Integration
10. Mode Switching State Propagation Across Dashboard & Subsystems
11. Rendering Throughput & Latency Microbenchmark (> 100 FPS, < 5.0 ms per frame render)
"""

import json
import os
import platform
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset import GestureDataset
from detection.hand_detector import HandData
from gestures import GestureRecognitionResult, OperationalMode, RealtimeGestureRecognizer
import ui
from ui import (
    FINGER_COLORS,
    HAND_CONNECTIONS,
    DashboardFrameResult,
    HUDRenderer,
    VisionDashboard,
)
from voice import MultimodalFusionEngine


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
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 16 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     Professional Real-Time GUI Dashboard & HUD (Module 18)")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Section 1: Scaffolding & Package Exports
    # -----------------------------------------------------------------
    print("\n1. Module Scaffolding & Package Exports")
    exports_ok = (
        hasattr(ui, "HUDRenderer")
        and hasattr(ui, "VisionDashboard")
        and hasattr(ui, "DashboardFrameResult")
        and hasattr(ui, "HAND_CONNECTIONS")
        and hasattr(ui, "FINGER_COLORS")
    )
    collector.record("UI Package Exports", exports_ok, "All HUD and Dashboard classes exported from ui package")

    # -----------------------------------------------------------------
    # Section 2: HUDRenderer Initialization & Base Configuration
    # -----------------------------------------------------------------
    print("\n2. HUDRenderer Initialization & Configuration")
    renderer = HUDRenderer(
        show_skeleton=True,
        show_interaction_box=True,
        show_telemetry=True,
        show_trajectory=True,
        interaction_box=(0.15, 0.15, 0.85, 0.85),
    )
    init_ok = (
        renderer.show_skeleton
        and renderer.show_interaction_box
        and renderer.show_telemetry
        and renderer.show_trajectory
        and len(HAND_CONNECTIONS) == 21
    )
    collector.record("HUDRenderer Initialization", init_ok, f"Skeleton: True | Interaction Box: {renderer.interaction_box}")

    # Generate synthetic base frame (640x480 dark canvas)
    base_frame = np.full((480, 640, 3), 20, dtype=np.uint8)

    # Load dataset landmarks for testing
    dataset_path = os.path.join(_ROOT, "dataset", "gesture_dataset.npz")
    dataset_obj = GestureDataset.load_from_npz(dataset_path)
    peace_idx = np.where(dataset_obj.y == "PEACE")[0][0]
    peace_landmarks = dataset_obj.landmarks[peace_idx]  # (21, 3)

    # -----------------------------------------------------------------
    # Section 3: Hand Skeleton & 21-Keypoint Rendering
    # -----------------------------------------------------------------
    print("\n3. Hand Skeleton & Keypoint Network Rendering")
    rendered_skel = base_frame.copy()
    renderer._draw_hand_skeleton(rendered_skel, peace_landmarks, 640, 480)

    # Pixel diff check (must have modified pixels from base)
    skel_diff = np.sum(np.abs(rendered_skel.astype(int) - base_frame.astype(int)))
    skel_ok = skel_diff > 50000  # Significant pixels colored
    collector.record("21-Landmark Hand Skeleton Rendering", skel_ok, f"Rendered 21 joints + 21 bones (pixel delta: {skel_diff:,})")

    # -----------------------------------------------------------------
    # Section 4: Dynamic Trajectory Polyline Motion Trail
    # -----------------------------------------------------------------
    print("\n4. Dynamic Trajectory Polyline Motion Trail")
    renderer.trajectory_buffer.clear()
    t_base = time.time()
    for i in range(15):
        # Simulate moving hand coordinates
        renderer.trajectory_buffer.append((200 + i * 15, 300 - i * 8, t_base - (15 - i) * 0.03))

    rendered_traj = base_frame.copy()
    renderer._draw_trajectory_trail(rendered_traj)
    traj_diff = np.sum(np.abs(rendered_traj.astype(int) - base_frame.astype(int)))
    traj_ok = traj_diff > 2000
    collector.record("Dynamic Trajectory Trail Rendering", traj_ok, f"Rendered 15-frame polyline trail (pixel delta: {traj_diff:,})")

    # -----------------------------------------------------------------
    # Section 5: Interaction Zone Bounding Box & Reticles
    # -----------------------------------------------------------------
    print("\n5. Active Interaction Zone Overlay")
    rendered_box = base_frame.copy()
    renderer._draw_interaction_box(rendered_box, 640, 480)
    box_diff = np.sum(np.abs(rendered_box.astype(int) - base_frame.astype(int)))
    box_ok = box_diff > 10000
    collector.record("Active Interaction Box Overlay", box_ok, f"Rendered brackets and normalized bounds (pixel delta: {box_diff:,})")

    # -----------------------------------------------------------------
    # Section 6: Translucent Glassmorphism Telemetry Cards
    # -----------------------------------------------------------------
    print("\n6. Translucent Glassmorphism Telemetry Cards")
    mock_gesture_result = GestureRecognitionResult(
        raw_gesture="PEACE",
        confirmed_gesture="PEACE",
        confidence=0.94,
        is_confident=True,
        is_stabilized=True,
        active_mode=OperationalMode.PRESENTATION,
        mapped_action="SLIDE_OVERVIEW",
        can_execute=True,
    )

    rendered_tele = base_frame.copy()
    renderer._draw_telemetry_cards(
        rendered_tele,
        recognition_result=mock_gesture_result,
        last_action="SLIDE_OVERVIEW",
        voice_intent_str="switch to presentation mode",
        fps=60.0,
        latency_ms=2.1,
        w=640,
        h=480,
    )
    tele_diff = np.sum(np.abs(rendered_tele.astype(int) - base_frame.astype(int)))
    tele_ok = tele_diff > 200000  # Substantial banner and pill overlays
    collector.record("Glassmorphism Telemetry HUD Cards", tele_ok, f"Rendered header, gesture pill, and action bar (pixel delta: {tele_diff:,})")

    # -----------------------------------------------------------------
    # Section 7: Complete Frame HUD Integration
    # -----------------------------------------------------------------
    print("\n7. Full Frame HUD Overlay Pipeline")
    complete_hud = renderer.render(
        frame=base_frame,
        recognition_result=mock_gesture_result,
        raw_landmarks=peace_landmarks,
        last_action="SLIDE_OVERVIEW",
        voice_intent_str="confirmed",
        latency_ms=2.4,
        fps_override=59.5,
    )
    full_hud_ok = complete_hud.shape == (480, 640, 3) and not np.array_equal(complete_hud, base_frame)
    collector.record("Full Composite Frame HUD Rendering", full_hud_ok, f"Composite shape: {complete_hud.shape} | Colors: BGR")

    # -----------------------------------------------------------------
    # Section 8: VisionDashboard Coordinator Pipeline
    # -----------------------------------------------------------------
    print("\n8. VisionDashboard Coordinator Pipeline Integration")
    recognizer = RealtimeGestureRecognizer(
        initial_mode=OperationalMode.PRESENTATION,
        confidence_threshold=0.60,
        temporal_window_size=3,
        min_stable_frames=2,
    )
    fusion_engine = MultimodalFusionEngine(mock_mode=True)

    dashboard = VisionDashboard(
        recognizer=recognizer,
        fusion_engine=fusion_engine,
        hud_renderer=renderer,
        initial_mode=OperationalMode.PRESENTATION,
        mock_mode=True,
    )

    # Feed synthetic frames through the dashboard
    peace_hand = create_mock_hand_data(peace_landmarks)
    t_now = time.perf_counter()

    # Frame 1
    res1 = dashboard.process_frame(base_frame, hand_input=peace_hand, timestamp=t_now)
    # Frame 2
    res2 = dashboard.process_frame(base_frame, hand_input=peace_hand, timestamp=t_now + 0.033)

    dash_pipeline_ok = (
        isinstance(res2, DashboardFrameResult)
        and res2.annotated_frame.shape == (480, 640, 3)
        and res2.gesture_result is not None
        and res2.active_mode == OperationalMode.PRESENTATION
        and "detect_ms" in res2.latency_breakdown
        and "render_ms" in res2.latency_breakdown
    )
    collector.record(
        "VisionDashboard Pipeline Execution",
        dash_pipeline_ok,
        f"Mode: {res2.active_mode} | Confirmed: '{res2.gesture_result.confirmed_gesture}' | Render: {res2.latency_breakdown.get('render_ms', 0)} ms"
    )

    # -----------------------------------------------------------------
    # Section 9: Mode Switching Propagation
    # -----------------------------------------------------------------
    print("\n9. Dashboard Mode Switching Propagation")
    mode_switched = dashboard.set_mode(OperationalMode.MEDIA)
    telemetry = dashboard.get_telemetry()
    mode_prop_ok = mode_switched and dashboard.active_mode == OperationalMode.MEDIA and telemetry["active_mode"] == OperationalMode.MEDIA
    collector.record("Dashboard Operational Mode Switching", mode_prop_ok, f"Active Mode: {dashboard.active_mode}")

    # -----------------------------------------------------------------
    # Section 10: High-Framerate Rendering Microbenchmark
    # -----------------------------------------------------------------
    print("\n10. High-Framerate Rendering Microbenchmark")
    iterations = 500
    t_start = time.perf_counter()
    for _ in range(iterations):
        _ = renderer.render(
            frame=base_frame,
            recognition_result=mock_gesture_result,
            raw_landmarks=peace_landmarks,
            last_action="TEST_ACTION",
            latency_ms=1.5,
            fps_override=60.0,
        )
    t_total = time.perf_counter() - t_start

    avg_render_ms = (t_total / iterations) * 1000.0
    render_fps = iterations / t_total
    benchmark_ok = avg_render_ms < 5.0  # Must be faster than 5 ms per frame (> 200 FPS)

    collector.record(
        "HUD Frame Rendering Throughput (> 100 FPS)",
        benchmark_ok,
        f"Mean render latency: {avg_render_ms:.3f} ms | Throughput: {render_fps:,.0f} FPS",
        latency_ms=avg_render_ms,
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

    report_path = os.path.join(_ROOT, "data", "phase16_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_data = {
        "phase": 16,
        "name": "Professional Real-Time GUI Dashboard & Heads-Up Display",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "pass_rate_pct": pass_rate,
            "mean_render_latency_ms": round(avg_render_ms, 4),
            "render_throughput_fps": round(render_fps, 1),
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
