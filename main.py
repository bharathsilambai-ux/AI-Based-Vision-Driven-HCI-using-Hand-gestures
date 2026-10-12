"""
AI-Based Vision-Driven Human-Computer Interaction Using Advanced Hand Gesture Recognition
Master System Entry Point and Unified CLI / GUI Driver (Phase 20).

Usage:
    python main.py                              # Launch Modern Desktop CustomTkinter GUI Dashboard
    python main.py --cli                        # Launch Lightweight OpenCV HUD Preview Window
    python main.py --cli --mock                 # Launch OpenCV HUD in Synthetic Mock Mode (no webcam needed)
    python main.py --cli --mode PRESENTATION    # Launch directly into Presentation Mode
    python main.py --cli --voice                # Launch with Voice + Gesture Multimodal Fusion
    python main.py --eval                       # Run Comprehensive Benchmark Evaluation & Latency Profiler
    python main.py --test                       # Run 20-Phase Master Verification Test Suite
    python main.py --help                       # Show command options and usage
"""

import argparse
import os
import platform
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from controllers import (
    BrowserController,
    MediaController,
    PresentationController,
    SystemController,
    VirtualMouseController,
)
from detection.hand_detector import HandData, HandDetector
from evaluation.benchmark_evaluator import BenchmarkEvaluator
from evaluation.latency_profiler import LatencyProfiler
from gestures.custom_mapping import CustomMappingEngine
from gestures.gesture_mapping import OperationalMode
from gestures.realtime_recognizer import RealtimeGestureRecognizer
from ui.dashboard import DashboardFrameResult, VisionDashboard
from ui.hud_renderer import HUDRenderer
from utils.logger import logger
from utils.safety_manager import SafetyManager
from voice.intent_classifier import IntentClassifier
from voice.multimodal_fusion import MultimodalFusionEngine
from voice.speech_engine import SpeechEngine


def run_test_suite() -> int:
    """Runs the 20-Phase Master Verification Test Suite and returns exit code."""
    print("=" * 78)
    print(" EXECUTING MASTER 20-PHASE INTEGRATION TEST SUITE")
    print("=" * 78)
    import test_phase20
    success = test_phase20.run_tests()
    return 0 if success else 1


def run_evaluation_suite(mock_mode: bool = True) -> int:
    """Runs full empirical ML benchmarking, paper comparative analysis, and latency profiling."""
    print("=" * 78)
    print(" AI-BASED VISION-DRIVEN HCI - EMPIRICAL BENCHMARK EVALUATOR")
    print("=" * 78)
    print(" 1. Machine Learning Accuracy & Dynamic Kinematic Recognition")
    print(" 2. Research Paper 10-Dimension Comparative Matrix")
    print(" 3. High-Resolution Multi-Stage Latency Profiling")
    print("=" * 78)

    evaluator = BenchmarkEvaluator()
    eval_report = evaluator.run_full_evaluation()

    static_res = eval_report.get("static_evaluation", {})
    dyn_res = eval_report.get("dynamic_evaluation", {})
    paper_res = eval_report.get("paper_comparison", {}).get("comparison_matrix", [])

    print("\n[ML STATIC CLASSIFICATION METRICS]")
    print(f"  Overall Accuracy:  {static_res.get('accuracy', 0.0)*100:.2f}%")
    print(f"  Macro Precision:   {static_res.get('macro_precision', 0.0)*100:.2f}%")
    print(f"  Macro Recall:      {static_res.get('macro_recall', 0.0)*100:.2f}%")
    print(f"  Macro F1-Score:    {static_res.get('macro_f1', 0.0)*100:.2f}%")

    print("\n[DYNAMIC TRAJECTORY RECOGNITION]")
    print(f"  Overall Accuracy:  {dyn_res.get('overall_accuracy', 0.0)*100:.2f}%")
    per_class = dyn_res.get("per_class_accuracy", {})
    for g_name, g_acc in per_class.items():
        print(f"    - {g_name:<14}: {g_acc*100:.1f}%")

    print("\n[RESEARCH PAPER COMPARATIVE BENCHMARK]")
    for item in paper_res[:5]:
        print(f"  * {item['dimension']}:")
        print(f"      Baseline: {item['baseline_paper']}")
        print(f"      Our Sys:  {item['our_system']} ({item['improvement']})")

    print("\n[HIGH-RESOLUTION LATENCY PROFILING (100 FRAMES)]")
    profiler = LatencyProfiler()
    prof_report = profiler.run_multi_stage_benchmark(n_frames=100, warmup_frames=20)
    summary = prof_report.get("summary", {})
    compliance = prof_report.get("compliance", {})
    print(f"  Mean End-to-End Latency:  {summary.get('mean_e2e_latency_ms', 0.0):.3f} ms")
    print(f"  Median (p50) Latency:     {summary.get('median_e2e_latency_ms', 0.0):.3f} ms")
    print(f"  95th Percentile (p95):    {summary.get('p95_e2e_latency_ms', 0.0):.3f} ms")
    print(f"  Throughput Capability:    {summary.get('achievable_fps', 0.0):.1f} FPS")
    print(f"  Jitter Stability Ratio:   {compliance.get('jitter_stability_ratio', 0.0):.2f}x (Threshold <= 2.0x)")
    print("=" * 78)
    print(" Empirical Evaluation Complete. Reports stored in data/phase18_evaluation_report.json")
    print(" and data/phase19_latency_report.json.")
    print("=" * 78)
    return 0


def generate_synthetic_frame(
    t: float,
    width: int = 640,
    height: int = 480
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a synthetic camera frame with moving hand landmarks for mock mode."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Subtle dark background grid
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (22, 24, 30), 1)
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (22, 24, 30), 1)

    # Simulated hand moving in a smooth lissajous / circular orbit
    cx = 0.5 + 0.25 * np.cos(t * 1.5)
    cy = 0.5 + 0.20 * np.sin(t * 2.0)

    # 21 synthetic landmarks around center
    lms = np.zeros((21, 3), dtype=np.float32)
    lms[0] = [cx, cy + 0.15, 0.0]  # Wrist
    # Thumb
    lms[1] = [cx - 0.06, cy + 0.10, 0.0]
    lms[2] = [cx - 0.09, cy + 0.05, 0.0]
    lms[3] = [cx - 0.11, cy + 0.01, 0.0]
    lms[4] = [cx - 0.12, cy - 0.03, 0.0]
    # Index finger extended
    lms[5] = [cx - 0.04, cy + 0.04, 0.0]
    lms[6] = [cx - 0.04, cy - 0.04, 0.0]
    lms[7] = [cx - 0.04, cy - 0.10, 0.0]
    lms[8] = [cx - 0.04, cy - 0.16, 0.0]  # Tip pointing
    # Middle, Ring, Pinky folded
    for base_i, offset_x in [(9, 0.0), (13, 0.04), (17, 0.08)]:
        lms[base_i] = [cx + offset_x, cy + 0.04, 0.0]
        lms[base_i + 1] = [cx + offset_x, cy + 0.00, 0.0]
        lms[base_i + 2] = [cx + offset_x, cy + 0.03, 0.0]
        lms[base_i + 3] = [cx + offset_x, cy + 0.06, 0.0]

    # Draw synthetic hand skeleton for mock visual
    px = (lms[:, 0] * width).astype(int)
    py = (lms[:, 1] * height).astype(int)
    for i in range(21):
        cv2.circle(frame, (px[i], py[i]), 4, (0, 200, 255), -1)

    return frame, lms


def run_cli_mode(
    mode: str = OperationalMode.MOUSE,
    mock_mode: bool = False,
    enable_voice: bool = False,
    max_frames: Optional[int] = None,
) -> int:
    """
    Runs lightweight OpenCV HUD preview mode with real-time telemetry and keyboard controls.
    """
    print("=" * 78)
    print(" AI-BASED VISION-DRIVEN HCI - REAL-TIME HUD PREVIEW")
    print("=" * 78)
    print(f" Initial Mode:    {mode}")
    print(f" Mock Mode:        {'ENABLED (Synthetic Simulation)' if mock_mode else 'DISABLED (Live Hardware)'}")
    print(f" Voice Fusion:     {'ENABLED' if enable_voice else 'DISABLED'}")
    print("-" * 78)
    print(" Keyboard Controls:")
    print("   [1-7]   - Switch Mode: 1:MOUSE, 2:SYSTEM, 3:MEDIA, 4:BROWSER, 5:PRESENTATION, 6:APPLICATION, 7:CUSTOM")
    print("   [Space] - Pause / Resume Tracking")
    print("   [Esc]   - Emergency Stop (Safety Lock)")
    print("   [r]     - Reset / Resume from Emergency Stop")
    print("   [q]     - Quit Preview")
    print("=" * 78)

    # Initialize Perception Subsystems
    hand_detector = None
    if not mock_mode:
        try:
            hand_detector = HandDetector(max_hands=1)
        except Exception as e:
            logger.warning(f"Could not initialize live HandDetector ({e}). Falling back to mock detector.")
            mock_mode = True

    recognizer = RealtimeGestureRecognizer(initial_mode=mode)
    hud_renderer = HUDRenderer()

    # Initialize Controllers
    mouse_ctrl = VirtualMouseController(mock_mode=mock_mode)
    system_ctrl = SystemController(mock_mode=mock_mode)
    media_ctrl = MediaController(mock_mode=mock_mode)
    browser_ctrl = BrowserController(mock_mode=mock_mode)
    pres_ctrl = PresentationController(mock_mode=mock_mode)
    custom_engine = CustomMappingEngine()
    safety_mgr = SafetyManager(mock_mode=mock_mode)

    # Multimodal Voice Fusion (Optional)
    fusion_engine = None
    if enable_voice:
        speech_engine = SpeechEngine(mock_mode=mock_mode)
        intent_classifier = IntentClassifier()
        fusion_engine = MultimodalFusionEngine(
            speech_engine=speech_engine,
            intent_classifier=intent_classifier,
            gesture_recognizer=recognizer,
            mouse_controller=mouse_ctrl,
            system_controller=system_ctrl,
            media_controller=media_ctrl,
            browser_controller=browser_ctrl,
            presentation_controller=pres_ctrl,
            mock_mode=mock_mode,
        )

    # Master Vision Dashboard
    dashboard = VisionDashboard(
        hand_detector=hand_detector,
        recognizer=recognizer,
        fusion_engine=fusion_engine,
        hud_renderer=hud_renderer,
        initial_mode=mode,
        mock_mode=mock_mode,
    )

    # Camera Acquisition
    cap = None
    if not mock_mode:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("Camera index 0 not available. Switching to mock synthetic simulation.")
            mock_mode = True
            if cap:
                cap.release()
                cap = None

    is_paused = False
    frame_idx = 0
    start_time = time.perf_counter()

    try:
        while True:
            t_now = time.perf_counter()
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # Frame Capture or Simulation
            mock_lms = None
            if mock_mode or cap is None:
                sim_t = t_now - start_time
                frame, mock_lms = generate_synthetic_frame(sim_t)
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                frame = cv2.flip(frame, 1)

            # If paused, render dimmed pause overlay
            if is_paused:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (10, 10, 20), -1)
                cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
                cv2.putText(frame, "PAUSED - PRESS SPACE TO RESUME", (140, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
            else:
                # Process through master perception pipeline
                dash_res: DashboardFrameResult = dashboard.process_frame(
                    frame=frame,
                    hand_input=mock_lms,
                    timestamp=t_now,
                )
                frame = dash_res.annotated_frame

                # Dispatch Action to Controllers (if not in emergency stop)
                if not safety_mgr.is_emergency_stopped and dash_res.gesture_result:
                    gres = dash_res.gesture_result
                    action = dash_res.last_action
                    cur_mode = dash_res.active_mode

                    if gres.can_execute and action and action != "NONE":
                        norm_pos = (gres.cursor_x, gres.cursor_y)
                        if cur_mode == OperationalMode.MOUSE:
                            mouse_ctrl.execute_action(action, tracking_point=norm_pos, timestamp=t_now)
                        elif cur_mode == OperationalMode.SYSTEM:
                            system_ctrl.execute_action(action)
                        elif cur_mode == OperationalMode.MEDIA:
                            media_ctrl.execute_action(action)
                        elif cur_mode == OperationalMode.BROWSER:
                            browser_ctrl.execute_action(action)
                        elif cur_mode == OperationalMode.PRESENTATION:
                            pres_ctrl.execute_action(action)
                        elif cur_mode == OperationalMode.APPLICATION:
                            system_ctrl.execute_action(action)
                        elif cur_mode == OperationalMode.CUSTOM:
                            custom_engine.execute_gesture(gres.confirmed_gesture)

            # Display preview window
            window_name = "AI Vision-Driven HCI (q:Quit, 1-7:Modes, Space:Pause, Esc:Stop)"
            try:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                # Headless environment where cv2.imshow is not supported
                if frame_idx % 30 == 0:
                    print(f"  [Headless Telemetry] Frame {frame_idx:04d} | Mode: {dashboard.active_mode:<12} | Action: {dashboard.last_action}")
                time.sleep(0.01)
                continue

            if key == ord('q'):
                break
            elif key == ord(' '):
                is_paused = not is_paused
                print(f"[{'PAUSED' if is_paused else 'RESUMED'}] System tracking toggled.")
            elif key == 27:  # Esc
                safety_mgr.emergency_stop()
                print("[EMERGENCY STOP] All OS controls halted immediately.")
            elif key == ord('r'):
                safety_mgr.resume_operation()
                print("[RESUMED] Emergency stop released. Operations normal.")
            elif ord('1') <= key <= ord('7'):
                modes = OperationalMode.ALL_MODES
                sel_idx = key - ord('1')
                if sel_idx < len(modes):
                    dashboard.set_mode(modes[sel_idx])
                    print(f"[MODE SWITCH] Active mode changed to: {modes[sel_idx]}")

    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[Preview] Exited cleanly.")

    return 0


def run_gui_mode() -> int:
    """Launches the modern CustomTkinter Desktop GUI Application with fallback to CLI."""
    try:
        import customtkinter as ctk
        from gui.dashboard import MainDashboard

        logger.info("Starting AI-Based Vision-Driven HCI Desktop Application...")
        app = MainDashboard()
        app.mainloop()
        return 0
    except Exception as e:
        logger.warning(f"Could not initialize CustomTkinter GUI ({e}). Launching OpenCV HUD Preview...")
        return run_cli_mode(mock_mode=False)


def main():
    parser = argparse.ArgumentParser(
        description="AI-Based Vision-Driven Human-Computer Interaction Using Hand Gesture Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                           # Full modern GUI Desktop Dashboard
    python main.py --cli                     # Lightweight OpenCV HUD preview mode
    python main.py --cli --mock              # Synthetic mock simulation (no webcam needed)
    python main.py --cli --mode MEDIA        # Start directly in Media Control mode
    python main.py --cli --voice             # Run with Voice + Gesture Multimodal Fusion
    python main.py --eval                    # Run ML evaluation & latency profiling
    python main.py --test                    # Run 20-phase master verification suite
        """
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in lightweight OpenCV HUD preview mode instead of full desktop GUI"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the desktop CustomTkinter GUI dashboard"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock/simulation mode (synthetic camera frames, mock OS controllers)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=OperationalMode.MOUSE,
        choices=OperationalMode.ALL_MODES,
        help="Initial operational mode (default: MOUSE)"
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable multimodal voice recognition and speech commands"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run comprehensive benchmark evaluation (accuracy, F1, paper comparison, latency)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the complete 20-Phase Master Verification Test Suite"
    )

    args = parser.parse_args()

    # Route based on explicit intent
    if args.test:
        sys.exit(run_test_suite())
    elif args.eval:
        sys.exit(run_evaluation_suite(mock_mode=args.mock))
    elif args.cli:
        sys.exit(run_cli_mode(mode=args.mode, mock_mode=args.mock, enable_voice=args.voice))
    elif args.gui:
        sys.exit(run_gui_mode())
    else:
        # Default behavior: Attempt GUI mode, with graceful fallback to CLI
        sys.exit(run_gui_mode())


if __name__ == "__main__":
    main()
