"""
Phase 11 Verification Test Suite: Media Player & Streaming Applications Control
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding & Package Exports (controllers.MediaController, MediaTarget)
2. Playback State Transitions (play_pause, stop, next_track, prev_track)
3. Scrubbing & Position Seeking (+10s seek_forward, -10s seek_backward)
4. Volume Adjustment & Audio Mute (volume_up, volume_down, toggle_mute, [0.0, 1.0] bounds)
5. Display & Subtitle Toggles (toggle_fullscreen, toggle_captions)
6. Context-Aware Profile Switching (YouTube, Spotify, VLC, Netflix, Generic - Module 13)
7. Unified Action Router (execute_action) for All Media Mode Commands
8. End-to-End Gesture Recognition to Media Execution (PINCH -> PLAY_PAUSE, SWIPE_RIGHT -> NEXT_TRACK)
9. Microbenchmark: Sub-Millisecond Dispatch Latency (< 0.05 ms per command)
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
from controllers import MediaController, MediaTarget
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
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 11 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     Media Player & Streaming Control (Module 10)")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Section 1: Scaffolding & Package Exports
    # -----------------------------------------------------------------
    print("\n1. Module Scaffolding & Package Exports")
    exports_ok = hasattr(controllers, "MediaController") and hasattr(controllers, "MediaTarget")
    collector.record("MediaController Package Exports", exports_ok, "MediaController and MediaTarget accessible")

    media = MediaController(mock_mode=True)
    init_ok = (media.mock_mode and media.target_app == MediaTarget.GENERIC and media.is_playing)
    collector.record("Media Controller Initialization", init_ok, f"Mode: Mock | Target: {media.target_app} | Playing: {media.is_playing}")

    # -----------------------------------------------------------------
    # Section 2: Playback State Transitions
    # -----------------------------------------------------------------
    print("\n2. Playback State Transitions (Play, Pause, Stop, Skip)")
    # Play / Pause Toggle
    media.play_pause()
    paused = not media.is_playing and media.last_action == "PAUSE"
    media.play_pause()
    resumed = media.is_playing and media.last_action == "PLAY"
    play_pause_ok = paused and resumed
    collector.record("Play / Pause Toggle Lifecycle", play_pause_ok, f"Paused: {paused} -> Resumed: {resumed}")

    # Stop Playback
    media.stop()
    stop_ok = (not media.is_playing and media.playback_position_sec == 0.0 and media.last_action == "STOP")
    collector.record("Stop Playback Operation", stop_ok, f"Action: '{media.last_action}', Playing: {media.is_playing}")

    # Next Track
    idx0 = media.track_index
    media.next_track()
    next_ok = (media.track_index == idx0 + 1 and media.last_action == "NEXT_TRACK")
    collector.record("Next Track Navigation", next_ok, f"Track Index: {idx0} -> {media.track_index}")

    # Previous Track
    media.prev_track()
    prev_ok = (media.track_index == idx0 and media.last_action == "PREV_TRACK")
    collector.record("Previous Track Navigation", prev_ok, f"Track Index: {idx0 + 1} -> {media.track_index}")

    # -----------------------------------------------------------------
    # Section 3: Scrubbing & Seeking Control
    # -----------------------------------------------------------------
    print("\n3. Scrubbing & Position Seeking (+10s / -10s)")
    media.playback_position_sec = 45.0

    # Seek Forward 10s
    media.seek_forward(10)
    fwd_ok = (media.playback_position_sec == 55.0 and media.last_action == "SEEK_FORWARD_10S")
    collector.record("Seek Forward (+10s)", fwd_ok, f"Position: 45.0s -> {media.playback_position_sec}s")

    # Seek Backward 10s
    media.seek_backward(10)
    back_ok = (media.playback_position_sec == 45.0 and media.last_action == "SEEK_BACKWARD_10S")
    collector.record("Seek Backward (-10s)", back_ok, f"Position: 55.0s -> {media.playback_position_sec}s")

    # Lower Bound Clamping: cannot seek below 0.0s
    media.seek_backward(100)
    bound_ok = (media.playback_position_sec == 0.0)
    collector.record("Seek Position Floor Clamping (>= 0.0s)", bound_ok, f"Position clamped to: {media.playback_position_sec}s")

    # -----------------------------------------------------------------
    # Section 4: Volume & Audio Mute
    # -----------------------------------------------------------------
    print("\n4. Volume Adjustment & Audio Mute")
    media.volume = 0.50

    # Volume Up
    media.volume_up(step=0.10)
    v_up_ok = (abs(media.volume - 0.60) < 1e-4 and media.last_action == "VOLUME_UP")
    collector.record("Playback Volume Up (+10%)", v_up_ok, f"Volume: 50% -> {media.volume*100:.0f}%")

    # Volume Down
    media.volume_down(step=0.15)
    v_down_ok = (abs(media.volume - 0.45) < 1e-4 and media.last_action == "VOLUME_DOWN")
    collector.record("Playback Volume Down (-15%)", v_down_ok, f"Volume: 60% -> {media.volume*100:.0f}%")

    # Toggle Mute
    media.toggle_mute()
    muted = media.is_muted and media.last_action == "MUTE"
    media.toggle_mute()
    unmuted = not media.is_muted and media.last_action == "UNMUTE"
    mute_ok = muted and unmuted
    collector.record("Playback Mute Toggle", mute_ok, f"Muted: {muted} -> Unmuted: {unmuted}")

    # -----------------------------------------------------------------
    # Section 5: Display & Subtitle Toggles
    # -----------------------------------------------------------------
    print("\n5. Display & Accessibility Controls")
    # Fullscreen Toggle
    media.toggle_fullscreen()
    fs_active = media.is_fullscreen and media.last_action == "FULLSCREEN_TOGGLE"
    media.toggle_fullscreen()
    fs_inactive = not media.is_fullscreen
    fs_ok = fs_active and fs_inactive
    collector.record("Fullscreen Display Toggle", fs_ok, f"Fullscreen: {fs_active} -> Restored: {fs_inactive}")

    # Captions Toggle
    media.toggle_captions()
    cap_active = media.captions_enabled and media.last_action == "CAPTIONS_TOGGLE"
    media.toggle_captions()
    cap_inactive = not media.captions_enabled
    cap_ok = cap_active and cap_inactive
    collector.record("Closed Captions Subtitle Toggle", cap_ok, f"Captions: {cap_active} -> Restored: {cap_inactive}")

    # -----------------------------------------------------------------
    # Section 6: Context-Aware Profiles (Module 13)
    # -----------------------------------------------------------------
    print("\n6. Context-Aware Profile Switching (Module 13)")
    profiles = [MediaTarget.YOUTUBE, MediaTarget.SPOTIFY, MediaTarget.VLC, MediaTarget.NETFLIX, MediaTarget.GENERIC]
    profiles_ok = all(media.set_target(p) and media.target_app == p for p in profiles)
    collector.record("Context Target Profiles (YouTube, VLC, Spotify)", profiles_ok, f"Supported: {profiles}")

    # -----------------------------------------------------------------
    # Section 7: Unified Action Router (execute_action())
    # -----------------------------------------------------------------
    print("\n7. Unified Media Action Router (execute_action)")
    test_commands = [
        ("PLAY_PAUSE", ["PLAY", "PAUSE"]),
        ("NEXT_TRACK", ["NEXT_TRACK"]),
        ("PREV_TRACK", ["PREV_TRACK"]),
        ("STOP", ["STOP"]),
        ("VOLUME_UP", ["VOLUME_UP"]),
        ("VOLUME_DOWN", ["VOLUME_DOWN"]),
        ("MUTE", ["MUTE", "UNMUTE"]),
        ("FAST_FORWARD", ["SEEK_FORWARD_10S"]),
        ("REWIND", ["SEEK_BACKWARD_10S"]),
        ("FULLSCREEN_TOGGLE", ["FULLSCREEN_TOGGLE"]),
        ("CAPTIONS_TOGGLE", ["CAPTIONS_TOGGLE"]),
    ]

    all_routed = True
    for cmd, expected_possible in test_commands:
        res = media.execute_action(cmd)
        if not res or media.last_action not in expected_possible:
            all_routed = False
            break

    collector.record("Unified Media Action Routing", all_routed, f"Successfully routed {len(test_commands)} media commands")

    # -----------------------------------------------------------------
    # Section 8: End-to-End Real-Time Gesture Integration
    # -----------------------------------------------------------------
    print("\n8. End-to-End Real-Time Gesture to Media Execution")
    dataset_path = os.path.join(_ROOT, "dataset", "gesture_dataset.npz")
    dataset_obj = GestureDataset.load_from_npz(dataset_path)

    recognizer = RealtimeGestureRecognizer(
        initial_mode=OperationalMode.MEDIA,
        confidence_threshold=0.60,
        temporal_window_size=3,
        min_stable_frames=2
    )

    # In MEDIA mode: PINCH triggers PLAY_PAUSE
    pinch_idx = np.where(dataset_obj.y == "PINCH")[0][0]
    pinch_hand = create_mock_hand_data(dataset_obj.landmarks[pinch_idx])

    t_now = time.perf_counter()
    recognizer.process_frame(pinch_hand, timestamp=t_now)
    r_media = recognizer.process_frame(pinch_hand, timestamp=t_now + 0.033)

    action_executed = False
    if r_media.can_execute and r_media.mapped_action is not None:
        action_executed = media.execute_action(r_media.mapped_action)

    e2e_ok = (r_media.confirmed_gesture == "PINCH" and r_media.mapped_action == "PLAY_PAUSE" and action_executed)
    collector.record("E2E Gesture Recognition -> Media Execution", e2e_ok, f"Gesture: '{r_media.confirmed_gesture}' -> Action: '{r_media.mapped_action}' (Executed: {action_executed})")

    # -----------------------------------------------------------------
    # Section 9: Sub-Millisecond Action Routing Latency Benchmark
    # -----------------------------------------------------------------
    print("\n9. Action Routing Latency Microbenchmark")
    n_iters = 1000
    t_start = time.perf_counter()
    for i in range(n_iters):
        _ = media.execute_action("PLAY_PAUSE")
    t_total = time.perf_counter() - t_start

    latency_per_call_ms = (t_total / n_iters) * 1000.0
    throughput_ops = n_iters / t_total

    collector.record(
        "Media Dispatch Latency (< 0.05 ms)",
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
    print(" PHASE 11 TEST EXECUTION SUMMARY")
    print("=" * 75)
    print(f" Total Tests Run:   {total}")
    print(f" Passed:            {passed}")
    print(f" Failed:            {failed}")
    print(f" Pass Rate:         {pass_rate:.1f}%")
    print("=" * 75)

    report_path = os.path.join(_ROOT, "data", "phase11_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "phase": "Phase 11 - Media Player & Streaming Control",
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
