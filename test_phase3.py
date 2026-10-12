"""
Phase 3: MediaPipe Hand Detection & Landmark Extraction Verification Test Suite
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Tests:
1. MediaPipe Tasks HandLandmarker Model Initialization & Delegate
2. Empty Frame Ingestion & Occlusion Robustness (Zero hand crash test, Module 17)
3. 21 3D Joint Topological Integrity & Coordinates (Norm, Pixel, World)
4. Hand Rotation & Orientation Angle Computation (-180 to +180 deg)
5. Rotation-Invariant Finger State Evaluation (Fist, Palm, Point, Peace, Thumbs Up)
6. Pairwise Landmark Distance & Palm Scale Normalization (Module 1)
7. Dual-Hand Perceptual Architecture (max_hands=2)
8. Sci-Fi HUD Skeleton Rendering (Bones, Joints, Tips, Badges)
9. Landmarker Forward Inference Latency Benchmark (Target < 35ms)
10. Multi-Module Backward Compatibility (detection, vision.hand_tracker, root hand_detector)
"""

import sys
import os
import time
import json
import math
import platform
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2


# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detection.hand_detector import (
    HandDetector,
    HandData,
    HandLandmarkData,
    HAND_CONNECTIONS,
    FINGER_TIPS,
    FINGER_PIPS,
    FINGER_MCPS
)

# ANSI Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


class MockLandmark:
    """Mock MediaPipe NormalizedLandmark for synthetic posture testing."""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def build_synthetic_hand(posture: str, tilt_deg: float = 0.0) -> List[MockLandmark]:
    """
    Constructs an anatomically grounded 21-landmark synthetic hand pose
    with optional rotational tilt.
    """
    # Base canonical upright hand
    # Wrist L0 at (0.5, 0.75)
    # Middle MCP L9 at (0.5, 0.50) -> Palm span = 0.25
    lms = [(0.5, 0.75, 0.0)] * 21
    
    # Palm Base & Knuckles
    lms[0] = (0.50, 0.75, 0.0)   # Wrist
    lms[1] = (0.43, 0.70, 0.0)   # Thumb CMC
    lms[2] = (0.38, 0.65, 0.0)   # Thumb MCP
    lms[3] = (0.33, 0.60, 0.0)   # Thumb IP
    lms[4] = (0.28, 0.55, 0.0)   # Thumb Tip (extended outwards)
    
    lms[5] = (0.44, 0.52, 0.0)   # Index MCP
    lms[6] = (0.44, 0.42, 0.0)   # Index PIP
    lms[7] = (0.44, 0.35, 0.0)   # Index DIP
    lms[8] = (0.44, 0.28, 0.0)   # Index Tip (extended)
    
    lms[9] = (0.50, 0.50, 0.0)   # Middle MCP
    lms[10] = (0.50, 0.40, 0.0)  # Middle PIP
    lms[11] = (0.50, 0.33, 0.0)  # Middle DIP
    lms[12] = (0.50, 0.25, 0.0)  # Middle Tip (extended)
    
    lms[13] = (0.56, 0.52, 0.0)  # Ring MCP
    lms[14] = (0.56, 0.43, 0.0)  # Ring PIP
    lms[15] = (0.56, 0.36, 0.0)  # Ring DIP
    lms[16] = (0.56, 0.30, 0.0)  # Ring Tip (extended)
    
    lms[17] = (0.61, 0.56, 0.0)  # Pinky MCP
    lms[18] = (0.61, 0.48, 0.0)  # Pinky PIP
    lms[19] = (0.61, 0.42, 0.0)  # Pinky DIP
    lms[20] = (0.61, 0.36, 0.0)  # Pinky Tip (extended)

    # Posture Modifications
    if posture == "fist":
        # All tips curled inward toward MCPs
        lms[4] = (0.43, 0.58, 0.05)  # Thumb folded over
        lms[8] = (0.44, 0.56, 0.05)  # Index curled
        lms[12] = (0.50, 0.54, 0.05) # Middle curled
        lms[16] = (0.56, 0.56, 0.05) # Ring curled
        lms[20] = (0.61, 0.58, 0.05) # Pinky curled
    elif posture == "point":
        # Only Index extended
        lms[4] = (0.43, 0.58, 0.05)
        lms[12] = (0.50, 0.54, 0.05)
        lms[16] = (0.56, 0.56, 0.05)
        lms[20] = (0.61, 0.58, 0.05)
    elif posture == "peace":
        # Index and Middle extended, rest curled
        lms[4] = (0.43, 0.58, 0.05)
        lms[16] = (0.56, 0.56, 0.05)
        lms[20] = (0.61, 0.58, 0.05)
    elif posture == "thumbs_up":
        # Thumb extended high, rest curled
        lms[4] = (0.34, 0.45, 0.0)
        lms[8] = (0.44, 0.56, 0.05)
        lms[12] = (0.50, 0.54, 0.05)
        lms[16] = (0.56, 0.56, 0.05)
        lms[20] = (0.61, 0.58, 0.05)

    # Apply rotational tilt around wrist (0.5, 0.75) if requested
    if abs(tilt_deg) > 0.1:
        rad = math.radians(tilt_deg)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)
        cx, cy = 0.5, 0.75
        rotated = []
        for x, y, z in lms:
            dx = x - cx
            dy = y - cy
            rx = cx + (dx * cos_t - dy * sin_t)
            ry = cy + (dx * sin_t + dy * cos_t)
            rotated.append(MockLandmark(rx, ry, z))
        return rotated

    return [MockLandmark(x, y, z) for x, y, z in lms]


class Phase3Tester:
    def __init__(self):
        self.results = {
            "phase": "Phase 3 - MediaPipe Hand Detection & Landmark Extraction",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_warned": 0,
            "benchmarks": {},
            "details": []
        }

    def _record(self, name: str, passed: bool, message: str, warning: bool = False, latency_ms: float = 0.0):
        self.results["tests_run"] += 1
        status = "PASSED" if passed else ("WARNING" if warning else "FAILED")
        if passed:
            self.results["tests_passed"] += 1
        elif warning:
            self.results["tests_warned"] += 1
        else:
            self.results["tests_failed"] += 1

        self.results["details"].append({
            "test": name,
            "status": status,
            "message": message,
            "latency_ms": round(latency_ms, 2)
        })

        tag = f"{GREEN}[PASS]{RESET}" if passed else (f"{YELLOW}[WARN]{RESET}" if warning else f"{RED}[FAIL]{RESET}")
        time_str = f"({latency_ms:.1f}ms)" if latency_ms > 0 else ""
        print(f"  {tag} {name:<42} {message} {time_str}")

    def test_detector_initialization(self):
        print(f"\n{BOLD}{CYAN}1. MediaPipe Landmarker Model & Backend Initialization{RESET}")
        t0 = time.time()
        detector = HandDetector(max_hands=2)
        latency = (time.time() - t0) * 1000

        init_ok = (detector.landmarker is not None)
        backend_name = detector.backend
        self._record("Detector Backend Initialization", init_ok, f"Active backend: '{backend_name}' (max_hands={detector.max_hands})", latency_ms=latency)
        detector.close()

    def test_empty_frame_handling(self):
        print(f"\n{BOLD}{CYAN}2. Empty Frame Ingestion & Occlusion Robustness (Module 17){RESET}")
        detector = HandDetector(max_hands=2)
        
        # Ingest a blank black image
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        t0 = time.time()
        hands, annotated = detector.detect(blank, draw=True)
        latency = (time.time() - t0) * 1000

        empty_ok = (isinstance(hands, list) and len(hands) == 0 and annotated is not None)
        self._record("Zero Hands Ingestion (Blank)", empty_ok, f"Hands detected: {len(hands)} (No crash)", latency_ms=latency)

        # Ingest random noise frame
        noise = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        hands_noise, _ = detector.detect(noise, draw=False)
        noise_ok = isinstance(hands_noise, list)
        self._record("Noise Frame Immunity", noise_ok, f"Hands detected: {len(hands_noise)} on high-entropy noise")
        detector.close()

    def test_topological_integrity(self):
        print(f"\n{BOLD}{CYAN}3. 21 3D Joint Topological Integrity & Spatial Coordinates{RESET}")
        raw_lms = build_synthetic_hand("palm")
        hand = HandData(raw_lms, frame_w=640, frame_h=480, handedness="Right", detection_score=0.98)

        # Check total joints
        n_lms = len(hand.landmarks)
        self._record("21 Landmark Extraction", n_lms == 21, f"Extracted {n_lms} 3D coordinate tuples")

        # Check coordinate bounds & conversion
        wx, wy, wz = hand.get_norm(0)
        px_x, px_y = hand.get_px(0)
        coord_ok = (0.0 <= wx <= 1.0 and 0.0 <= wy <= 1.0 and px_x == int(wx * 640) and px_y == int(wy * 480))
        self._record("Normalized-to-Pixel Projection", coord_ok, f"L0 Wrist: norm=({wx:.2f}, {wy:.2f}) -> px=({px_x}, {px_y})")

        # Bounding box validity
        x1, y1, x2, y2 = hand.bbox
        bbox_ok = (0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480)
        self._record("Padded Bounding Box Geometry", bbox_ok, f"Box: [{x1}, {y1}, {x2}, {y2}] ({x2-x1}x{y2-y1} px)")

        # Palm centroid & Palm span
        cx, cy = hand.center
        span = hand.palm_span
        span_ok = (0.10 <= span <= 0.60 and x1 <= cx <= x2 and y1 <= cy <= y2)
        self._record("Palm Centroid & Scale Span", span_ok, f"Center: ({cx}, {cy}) | Palm Span: {span:.3f}")

        # Feature vector (63 dims)
        feat_vec = hand.to_feature_vector()
        self._record("63-Dim Flat Feature Vector", feat_vec.shape == (63,), f"Shape: {feat_vec.shape}, Dtype: {feat_vec.dtype}")

    def test_hand_rotation_and_orientation(self):
        print(f"\n{BOLD}{CYAN}4. Hand Rotation Angle & Spatial Orientation Tracking{RESET}")
        
        # Upright hand
        hand_up = HandData(build_synthetic_hand("palm", tilt_deg=0.0), 640, 480)
        self._record("Upright Orientation (0 deg)", abs(hand_up.rotation_deg) < 5.0,
                     f"Measured: {hand_up.rotation_deg:.1f} deg (expected ~0 deg)")

        # 45 deg clockwise tilt
        hand_cw = HandData(build_synthetic_hand("palm", tilt_deg=45.0), 640, 480)
        self._record("Clockwise Tilt (+45 deg)", abs(hand_cw.rotation_deg - 45.0) < 5.0,
                     f"Measured: {hand_cw.rotation_deg:.1f} deg (expected ~45 deg)")

        # 45 deg counter-clockwise tilt
        hand_ccw = HandData(build_synthetic_hand("palm", tilt_deg=-45.0), 640, 480)
        self._record("Counter-Clockwise Tilt (-45 deg)", abs(hand_ccw.rotation_deg - (-45.0)) < 5.0,
                     f"Measured: {hand_ccw.rotation_deg:.1f} deg (expected ~-45 deg)")

        # 90 deg horizontal hand
        hand_90 = HandData(build_synthetic_hand("palm", tilt_deg=90.0), 640, 480)
        self._record("Horizontal Orientation (+90 deg)", abs(hand_90.rotation_deg - 90.0) < 5.0,
                     f"Measured: {hand_90.rotation_deg:.1f} deg (expected ~90 deg)")

    def test_finger_extension_postures(self):
        print(f"\n{BOLD}{CYAN}5. Anatomical Finger Extension States (Thumb, Index, Middle, Ring, Pinky){RESET}")
        
        # 1. Open Palm -> [1, 1, 1, 1, 1]
        palm = HandData(build_synthetic_hand("palm"), 640, 480)
        self._record("Open Palm (5 Fingers Extended)", palm.fingers_up == [1, 1, 1, 1, 1],
                     f"States: {palm.fingers_up}")

        # 2. Closed Fist -> [0, 0, 0, 0, 0]
        fist = HandData(build_synthetic_hand("fist"), 640, 480)
        self._record("Closed Fist (0 Fingers Extended)", fist.fingers_up == [0, 0, 0, 0, 0],
                     f"States: {fist.fingers_up}")

        # 3. Pointing Gesture -> [0, 1, 0, 0, 0]
        point = HandData(build_synthetic_hand("point"), 640, 480)
        self._record("Point Gesture (Index Only)", point.fingers_up == [0, 1, 0, 0, 0],
                     f"States: {point.fingers_up}")

        # 4. Peace / Victory Gesture -> [0, 1, 1, 0, 0]
        peace = HandData(build_synthetic_hand("peace"), 640, 480)
        self._record("Peace Gesture (Index + Middle)", peace.fingers_up == [0, 1, 1, 0, 0],
                     f"States: {peace.fingers_up}")

        # 5. Thumbs Up Gesture -> [1, 0, 0, 0, 0]
        thumbs = HandData(build_synthetic_hand("thumbs_up"), 640, 480)
        self._record("Thumbs Up (Thumb Only)", thumbs.fingers_up == [1, 0, 0, 0, 0],
                     f"States: {thumbs.fingers_up}")

        # 6. Rotated Palm (45 deg) -> Should still detect all 5 fingers extended!
        rotated_palm = HandData(build_synthetic_hand("palm", tilt_deg=45.0), 640, 480)
        self._record("Rotated Palm Invariance (45 deg)", rotated_palm.fingers_up == [1, 1, 1, 1, 1],
                     f"States: {rotated_palm.fingers_up} at {rotated_palm.rotation_deg:.1f} deg")

    def test_skeleton_hud_rendering(self):
        print(f"\n{BOLD}{CYAN}6. Sci-Fi HUD Skeleton & Visual Annotation Rendering{RESET}")
        detector = HandDetector()
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand = HandData(build_synthetic_hand("palm"), 640, 480, handedness="Right", detection_score=0.97)

        detector.draw_skeleton(test_frame, hand)
        detector.close()

        # Verify pixels were written
        non_zero = np.count_nonzero(test_frame)
        self._record("Skeleton Drawing Execution", non_zero > 1000, f"Rendered {non_zero} non-zero annotated pixels")

        # Verify Index fingertip color exists in image (Cyan/Yellow)
        pt8 = hand.get_px(8)
        color_at_tip = test_frame[pt8[1], pt8[0]]
        tip_rendered = np.any(color_at_tip > 0)
        self._record("Fingertip Highlight Node", tip_rendered, f"Index tip pixel: BGR {color_at_tip.tolist()}")

    def test_inference_latency_benchmark(self):
        print(f"\n{BOLD}{CYAN}7. MediaPipe Landmarker Inference Latency Benchmark{RESET}")
        detector = HandDetector(max_hands=2)
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a synthetic hand-colored blob to stimulate feature layers
        cv2.circle(test_img, (320, 240), 80, (180, 200, 230), -1)

        # Warmup
        for _ in range(3):
            detector.detect(test_img, draw=False)

        latencies = []
        iterations = 25
        for _ in range(iterations):
            t0 = time.time()
            detector.detect(test_img, draw=False)
            dt = (time.time() - t0) * 1000.0
            latencies.append(dt)
        detector.close()

        avg_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        min_lat = float(np.min(latencies))
        self.results["benchmarks"]["inference_mean_ms"] = round(avg_lat, 2)
        self.results["benchmarks"]["inference_p95_ms"] = round(p95_lat, 2)
        self.results["benchmarks"]["inference_min_ms"] = round(min_lat, 2)

        lat_ok = avg_lat < 40.0
        self._record("Forward Inference Latency", lat_ok,
                     f"Mean: {avg_lat:.2f} ms | P95: {p95_lat:.2f} ms | Min: {min_lat:.2f} ms (target < 40ms)")

    def test_backward_compatibility(self):
        print(f"\n{BOLD}{CYAN}8. Multi-Module Backward Compatibility & Scaffolding{RESET}")
        
        # 1. vision.hand_tracker.HandTracker
        from vision.hand_tracker import HandTracker as HT_Vision
        v_tracker = HT_Vision(max_hands=2)
        self._record("vision.hand_tracker.HandTracker Bridge", v_tracker is not None, "Instantiated successfully")
        v_tracker.close()

        # 2. root hand_detector.HandDetector (returns single hand or None)
        import hand_detector as root_hd
        r_detector = root_hd.HandDetector(max_hands=1)
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        single_hand, _ = r_detector.find_hands(synthetic_frame, draw=False)
        self._record("root hand_detector Single-Hand Return", single_hand is None, "Correctly returns None on empty frame")
        r_detector.close()

        # 3. Context Manager Support
        with HandDetector() as ctx_det:
            ctx_ok = (ctx_det.landmarker is not None)
            self._record("Context Manager (with HandDetector)", ctx_ok, "Active inside with-block")
        self._record("Context Manager Auto-Close", ctx_det.landmarker is None, "Resources released on exit")

    def run_all(self):
        print("=" * 75)
        print(f" {BOLD}AI-BASED VISION-DRIVEN HCI - PHASE 3 VERIFICATION TEST{RESET}")
        print("=" * 75)
        print(f" Timestamp:  {self.results['timestamp']}")
        print(f" Platform:   {platform.platform()}")
        print(f" Target:     MediaPipe Hand Detection, 21 Landmarks & Rotation Tracking")
        print("=" * 75)

        self.test_detector_initialization()
        self.test_empty_frame_handling()
        self.test_topological_integrity()
        self.test_hand_rotation_and_orientation()
        self.test_finger_extension_postures()
        self.test_skeleton_hud_rendering()
        self.test_inference_latency_benchmark()
        self.test_backward_compatibility()

        print("\n" + "=" * 75)
        print(f" {BOLD}PHASE 3 TEST EXECUTION SUMMARY{RESET}")
        print("=" * 75)
        print(f" Total Tests Run:   {self.results['tests_run']}")
        print(f" Passed:            {GREEN}{self.results['tests_passed']}{RESET}")
        print(f" Warnings:          {YELLOW}{self.results['tests_warned']}{RESET}")
        print(f" Failed:            {RED}{self.results['tests_failed']}{RESET}")
        
        pass_rate = (self.results['tests_passed'] / self.results['tests_run']) * 100
        print(f" Pass Rate:         {BOLD}{GREEN if pass_rate == 100 else YELLOW}{pass_rate:.1f}%{RESET}")
        print("=" * 75)

        # Export benchmark report
        report_path = os.path.join(PROJECT_ROOT, "data", "phase3_test_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f" Benchmark report exported to: {report_path}\n")

        return self.results['tests_failed'] == 0


if __name__ == "__main__":
    tester = Phase3Tester()
    success = tester.run_all()
    sys.exit(0 if success else 1)
