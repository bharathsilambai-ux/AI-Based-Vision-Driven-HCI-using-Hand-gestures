"""
Phase 4 Verification Test Suite: Landmark Normalization, Kinematics & High-Dimensional ML Features
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding, Package Exports & Backward-Compatibility Bridging
2. Translation Invariance (Wrist L0 = [0, 0, 0] across arbitrary translations)
3. Scale Invariance (Wrist-to-Middle-MCP ||L9 - L0|| = 1.0000 across zoom/distances)
4. Rotation Alignment & Invariant Coordinate Representation
5. 3D Joint Angles (Theta in [0, pi], collinear, right angles, zero-magnitude guard)
6. Canonical 14 Pairwise Euclidean Distances (Finger curl, pinch, span)
7. 3D Palm Normal, Roll, Pitch, Yaw & Facing-Camera Detection
8. Kinematic Velocity, Speed, Direction & Acceleration Tracking (EMA smoothing)
9. Dual-Hand Spatial Interaction (Inter-wrist distance, relative angle, zoom delta, symmetry)
10. High-Dimensional Feature Vectorization (92-D Canonical, 92-D Rotation-Invariant, 128-D Extended)
11. Input Polymorphism (HandData, HandDetectionResult, NumPy (21,3), tuple lists)
12. Microbenchmark: Processing Latency (< 0.5 ms per hand) & Throughput (> 2,000 FPS)
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

from detection.hand_detector import HandData as DetectionHandData
from features.landmark_processor import (
    CANONICAL_ANGLE_TRIPLETS,
    CANONICAL_DISTANCE_PAIRS,
    EXTENDED_ANGLE_TRIPLETS,
    DualHandFeatures,
    FeatureExtractor,
    HandFeatures,
    LandmarkProcessor,
    compute_angle_3d,
)
from vision.hand_tracker import HandData as VisionHandData


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


def make_synthetic_hand_coords(
    wrist_xy: Tuple[float, float] = (0.5, 0.7),
    scale: float = 1.0,
    rotation_rad: float = 0.0
) -> List[Tuple[float, float, float]]:
    """Generates synthetic 21 3D landmarks with controlled translation, scale, and planar rotation."""
    base_offsets = [
        (0.00, 0.00, 0.00),    # 0: Wrist
        # Thumb (1-4)
        (-0.04, -0.05, -0.01),
        (-0.07, -0.10, -0.02),
        (-0.09, -0.15, -0.03),
        (-0.10, -0.20, -0.04),
        # Index (5-8)
        (-0.02, -0.18, -0.01),
        (-0.02, -0.26, -0.02),
        (-0.02, -0.32, -0.03),
        (-0.02, -0.40, -0.04),
        # Middle (9-12)
        (0.01, -0.19, -0.01),
        (0.01, -0.27, -0.02),
        (0.01, -0.33, -0.03),
        (0.01, -0.41, -0.04),
        # Ring (13-16)
        (0.04, -0.17, -0.01),
        (0.04, -0.24, -0.02),
        (0.04, -0.29, -0.03),
        (0.04, -0.36, -0.04),
        # Pinky (17-20)
        (0.07, -0.14, -0.01),
        (0.07, -0.20, -0.02),
        (0.07, -0.24, -0.03),
        (0.07, -0.28, -0.04)
    ]

    cos_r = math.cos(rotation_rad)
    sin_r = math.sin(rotation_rad)
    transformed: List[Tuple[float, float, float]] = []

    for dx, dy, dz in base_offsets:
        # Scale
        sx = dx * scale
        sy = dy * scale
        sz = dz * scale
        # Rotate
        rx = sx * cos_r - sy * sin_r
        ry = sx * sin_r + sy * cos_r
        rz = sz
        # Translate
        tx = wrist_xy[0] + rx
        ty = wrist_xy[1] + ry
        tz = rz
        transformed.append((tx, ty, tz))

    return transformed


def create_synthetic_hand_object(
    coords: List[Tuple[float, float, float]],
    handedness: str = "Right",
    cls=VisionHandData,
    w: int = 640,
    h: int = 480
):
    class FakeLM:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    raw_lms = [FakeLM(c[0], c[1], c[2]) for c in coords]
    return cls(raw_lms, w, h, handedness=handedness)


def run_tests():
    print("=" * 75)
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 4 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     Landmark Normalization, Kinematics & High-Dim Features")
    print("=" * 75)

    processor = LandmarkProcessor()

    # -----------------------------------------------------------------
    # Section 1: Scaffolding, Package Exports & Aliases
    # -----------------------------------------------------------------
    print("\n1. Scaffolding, Package Exports & Aliases")
    t0 = time.perf_counter()
    import features
    import vision.feature_extractor as vfe

    p_init = hasattr(features, "LandmarkProcessor") and hasattr(features, "FeatureExtractor")
    collector.record("Features Package Exports", p_init, "LandmarkProcessor, FeatureExtractor accessible")

    bridge_ok = (
        vfe.LandmarkProcessor is LandmarkProcessor and
        vfe.FeatureExtractor is FeatureExtractor and
        vfe.HandFeatures is HandFeatures and
        vfe.DualHandFeatures is DualHandFeatures and
        vfe.compute_angle_3d is compute_angle_3d
    )
    collector.record("Vision FeatureExtractor Bridge", bridge_ok, "100% identical symbols re-exported")

    constants_ok = len(CANONICAL_DISTANCE_PAIRS) == 14 and len(CANONICAL_ANGLE_TRIPLETS) == 8
    collector.record("Canonical Constants (14 Dist, 8 Angle)", constants_ok, "14 pairs, 8 triplets defined")

    # -----------------------------------------------------------------
    # Section 2: Translation Invariance Verification
    # -----------------------------------------------------------------
    print("\n2. Translation Invariance (Wrist L0 Centering)")
    t0 = time.perf_counter()
    coords_center = make_synthetic_hand_coords(wrist_xy=(0.50, 0.50))
    coords_shifted = make_synthetic_hand_coords(wrist_xy=(0.85, 0.20))

    hand_center = create_synthetic_hand_object(coords_center)
    hand_shifted = create_synthetic_hand_object(coords_shifted)

    feat_center = processor.extract_single_hand(hand_center)
    feat_shifted = processor.extract_single_hand(hand_shifted)

    wrist_zero_center = np.allclose(feat_center.normalized_lms[0], [0.0, 0.0, 0.0], atol=1e-6)
    wrist_zero_shifted = np.allclose(feat_shifted.normalized_lms[0], [0.0, 0.0, 0.0], atol=1e-6)
    collector.record("Wrist Centered to Origin (0,0,0)", wrist_zero_center and wrist_zero_shifted, "L0 = [0, 0, 0] at both positions")

    # The normalized relative coordinates should be identical regardless of frame translation
    trans_diff = np.max(np.abs(feat_center.normalized_lms - feat_shifted.normalized_lms))
    collector.record("Translation Invariance Across Frame", trans_diff < 1e-4, f"Max coordinate delta: {trans_diff:.6f}")

    # -----------------------------------------------------------------
    # Section 3: Scale Invariance Verification
    # -----------------------------------------------------------------
    print("\n3. Scale Invariance (Palm Span L0-L9 Normalization)")
    coords_scale1 = make_synthetic_hand_coords(scale=1.0)
    coords_scale2 = make_synthetic_hand_coords(scale=2.2)   # Hand closer to camera
    coords_scale3 = make_synthetic_hand_coords(scale=0.45)  # Hand farther from camera

    feat_scale1 = processor.extract_single_hand(create_synthetic_hand_object(coords_scale1))
    feat_scale2 = processor.extract_single_hand(create_synthetic_hand_object(coords_scale2))
    feat_scale3 = processor.extract_single_hand(create_synthetic_hand_object(coords_scale3))

    d09_1 = np.linalg.norm(feat_scale1.normalized_lms[9])
    d09_2 = np.linalg.norm(feat_scale2.normalized_lms[9])
    d09_3 = np.linalg.norm(feat_scale3.normalized_lms[9])

    all_unit_span = (
        abs(d09_1 - 1.0) < 1e-4 and
        abs(d09_2 - 1.0) < 1e-4 and
        abs(d09_3 - 1.0) < 1e-4
    )
    collector.record("Unit Palm Span (||L9 - L0|| = 1.0000)", all_unit_span, f"Spans: [{d09_1:.4f}, {d09_2:.4f}, {d09_3:.4f}]")

    # Pairwise distances should match across all scales
    scale_dist_diff = np.max(np.abs(
        np.array(feat_scale1.pairwise_distances) - np.array(feat_scale2.pairwise_distances)
    ))
    collector.record("Scale Invariant Pairwise Distances", scale_dist_diff < 1e-4, f"Max distance delta: {scale_dist_diff:.6f}")

    # -----------------------------------------------------------------
    # Section 4: Rotation Alignment & Orientation Invariance
    # -----------------------------------------------------------------
    print("\n4. Rotation Alignment & Invariant Coordinates")
    coords_rot0 = make_synthetic_hand_coords(rotation_rad=0.0)
    coords_rot45 = make_synthetic_hand_coords(rotation_rad=math.radians(45.0))
    coords_rot90 = make_synthetic_hand_coords(rotation_rad=math.radians(90.0))

    feat_rot0 = processor.extract_single_hand(create_synthetic_hand_object(coords_rot0))
    feat_rot45 = processor.extract_single_hand(create_synthetic_hand_object(coords_rot45))
    feat_rot90 = processor.extract_single_hand(create_synthetic_hand_object(coords_rot90))

    # In rotation-aligned landmarks, middle MCP (index 9) must have x ~ 0 and y < 0 (pointing straight up)
    y_aligned_0 = feat_rot0.rotation_aligned_lms[9]
    y_aligned_45 = feat_rot45.rotation_aligned_lms[9]
    y_aligned_90 = feat_rot90.rotation_aligned_lms[9]

    aligned_upright = (
        abs(y_aligned_0[0]) < 1e-4 and y_aligned_0[1] < -0.99 and
        abs(y_aligned_45[0]) < 1e-4 and y_aligned_45[1] < -0.99 and
        abs(y_aligned_90[0]) < 1e-4 and y_aligned_90[1] < -0.99
    )
    collector.record("Palm Vector Vertical Alignment", aligned_upright, f"L9 x-coords: [{y_aligned_0[0]:.4f}, {y_aligned_45[0]:.4f}, {y_aligned_90[0]:.4f}]")

    rot_vec0 = feat_rot0.to_rotation_invariant_vector()
    rot_vec45 = feat_rot45.to_rotation_invariant_vector()
    rot_diff = np.max(np.abs(rot_vec0[:63] - rot_vec45[:63]))
    collector.record("Rotation-Invariant Feature Vector", rot_diff < 1e-3, f"Max diff between 0 and 45 deg: {rot_diff:.6f}")

    # -----------------------------------------------------------------
    # Section 5: 3D Joint Angle Calculations
    # -----------------------------------------------------------------
    print("\n5. 3D Joint Angles (theta in [0, pi])")
    # Collinear opposite rays -> 180 degrees (pi radians)
    p_a = (0.0, 1.0, 0.0)
    p_b = (0.0, 0.0, 0.0)
    p_c = (0.0, -1.0, 0.0)
    angle_straight = compute_angle_3d(p_a, p_b, p_c)
    collector.record("Straight Ray Angle (180 deg / pi rad)", abs(angle_straight - math.pi) < 1e-5, f"Angle: {angle_straight:.4f} rad ({math.degrees(angle_straight):.1f} deg)")

    # Right angle -> 90 degrees (pi/2 radians)
    p_d = (1.0, 0.0, 0.0)
    angle_right = compute_angle_3d(p_a, p_b, p_d)
    collector.record("Right Ray Angle (90 deg / pi/2 rad)", abs(angle_right - math.pi / 2.0) < 1e-5, f"Angle: {angle_right:.4f} rad ({math.degrees(angle_right):.1f} deg)")

    # Collinear identical rays -> 0 degrees
    angle_zero = compute_angle_3d(p_a, p_b, p_a)
    collector.record("Coincident Ray Angle (0 deg)", abs(angle_zero) < 1e-5, f"Angle: {angle_zero:.4f} rad")

    # Degenerate zero vector guard
    angle_degen = compute_angle_3d(p_b, p_b, p_a)
    collector.record("Zero-Magnitude Vector Guard", angle_degen == 0.0, "Safely returns 0.0 without ZeroDivisionError")

    # All canonical joint angles in [0, pi]
    angles = feat_scale1.joint_angles
    all_in_bounds = len(angles) == 8 and all(0.0 <= a <= math.pi for a in angles)
    collector.record("Canonical 8 Joint Angles Bounded [0, pi]", all_in_bounds, f"Values: {[round(a, 2) for a in angles[:4]]}...")

    # Extended 15 joint angles
    ext_angles = feat_scale1.extended_angles
    ext_ok = len(ext_angles) == 15 and all(0.0 <= a <= math.pi for a in ext_angles.values())
    collector.record("Extended 15 Finger Joint Angles", ext_ok, f"Total joints: {len(ext_angles)}")

    # -----------------------------------------------------------------
    # Section 6: Canonical Pairwise Euclidean Distances
    # -----------------------------------------------------------------
    print("\n6. Canonical Pairwise Euclidean Distances")
    dists = feat_scale1.pairwise_distances
    dist_valid = len(dists) == 14 and all(d >= 0.0 for d in dists)
    collector.record("14 Canonical Distances Non-Negative", dist_valid, f"Count: {len(dists)} | Min: {min(dists):.3f} | Max: {max(dists):.3f}")

    # Pinch Index Distance Check
    coords_pinch = make_synthetic_hand_coords()
    coords_pinch[4] = coords_pinch[8]  # Thumb tip touches index tip
    feat_pinch = processor.extract_single_hand(create_synthetic_hand_object(coords_pinch))
    d_thumb_idx_pinch = feat_pinch.pairwise_distances[0]  # Pair (4, 8)
    collector.record("Pinch Index Touch Detection", d_thumb_idx_pinch < 1e-4, f"Distance: {d_thumb_idx_pinch:.6f}")

    # -----------------------------------------------------------------
    # Section 7: 3D Palm Normal, Roll, Pitch, Yaw & Facing Camera
    # -----------------------------------------------------------------
    print("\n7. Palm 3D Normal & Orientation Euler Angles")
    norm_len = np.linalg.norm(feat_scale1.palm_normal)
    collector.record("Unit Palm Normal Vector (||n|| = 1.0)", abs(norm_len - 1.0) < 1e-4, f"Normal: {[round(float(v), 3) for v in feat_scale1.palm_normal]}")

    # Facing camera: in default synthetic hand, palm faces camera
    facing_cam = feat_scale1.palm_facing_camera
    collector.record("Palm Facing Camera Boolean", isinstance(facing_cam, bool), f"Result: {facing_cam}")

    roll_deg = feat_scale1.palm_roll_deg
    pitch_deg = feat_scale1.palm_pitch_deg
    yaw_deg = feat_scale1.palm_yaw_deg
    collector.record("Euler Angles (Roll, Pitch, Yaw)", -180.0 <= roll_deg <= 180.0, f"Roll: {roll_deg:.1f} deg | Pitch: {pitch_deg:.1f} deg | Yaw: {yaw_deg:.1f} deg")

    # -----------------------------------------------------------------
    # Section 8: Kinematic Velocity, Speed & Acceleration Tracking
    # -----------------------------------------------------------------
    print("\n8. Kinematic Velocity & Acceleration Tracking")
    processor.reset()
    hand_f1 = create_synthetic_hand_object(coords_center)
    f1 = processor.extract_single_hand(hand_f1)
    collector.record("Initial Frame Velocity (Zero)", f1.speed == 0.0, f"Speed: {f1.speed:.1f} px/s")

    # Second frame with 50 px movement
    coords_moved = make_synthetic_hand_coords(wrist_xy=(0.55, 0.50))
    hand_f2 = create_synthetic_hand_object(coords_moved)
    time.sleep(0.02)  # small delta t
    f2 = processor.extract_single_hand(hand_f2)

    has_vel = f2.speed > 0.0 and -180.0 <= f2.direction_deg <= 180.0
    collector.record("Kinematic Linear Velocity & Speed", has_vel, f"Speed: {f2.speed:.1f} px/s | Dir: {f2.direction_deg:.1f} deg")

    # Acceleration on third frame
    time.sleep(0.02)
    f3 = processor.extract_single_hand(create_synthetic_hand_object(coords_center))
    collector.record("Kinematic Acceleration", f3.acceleration >= 0.0, f"Acceleration: {f3.acceleration:.1f} px/s^2")

    processor.reset()
    f_reset = processor.extract_single_hand(hand_f1)
    collector.record("Velocity History Reset", f_reset.speed == 0.0, "History cleanly cleared")

    # -----------------------------------------------------------------
    # Section 9: Dual-Hand Spatial Interaction Features
    # -----------------------------------------------------------------
    print("\n9. Dual-Hand Spatial Interaction & Zoom Deltas")
    processor.reset()
    left_coords = make_synthetic_hand_coords(wrist_xy=(0.25, 0.60))
    right_coords = make_synthetic_hand_coords(wrist_xy=(0.75, 0.60))

    h_left = create_synthetic_hand_object(left_coords, handedness="Left")
    h_right = create_synthetic_hand_object(right_coords, handedness="Right")

    dual_1 = processor.extract_dual_hands([h_left, h_right])
    dual_ok = dual_1 is not None and dual_1.hand_distance_norm > 0.40
    collector.record("Dual Hand Extraction (Wrist Dist)", dual_ok, f"Distance: {dual_1.hand_distance_norm:.4f} | Center: {dual_1.center_distance_px:.1f} px")

    collector.record("Dual Hand Screen Plane Angle", abs(dual_1.angle_between_hands_deg) < 15.0, f"Angle: {dual_1.angle_between_hands_deg:.1f} deg")

    # Expand hands apart (Zoom In / Pinch Expand)
    left_expanded = make_synthetic_hand_coords(wrist_xy=(0.15, 0.60))
    right_expanded = make_synthetic_hand_coords(wrist_xy=(0.85, 0.60))
    h_left_exp = create_synthetic_hand_object(left_expanded, handedness="Left")
    h_right_exp = create_synthetic_hand_object(right_expanded, handedness="Right")

    dual_2 = processor.extract_dual_hands([h_left_exp, h_right_exp])
    is_expanding = dual_2.delta_distance > 0.0
    collector.record("Temporal Delta Distance (Expansion)", is_expanding, f"Delta dist: +{dual_2.delta_distance:.4f} (Zoom: {dual_2.zoom_factor:.2f}x)")

    # Symmetry score
    collector.record("Two-Hand Symmetry Score", 0.0 <= dual_2.symmetry_score <= 1.0, f"Symmetry: {dual_2.symmetry_score:.2f}")

    # -----------------------------------------------------------------
    # Section 10: High-Dimensional Feature Vectorization
    # -----------------------------------------------------------------
    print("\n10. High-Dimensional Feature Vectorization")
    ml_vec = feat_scale1.to_ml_vector()
    collector.record("Canonical 92-D ML Feature Vector", len(ml_vec) == 92 and ml_vec.dtype == np.float32, f"Shape: {ml_vec.shape}, Dtype: {ml_vec.dtype}")

    collector.record("No NaN / Inf in 92-D Vector", not np.isnan(ml_vec).any() and not np.isinf(ml_vec).any(), "Clean floating point vector")

    rot_ml_vec = feat_scale1.to_rotation_invariant_vector()
    collector.record("Rotation-Invariant 92-D ML Vector", len(rot_ml_vec) == 92 and not np.isnan(rot_ml_vec).any(), f"Shape: {rot_ml_vec.shape}")

    ext_ml_vec = feat_scale1.to_extended_ml_vector()
    collector.record("Extended 128-D Multi-Modal Vector", len(ext_ml_vec) == 128 and not np.isnan(ext_ml_vec).any(), f"Shape: {ext_ml_vec.shape}")

    dual_vec = dual_2.to_vector()
    collector.record("Dual-Hand Interaction Vector (10-D)", len(dual_vec) == 10 and not np.isnan(dual_vec).any(), f"Shape: {dual_vec.shape}")

    # -----------------------------------------------------------------
    # Section 11: Multi-Type Input Polymorphism
    # -----------------------------------------------------------------
    print("\n11. Multi-Type Input Polymorphism")
    # Vision HandData
    f_vis = processor.extract_single_hand(hand_center)
    collector.record("vision.hand_tracker.HandData Support", f_vis.normalized_lms.shape == (21, 3), "Processed successfully")

    # Detection HandData
    det_hand = create_synthetic_hand_object(coords_center, cls=DetectionHandData)
    f_det = processor.extract_single_hand(det_hand)
    collector.record("detection.hand_detector.HandData Support", f_det.normalized_lms.shape == (21, 3), "Processed successfully")

    # Raw NumPy array (21, 3)
    raw_np = np.array(coords_center, dtype=np.float32)
    f_np = processor.extract_single_hand(raw_np)
    collector.record("Raw NumPy Array (21, 3) Support", f_np.normalized_lms.shape == (21, 3), "Processed successfully")

    # Tuple list
    f_list = processor.extract_single_hand(coords_center)
    collector.record("Raw Coordinate Tuple List Support", f_list.normalized_lms.shape == (21, 3), "Processed successfully")

    # -----------------------------------------------------------------
    # Section 12: Microbenchmark: Latency & Throughput
    # -----------------------------------------------------------------
    print("\n12. Microbenchmark: Extraction Latency & Throughput")
    n_iters = 1000
    t_start = time.perf_counter()
    for _ in range(n_iters):
        _ = processor.extract_single_hand(hand_center)
    t_total = time.perf_counter() - t_start

    latency_per_call_ms = (t_total / n_iters) * 1000.0
    throughput_fps = n_iters / t_total

    collector.record(
        "Single-Hand Processing Latency (< 0.5 ms)",
        latency_per_call_ms < 0.50,
        f"Mean: {latency_per_call_ms:.3f} ms | Throughput: {throughput_fps:,.0f} hands/sec"
    )

    t_start_dual = time.perf_counter()
    for _ in range(n_iters):
        _ = processor.extract_dual_hands([h_left, h_right])
    t_total_dual = time.perf_counter() - t_start_dual

    dual_latency_ms = (t_total_dual / n_iters) * 1000.0
    dual_fps = n_iters / t_total_dual
    collector.record(
        "Dual-Hand Processing Latency (< 0.2 ms)",
        dual_latency_ms < 0.20,
        f"Mean: {dual_latency_ms:.3f} ms | Throughput: {dual_fps:,.0f} pairs/sec"
    )

    # -----------------------------------------------------------------
    # Summary & Export
    # -----------------------------------------------------------------
    total = len(collector.results)
    passed = sum(1 for r in collector.results if r["status"] == "PASS")
    failed = total - passed
    pass_rate = (passed / total) * 100.0 if total > 0 else 0.0

    print("\n" + "=" * 75)
    print(" PHASE 4 TEST EXECUTION SUMMARY")
    print("=" * 75)
    print(f" Total Tests Run:   {total}")
    print(f" Passed:            {passed}")
    print(f" Failed:            {failed}")
    print(f" Pass Rate:         {pass_rate:.1f}%")
    print("=" * 75)

    report_path = os.path.join(_ROOT, "data", "phase4_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "phase": "Phase 4 - Landmark Processing, Normalization & High-Dimensional ML Features",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.platform(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": pass_rate,
            "mean_latency_ms": latency_per_call_ms,
            "throughput_fps": throughput_fps,
            "results": collector.results
        }, f, indent=2)

    print(f" Benchmark report exported to: {report_path}\n")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
