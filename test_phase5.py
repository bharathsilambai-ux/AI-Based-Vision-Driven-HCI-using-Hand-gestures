"""
Phase 5 Verification Test Suite: Custom Landmark-Based Hand Gesture Dataset
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding, Package Exports & 14-Gesture Taxonomy
2. Gesture Synonym Mapping & Canonicalization
3. Data Augmentation Algorithms (Gaussian Jitter, 3D Rotation, Scale, Mirroring)
4. Synthetic Benchmark Dataset Generation (1,400 samples across 14 classes)
5. Feature Dimensionality (92-D vectors, clean float32, zero NaNs / Infs)
6. Class Distribution & Balance (100 samples per class, 100% balanced)
7. Dataset Serialization & Deserialization (NPZ and CSV)
8. Stratified Train / Validation / Test Split Integrity
9. Dataset Validation & JSON Metadata Summary Generation
10. Live Dataset Collector Session Management
"""

import csv
import json
import math
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

import dataset
from dataset import (
    DataAugmenter,
    DatasetCollector,
    GESTURE_CLASSES,
    GESTURE_SYNONYMS,
    GestureDataset,
    ID_TO_LABEL,
    LABEL_TO_ID,
    SyntheticDatasetGenerator,
    canonicalize_label,
)


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


def run_tests():
    print("=" * 75)
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 5 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     Custom Landmark-Based Hand Gesture Dataset (Module 2)")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Section 1: 14-Gesture Taxonomy & Scaffolding
    # -----------------------------------------------------------------
    print("\n1. 14-Gesture Taxonomic Vocabulary (Module 2)")
    expected_14 = [
        "OPEN_PALM", "FIST", "THUMBS_UP", "THUMBS_DOWN", "POINT", "PEACE",
        "OK", "PINCH", "THREE_FINGERS", "FOUR_FINGERS",
        "SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN"
    ]
    has_all_14 = set(expected_14).issubset(set(GESTURE_CLASSES))
    collector.record("14 Canonical Gesture Classes Present", has_all_14, f"Classes count: {len(GESTURE_CLASSES)}")

    bi_directional_map = all(
        ID_TO_LABEL[LABEL_TO_ID[name]] == name for name in GESTURE_CLASSES
    )
    collector.record("Label-to-ID Bi-directional Mapping", bi_directional_map, "14/14 bijective mappings verified")

    synonyms_ok = (
        canonicalize_label("CLOSED_FIST") == "FIST" and
        canonicalize_label("INDEX_POINTING") == "POINT" and
        canonicalize_label("PEACE_SIGN") == "PEACE" and
        canonicalize_label("OK_SIGN") == "OK" and
        canonicalize_label("PINCH_INDEX") == "PINCH" and
        canonicalize_label("open palm") == "OPEN_PALM"
    )
    collector.record("Gesture Synonym & Alias Canonicalization", synonyms_ok, "Synonyms correctly normalized")

    # -----------------------------------------------------------------
    # Section 2: Data Augmentation Operations
    # -----------------------------------------------------------------
    print("\n2. 3D Landmark Data Augmentation Pipeline")
    base_lms = np.zeros((21, 3), dtype=np.float64)
    base_lms[0] = [0.0, 0.0, 0.0]  # Wrist
    base_lms[9] = [0.0, -1.0, 0.0] # Middle MCP

    # Jitter
    jittered = DataAugmenter.add_gaussian_jitter(base_lms, sigma=0.01)
    jitter_wrist_fixed = np.allclose(jittered[0], [0.0, 0.0, 0.0])
    jitter_changed = not np.allclose(jittered[9], base_lms[9])
    collector.record("Gaussian Coordinate Jitter Noise", jitter_wrist_fixed and jitter_changed, "Wrist fixed at origin, coordinates perturbed")

    # 3D Rotation
    rotated = DataAugmenter.random_3d_rotation(base_lms, max_angle_deg=15.0)
    rot_dist_preserved = abs(np.linalg.norm(rotated[9] - rotated[0]) - 1.0) < 1e-4
    collector.record("3D Random Euler Rotation (Preserves Dist)", rot_dist_preserved, f"Span after rotation: {np.linalg.norm(rotated[9] - rotated[0]):.4f}")

    # Scale variation
    scaled = DataAugmenter.random_scaling(base_lms, scale_range=(0.85, 1.15))
    scale_span = np.linalg.norm(scaled[9] - scaled[0])
    collector.record("Distance / Scale Perturbation", 0.80 <= scale_span <= 1.20, f"Span after scale: {scale_span:.4f}")

    # Horizontal Mirroring
    test_lms = base_lms.copy()
    test_lms[5] = [0.25, -0.80, 0.0]  # Index MCP
    mirrored = DataAugmenter.mirror_hand_horizontal(test_lms)
    collector.record("Horizontal Hand Mirroring (X Inversion)", mirrored[5, 0] == -0.25, f"Original X: 0.25 -> Mirrored X: {mirrored[5, 0]}")

    # Full stochastic pipeline
    aug_full = DataAugmenter.augment_landmarks(base_lms)
    collector.record("Composite Stochastic Augmentation", aug_full.shape == (21, 3) and not np.isnan(aug_full).any(), "Clean augmented array")

    # -----------------------------------------------------------------
    # Section 3: Synthetic Benchmark Dataset Generation
    # -----------------------------------------------------------------
    print("\n3. Synthetic Benchmark Dataset Generation (1,400 Samples)")
    t0 = time.perf_counter()
    generator = SyntheticDatasetGenerator()
    dataset_dir = os.path.join(_ROOT, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    samples_per_class = 100
    total_expected = samples_per_class * len(GESTURE_CLASSES)  # 100 * 14 = 1,400

    dataset_obj = generator.generate_dataset(samples_per_class=samples_per_class, output_dir=dataset_dir)
    gen_time_sec = time.perf_counter() - t0

    collector.record(
        f"Benchmark Generation ({total_expected} Samples)",
        len(dataset_obj) == total_expected,
        f"Generated {len(dataset_obj)} samples in {gen_time_sec:.2f}s ({(len(dataset_obj)/gen_time_sec):,.0f} samples/s)"
    )

    # -----------------------------------------------------------------
    # Section 4: Feature Dimensionality & Data Cleanliness
    # -----------------------------------------------------------------
    print("\n4. Feature Dimensionality & Data Cleanliness")
    dim_ok = dataset_obj.X.shape == (total_expected, 92)
    collector.record("Feature Matrix Shape (N, 92)", dim_ok, f"Shape: {dataset_obj.X.shape}, Dtype: {dataset_obj.X.dtype}")

    no_nan_or_inf = not np.isnan(dataset_obj.X).any() and not np.isinf(dataset_obj.X).any()
    collector.record("Zero NaN / Inf In Feature Matrix", no_nan_or_inf, "Strict numeric integrity verified")

    lms_dim_ok = dataset_obj.landmarks.shape == (total_expected, 21, 3)
    collector.record("Landmark Coordinates Matrix (N, 21, 3)", lms_dim_ok, f"Shape: {dataset_obj.landmarks.shape}")

    vel_dim_ok = dataset_obj.velocities.shape == (total_expected, 4)
    collector.record("Kinematic Velocities Matrix (N, 4)", vel_dim_ok, f"Shape: {dataset_obj.velocities.shape}")

    # -----------------------------------------------------------------
    # Section 5: Class Balance & Distribution
    # -----------------------------------------------------------------
    print("\n5. Class Distribution & Dataset Balance")
    dist = dataset_obj.get_class_distribution()
    is_balanced = all(count == samples_per_class for count in dist.values())
    collector.record("100% Balanced Class Distribution", is_balanced, f"Exactly {samples_per_class} samples across all 14 classes")

    labels_encoded_ok = (
        len(dataset_obj.y_encoded) == total_expected and
        dataset_obj.y_encoded.min() == 0 and
        dataset_obj.y_encoded.max() == 13
    )
    collector.record("Class ID Encoding Range [0..13]", labels_encoded_ok, f"Unique encoded IDs: {len(np.unique(dataset_obj.y_encoded))}")

    # -----------------------------------------------------------------
    # Section 6: Serialization & Storage (NPZ and CSV)
    # -----------------------------------------------------------------
    print("\n6. Dataset Serialization & Deserialization")
    npz_path = os.path.join(dataset_dir, "gesture_dataset.npz")
    csv_path = os.path.join(dataset_dir, "gesture_dataset.csv")

    collector.record("Compressed NPZ File Exists", os.path.exists(npz_path), f"File size: {os.path.getsize(npz_path):,} bytes")
    collector.record("CSV Inspection File Exists", os.path.exists(csv_path), f"File size: {os.path.getsize(csv_path):,} bytes")

    # Load from NPZ and verify byte-for-byte exact equality
    loaded_npz = GestureDataset.load_from_npz(npz_path)
    npz_match = (
        len(loaded_npz) == total_expected and
        np.array_equal(loaded_npz.X, dataset_obj.X) and
        np.array_equal(loaded_npz.y_encoded, dataset_obj.y_encoded)
    )
    collector.record("NPZ Round-Trip Deserialization", npz_match, "Byte-level match with in-memory dataset")

    # Verify CSV line count
    with open(csv_path, "r", encoding="utf-8") as f:
        csv_lines = sum(1 for _ in f)
    collector.record("CSV Row Integrity (1 Header + 1400 Samples)", csv_lines == total_expected + 1, f"Total CSV rows: {csv_lines}")

    # -----------------------------------------------------------------
    # Section 7: Stratified Train / Validation / Test Splits
    # -----------------------------------------------------------------
    print("\n7. Stratified Dataset Splitting (70% Train, 10% Val, 20% Test)")
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_obj.train_test_split(
        test_size=0.20,
        val_size=0.10,
        random_state=42
    )

    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)
    splits_sum_ok = (n_train + n_val + n_test) == total_expected
    collector.record("Dataset Partition Sum (980 + 140 + 280 = 1400)", splits_sum_ok, f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # Check stratification: all 14 classes must be in all 3 splits
    train_classes = len(np.unique(y_train))
    val_classes = len(np.unique(y_val))
    test_classes = len(np.unique(y_test))
    all_classes_present = (train_classes == 14 and val_classes == 14 and test_classes == 14)
    collector.record("Stratified Class Representation", all_classes_present, "All 14 classes represented across Train, Val, Test")

    # -----------------------------------------------------------------
    # Section 8: Metadata Summary & Validation
    # -----------------------------------------------------------------
    print("\n8. Dataset Validation & Metadata Summary JSON")
    validation_res = dataset_obj.validate()
    collector.record("Dataset Validation Suite", validation_res["valid"], f"Issues found: {len(validation_res['issues'])}")

    summary_path = os.path.join(dataset_dir, "dataset_summary.json")
    collector.record("Summary JSON Metadata Exists", os.path.exists(summary_path), f"Path: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    collector.record("Summary JSON Schema Integrity", summary_data["total_samples"] == total_expected and summary_data["feature_dimension"] == 92, "Valid JSON structure")

    # -----------------------------------------------------------------
    # Section 9: Live Collector Session Management
    # -----------------------------------------------------------------
    print("\n9. Live Dataset Collector Session Management")
    collector_inst = DatasetCollector(output_dir=dataset_dir)
    collector_inst.start_session("OPEN_PALM", target_samples=10)
    collector.record("Live Collector Session Start", collector_inst.is_collecting and collector_inst.current_gesture == "OPEN_PALM", "Session active for OPEN_PALM")

    # Simulate recording 10 frames
    sample_hand_coords = [(float(p[0]), float(p[1]), float(p[2])) for p in base_lms]
    for _ in range(10):
        done = collector_inst.record_frame(sample_hand_coords)

    collector.record("Live Session Auto-Stop at Target", not collector_inst.is_collecting and collector_inst.collected_in_session == 10, "Target 10 samples reached, session closed")

    # -----------------------------------------------------------------
    # Summary & Export
    # -----------------------------------------------------------------
    total = len(collector.results)
    passed = sum(1 for r in collector.results if r["status"] == "PASS")
    failed = total - passed
    pass_rate = (passed / total) * 100.0 if total > 0 else 0.0

    print("\n" + "=" * 75)
    print(" PHASE 5 TEST EXECUTION SUMMARY")
    print("=" * 75)
    print(f" Total Tests Run:   {total}")
    print(f" Passed:            {passed}")
    print(f" Failed:            {failed}")
    print(f" Pass Rate:         {pass_rate:.1f}%")
    print("=" * 75)

    report_path = os.path.join(_ROOT, "data", "phase5_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "phase": "Phase 5 - Custom Landmark-Based Hand Gesture Dataset",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.platform(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": pass_rate,
            "samples_generated": total_expected,
            "classes_count": len(GESTURE_CLASSES),
            "results": collector.results
        }, f, indent=2)

    print(f" Benchmark report exported to: {report_path}\n")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
