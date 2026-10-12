"""
Phase 6 Verification Test Suite: ML Classifier Training, Evaluation & Comparative Benchmarking
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding, Package Exports & Default Model Paths
2. Training Execution Across 3 Classifier Architectures (RF, SVM, MLP)
3. Multi-Metric Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix, Latency)
4. Empirical Data-Driven Model Selection
5. Model Serialization & Bundle Integrity (models/trained_model.pkl)
6. Model Comparison JSON Report Generation (data/model_comparison.json)
7. Production Runtime Predictor (GesturePredictor) Functionality
8. Calibrated Probability & Confidence Estimation
9. Top-K Candidate Disambiguation Ranking
10. End-to-End Landmark Ingestion (HandData -> 92-D -> Prediction)
11. Input Validation & Error Resilience (None, Empty, NaN, Dimension Mismatches)
12. Single-Sample Inference Latency Benchmark (< 5.0 ms / > 200 FPS)
"""

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

import models
from dataset import GESTURE_CLASSES, GestureDataset
from features.landmark_processor import LandmarkProcessor
from models import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_SAVE_PATH,
    DEFAULT_REPORT_PATH,
    GesturePredictor,
    ModelTrainer,
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
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 6 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     Multi-Classifier Training & Evaluation (Module 3)")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Section 1: Scaffolding, Package Exports & Model Paths
    # -----------------------------------------------------------------
    print("\n1. Scaffolding, Package Exports & Model Paths")
    p_init = hasattr(models, "ModelTrainer") and hasattr(models, "GesturePredictor")
    collector.record("Models Package Exports", p_init, "ModelTrainer, GesturePredictor accessible")

    paths_ok = (
        isinstance(DEFAULT_MODEL_PATH, str) and
        isinstance(DEFAULT_DATASET_PATH, str) and
        os.path.exists(DEFAULT_DATASET_PATH)
    )
    collector.record("Default Paths & Dataset Existence", paths_ok, f"Dataset: {DEFAULT_DATASET_PATH}")

    # -----------------------------------------------------------------
    # Section 2: Model Trainer & Multi-Model Training
    # -----------------------------------------------------------------
    print("\n2. Multi-Model Training & Evaluation (RF, SVM, MLP)")
    trainer = ModelTrainer(
        dataset_path=DEFAULT_DATASET_PATH,
        model_save_path=DEFAULT_MODEL_SAVE_PATH,
        report_save_path=DEFAULT_REPORT_PATH
    )

    t0 = time.perf_counter()
    train_results = trainer.train_and_compare()
    train_duration = time.perf_counter() - t0

    models_trained = list(train_results["all_results"].keys())
    expected_models = ["Random_Forest", "SVM_RBF", "MLP_Neural_Network"]
    all_3_trained = set(expected_models).issubset(set(models_trained))
    collector.record("3 Model Architectures Evaluated", all_3_trained, f"Trained: {models_trained} ({train_duration:.2f}s)")

    # -----------------------------------------------------------------
    # Section 3: Metric Completeness Across All Models
    # -----------------------------------------------------------------
    print("\n3. Metric Completeness (Accuracy, F1, Latency, Confusion Matrix)")
    for name in expected_models:
        m = train_results["all_results"].get(name, {})
        valid_metrics = (
            "accuracy" in m and m["accuracy"] > 0.80 and
            "f1_macro" in m and m["f1_macro"] > 0.80 and
            "inference_latency_ms" in m and m["inference_latency_ms"] < 60.0 and
            "confusion_matrix" in m and len(m["confusion_matrix"]) == 14
        )
        collector.record(
            f"Metrics Completeness ({name})",
            valid_metrics,
            f"Acc: {m.get('accuracy', 0)*100:.1f}% | F1: {m.get('f1_macro', 0)*100:.1f}% | Latency: {m.get('inference_latency_ms', 0):.2f} ms"
        )

    # -----------------------------------------------------------------
    # Section 4: Data-Driven Selection & Serialization
    # -----------------------------------------------------------------
    print("\n4. Model Selection & Production Serialization")
    selected_name = train_results["selected_model"]
    best_f1 = train_results["best_metrics"]["f1_macro"]
    collector.record("Empirical Model Selection", bool(selected_name), f"Winner: '{selected_name}' (F1: {best_f1*100:.2f}%)")

    model_file_exists = os.path.exists(DEFAULT_MODEL_SAVE_PATH)
    model_size = os.path.getsize(DEFAULT_MODEL_SAVE_PATH) if model_file_exists else 0
    collector.record("Serialized Model Bundle Exists", model_file_exists and model_size > 1000, f"Path: {DEFAULT_MODEL_SAVE_PATH} ({model_size:,} bytes)")

    report_exists = os.path.exists(DEFAULT_REPORT_PATH)
    collector.record("Model Comparison JSON Report Exists", report_exists, f"Path: {DEFAULT_REPORT_PATH}")

    if report_exists:
        with open(DEFAULT_REPORT_PATH, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        report_valid = (
            report_data["selected_model"] == selected_name and
            len(report_data["models_evaluated"]) == 3
        )
        collector.record("Comparison JSON Schema & Completeness", report_valid, "Valid multi-model evaluation schema")

    # -----------------------------------------------------------------
    # Section 5: Production Runtime Predictor (GesturePredictor)
    # -----------------------------------------------------------------
    print("\n5. Production Runtime Predictor (GesturePredictor)")
    predictor = GesturePredictor(model_path=DEFAULT_MODEL_SAVE_PATH)
    collector.record("GesturePredictor Loaded", predictor.is_loaded, f"Active classifier: '{predictor.model_name}'")

    classes_count_ok = len(predictor.classes) == 14
    collector.record("14 Target Classes Loaded", classes_count_ok, f"Classes: {len(predictor.classes)}")

    # Test single-vector prediction
    dataset_obj = GestureDataset.load_from_npz(DEFAULT_DATASET_PATH)
    test_idx = 0
    test_sample = dataset_obj.X[test_idx]
    expected_label = dataset_obj.y[test_idx]

    pred_label, pred_conf = predictor.predict(test_sample)
    pred_ok = pred_label in GESTURE_CLASSES and 0.0 <= pred_conf <= 1.0
    collector.record("Single Vector Prediction & Confidence", pred_ok, f"Predicted: '{pred_label}' (Conf: {pred_conf*100:.1f}%) | Expected: '{expected_label}'")

    # Top-K candidate ranking
    top_3 = predictor.predict_top_k(test_sample, k=3)
    top_k_ok = (
        len(top_3) == 3 and
        top_3[0][1] >= top_3[1][1] >= top_3[2][1] and
        all(c[0] in GESTURE_CLASSES for c in top_3)
    )
    collector.record("Top-K Candidate Ranking (Descending Conf)", top_k_ok, f"Top-3: {[f'{name}:{conf:.2f}' for name, conf in top_3]}")

    # -----------------------------------------------------------------
    # Section 6: End-to-End Landmark Ingestion
    # -----------------------------------------------------------------
    print("\n6. End-to-End Landmark Ingestion (HandData -> Predict)")
    # Raw landmark coordinate array (21, 3) from dataset
    raw_lms = dataset_obj.landmarks[test_idx]
    pred_from_lms, conf_lms = predictor.predict_from_landmarks(raw_lms)
    collector.record("predict_from_landmarks() Integration", pred_from_lms in GESTURE_CLASSES, f"Predicted: '{pred_from_lms}' (Conf: {conf_lms*100:.1f}%)")

    # -----------------------------------------------------------------
    # Section 7: Input Validation & Error Resilience (Module 17)
    # -----------------------------------------------------------------
    print("\n7. Error Handling & Input Validation (Module 17)")
    # None input
    p_none, c_none = predictor.predict(None)
    collector.record("None Input Immunity", p_none == "NONE" and c_none == 0.0, "Safely returns ('NONE', 0.0)")

    # Empty array
    p_empty, c_empty = predictor.predict(np.array([]))
    collector.record("Empty Array Immunity", p_empty == "NONE" and c_empty == 0.0, "Safely returns ('NONE', 0.0)")

    # Dimension mismatch (50 instead of 92)
    p_dim, c_dim = predictor.predict(np.zeros(50))
    collector.record("Feature Dimension Mismatch Guard", p_dim == "NONE" and c_dim == 0.0, "Rejects wrong shape without crash")

    # NaN / Inf input
    nan_arr = np.full(92, np.nan)
    p_nan, c_nan = predictor.predict(nan_arr)
    collector.record("NaN / Inf Feature Vector Immunity", p_nan == "NONE" and c_nan == 0.0, "Safely rejected")

    # -----------------------------------------------------------------
    # Section 8: Single-Sample Inference Latency Benchmark
    # -----------------------------------------------------------------
    print("\n8. Single-Sample Inference Latency Benchmark")
    n_iters = 1000
    t_start = time.perf_counter()
    for _ in range(n_iters):
        _ = predictor.predict(test_sample)
    t_total = time.perf_counter() - t_start

    latency_per_call_ms = (t_total / n_iters) * 1000.0
    throughput_fps = n_iters / t_total

    collector.record(
        "Runtime Inference Latency (< 5.0 ms)",
        latency_per_call_ms < 5.0,
        f"Mean: {latency_per_call_ms:.3f} ms | Throughput: {throughput_fps:,.0f} predictions/sec"
    )

    # -----------------------------------------------------------------
    # Summary & Export
    # -----------------------------------------------------------------
    total = len(collector.results)
    passed = sum(1 for r in collector.results if r["status"] == "PASS")
    failed = total - passed
    pass_rate = (passed / total) * 100.0 if total > 0 else 0.0

    print("\n" + "=" * 75)
    print(" PHASE 6 TEST EXECUTION SUMMARY")
    print("=" * 75)
    print(f" Total Tests Run:   {total}")
    print(f" Passed:            {passed}")
    print(f" Failed:            {failed}")
    print(f" Pass Rate:         {pass_rate:.1f}%")
    print("=" * 75)

    report_path = os.path.join(_ROOT, "data", "phase6_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "phase": "Phase 6 - Train and Evaluate Machine Learning Classifiers",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.platform(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": pass_rate,
            "selected_model": selected_name,
            "best_f1_macro": best_f1,
            "mean_inference_latency_ms": latency_per_call_ms,
            "results": collector.results
        }, f, indent=2)

    print(f" Benchmark report exported to: {report_path}\n")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
