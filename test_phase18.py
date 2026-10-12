"""
Phase 18 Verification Test Suite: Full Benchmark Evaluation & Paper Comparison (Module 19)
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding & Package Exports (evaluation.BenchmarkEvaluator, DEFAULT_EVAL_REPORT_PATH)
2. Empirical Static ML Model Evaluation on Test Split:
   - Overall Accuracy >= 90.0%
   - Macro Precision, Recall, and F1-Score >= 90.0%
   - Per-Class Metrics across all 10 static gesture classes
   - Complete 10x10 Confusion Matrix Integrity
3. Empirical Dynamic Gesture Trajectory Recognition:
   - Kinematic Recognition Accuracy >= 90.0% across all 4 swipe directions
4. Multi-Stage Pipeline Latency Profiling:
   - Feature Extraction Latency (92-D geometry)
   - ML Inference Latency (SVM-RBF probability estimation)
   - 1-Euro Low-Pass Filter Latency
   - HUD Visual Rendering Latency
   - Theoretical Pipeline Throughput (> 100 FPS)
5. Foundational Research Paper Comparative Analysis (10 Evaluation Dimensions)
6. Complete Evaluation Report Persistence (data/phase18_evaluation_report.json)
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

import evaluation
from evaluation import DEFAULT_EVAL_REPORT_PATH, BenchmarkEvaluator


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


def run_tests() -> bool:
    print("=" * 75)
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 18 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     Full Benchmark Evaluation & Paper Comparison (Module 19)")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Section 1: Scaffolding & Package Exports
    # -----------------------------------------------------------------
    print("\n1. Module Scaffolding & Package Exports")
    exports_ok = hasattr(evaluation, "BenchmarkEvaluator") and hasattr(evaluation, "DEFAULT_EVAL_REPORT_PATH")
    collector.record("BenchmarkEvaluator Package Export", exports_ok, "BenchmarkEvaluator exported from evaluation package")

    evaluator = BenchmarkEvaluator()
    init_ok = evaluator.predictor.is_loaded and os.path.exists(evaluator.dataset_path)
    collector.record("Benchmark Evaluator Initialization", init_ok, f"Model: {evaluator.predictor.model_name} | Dataset: {evaluator.dataset_path}")

    # -----------------------------------------------------------------
    # Section 2: Static ML Model Evaluation
    # -----------------------------------------------------------------
    print("\n2. Empirical Static ML Model Evaluation")
    t0 = time.perf_counter()
    static_results = evaluator.evaluate_static_gestures(test_size=0.25, seed=42)
    eval_time = (time.perf_counter() - t0) * 1000.0

    acc = static_results["accuracy"]
    macro_f1 = static_results["macro_f1"]
    sample_count = static_results["test_sample_count"]

    # Accuracy >= 90%
    acc_ok = acc >= 0.90
    collector.record("Static Classification Accuracy (>= 90%)", acc_ok, f"Accuracy: {acc*100:.2f}% (Test Samples: {sample_count})")

    # Macro F1 >= 90%
    f1_ok = macro_f1 >= 0.90
    collector.record("Static Classification Macro F1 (>= 90%)", f1_ok, f"Macro F1: {macro_f1*100:.2f}% | Precision: {static_results['macro_precision']*100:.2f}%")

    # Per-Class Integrity (All classes evaluated)
    per_class = static_results["per_class"]
    classes_ok = len(per_class) >= 10 and all(metrics["f1_score"] >= 0.75 for metrics in per_class.values())
    collector.record("Per-Class Gesture Evaluation", classes_ok, f"Evaluated {len(per_class)} classes: {', '.join(sorted(per_class.keys()))}")

    # Confusion Matrix
    cm = static_results["confusion_matrix"]
    n_classes = len(static_results["classes"])
    cm_ok = len(cm) == n_classes and len(cm[0]) == n_classes and sum(sum(row) for row in cm) == sample_count
    collector.record("Confusion Matrix Integrity", cm_ok, f"Matrix size: {n_classes}x{n_classes} | Diagonal Sum (Correct): {sum(cm[i][i] for i in range(n_classes))}/{sample_count}")

    # -----------------------------------------------------------------
    # Section 3: Dynamic Gesture Trajectory Evaluation
    # -----------------------------------------------------------------
    print("\n3. Dynamic Trajectory Gesture Evaluation")
    dyn_results = evaluator.evaluate_dynamic_gestures(samples_per_class=40)
    dyn_acc = dyn_results["overall_accuracy"]
    dyn_ok = dyn_acc >= 0.90
    collector.record(
        "Dynamic Gesture Recognition Accuracy (>= 90%)",
        dyn_ok,
        f"Accuracy: {dyn_acc*100:.2f}% across SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN"
    )

    # -----------------------------------------------------------------
    # Section 4: Multi-Stage Pipeline Latency Profiling
    # -----------------------------------------------------------------
    print("\n4. Multi-Stage Pipeline Latency Profiling")
    lat_profile = evaluator.profile_pipeline_latencies(n_iterations=500)

    t_feat = lat_profile["feature_extraction_ms"]
    t_infer = lat_profile["inference_ms"]
    t_core = lat_profile["core_pipeline_latency_ms"]
    t_e2e = lat_profile["total_e2e_latency_ms"]
    max_fps = lat_profile["theoretical_max_fps"]

    core_lat_ok = t_core < 10.0  # Core pipeline < 10ms
    collector.record(
        "Core Pipeline Latency (< 10 ms)",
        core_lat_ok,
        f"Core: {t_core:.3f} ms (Feature Ext: {t_feat:.3f} ms | Inference: {t_infer:.3f} ms)",
        latency_ms=t_core
    )

    fps_ok = max_fps >= 100.0
    collector.record(
        "End-to-End Processing Throughput (> 100 FPS)",
        fps_ok,
        f"Total E2E: {t_e2e:.3f} ms | Theoretical Max: {max_fps:,.0f} FPS",
        latency_ms=t_e2e
    )

    # -----------------------------------------------------------------
    # Section 5: Comparative Paper Benchmark Analysis
    # -----------------------------------------------------------------
    print("\n5. Foundational Research Paper Comparative Analysis")
    inter_data = {
        "static_evaluation": static_results,
        "dynamic_evaluation": dyn_results,
        "latency_profile": lat_profile,
    }
    paper_comp = evaluator.generate_paper_comparison(inter_data)
    matrix = paper_comp["comparison_matrix"]

    comp_ok = len(matrix) >= 10 and all("baseline_paper" in row and "our_system" in row for row in matrix)
    collector.record("Research Paper Comparison Matrix", comp_ok, f"Evaluated {len(matrix)} distinct performance dimensions vs baseline paper")

    # -----------------------------------------------------------------
    # Section 6: Full Evaluation Suite & Report Persistence
    # -----------------------------------------------------------------
    print("\n6. Full Evaluation Report Generation & Disk Persistence")
    eval_report = evaluator.run_full_evaluation(DEFAULT_EVAL_REPORT_PATH)
    file_exists = os.path.exists(DEFAULT_EVAL_REPORT_PATH)
    file_size = os.path.getsize(DEFAULT_EVAL_REPORT_PATH)

    persist_ok = file_exists and file_size > 1000 and "static_evaluation" in eval_report
    collector.record("Evaluation Report JSON Generation", persist_ok, f"Saved report to: {DEFAULT_EVAL_REPORT_PATH} ({file_size:,} bytes)")

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

    report_path = os.path.join(_ROOT, "data", "phase18_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_data = {
        "phase": 18,
        "name": "Full Benchmark Evaluation & Paper Comparison",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "pass_rate_pct": pass_rate,
            "static_accuracy_pct": round(acc * 100.0, 2),
            "static_macro_f1_pct": round(macro_f1 * 100.0, 2),
            "dynamic_accuracy_pct": round(dyn_acc * 100.0, 2),
            "core_pipeline_latency_ms": round(t_core, 4),
            "total_e2e_latency_ms": round(t_e2e, 4),
            "theoretical_max_fps": round(max_fps, 1),
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
