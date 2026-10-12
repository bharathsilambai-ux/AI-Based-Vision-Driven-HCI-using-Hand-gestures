"""
Phase 19 Verification Test Suite: Latency Profiling & Performance Optimization (Module 20)
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding & Package Exports (evaluation.LatencyProfiler, DEFAULT_LATENCY_REPORT_PATH)
2. Profiler Recording Mechanics & Statistical Reductions (Mean, Median, p90, p95, p99, Std, Jitter)
3. Multi-Stage Pipeline Benchmark Execution Across All 7 Stages:
   - Stage 1: Frame I/O & Color Conversion (BGR -> RGB)
   - Stage 2: Hand Detection & Joint Parsing
   - Stage 3: 92-D Geometric Feature Vector Extraction
   - Stage 4: Machine Learning Inference (SVM-RBF)
   - Stage 5: Temporal Stabilization & 1-Euro Low-Pass Filter
   - Stage 6: OS Control Dispatch (Direct Win32 Hardware Cursor & Key Injection)
   - Stage 7: HUD Visual Rendering (Glassmorphism Overlay)
4. Core Computational Latency Compliance (< 5.0 ms / > 200 FPS Core Capability)
5. End-to-End Interactive Latency Compliance (< 33.3 ms / > 30 FPS Real-Time Standard)
6. Jitter Variance & Latency Spike Stability (p95 <= 2.5x Mean)
7. Full Latency Profiling Report Persistence (data/phase19_latency_report.json)
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
from evaluation import DEFAULT_LATENCY_REPORT_PATH, LatencyProfiler


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
        print(f"  {badge} {test_name:<50} {details}")


collector = TestResultCollector()


def run_tests() -> bool:
    print("=" * 75)
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 19 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     Latency Profiling & Performance Optimization (Module 20)")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Section 1: Scaffolding & Package Exports
    # -----------------------------------------------------------------
    print("\n1. Module Scaffolding & Package Exports")
    exports_ok = hasattr(evaluation, "LatencyProfiler") and hasattr(evaluation, "DEFAULT_LATENCY_REPORT_PATH")
    collector.record("LatencyProfiler Package Export", exports_ok, "LatencyProfiler exported from evaluation package")

    profiler = LatencyProfiler()
    init_ok = len(profiler.stage_timings) == 0
    collector.record("LatencyProfiler Initialization", init_ok, "Buffer initialized and clean")

    # -----------------------------------------------------------------
    # Section 2: Profiler Statistics Engine
    # -----------------------------------------------------------------
    print("\n2. Statistical Reduction & Percentile Engine")
    test_latencies = [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 4.5, 8.0]
    for val in test_latencies:
        profiler.record_measurement("test_metric", val)

    stats = profiler.compute_stage_statistics("test_metric")
    stats_ok = (
        stats["count"] == 10
        and stats["mean_ms"] == round(float(np.mean(test_latencies)), 4)
        and stats["median_p50_ms"] == round(float(np.median(test_latencies)), 4)
        and stats["p95_ms"] > stats["median_p50_ms"]
        and stats["min_ms"] == 1.0
        and stats["max_ms"] == 8.0
    )
    collector.record("Percentile Statistics Engine", stats_ok, f"Mean: {stats['mean_ms']} ms | p50: {stats['median_p50_ms']} ms | p95: {stats['p95_ms']} ms")

    # -----------------------------------------------------------------
    # Section 3: Multi-Stage Continuous Benchmark Execution
    # -----------------------------------------------------------------
    print("\n3. Continuous End-to-End Multi-Stage Benchmark")
    report = profiler.run_multi_stage_benchmark(n_frames=300, warmup_frames=30)

    stages = report.get("stages", {})
    all_stages_present = (
        len(stages) >= 8
        and "stage1_frame_io_ms" in stages
        and "stage2_hand_parsing_ms" in stages
        and "stage3_feature_extraction_ms" in stages
        and "stage4_ml_inference_ms" in stages
        and "stage5_stabilization_ms" in stages
        and "stage6_control_dispatch_ms" in stages
        and "stage7_hud_rendering_ms" in stages
        and "total_end_to_end_latency_ms" in stages
    )
    collector.record("All 7 Pipeline Stages Measured", all_stages_present, f"Captured 300 iterations across {len(stages)} telemetry channels")

    # -----------------------------------------------------------------
    # Section 4: Core Computational Latency Verification
    # -----------------------------------------------------------------
    print("\n4. Core Computational Latency Verification")
    core_mean = report["summary"]["core_computational_latency_ms"]
    core_max_fps = report["summary"]["core_max_fps"]

    core_lat_ok = core_mean < 5.0
    collector.record(
        "Core Computational Latency (< 5.0 ms)",
        core_lat_ok,
        f"Core Latency: {core_mean:.3f} ms | Max Computational Speed: {core_max_fps:,.0f} FPS",
        latency_ms=core_mean
    )

    # -----------------------------------------------------------------
    # Section 5: End-to-End Interactive Latency Compliance
    # -----------------------------------------------------------------
    print("\n5. End-to-End Interactive Latency Compliance")
    total_mean = report["summary"]["mean_e2e_latency_ms"]
    total_median = report["summary"]["median_e2e_latency_ms"]
    total_fps = report["summary"]["achievable_fps"]

    interactive_ok = total_mean < 33.33  # 30 FPS threshold
    collector.record(
        "End-to-End Latency Compliance (< 33.3 ms)",
        interactive_ok,
        f"Mean E2E: {total_mean:.3f} ms (p50: {total_median:.3f} ms) | Throughput: {total_fps:,.0f} FPS",
        latency_ms=total_mean
    )

    # -----------------------------------------------------------------
    # Section 6: Latency Percentile Consistency & Jitter Stability
    # -----------------------------------------------------------------
    print("\n6. Jitter Stability & Percentile Consistency")
    p95_val = report["summary"]["p95_e2e_latency_ms"]
    p99_val = report["summary"]["p99_e2e_latency_ms"]
    jitter_ratio = report["compliance"]["jitter_stability_ratio"]
    jitter_stable = report["compliance"]["jitter_stable"]

    collector.record(
        "Latency Jitter Stability (p95 <= 2.0x Mean)",
        jitter_stable,
        f"p95: {p95_val:.3f} ms | p99: {p99_val:.3f} ms | Ratio p95/mean: {jitter_ratio:.2f}x"
    )

    # -----------------------------------------------------------------
    # Section 7: Stage Contributions & Bottleneck Identification
    # -----------------------------------------------------------------
    print("\n7. Stage Percentage Contributions & Bottlenecks")
    contribs = report.get("stage_contributions_pct", {})
    sum_contribs = sum(contribs.values())
    contrib_ok = len(contribs) == 7 and 95.0 <= sum_contribs <= 105.0

    # Find dominant stage
    dominant_stage = max(contribs.items(), key=lambda x: x[1])
    collector.record(
        "Stage Latency Percentage Decomposition",
        contrib_ok,
        f"Top consumer: {dominant_stage[0]} ({dominant_stage[1]}%) | Sum: {sum_contribs:.1f}%"
    )

    # -----------------------------------------------------------------
    # Section 8: Report Generation & Persistence
    # -----------------------------------------------------------------
    print("\n8. Latency Profile Report Persistence")
    saved = profiler.save_report(DEFAULT_LATENCY_REPORT_PATH)
    file_exists = os.path.exists(DEFAULT_LATENCY_REPORT_PATH)
    file_size = os.path.getsize(DEFAULT_LATENCY_REPORT_PATH) if file_exists else 0
    persist_ok = saved and file_exists and file_size > 1000
    collector.record(
        "Latency Profile Report JSON Persistence",
        persist_ok,
        f"Saved to: {DEFAULT_LATENCY_REPORT_PATH} ({file_size:,} bytes)"
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

    report_path = os.path.join(_ROOT, "data", "phase19_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_data = {
        "phase": 19,
        "name": "Latency Profiling & Performance Optimization",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "pass_rate_pct": pass_rate,
            "mean_e2e_latency_ms": total_mean,
            "core_latency_ms": core_mean,
            "achievable_fps": total_fps,
            "core_max_fps": core_max_fps,
            "p95_latency_ms": p95_val,
            "p99_latency_ms": p99_val,
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
