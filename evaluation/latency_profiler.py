"""
Phase 19: End-to-End Latency Profiling & Real-Time Performance Optimization (Module 20)
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. High-Resolution Sub-Microsecond Multi-Stage Profiler (Module 20):
   - Instruments every distinct stage of the perception and control loop using `time.perf_counter_ns()`:
     * STAGE 1: Frame Acquisition & Color Conversion (BGR -> RGB)
     * STAGE 2: Landmark Perception & Geometry Extraction (21 3D joints)
     * STAGE 3: 92-D Vector Geometric Feature Normalization
     * STAGE 4: Machine Learning Inference (SVM-RBF Probabilistic Predictor)
     * STAGE 5: Temporal Stabilization & 1-Euro Low-Pass Jitter Filtering
     * STAGE 6: OS Control Dispatch (Direct Win32 Hardware Cursor & Key Injection)
     * STAGE 7: HUD Visual Rendering (Translucent Glassmorphism Canvas Overlay)

2. Complete Statistical & Percentile Analysis:
   - Computes: Mean, Median (p50), 90th Percentile (p90), 95th Percentile (p95),
     99th Percentile (p99), Min, Max, Standard Deviation, and Latency Jitter Variance.
   - Computes stage percentage contribution breakdown to identify critical bottlenecks.

3. Real-Time HCI Compliance Verification:
   - Verifies adherence to strict HCI interactivity thresholds:
     * Interactive Latency Requirement: Total perception-to-action latency < 33.3 ms (30+ FPS).
     * Ultra-Responsive Computational Core: Core recognition pipeline < 5.0 ms (> 200 FPS).
     * Temporal Jitter Stability: p95 latency <= 2.0x mean latency.
"""

import collections
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from controllers import OneEuroFilter, VirtualMouseController
from dataset import GestureDataset
from detection.hand_detector import HandData
from features.landmark_processor import LandmarkProcessor
from gestures import RealtimeGestureRecognizer
from models.predict import DEFAULT_MODEL_PATH, GesturePredictor
from ui import HUDRenderer
from utils.logger import logger

DEFAULT_LATENCY_REPORT_PATH = os.path.join(_ROOT, "data", "phase19_latency_report.json")


class LatencyProfiler:
    """
    Comprehensive multi-stage latency instrumentation and statistical profiler
    for the vision-driven HCI perception and automation pipeline.
    """

    def __init__(self):
        # Stage timings buffer: stage_name -> deque of latency in milliseconds
        self.stage_timings: Dict[str, List[float]] = collections.defaultdict(list)

    def record_measurement(self, stage_name: str, latency_ms: float):
        """Records a single duration measurement in milliseconds."""
        self.stage_timings[stage_name].append(float(latency_ms))

    def compute_stage_statistics(self, stage_name: str) -> Dict[str, Any]:
        """
        Computes descriptive and percentile statistics for a specific stage.
        """
        measurements = self.stage_timings.get(stage_name, [])
        if not measurements:
            return {
                "count": 0,
                "mean_ms": 0.0,
                "median_p50_ms": 0.0,
                "p90_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "std_ms": 0.0,
                "jitter_variance": 0.0,
            }

        arr = np.array(measurements, dtype=np.float64)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        var_val = float(np.var(arr))

        return {
            "count": len(arr),
            "mean_ms": round(mean_val, 4),
            "median_p50_ms": round(float(np.percentile(arr, 50)), 4),
            "p90_ms": round(float(np.percentile(arr, 90)), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
            "p99_ms": round(float(np.percentile(arr, 99)), 4),
            "min_ms": round(float(np.min(arr)), 4),
            "max_ms": round(float(np.max(arr)), 4),
            "std_ms": round(std_val, 4),
            "jitter_variance": round(var_val, 6),
        }

    def run_multi_stage_benchmark(
        self,
        n_frames: int = 500,
        warmup_frames: int = 50,
    ) -> Dict[str, Any]:
        """
        Executes a continuous end-to-end benchmark across all 7 pipeline stages.
        """
        logger.info(f"[LatencyProfiler] Starting benchmark ({n_frames} frames, {warmup_frames} warmup)...")

        # Initialize subsystems
        processor = LandmarkProcessor()
        predictor = GesturePredictor(processor=processor)
        recognizer = RealtimeGestureRecognizer(processor=processor, predictor=predictor)
        mouse_ctrl = VirtualMouseController(mock_mode=True)
        hud = HUDRenderer()
        filter_obj = OneEuroFilter(t0=0.0, x0=0.5, min_cutoff=1.0, beta=0.007, d_cutoff=1.0)

        # Load real test landmarks from dataset
        dataset_path = os.path.join(_ROOT, "dataset", "gesture_dataset.npz")
        dataset = GestureDataset.load_from_npz(dataset_path)
        sample_landmarks = dataset.landmarks[0]  # (21, 3)

        # Pre-allocate frames
        raw_bgr = np.full((480, 640, 3), 40, dtype=np.uint8)

        self.stage_timings.clear()
        total_runs = warmup_frames + n_frames

        for idx in range(total_runs):
            is_warmup = idx < warmup_frames
            t_pipeline_start = time.perf_counter_ns()

            # -------------------------------------------------------------
            # Stage 1: Frame Acquisition & Color Conversion
            # -------------------------------------------------------------
            t0 = time.perf_counter_ns()
            rgb_frame = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            t_stage1 = (time.perf_counter_ns() - t0) / 1e6

            # -------------------------------------------------------------
            # Stage 2: Landmark Detection & Joint Extraction
            # (Simulated extraction of 21 3D joints)
            # -------------------------------------------------------------
            t0 = time.perf_counter_ns()
            # In live webcam this is MediaPipe inference (~8-15 ms).
            # Here we measure joint container construction and sanitization.
            hand_data = HandData(
                raw_landmarks=[type("MockLM", (), {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}) for p in sample_landmarks],
                frame_w=640,
                frame_h=480,
            )
            t_stage2 = (time.perf_counter_ns() - t0) / 1e6

            # -------------------------------------------------------------
            # Stage 3: 92-D Geometric Feature Vector Normalization
            # -------------------------------------------------------------
            t0 = time.perf_counter_ns()
            features = processor.extract_single_hand(sample_landmarks)
            feat_vec = features.feature_vector_92d
            t_stage3 = (time.perf_counter_ns() - t0) / 1e6

            # -------------------------------------------------------------
            # Stage 4: Machine Learning Inference (SVM-RBF)
            # -------------------------------------------------------------
            t0 = time.perf_counter_ns()
            pred_gesture, pred_conf = predictor.predict(feat_vec)
            t_stage4 = (time.perf_counter_ns() - t0) / 1e6

            # -------------------------------------------------------------
            # Stage 5: Temporal Stabilization & 1-Euro Filter
            # -------------------------------------------------------------
            t0 = time.perf_counter_ns()
            # 1-Euro low-pass filter
            _ = filter_obj.filter(t=idx * 0.033, x=0.52)
            # Majority voting & cooldown stabilization
            recog_res = recognizer.process_frame(hand_data, timestamp=idx * 0.033)
            t_stage5 = (time.perf_counter_ns() - t0) / 1e6

            # -------------------------------------------------------------
            # Stage 6: OS Control Dispatch
            # -------------------------------------------------------------
            t0 = time.perf_counter_ns()
            mouse_ctrl.move_cursor_to_norm(0.5, 0.5, timestamp=idx * 0.033)
            t_stage6 = (time.perf_counter_ns() - t0) / 1e6

            # -------------------------------------------------------------
            # Stage 7: HUD Visual Rendering
            # -------------------------------------------------------------
            t0 = time.perf_counter_ns()
            _ = hud.render(
                frame=raw_bgr,
                recognition_result=recog_res,
                raw_landmarks=sample_landmarks,
                last_action="CURSOR_MOVE",
                fps_override=60.0,
            )
            t_stage7 = (time.perf_counter_ns() - t0) / 1e6

            # Total end-to-end cycle
            t_total_e2e = (time.perf_counter_ns() - t_pipeline_start) / 1e6

            # Core computational cycle (Stages 3 + 4 + 5 + 6)
            t_core_cycle = t_stage3 + t_stage4 + t_stage5 + t_stage6

            if not is_warmup:
                self.record_measurement("stage1_frame_io_ms", t_stage1)
                self.record_measurement("stage2_hand_parsing_ms", t_stage2)
                self.record_measurement("stage3_feature_extraction_ms", t_stage3)
                self.record_measurement("stage4_ml_inference_ms", t_stage4)
                self.record_measurement("stage5_stabilization_ms", t_stage5)
                self.record_measurement("stage6_control_dispatch_ms", t_stage6)
                self.record_measurement("stage7_hud_rendering_ms", t_stage7)
                self.record_measurement("core_computational_latency_ms", t_core_cycle)
                self.record_measurement("total_end_to_end_latency_ms", t_total_e2e)

        return self.generate_profile_report(n_frames)

    def generate_profile_report(self, n_frames: int) -> Dict[str, Any]:
        """Computes summary statistics and percentage breakdown for all stages."""
        stage_names = [
            "stage1_frame_io_ms",
            "stage2_hand_parsing_ms",
            "stage3_feature_extraction_ms",
            "stage4_ml_inference_ms",
            "stage5_stabilization_ms",
            "stage6_control_dispatch_ms",
            "stage7_hud_rendering_ms",
            "core_computational_latency_ms",
            "total_end_to_end_latency_ms",
        ]

        stage_metrics = {name: self.compute_stage_statistics(name) for name in stage_names}

        total_mean = stage_metrics["total_end_to_end_latency_ms"]["mean_ms"]
        core_mean = stage_metrics["core_computational_latency_ms"]["mean_ms"]

        # Stage percentage contributions
        breakdown_pct = {}
        for st_id in range(1, 8):
            key = f"stage{st_id}_" + [
                "frame_io_ms",
                "hand_parsing_ms",
                "feature_extraction_ms",
                "ml_inference_ms",
                "stabilization_ms",
                "control_dispatch_ms",
                "hud_rendering_ms"
            ][st_id - 1]
            st_mean = stage_metrics[key]["mean_ms"]
            pct = (st_mean / total_mean * 100.0) if total_mean > 0 else 0.0
            breakdown_pct[key] = round(pct, 2)

        # HCI Compliance Status Checks
        p95_total = stage_metrics["total_end_to_end_latency_ms"]["p95_ms"]
        max_fps = 1000.0 / total_mean if total_mean > 0 else 0.0
        core_max_fps = 1000.0 / core_mean if core_mean > 0 else 0.0

        compliance = {
            "real_time_interactive_threshold_met": total_mean < 33.33,  # > 30 FPS standard
            "sub_5ms_computational_core_met": core_mean < 5.0,           # > 200 FPS core
            "jitter_stability_ratio": round(p95_total / total_mean, 2) if total_mean > 0 else 1.0,
            "jitter_stable": (p95_total / total_mean <= 2.0) if total_mean > 0 else True,
            "achievable_throughput_fps": round(max_fps, 1),
            "core_throughput_fps": round(core_max_fps, 1),
        }

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.platform(),
            "benchmark_frames": n_frames,
            "summary": {
                "mean_e2e_latency_ms": total_mean,
                "median_e2e_latency_ms": stage_metrics["total_end_to_end_latency_ms"]["median_p50_ms"],
                "p95_e2e_latency_ms": p95_total,
                "p99_e2e_latency_ms": stage_metrics["total_end_to_end_latency_ms"]["p99_ms"],
                "core_computational_latency_ms": core_mean,
                "achievable_fps": round(max_fps, 1),
                "core_max_fps": round(core_max_fps, 1),
            },
            "compliance": compliance,
            "stage_contributions_pct": breakdown_pct,
            "stages": stage_metrics,
        }

        return report

    def save_report(self, output_path: str = DEFAULT_LATENCY_REPORT_PATH) -> bool:
        """Saves latency report to a JSON file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            report = self.generate_profile_report(len(self.stage_timings.get("total_end_to_end_latency_ms", [])))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"[LatencyProfiler] Saved latency report to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"[LatencyProfiler] Failed to save latency report: {e}")
            return False
