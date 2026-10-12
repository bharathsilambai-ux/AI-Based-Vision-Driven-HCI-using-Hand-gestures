"""
Evaluation Package: AI-Based Vision-Driven HCI System.
Provides benchmark evaluation, latency profiling, and comparative analysis (Modules 19 & 20).
"""

from .benchmark_evaluator import DEFAULT_EVAL_REPORT_PATH, BenchmarkEvaluator
from .latency_profiler import DEFAULT_LATENCY_REPORT_PATH, LatencyProfiler

__all__ = [
    "BenchmarkEvaluator",
    "DEFAULT_EVAL_REPORT_PATH",
    "LatencyProfiler",
    "DEFAULT_LATENCY_REPORT_PATH",
]
