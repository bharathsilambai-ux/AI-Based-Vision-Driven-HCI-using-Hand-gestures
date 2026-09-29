"""
Performance Monitoring and Telemetry Module.
Calculates FPS, end-to-end processing latency, recognition accuracy,
and logs gesture stability metrics.
"""

from collections import deque
import json
import os
import time
from typing import Any, Dict, Optional


class PerformanceTracker:
    """Tracks latency, frame rate, and gesture recognition metrics."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._frame_times = deque(maxlen=window_size)
        self._latencies = deque(maxlen=window_size)
        self._start_time = time.time()

        # Cumulative Metrics
        self.total_frames = 0
        self.total_recognitions = 0
        self.correct_recognitions = 0
        self.false_triggers = 0

        # Per-gesture metrics dictionary: { gesture_name: {"attempts": int, "correct": int, "latency_sum": float} }
        self.gesture_metrics: Dict[str, Dict[str, Any]] = {}

        # Timers
        self._current_frame_start: Optional[float] = None

    def start_frame(self):
        """Call at the beginning of each frame processing iteration."""
        self._current_frame_start = time.perf_counter()

    def end_frame(self):
        """Call at the end of each frame processing cycle."""
        if self._current_frame_start is not None:
            latency = (time.perf_counter() - self._current_frame_start) * 1000.0  # in ms
            self._latencies.append(latency)
            self._frame_times.append(time.perf_counter())
            self.total_frames += 1
            self._current_frame_start = None
            return latency
        return 0.0

    @property
    def fps(self) -> float:
        """Computes instantaneous FPS over the sliding window."""
        if len(self._frame_times) < 2:
            return 0.0
        delta = self._frame_times[-1] - self._frame_times[0]
        if delta <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / delta

    @property
    def avg_latency_ms(self) -> float:
        """Returns average processing latency in milliseconds."""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def current_latency_ms(self) -> float:
        """Returns the most recent frame latency in milliseconds."""
        if not self._latencies:
            return 0.0
        return self._latencies[-1]

    def record_gesture_event(self, gesture_name: str, is_correct: bool = True, latency_ms: float = 0.0):
        """Records an individual gesture recognition event."""
        self.total_recognitions += 1
        if is_correct:
            self.correct_recognitions += 1
        else:
            self.false_triggers += 1

        if gesture_name not in self.gesture_metrics:
            self.gesture_metrics[gesture_name] = {
                "attempts": 0,
                "correct": 0,
                "false_triggers": 0,
                "latency_sum": 0.0
            }

        gm = self.gesture_metrics[gesture_name]
        gm["attempts"] += 1
        if is_correct:
            gm["correct"] += 1
        else:
            gm["false_triggers"] += 1
        gm["latency_sum"] += latency_ms

    @property
    def accuracy(self) -> float:
        """Computes overall accuracy percentage."""
        if self.total_recognitions == 0:
            return 100.0
        return (self.correct_recognitions / self.total_recognitions) * 100.0

    def generate_report(self) -> Dict[str, Any]:
        """Generates a structured performance benchmark dictionary."""
        per_gesture_summary = {}
        for name, data in self.gesture_metrics.items():
            att = data["attempts"]
            acc = (data["correct"] / att * 100.0) if att > 0 else 0.0
            avg_lat = (data["latency_sum"] / att) if att > 0 else 0.0
            per_gesture_summary[name] = {
                "attempts": att,
                "correct": data["correct"],
                "false_triggers": data["false_triggers"],
                "accuracy_percent": round(acc, 2),
                "avg_latency_ms": round(avg_lat, 2)
            }

        return {
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "total_frames_processed": self.total_frames,
            "average_fps": round(self.fps, 2),
            "average_latency_ms": round(self.avg_latency_ms, 2),
            "total_recognitions": self.total_recognitions,
            "correct_recognitions": self.correct_recognitions,
            "false_triggers": self.false_triggers,
            "overall_accuracy_percent": round(self.accuracy, 2),
            "gestures": per_gesture_summary
        }

    def save_report_to_disk(self, filepath: str):
        """Saves current report to a JSON file."""
        report = self.generate_report()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        return report


# Global performance tracker instance
perf_tracker = PerformanceTracker()
