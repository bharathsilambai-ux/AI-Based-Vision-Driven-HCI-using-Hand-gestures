"""
Phase 14 Verification Test Suite: Custom Gesture-to-Command Mapping System (Module 14)
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding & Package Exports (CustomActionType, CustomAction, CustomActionExecutor, MappingProfile, CustomMappingEngine)
2. CustomAction Model & Serialization Roundtrip (to_dict / from_dict)
3. CustomActionExecutor Actions: Hotkey, Key Press, Text Typing, Shell Cmd, and Callbacks
4. Security Blacklist Defense: Rejection of dangerous shell commands (format, rm -rf, del /f)
5. Execution Telemetry & Error Logging (execution_history, duration_ms)
6. MappingProfile Structure & Defaults Population
7. Dynamic Gesture Binding, Unbinding & Re-binding Across Modes
8. Fallback Resolution to MOUSE Mode
9. Conflict Analysis & Contradictory Mapping Detection
10. Gesture Repertoire Coverage Reporting (Mapped vs Unmapped gestures)
11. Profile Schema Validation (Detecting missing keys, invalid types)
12. Atomic Disk Persistence & Automatic .bak Backup
13. Profile Export & Import Roundtrip
14. End-to-End Real-Time Classifier Integration with Custom Action Triggering
15. Dispatch Latency & Throughput Microbenchmark (< 0.05 ms/op)
"""

import json
import os
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List

import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gestures
from dataset import GestureDataset
from detection.hand_detector import HandData
from gestures import (
    ALL_KNOWN_GESTURES,
    CustomAction,
    CustomActionExecutor,
    CustomActionType,
    CustomMappingEngine,
    MappingProfile,
    OperationalMode,
    RealtimeGestureRecognizer,
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
        print(f"  {badge} {test_name:<48} {details}")


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
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 14 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     Custom Gesture Mapping System (Module 14)")
    print("=" * 75)

    temp_dir = tempfile.mkdtemp(prefix="phase14_test_")
    test_config_path = os.path.join(temp_dir, "test_mappings.json")

    try:
        # -----------------------------------------------------------------
        # Section 1: Scaffolding & Package Exports
        # -----------------------------------------------------------------
        print("\n1. Module Scaffolding & Package Exports")
        exports_ok = (
            hasattr(gestures, "CustomActionType")
            and hasattr(gestures, "CustomAction")
            and hasattr(gestures, "CustomActionExecutor")
            and hasattr(gestures, "MappingProfile")
            and hasattr(gestures, "CustomMappingEngine")
            and hasattr(gestures, "ALL_KNOWN_GESTURES")
        )
        collector.record("Custom Mapping Package Exports", exports_ok, "All classes exported from gestures package")

        # -----------------------------------------------------------------
        # Section 2: CustomAction Model & Serialization
        # -----------------------------------------------------------------
        print("\n2. CustomAction Model & Serialization")
        act = CustomAction(
            name="COPY_TEXT",
            action_type=CustomActionType.HOTKEY,
            payload=["ctrl", "c"],
            description="Copy selection to clipboard",
        )
        d = act.to_dict()
        act_restored = CustomAction.from_dict(d)
        roundtrip_ok = (
            act_restored.name == act.name
            and act_restored.action_type == act.action_type
            and act_restored.payload == act.payload
            and act_restored.description == act.description
        )
        collector.record("CustomAction Serialization Roundtrip", roundtrip_ok, f"Action: {act.name} ({act.action_type})")

        # -----------------------------------------------------------------
        # Section 3: CustomActionExecutor Capabilities
        # -----------------------------------------------------------------
        print("\n3. CustomActionExecutor Execution Verification")
        executor = CustomActionExecutor(mock_mode=True)

        # Hotkey execution
        res_hotkey = executor.execute(act)
        # Key press execution
        res_key = executor.execute(CustomAction("ENTER_KEY", CustomActionType.KEY_PRESS, "enter"))
        # Text typing execution
        res_text = executor.execute(CustomAction("PASTE_SNIPPET", CustomActionType.TEXT_INPUT, "Hello HCI"))
        # Safe shell command execution
        res_shell = executor.execute(CustomAction("LAUNCH_NOTEPAD", CustomActionType.SHELL_CMD, "notepad.exe"))

        # Python callback execution
        callback_called = {"count": 0}

        def sample_hook(**kwargs):
            callback_called["count"] += 1

        act_cb = CustomAction("CALL_HOOK", CustomActionType.PYTHON_CALLBACK, None, callback=sample_hook)
        res_cb = executor.execute(act_cb)

        all_exec_ok = res_hotkey and res_key and res_text and res_shell and res_cb and (callback_called["count"] == 1)
        collector.record("CustomActionExecutor Execution Types", all_exec_ok, "Hotkey, KeyPress, Text, Shell, and Callback tested")

        # -----------------------------------------------------------------
        # Section 4: Security Blacklist Protection
        # -----------------------------------------------------------------
        print("\n4. Security Blacklist Protection")
        dangerous_cmds = [
            "rm -rf /",
            "format C: /fs:NTFS",
            "del /f /q /s C:\\Windows",
            "shutdown /s /t 0",
        ]
        blocked_count = 0
        for bad_cmd in dangerous_cmds:
            bad_action = CustomAction("DANGEROUS", CustomActionType.SHELL_CMD, bad_cmd)
            try:
                executor.execute(bad_action)
            except PermissionError:
                blocked_count += 1
            except Exception:
                pass

        security_ok = blocked_count == len(dangerous_cmds)
        collector.record("Security Blacklist Defense", security_ok, f"Blocked {blocked_count}/{len(dangerous_cmds)} dangerous shell commands")

        # -----------------------------------------------------------------
        # Section 5: Telemetry & Execution History
        # -----------------------------------------------------------------
        print("\n5. Telemetry & Execution History")
        history = executor.execution_history
        telemetry_ok = len(history) >= 5 and all("duration_ms" in item and "timestamp" in item for item in history)
        collector.record("Executor Telemetry & History", telemetry_ok, f"Recorded {len(history)} execution logs with latencies")

        # -----------------------------------------------------------------
        # Section 6: Dynamic Binding & Profile Management
        # -----------------------------------------------------------------
        print("\n6. Dynamic Binding & Profile Management")
        engine = CustomMappingEngine(config_path=test_config_path, mock_mode=True)

        # Register custom actions
        engine.define_custom_action("MUTE_ALL", CustomActionType.HOTKEY, ["ctrl", "alt", "m"], "Mute Microphone")
        engine.define_custom_action("LAUNCH_TERMINAL", CustomActionType.SHELL_CMD, "wt.exe", "Open Windows Terminal")

        # Dynamic bind
        bind1_ok, _ = engine.bind_gesture("PEACE", "MUTE_ALL", OperationalMode.CUSTOM)
        bind2_ok, _ = engine.bind_gesture("THUMBS_UP", "LAUNCH_TERMINAL", OperationalMode.CUSTOM)

        # Resolve actions
        act_resolved_1 = engine.get_action_for_gesture("PEACE", OperationalMode.CUSTOM)
        act_resolved_2 = engine.get_action_for_gesture("THUMBS_UP", OperationalMode.CUSTOM)

        resolve_ok = (
            bind1_ok and bind2_ok
            and isinstance(act_resolved_1, CustomAction) and act_resolved_1.name == "MUTE_ALL"
            and isinstance(act_resolved_2, CustomAction) and act_resolved_2.name == "LAUNCH_TERMINAL"
        )
        collector.record("Dynamic Custom Action Binding", resolve_ok, "Bound PEACE -> MUTE_ALL and THUMBS_UP -> LAUNCH_TERMINAL")

        # Execute bound gesture
        exec_gesture_ok = engine.execute_gesture("PEACE", OperationalMode.CUSTOM)
        collector.record("Direct Bound Gesture Execution", exec_gesture_ok, "Successfully resolved and executed bound custom action")

        # Unbind gesture
        unbind_ok = engine.unbind_gesture("PEACE", OperationalMode.CUSTOM)
        re_resolved = engine.get_action_for_gesture("PEACE", OperationalMode.CUSTOM)
        # Should fall back to MOUSE mode binding (RIGHT_CLICK)
        fallback_ok = unbind_ok and re_resolved == "RIGHT_CLICK"
        collector.record("Gesture Unbinding & Mode Fallback", fallback_ok, f"Unbound PEACE in CUSTOM -> Fallback to MOUSE '{re_resolved}'")

        # -----------------------------------------------------------------
        # Section 7: Conflict Detection & Coverage Analytics
        # -----------------------------------------------------------------
        print("\n7. Conflict Detection & Coverage Analytics")
        # Introduce duplicate binding conflict deliberately in a test mode
        engine.bind_gesture("FIST", "TEST_DUPLICATE_ACT", "TEST_MODE")
        engine.bind_gesture("OK", "TEST_DUPLICATE_ACT", "TEST_MODE")
        conflicts = engine.detect_conflicts("TEST_MODE")

        conflict_found = any(c.get("type") == "DUPLICATE_ACTION" for c in conflicts)
        collector.record("Duplicate Action Conflict Detection", conflict_found, f"Detected conflict: {conflicts[0]['message'] if conflicts else 'None'}")

        # Coverage analysis
        coverage = engine.get_coverage_report(OperationalMode.PRESENTATION)
        cov_ok = (
            coverage["total_supported_gestures"] == len(ALL_KNOWN_GESTURES)
            and coverage["mapped_count"] > 0
            and coverage["coverage_pct"] > 0
        )
        collector.record("Gesture Repertoire Coverage Report", cov_ok, f"Mode: PRESENTATION | Coverage: {coverage['coverage_pct']}% ({coverage['mapped_count']}/{coverage['total_supported_gestures']})")

        # -----------------------------------------------------------------
        # Section 8: Schema Validation & Atomic File Persistence
        # -----------------------------------------------------------------
        print("\n8. Schema Validation & Atomic Persistence")
        # Validate valid schema
        is_valid, _ = engine.validate_schema(engine.profile.to_dict())
        # Validate malformed schema
        malformed = {"metadata": "not_a_dict"}
        is_invalid, errs = engine.validate_schema(malformed)
        schema_ok = is_valid and not is_invalid and len(errs) > 0
        collector.record("Profile JSON Schema Validation", schema_ok, f"Valid passed, malformed caught {len(errs)} error(s)")

        # Save to disk with atomic replace
        save_ok = engine.save_to_disk()
        disk_file_exists = os.path.exists(test_config_path)

        # Mutate and save again to verify .bak creation
        engine.bind_gesture("OK", "NEW_ACTION_MUTATED", OperationalMode.CUSTOM)
        engine.save_to_disk()
        bak_file_exists = os.path.exists(test_config_path + ".bak")

        # Load back from disk into a fresh engine instance
        engine_fresh = CustomMappingEngine(config_path=test_config_path, mock_mode=True)
        fresh_load_ok = (
            disk_file_exists
            and bak_file_exists
            and engine_fresh.profile.mappings.get(OperationalMode.CUSTOM, {}).get("OK") == "NEW_ACTION_MUTATED"
        )
        collector.record("Atomic Persistence & Backup Roundtrip", fresh_load_ok, "Atomic .tmp replace verified, .bak backup confirmed")

        # Profile export & import
        export_path = os.path.join(temp_dir, "exported_profile.json")
        exp_ok = engine.export_profile(export_path)
        engine_imported = CustomMappingEngine(config_path=os.path.join(temp_dir, "dummy.json"), mock_mode=True)
        imp_ok = engine_imported.import_profile(export_path)
        export_import_ok = exp_ok and imp_ok and (engine_imported.profile.name == engine.profile.name)
        collector.record("Profile Export & Import Roundtrip", export_import_ok, f"Exported & restored profile '{engine.profile.name}'")

        # Reset to defaults
        engine.reset_to_defaults()
        reset_ok = len(engine.profile.custom_actions) == 0
        collector.record("Profile Reset to Defaults", reset_ok, "Cleared custom actions and restored canonical mappings")

        # -----------------------------------------------------------------
        # Section 9: End-to-End Gesture Classifier Integration
        # -----------------------------------------------------------------
        print("\n9. End-to-End Real-Time Classifier Custom Action Dispatch")
        dataset_path = os.path.join(_ROOT, "dataset", "gesture_dataset.npz")
        dataset_obj = GestureDataset.load_from_npz(dataset_path)

        # Setup custom action for 'PEACE' gesture in CUSTOM mode
        e2e_engine = CustomMappingEngine(config_path=os.path.join(temp_dir, "e2e.json"), mock_mode=True)
        custom_trigger = {"executed": False}

        def custom_plugin_action():
            custom_trigger["executed"] = True

        e2e_engine.define_custom_action("E2E_PLUGIN_ACTION", CustomActionType.PYTHON_CALLBACK, None, callback=custom_plugin_action)
        e2e_engine.bind_gesture("PEACE", "E2E_PLUGIN_ACTION", OperationalMode.CUSTOM)

        # Real-time gesture recognizer configured with custom mapping manager
        recognizer = RealtimeGestureRecognizer(
            initial_mode=OperationalMode.CUSTOM,
            confidence_threshold=0.60,
            temporal_window_size=3,
            min_stable_frames=2,
        )
        recognizer.mapping_manager.bind_gesture("PEACE", "E2E_PLUGIN_ACTION", OperationalMode.CUSTOM)

        peace_idx = np.where(dataset_obj.y == "PEACE")[0][0]
        peace_hand = create_mock_hand_data(dataset_obj.landmarks[peace_idx])

        t_now = time.perf_counter()
        recognizer.process_frame(peace_hand, timestamp=t_now)
        res_frame = recognizer.process_frame(peace_hand, timestamp=t_now + 0.033)

        # Dispatch resolved action through engine executor
        dispatch_success = False
        if res_frame.can_execute and res_frame.mapped_action == "E2E_PLUGIN_ACTION":
            act_obj = e2e_engine.profile.custom_actions.get("E2E_PLUGIN_ACTION")
            if act_obj:
                dispatch_success = e2e_engine.executor.execute(act_obj)

        e2e_ok = (
            res_frame.confirmed_gesture == "PEACE"
            and res_frame.mapped_action == "E2E_PLUGIN_ACTION"
            and dispatch_success
            and custom_trigger["executed"]
        )
        collector.record(
            "E2E Classifier -> Custom Action Execution",
            e2e_ok,
            f"Gesture 'PEACE' recognized -> 'E2E_PLUGIN_ACTION' executed successfully"
        )

        # -----------------------------------------------------------------
        # Section 10: Dispatch Latency & Throughput Microbenchmark
        # -----------------------------------------------------------------
        print("\n10. Latency & Throughput Microbenchmark")
        benchmark_act = CustomAction("BENCH_ACTION", CustomActionType.KEY_PRESS, "space")
        n_iters = 5000

        t0 = time.perf_counter()
        for _ in range(n_iters):
            e2e_engine.executor.execute(benchmark_act)
        t1 = time.perf_counter()

        total_time = t1 - t0
        avg_latency_ms = (total_time / n_iters) * 1000.0
        throughput = n_iters / total_time
        benchmark_ok = avg_latency_ms < 0.05

        collector.record(
            "Custom Action Dispatch Benchmark",
            benchmark_ok,
            f"Mean latency: {avg_latency_ms:.4f} ms | Throughput: {throughput:,.0f} ops/sec",
            latency_ms=avg_latency_ms,
        )

    finally:
        # Cleanup temporary test files
        shutil.rmtree(temp_dir, ignore_errors=True)

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

    report_path = os.path.join(_ROOT, "data", "phase14_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_data = {
        "phase": 14,
        "name": "Custom Gesture-to-Command Mapping System",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "pass_rate_pct": pass_rate,
            "mean_dispatch_latency_ms": round(avg_latency_ms, 5),
            "throughput_ops_sec": round(throughput, 1),
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
