"""
Phase 15 Verification Test Suite: Voice + Gesture Multimodal Control Subsystem (Module 15)
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Validates:
1. Module Scaffolding & Package Exports (SpeechEngine, IntentClassifier, MultimodalFusionEngine, ConfirmationGateState)
2. SpeechEngine Initialization, Audio Simulation & Text-to-Speech Feedback
3. IntentClassifier Accuracy Across Canonical Categories:
   - Operational Mode Transitions (Mouse, System, Media, Browser, Presentation)
   - Hardware Controls (Volume, Brightness, Screenshot, Lock Screen)
   - Media Playback Controls (Play/Pause, Next Track, Fullscreen)
   - Web Navigation Controls (Scroll Down, History Back, New Tab)
   - Presentation Slideshow Controls (Next Slide, Start Slideshow, Laser)
   - Confirmation Gate Responses (Confirm, Cancel, Abort)
   - Text Dictation Parsing ("type <message>")
4. Multimodal Fusion Engine State & Controller Integration
5. Voice-Assisted Mode Switching (Voice command dynamically updates RealtimeGestureRecognizer mode)
6. Safety Confirmation Gate State Machine:
   - Triggering critical action (LOCK_SCREEN) engages AWAITING_CONFIRMATION
   - Spoken "confirm" validates confirmation and executes action
   - Spoken "cancel" cleanly cancels pending action
   - Temporal timeout expiration auto-aborts pending action
7. Compound Multimodal Interaction: Hand Pointing + Spoken Click ("click here" at pointer coordinate)
8. Direct Cross-Controller Voice Command Dispatch (System, Media, Presentation, Browser)
9. Spoken Dictation Automated Typing
10. Latency & Throughput Microbenchmark (< 2.0 ms per fusion cycle)
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

from controllers import (
    BrowserController,
    MediaController,
    PresentationController,
    SystemController,
    VirtualMouseController,
)
from gestures import GestureRecognitionResult, OperationalMode, RealtimeGestureRecognizer
import voice
from voice import (
    CRITICAL_ACTIONS,
    ConfirmationGateState,
    IntentClassifier,
    MultimodalFusionEngine,
    SpeechEngine,
    VoiceIntentResult,
    VoiceIntentType,
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


def run_tests() -> bool:
    print("=" * 75)
    print(" AI-BASED VISION-DRIVEN HCI - PHASE 15 VERIFICATION TEST")
    print("=" * 75)
    print(f" Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Platform:   {platform.platform()}")
    print(" Target:     Voice + Gesture Multimodal Control (Module 15)")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Section 1: Scaffolding & Package Exports
    # -----------------------------------------------------------------
    print("\n1. Module Scaffolding & Package Exports")
    exports_ok = (
        hasattr(voice, "SpeechEngine")
        and hasattr(voice, "IntentClassifier")
        and hasattr(voice, "VoiceIntentType")
        and hasattr(voice, "VoiceIntentResult")
        and hasattr(voice, "MultimodalFusionEngine")
        and hasattr(voice, "ConfirmationGateState")
        and hasattr(voice, "CRITICAL_ACTIONS")
    )
    collector.record("Voice Package Exports", exports_ok, "All speech, intent, and fusion classes exported")

    # -----------------------------------------------------------------
    # Section 2: SpeechEngine Initialization & Simulation
    # -----------------------------------------------------------------
    print("\n2. SpeechEngine Audio Simulation & TTS Synthesis")
    speech_eng = SpeechEngine(mock_mode=True)
    tts_ok = speech_eng.speak("Testing Text-to-Speech synthesis feedback")

    # Audio simulation
    sim_transcript = speech_eng.simulate_speech("switch to presentation mode")
    history_ok = len(speech_eng.recognition_history) == 1 and speech_eng.last_transcript == "switch to presentation mode"
    collector.record("SpeechEngine Audio Simulation & TTS", tts_ok and history_ok, f"Simulated: \"{sim_transcript}\" | TTS: {tts_ok}")

    # -----------------------------------------------------------------
    # Section 3: Spoken Intent Classification Accuracy
    # -----------------------------------------------------------------
    print("\n3. Spoken Intent Classification Accuracy")
    classifier = IntentClassifier()

    test_phrases = [
        # (Input phrase, Expected Intent Type, Expected Canonical Action, Expected Parameter Key, Expected Parameter Val)
        ("switch to presentation mode", VoiceIntentType.MODE_SWITCH, "SWITCH_MODE", "target_mode", OperationalMode.PRESENTATION),
        ("browser mode", VoiceIntentType.MODE_SWITCH, "SWITCH_MODE", "target_mode", OperationalMode.BROWSER),
        ("media mode", VoiceIntentType.MODE_SWITCH, "SWITCH_MODE", "target_mode", OperationalMode.MEDIA),
        ("volume up", VoiceIntentType.SYSTEM_CONTROL, "VOLUME_UP", None, None),
        ("mute volume", VoiceIntentType.SYSTEM_CONTROL, "MUTE", None, None),
        ("take screenshot", VoiceIntentType.SYSTEM_CONTROL, "SCREENSHOT", None, None),
        ("lock screen", VoiceIntentType.SYSTEM_CONTROL, "LOCK_SCREEN", None, None),
        ("play music", VoiceIntentType.MEDIA_CONTROL, "PLAY_PAUSE", None, None),
        ("next track", VoiceIntentType.MEDIA_CONTROL, "NEXT_TRACK", None, None),
        ("scroll down", VoiceIntentType.BROWSER_CONTROL, "SCROLL_DOWN", None, None),
        ("new tab", VoiceIntentType.BROWSER_CONTROL, "NEW_TAB", None, None),
        ("next slide", VoiceIntentType.PRESENTATION_CONTROL, "NEXT_SLIDE", None, None),
        ("start slideshow", VoiceIntentType.PRESENTATION_CONTROL, "START_PRESENTATION", None, None),
        ("laser pointer", VoiceIntentType.PRESENTATION_CONTROL, "LASER_POINTER", None, None),
        ("confirm", VoiceIntentType.CONFIRMATION, "CONFIRM_ACTION", None, None),
        ("cancel", VoiceIntentType.CONFIRMATION, "CANCEL_ACTION", None, None),
        ("click here", VoiceIntentType.COMPOUND_ACTION, "LEFT_CLICK", None, None),
        ("type meeting agenda", VoiceIntentType.DICTATION, "TYPE_TEXT", "text", "meeting agenda"),
    ]

    correct_classifications = 0
    for phrase, exp_type, exp_act, exp_param_k, exp_param_v in test_phrases:
        res = classifier.classify(phrase)
        type_match = res.intent_type == exp_type
        action_match = res.canonical_action == exp_act
        param_match = True
        if exp_param_k:
            param_match = res.parameters.get(exp_param_k) == exp_param_v

        if type_match and action_match and param_match:
            correct_classifications += 1
        else:
            print(f"    MISMATCH on '{phrase}': got {res.intent_type}:{res.canonical_action}, expected {exp_type}:{exp_act}")

    classification_acc = (correct_classifications / len(test_phrases)) * 100.0
    collector.record(
        "Intent Classification Accuracy",
        correct_classifications == len(test_phrases),
        f"Passed {correct_classifications}/{len(test_phrases)} phrases ({classification_acc:.1f}%)"
    )

    # -----------------------------------------------------------------
    # Section 4: Multimodal Fusion Engine Initialization
    # -----------------------------------------------------------------
    print("\n4. Multimodal Fusion Engine Setup")
    recognizer = RealtimeGestureRecognizer(initial_mode=OperationalMode.MOUSE)
    mouse_ctrl = VirtualMouseController(mock_mode=True)
    sys_ctrl = SystemController(mock_mode=True)
    media_ctrl = MediaController(mock_mode=True)
    browser_ctrl = BrowserController(mock_mode=True)
    pres_ctrl = PresentationController(mock_mode=True)

    fusion = MultimodalFusionEngine(
        speech_engine=speech_eng,
        intent_classifier=classifier,
        gesture_recognizer=recognizer,
        mouse_controller=mouse_ctrl,
        system_controller=sys_ctrl,
        media_controller=media_ctrl,
        browser_controller=browser_ctrl,
        presentation_controller=pres_ctrl,
        coincidence_window_s=1.5,
        confirmation_timeout_s=2.0,
        mock_mode=True,
    )

    init_status = fusion.get_status()
    setup_ok = (
        init_status["active_mode"] == OperationalMode.MOUSE
        and init_status["gate_state"] == ConfirmationGateState.IDLE
        and init_status["mock_mode"] is True
    )
    collector.record("Multimodal Fusion Engine Initialization", setup_ok, f"Mode: {init_status['active_mode']} | Gate: {init_status['gate_state']}")

    # -----------------------------------------------------------------
    # Section 5: Voice-Assisted Mode Switching
    # -----------------------------------------------------------------
    print("\n5. Voice-Assisted Operational Mode Switching")
    ev_mode = fusion.feed_voice_transcript("switch to presentation mode")
    mode_switch_ok = (
        fusion.active_mode == OperationalMode.PRESENTATION
        and recognizer.get_mode() == OperationalMode.PRESENTATION
        and ev_mode["disposition"] == "MODE_SWITCHED_PRESENTATION"
    )
    collector.record("Voice Mode Switch -> Presentation", mode_switch_ok, f"Engine Mode: {fusion.active_mode} | Recognizer Mode: {recognizer.get_mode()}")

    # Switch back to System mode
    ev_sys = fusion.feed_voice_transcript("system mode")
    mode_sys_ok = (
        fusion.active_mode == OperationalMode.SYSTEM
        and recognizer.get_mode() == OperationalMode.SYSTEM
    )
    collector.record("Voice Mode Switch -> System", mode_sys_ok, f"Engine Mode: {fusion.active_mode} | Recognizer Mode: {recognizer.get_mode()}")

    # -----------------------------------------------------------------
    # Section 6: Safety Confirmation Gate State Machine
    # -----------------------------------------------------------------
    print("\n6. Safety Confirmation Gate State Machine (Critical Actions)")
    # Scenario A: Gesture triggers LOCK_SCREEN -> Gate Engaged -> Voice "confirm" -> Executed
    t_gate_start = time.perf_counter()
    critical_gesture_res = GestureRecognitionResult(
        raw_gesture="OPEN_PALM",
        confirmed_gesture="OPEN_PALM",
        confidence=0.92,
        is_confident=True,
        is_stabilized=True,
        active_mode=OperationalMode.SYSTEM,
        mapped_action="LOCK_SCREEN",
        can_execute=True,
    )

    ev_gate = fusion.feed_gesture(critical_gesture_res, timestamp=t_gate_start)
    gate_engaged = (
        fusion.gate_state == ConfirmationGateState.AWAITING_CONFIRMATION
        and fusion.pending_action == "LOCK_SCREEN"
        and ev_gate["disposition"] == "GATE_PENDING_CONFIRMATION"
    )
    collector.record("Confirmation Gate Engagement (Lock Screen)", gate_engaged, f"Gate State: {fusion.gate_state} | Pending: {fusion.pending_action}")

    # Confirm via voice
    ev_confirm = fusion.feed_voice_transcript("yes confirm", timestamp=t_gate_start + 0.5)
    confirmed_ok = (
        fusion.gate_state == ConfirmationGateState.IDLE
        and fusion.pending_action is None
        and "CONFIRMED_AND_EXECUTED_LOCK_SCREEN" in ev_confirm["disposition"]
        and sys_ctrl.last_action == "LOCK_SCREEN"
    )
    collector.record("Voice Confirmation Execution", confirmed_ok, f"Disposition: {ev_confirm['disposition']} | Last Action: {sys_ctrl.last_action}")

    # Scenario B: Critical Action -> Spoken "cancel" -> Aborted
    t_cancel = time.perf_counter()
    fusion.feed_gesture(critical_gesture_res, timestamp=t_cancel)
    ev_cancel = fusion.feed_voice_transcript("cancel action", timestamp=t_cancel + 0.3)
    cancelled_ok = (
        fusion.gate_state == ConfirmationGateState.IDLE
        and fusion.pending_action is None
        and "CANCELLED_LOCK_SCREEN" in ev_cancel["disposition"]
    )
    collector.record("Voice Cancellation Safe Abort", cancelled_ok, f"Disposition: {ev_cancel['disposition']}")

    # Scenario C: Critical Action -> Timeout Expiration -> Aborted
    t_timeout = time.perf_counter()
    fusion.feed_gesture(critical_gesture_res, timestamp=t_timeout)
    # Feed an event well past the 2.0s confirmation timeout
    fusion.feed_voice_transcript("what is the weather", timestamp=t_timeout + 3.0)
    timeout_ok = (
        fusion.gate_state == ConfirmationGateState.IDLE
        and fusion.pending_action is None
    )
    collector.record("Confirmation Gate Timeout Expiration", timeout_ok, f"Auto-cleared expired critical action safely")

    # -----------------------------------------------------------------
    # Section 7: Compound Multimodal Interaction (Point + Voice)
    # -----------------------------------------------------------------
    print("\n7. Compound Multimodal Interaction (Spatial Hand + Spoken Action)")
    t_point = time.perf_counter()
    point_gesture_res = GestureRecognitionResult(
        raw_gesture="POINT",
        confirmed_gesture="POINT",
        confidence=0.96,
        is_confident=True,
        is_stabilized=True,
        active_mode=OperationalMode.MOUSE,
        mapped_action="CURSOR_MOVE",
        can_execute=True,
        tracking_point=(520.0, 340.0),
    )
    fusion.feed_gesture(point_gesture_res, timestamp=t_point)

    # Within coincidence window (0.4s later), user speaks "click here"
    mouse_ctrl.last_action = "NONE"
    ev_compound = fusion.feed_voice_transcript("click here", timestamp=t_point + 0.4)
    compound_ok = (
        "COMPOUND_LEFT_CLICK" in ev_compound["disposition"]
        and mouse_ctrl.last_action == "LEFT_CLICK"
    )
    collector.record("Compound Spatial Point + Voice Click", compound_ok, f"Disposition: {ev_compound['disposition']} | Mouse Action: {mouse_ctrl.last_action}")

    # -----------------------------------------------------------------
    # Section 8: Direct Controller Voice Execution
    # -----------------------------------------------------------------
    print("\n8. Direct Cross-Controller Voice Command Dispatch")
    # System volume
    ev_vol = fusion.feed_voice_transcript("volume up")
    vol_ok = "VOLUME_UP" in ev_vol["disposition"] and sys_ctrl.last_action == "VOLUME_UP"

    # Presentation next slide
    ev_slide = fusion.feed_voice_transcript("next slide")
    slide_ok = "NEXT_SLIDE" in ev_slide["disposition"] and pres_ctrl.last_action == "NEXT_SLIDE"

    # Media next track
    ev_track = fusion.feed_voice_transcript("next track")
    track_ok = "NEXT_TRACK" in ev_track["disposition"] and media_ctrl.last_action == "NEXT_TRACK"

    # Browser scroll down
    ev_scroll = fusion.feed_voice_transcript("scroll down")
    scroll_ok = "SCROLL_DOWN" in ev_scroll["disposition"] and browser_ctrl.last_action == "SCROLL_DOWN"

    direct_ok = vol_ok and slide_ok and track_ok and scroll_ok
    collector.record("Direct Controller Voice Dispatch", direct_ok, "Dispatched VolumeUp, NextSlide, NextTrack, ScrollDown")

    # -----------------------------------------------------------------
    # Section 9: Spoken Dictation Automated Typing
    # -----------------------------------------------------------------
    print("\n9. Spoken Dictation Automated Typing")
    ev_dict = fusion.feed_voice_transcript("type Hello HCI World")
    dict_ok = ev_dict["disposition"] == "DICTATION_TYPED" and ev_dict["action"] == "TYPE_TEXT"
    collector.record("Spoken Dictation Typing Dispatch", dict_ok, f"Disposition: {ev_dict['disposition']}")

    # -----------------------------------------------------------------
    # Section 10: Fusion Latency & Throughput Microbenchmark
    # -----------------------------------------------------------------
    print("\n10. Latency & Throughput Microbenchmark")
    iterations = 5000
    t0 = time.perf_counter()
    for _ in range(iterations):
        fusion.feed_voice_transcript("next slide")
    t1 = time.perf_counter()

    total_time = t1 - t0
    avg_latency_ms = (total_time / iterations) * 1000.0
    throughput = iterations / total_time
    benchmark_ok = avg_latency_ms < 2.0

    collector.record(
        "Multimodal Fusion Latency Benchmark",
        benchmark_ok,
        f"Mean latency: {avg_latency_ms:.4f} ms | Throughput: {throughput:,.0f} ops/sec",
        latency_ms=avg_latency_ms,
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

    report_path = os.path.join(_ROOT, "data", "phase15_test_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report_data = {
        "phase": 15,
        "name": "Voice + Gesture Multimodal Control Subsystem",
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
