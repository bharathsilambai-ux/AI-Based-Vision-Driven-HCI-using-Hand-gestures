"""
Phase 15: Speech Recognition & Text-to-Speech Engine Subsystem (Module 15)
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Speech Recognition (SR):
   - Wraps speech_recognition.Recognizer with configurable energy threshold, dynamic ambient noise
     adjustment, and timeout control.
   - Full mock_mode support for deterministic, headless unit and integration testing without microphone hardware.
   - Simulated speech injection API (simulate_speech) for reproducible testing pipelines.

2. Text-to-Speech (TTS):
   - Auditory feedback using pyttsx3 (speech synthesizer).
   - Non-blocking auditory alerts (e.g. "Mode switched to presentation", "Lock screen confirmed").
   - Thread-safe TTS with mock mode bypass to prevent audio device lockups in CI.

3. Telemetry & History:
   - Tracks speech recognition events, confidence, latency, and transcribed audio logs.
"""

import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import speech_recognition as sr
from utils.logger import logger

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except Exception:
    HAS_PYTTSX3 = False


class SpeechEngine:
    """
    Subsystem for capturing spoken commands, performing Speech-to-Text (STT),
    and providing Text-to-Speech (TTS) auditory feedback.
    """

    def __init__(
        self,
        mock_mode: bool = False,
        energy_threshold: int = 300,
        pause_threshold: float = 0.8,
        dynamic_energy_threshold: bool = True,
        enable_tts: bool = True,
    ):
        self.mock_mode = mock_mode
        self.enable_tts = enable_tts and HAS_PYTTSX3

        # Speech Recognizer Configuration
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = dynamic_energy_threshold

        # Text-to-Speech Synthesizer
        self._tts_engine = None
        if self.enable_tts and not self.mock_mode:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty("rate", 175)  # Words per minute
                self._tts_engine.setProperty("volume", 0.9)
            except Exception as e:
                logger.warning(f"[SpeechEngine] Failed to initialize pyttsx3 TTS: {e}")
                self.enable_tts = False

        self.last_transcript: str = ""
        self.last_audio_duration_s: float = 0.0
        self.recognition_history: List[Dict[str, Any]] = []
        self._is_listening: bool = False

    def speak(self, text: str, async_mode: bool = True) -> bool:
        """
        Synthesizes speech output for auditory user feedback.

        Args:
            text: Message string to speak.
            async_mode: If True, executes speech in background thread to avoid blocking.
        """
        if not text or not text.strip():
            return False

        msg = text.strip()
        logger.info(f"[SpeechEngine TTS] Speaking: \"{msg}\"")

        if self.mock_mode or not self.enable_tts or self._tts_engine is None:
            return True

        def _run_tts():
            try:
                self._tts_engine.say(msg)
                self._tts_engine.runAndWait()
            except Exception as e:
                logger.warning(f"[SpeechEngine TTS] Speech synthesis failed: {e}")

        if async_mode:
            t = threading.Thread(target=_run_tts, daemon=True)
            t.start()
        else:
            _run_tts()

        return True

    def simulate_speech(self, text: str, confidence: float = 0.95) -> str:
        """
        Injects a simulated speech phrase directly into the engine.
        Essential for automated test suites, CI testing, and offline evaluation.
        """
        t0 = time.perf_counter()
        normalized = text.lower().strip()
        self.last_transcript = normalized

        duration_ms = (time.perf_counter() - t0) * 1000.0
        self.recognition_history.append({
            "timestamp": time.time(),
            "transcript": normalized,
            "confidence": confidence,
            "simulated": True,
            "latency_ms": round(duration_ms, 3),
        })

        logger.info(f"[SpeechEngine] Simulated transcript: \"{normalized}\" (confidence={confidence:.2f})")
        return normalized

    def listen_and_recognize(self, timeout: float = 3.0, phrase_time_limit: float = 4.0) -> Optional[str]:
        """
        Captures microphone audio and performs speech recognition.
        In mock_mode, returns the last simulated transcript.
        """
        if self.mock_mode:
            return self.last_transcript if self.last_transcript else None

        t0 = time.perf_counter()
        try:
            with sr.Microphone() as source:
                logger.info("[SpeechEngine] Listening for voice command...")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )

            # Recognize using Sphinx (offline) or Google Web Speech API (online)
            try:
                transcript = self.recognizer.recognize_google(audio)
            except sr.RequestError:
                # Fallback to offline Sphinx if available
                transcript = self.recognizer.recognize_sphinx(audio)

            normalized = transcript.lower().strip()
            self.last_transcript = normalized
            latency_ms = (time.perf_counter() - t0) * 1000.0

            self.recognition_history.append({
                "timestamp": time.time(),
                "transcript": normalized,
                "confidence": 0.90,
                "simulated": False,
                "latency_ms": round(latency_ms, 3),
            })

            logger.info(f"[SpeechEngine] Recognized: \"{normalized}\" ({latency_ms:.1f} ms)")
            return normalized

        except sr.WaitTimeoutError:
            logger.debug("[SpeechEngine] Listening timed out (no speech detected).")
            return None
        except sr.UnknownValueError:
            logger.debug("[SpeechEngine] Speech detected but could not be understood.")
            return None
        except Exception as e:
            logger.error(f"[SpeechEngine] Speech recognition error: {e}")
            return None
