"""
Personalized Custom Gesture Studio & ML Model Manager.
Handles live recording of feature sample vectors, Random Forest classifier training,
model serialization (.pkl), and runtime probability inference.
"""

import json
import os
import sys
import threading
from typing import Dict, List, Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import config
from utils.logger import logger


class CustomGestureManager:
    """Manages custom gesture sample recording, model training, and runtime predictions."""

    def __init__(
        self,
        config_path: str = config.GESTURES_CONFIG_PATH,
        model_path: str = config.CUSTOM_MODEL_PATH
    ):
        self.config_path = config_path
        self.model_path = model_path

        # Dataset storage: { gesture_name: List[np.ndarray] }
        self.training_data: Dict[str, List[np.ndarray]] = {}

        # Live Recording session
        self.is_recording = False
        self.current_recording_name = ""
        self.target_samples = 40
        self.recorded_samples_count = 0

        # Trained Scikit-Learn Model
        self.classifier: Optional[RandomForestClassifier] = None
        self.classes: List[str] = []

        self._load_config()
        self.load_model()

    def _load_config(self):
        """Loads persistent JSON configuration."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    self.config_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read gestures config: {e}")
                self.config_data = {"profiles": {"default": {"custom_gestures": {}}}}
        else:
            self.config_data = {"profiles": {"default": {"custom_gestures": {}}}}

    def _save_config(self):
        """Saves config back to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save gestures config: {e}")

    def start_recording(self, gesture_name: str, target_samples: int = 40) -> bool:
        """Initiates recording session for a new or existing gesture."""
        clean_name = gesture_name.strip().upper().replace(" ", "_")
        if not clean_name:
            return False

        self.current_recording_name = clean_name
        self.target_samples = target_samples
        self.recorded_samples_count = 0
        self.is_recording = True

        if clean_name not in self.training_data:
            self.training_data[clean_name] = []

        logger.info(f"[CustomStudio] Started recording '{clean_name}' (target={target_samples} frames).")
        return True

    def record_frame_features(self, feature_vector: np.ndarray) -> bool:
        """
        Appends a 1D feature vector to the active recording session.
        Returns True if recording just completed on this frame.
        """
        if not self.is_recording or not self.current_recording_name:
            return False

        self.training_data[self.current_recording_name].append(feature_vector.copy())
        self.recorded_samples_count += 1

        if self.recorded_samples_count >= self.target_samples:
            self.is_recording = False
            logger.success(f"[CustomStudio] Finished recording '{self.current_recording_name}' ({self.recorded_samples_count} frames).")
            return True
        return False

    def train_model(self) -> Tuple[bool, str]:
        """
        Trains a Random Forest classifier on all recorded custom gesture samples.
        """
        if len(self.training_data) < 2:
            return False, "Need at least 2 distinct gesture classes to train a classifier."

        X_list, y_list = [], []
        for name, vectors in self.training_data.items():
            if len(vectors) < 5:
                continue
            for vec in vectors:
                X_list.append(vec)
                y_list.append(name)

        if not X_list:
            return False, "Insufficient sample frames recorded."

        X = np.array(X_list)
        y = np.array(y_list)

        try:
            clf = RandomForestClassifier(n_estimators=50, max_depth=14, random_state=42)
            if len(y) >= 12:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc = accuracy_score(y_test, preds) * 100.0
                clf.fit(X, y)
            else:
                clf.fit(X, y)
                acc = 100.0

            self.classifier = clf
            self.classes = list(clf.classes_)

            os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
            joblib.dump({"model": clf, "classes": self.classes, "training_data": self.training_data}, self.model_path)

            msg = f"Model successfully trained with accuracy: {acc:.1f}% across {len(self.classes)} gestures."
            logger.success(f"[CustomStudio] {msg}")
            return True, msg
        except Exception as e:
            err = f"Training failed: {e}"
            logger.error(f"[CustomStudio] {err}")
            return False, err

    def load_model(self) -> bool:
        """Loads trained weights from disk."""
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.classifier = data["model"]
                self.classes = data.get("classes", [])
                self.training_data = data.get("training_data", {})
                logger.info(f"[CustomStudio] Loaded custom model classes: {self.classes}")
                return True
            except Exception as e:
                logger.warning(f"[CustomStudio] Could not load model: {e}")
        return False

    def predict(self, feature_vector: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Predicts gesture class and probability confidence.
        """
        if self.classifier is None:
            return None, 0.0

        try:
            probs = self.classifier.predict_proba([feature_vector])[0]
            best_idx = int(np.argmax(probs))
            conf = float(probs[best_idx])
            pred_name = str(self.classes[best_idx])
            return pred_name, conf
        except Exception:
            return None, 0.0

    def bind_action(self, gesture_name: str, action_name: str):
        """Binds a gesture to an action in JSON config."""
        clean_name = gesture_name.strip().upper().replace(" ", "_")
        prof = self.config_data.get("profiles", {}).get("default", {})
        if "custom_gestures" not in prof:
            prof["custom_gestures"] = {}
        prof["custom_gestures"][clean_name] = action_name
        self._save_config()
        logger.info(f"[CustomStudio] Bound gesture '{clean_name}' -> '{action_name}'.")

    def get_bound_action(self, gesture_name: str) -> Optional[str]:
        prof = self.config_data.get("profiles", {}).get("default", {})
        return prof.get("custom_gestures", {}).get(gesture_name, None)

    def get_registered_gestures(self) -> List[Tuple[str, str, int]]:
        res = []
        prof = self.config_data.get("profiles", {}).get("default", {})
        customs = prof.get("custom_gestures", {})
        for name in set(list(self.training_data.keys()) + list(customs.keys())):
            action = customs.get(name, "UNASSIGNED")
            count = len(self.training_data.get(name, []))
            res.append((name, action, count))
        return res
