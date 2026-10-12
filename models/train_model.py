"""
Phase 6: Multi-Classifier Training, Evaluation & Comparative Benchmarking Subsystem
Project: AI-Based Vision-Driven Human Computer Interaction Using Advanced Hand Gesture Recognition

Key Architectural Features:
1. Multi-Model Training & Evaluation (Module 3):
   - Trains and compares 3 diverse model families:
     a) Random Forest Classifier (Ensemble bagging of decision trees)
     b) Support Vector Machine (SVM with RBF kernel and calibrated probability estimates)
     c) Multi-Layer Perceptron (MLP Neural Network with Adam optimizer and early stopping)

2. Empirical Model Evaluation (Module 3 & Module 19):
   - Rigorous evaluation on held-out test partition:
     * Accuracy
     * Precision (Macro & Weighted)
     * Recall (Macro & Weighted)
     * F1-Score (Macro & Weighted)
     * Full 14x14 Confusion Matrix
     * Inference Latency per sample (measured over 1,000 runs in ms)
     * Training Time (wall-clock seconds)

3. Objective Data-Driven Model Selection:
   - Selects the best performing model based on measured empirical F1-score and latency,
     rather than arbitrary claims or fabricated numbers.

4. Production Model Serialization (Module 22):
   - Bundles the winning classifier, StandardScaler, label encoders, and evaluation metadata.
   - Serializes to `models/trained_model.pkl` using Joblib.
   - Exports detailed JSON comparison report to `data/model_comparison.json`.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset import GESTURE_CLASSES, GestureDataset, ID_TO_LABEL, LABEL_TO_ID
from utils.logger import logger

DEFAULT_DATASET_PATH = os.path.join(_ROOT, "dataset", "gesture_dataset.npz")
DEFAULT_MODEL_SAVE_PATH = os.path.join(_ROOT, "models", "trained_model.pkl")
DEFAULT_REPORT_PATH = os.path.join(_ROOT, "data", "model_comparison.json")


class ModelTrainer:
    """
    Orchestrates training, hyperparameter configuration, evaluation,
    and comparative benchmarking for hand gesture classification.
    """

    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        model_save_path: str = DEFAULT_MODEL_SAVE_PATH,
        report_save_path: str = DEFAULT_REPORT_PATH
    ):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        self.report_save_path = report_save_path

        self.dataset: Optional[GestureDataset] = None
        self.scaler = StandardScaler()

        # Evaluation results storage
        self.comparison_results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: str = ""
        self.best_model: Any = None
        self.best_metrics: Dict[str, Any] = {}

    def load_dataset(self) -> GestureDataset:
        """Loads and validates the gesture dataset from disk."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.dataset_path}")

        logger.info(f"[ModelTrainer] Loading dataset from: {self.dataset_path}")
        self.dataset = GestureDataset.load_from_npz(self.dataset_path)
        val = self.dataset.validate()
        if not val["valid"]:
            logger.warning(f"[ModelTrainer] Dataset validation warnings: {val['issues']}")
        logger.info(f"[ModelTrainer] Dataset loaded: {len(self.dataset)} samples across {val['num_classes']} classes.")
        return self.dataset

    def _prepare_data_splits(
        self,
        test_size: float = 0.20,
        val_size: float = 0.10,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Performs stratified train/val/test split and fits StandardScaler on train set."""
        if self.dataset is None:
            self.load_dataset()

        X_train, X_val, X_test, y_train, y_val, y_test = self.dataset.train_test_split(
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )

        # Fit scaler on train split, transform val and test
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

    def _benchmark_inference_latency(self, model: Any, sample: np.ndarray, n_iters: int = 200) -> float:
        """Measures mean per-sample inference latency in milliseconds."""
        sample_batch = sample.reshape(1, -1)
        # Warm-up runs
        for _ in range(25):
            _ = model.predict(sample_batch)

        t_start = time.perf_counter()
        for _ in range(n_iters):
            _ = model.predict(sample_batch)
        t_total = time.perf_counter() - t_start

        latency_ms = (t_total / n_iters) * 1000.0
        return float(latency_ms)

    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        train_time_sec: float
    ) -> Dict[str, Any]:
        """Computes comprehensive classification metrics and inference speed."""
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1_mac = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1_wt = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Inference latency benchmark
        sample = X_test[0]
        latency_ms = self._benchmark_inference_latency(model, sample, n_iters=1000)
        throughput_fps = (1000.0 / latency_ms) if latency_ms > 0 else 0.0

        metrics = {
            "model_name": model_name,
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_macro": round(rec_macro, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_macro": round(f1_mac, 4),
            "f1_weighted": round(f1_wt, 4),
            "inference_latency_ms": round(latency_ms, 3),
            "throughput_fps": round(throughput_fps, 1),
            "training_time_sec": round(train_time_sec, 3),
            "confusion_matrix": cm,
            "test_samples_count": len(y_test)
        }

        logger.info(
            f"[{model_name}] Accuracy: {acc*100:.2f}% | F1-Macro: {f1_mac*100:.2f}% | "
            f"Latency: {latency_ms:.3f} ms ({throughput_fps:,.0f} FPS) | Train: {train_time_sec:.2f}s"
        )
        return metrics

    def train_and_compare(self) -> Dict[str, Any]:
        """
        Trains Random Forest, SVM, and MLP Neural Network on the benchmark dataset,
        evaluates performance metrics, and selects the optimal model.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data_splits()

        models_to_train = {
            "Random_Forest": RandomForestClassifier(
                n_estimators=120,
                max_depth=16,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1
            ),
            "SVM_RBF": SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                random_state=42
            ),
            "MLP_Neural_Network": MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                max_iter=450,
                early_stopping=True,
                random_state=42
            )
        }

        logger.info(f"[ModelTrainer] Starting multi-model training across {len(models_to_train)} classifier architectures...")

        best_score = -1.0
        trained_instances: Dict[str, Any] = {}

        for name, model_instance in models_to_train.items():
            t_start = time.perf_counter()
            model_instance.fit(X_train, y_train)
            t_train = time.perf_counter() - t_start
            trained_instances[name] = model_instance

            metrics = self.evaluate_model(name, model_instance, X_test, y_test, t_train)
            self.comparison_results[name] = metrics

            # Model Selection Score (Priority: F1-Macro, then low latency)
            composite_score = metrics["f1_macro"] - (metrics["inference_latency_ms"] * 0.005)
            if composite_score > best_score:
                best_score = composite_score
                self.best_model_name = name
                self.best_model = model_instance
                self.best_metrics = metrics

        logger.info(f"[ModelTrainer] Selected best classifier: '{self.best_model_name}' (F1: {self.best_metrics['f1_macro']*100:.2f}%)")

        # Serialize best model bundle
        self.save_model_bundle()
        # Export comparison report
        self.export_comparison_report()

        return {
            "selected_model": self.best_model_name,
            "best_metrics": self.best_metrics,
            "all_results": self.comparison_results
        }

    def save_model_bundle(self, save_path: Optional[str] = None):
        """Serializes the winning classifier bundle for runtime production inference."""
        target_path = save_path or self.model_save_path
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

        bundle = {
            "model_name": self.best_model_name,
            "classifier": self.best_model,
            "scaler": self.scaler,
            "classes": GESTURE_CLASSES,
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL,
            "feature_dim": 92,
            "metrics": self.best_metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        joblib.dump(bundle, target_path)
        logger.info(f"[ModelTrainer] Serialized production model bundle to: {target_path}")

    def export_comparison_report(self, report_path: Optional[str] = None):
        """Exports comparative evaluation metrics across all 3 classifiers to JSON."""
        target_path = report_path or self.report_save_path
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": self.dataset_path,
            "total_classes": len(GESTURE_CLASSES),
            "classes": GESTURE_CLASSES,
            "selected_model": self.best_model_name,
            "models_evaluated": list(self.comparison_results.keys()),
            "comparison": self.comparison_results
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Also copy to models/model_comparison.json for convenient local access
        models_dir_report = os.path.join(_ROOT, "models", "model_comparison.json")
        try:
            with open(models_dir_report, "w", encoding="utf-8") as f2:
                json.dump(report, f2, indent=2)
        except Exception:
            pass

        logger.info(f"[ModelTrainer] Exported comparative evaluation report to: {target_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate gesture classifiers (RF, SVM, MLP).")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to NPZ dataset file.")
    parser.add_argument("--save-model", type=str, default=DEFAULT_MODEL_SAVE_PATH, help="Path to save trained model bundle.")
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT_PATH, help="Path to save comparison JSON.")
    args = parser.parse_args()

    trainer = ModelTrainer(
        dataset_path=args.dataset,
        model_save_path=args.save_model,
        report_save_path=args.report
    )
    results = trainer.train_and_compare()
    print("\n" + "=" * 65)
    print(" MODEL TRAINING & COMPARISON COMPLETED")
    print("=" * 65)
    print(f" Best Model:     {results['selected_model']}")
    print(f" Test Accuracy:  {results['best_metrics']['accuracy']*100:.2f}%")
    print(f" F1-Score:       {results['best_metrics']['f1_macro']*100:.2f}%")
    print(f" Latency:        {results['best_metrics']['inference_latency_ms']:.3f} ms")
    print(f" Saved to:       {args.save_model}")
    print("=" * 65)


if __name__ == "__main__":
    main()
