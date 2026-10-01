"""
Machine Learning Model Training & Validation Pipeline
=====================================================
Trains and compares Random Forest, Support Vector Machine (SVM),
and K-Nearest Neighbors (KNN) classifiers on hand landmark features.
Saves the highest-performing model and comprehensive metrics report.
"""

import sys
import json
import time
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config


def train_and_evaluate():
    print("=" * 70)
    print("  GESTURE CLASSIFICATION MODEL TRAINING & BENCHMARKING")
    print("=" * 70)

    csv_path = config.DATASET_CSV_PATH
    if not csv_path.exists():
        print(f"[ERROR] Dataset not found at: {csv_path}")
        print("Please run 'python ml/collect_data.py' or 'python ml/generate_sample_dataset.py' first.")
        return

    # 1. Load Dataset
    print(f"[*] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[*] Total Records Loaded: {len(df)}")
    print(f"[*] Class Distribution:\n{df['label'].value_counts().to_string()}\n")

    # 2. Extract Features and Labels
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y_raw = df["label"].values

    if X.shape[1] != 63:
        print(f"[WARNING] Expected 63 features, got {X.shape[1]}.")

    # Encode string labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)

    # 3. Stratified Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Train Set Size: {len(X_train)} | Test Set Size: {len(X_test)}")

    # 4. Define Candidate Classifiers
    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Support Vector Machine (RBF)": SVC(
            kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42
        ),
        "K-Nearest Neighbors (k=5)": KNeighborsClassifier(
            n_neighbors=5, weights="distance"
        ),
    }

    # 5. Cross-Validation & Test Benchmarking
    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model_instance = None

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "-" * 70)
    print("  MODEL COMPARISON & 5-FOLD CROSS-VALIDATION")
    print("-" * 70)

    for name, model in candidate_models.items():
        print(f"\n[*] Evaluating: {name}...")
        t0 = time.time()

        # 5-fold CV
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
        train_time = time.time() - t0

        # Fit on full training set
        model.fit(X_train, y_train)

        # Predict on held-out test set
        y_pred = model.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"    5-Fold CV Accuracy : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
        print(f"    Test Accuracy      : {test_acc * 100:.2f}%")
        print(f"    Precision (Macro)  : {prec_macro * 100:.2f}%")
        print(f"    Recall (Macro)     : {rec_macro * 100:.2f}%")
        print(f"    F1-Score (Macro)   : {f1_macro * 100:.2f}%")
        print(f"    Fit Time           : {train_time:.2f}s")

        results[name] = {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "test_accuracy": float(test_acc),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "classification_report": classification_report(
                y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
            ),
        }

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_name = name
            best_model_instance = model

    print("\n" + "=" * 70)
    print(f"  WINNING MODEL: {best_model_name} (F1: {best_f1 * 100:.2f}%)")
    print("=" * 70)

    # Detailed report for winning model
    best_pred = best_model_instance.predict(X_test)
    print("\nDetailed Classification Report for Winner:")
    print(classification_report(y_test, best_pred, target_names=class_names, digits=4))

    # 6. Save Model Artifacts
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model_instance, config.MODEL_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

    report_payload = {
        "best_model": best_model_name,
        "classes": class_names,
        "results": results,
    }
    with open(config.EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=4)

    print(f"[+] Serialized model saved to        : {config.MODEL_PATH}")
    print(f"[+] Serialized label encoder saved to: {config.LABEL_ENCODER_PATH}")
    print(f"[+] Evaluation metrics saved to      : {config.EVALUATION_METRICS_PATH}")
    print("\n[SUCCESS] Model training complete and ready for real-time inference!")


if __name__ == "__main__":
    train_and_evaluate()
