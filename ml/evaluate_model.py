"""
Model Evaluation & Visual Analytics Module
==========================================
Generates high-resolution Confusion Matrix heatmaps, per-class
performance radar/bar charts, and algorithm comparison figures.
"""

import sys
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/clean plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config


def generate_evaluation_visualizations():
    print("=" * 70)
    print("  GENERATING VISUAL EVALUATION ARTIFACTS")
    print("=" * 70)

    if not config.MODEL_PATH.exists() or not config.LABEL_ENCODER_PATH.exists():
        print("[ERROR] Model artifacts not found. Run 'python ml/train_model.py' first.")
        return

    # 1. Load Data & Models
    df = pd.read_csv(config.DATASET_CSV_PATH)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y_raw = df["label"].values

    model = joblib.load(config.MODEL_PATH)
    encoder = joblib.load(config.LABEL_ENCODER_PATH)
    y = encoder.transform(y_raw)
    class_names = list(encoder.classes_)

    # Re-create identical test split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(style="white")

    # Format class display labels
    readable_labels = [c.replace("_", " ").title() for c in class_names]

    ax = sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=readable_labels,
        yticklabels=readable_labels,
        cbar=True,
        linewidths=0.5,
        linecolor="#e2e8f0",
    )
    plt.title("Normalized Confusion Matrix — Hand Gesture Recognition", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Gesture", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("True Ground Truth Gesture", fontsize=12, fontweight="bold", labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_output_path = config.MODELS_DIR / "confusion_matrix.png"
    plt.savefig(cm_output_path, dpi=300)
    plt.close()
    print(f"[+] Confusion Matrix saved to: {cm_output_path}")

    # 3. Model Comparison Chart (from evaluation_report.json)
    if config.EVALUATION_METRICS_PATH.exists():
        with open(config.EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", {})
        models = list(results.keys())
        accs = [results[m]["test_accuracy"] * 100 for m in models]
        f1s = [results[m]["f1_macro"] * 100 for m in models]
        precs = [results[m]["precision_macro"] * 100 for m in models]
        recs = [results[m]["recall_macro"] * 100 for m in models]

        x = np.arange(len(models))
        width = 0.20

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.bar(x - 1.5 * width, accs, width, label="Accuracy", color="#3b82f6")
        ax.bar(x - 0.5 * width, precs, width, label="Precision (Macro)", color="#10b981")
        ax.bar(x + 0.5 * width, recs, width, label="Recall (Macro)", color="#f59e0b")
        ax.bar(x + 1.5 * width, f1s, width, label="F1-Score (Macro)", color="#8b5cf6")

        ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
        ax.set_title("Algorithm Performance Comparison (Test Set)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10, fontweight="bold")
        ax.set_ylim([85, 103])
        ax.legend(loc="lower right", framealpha=0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        comparison_chart_path = config.MODELS_DIR / "model_comparison.png"
        plt.savefig(comparison_chart_path, dpi=300)
        plt.close()
        print(f"[+] Comparison Chart saved to: {comparison_chart_path}")

    print("\n[SUCCESS] Visual evaluation artifacts generated successfully!")


if __name__ == "__main__":
    generate_evaluation_visualizations()
