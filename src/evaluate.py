"""
evaluate.py — Compute metrics and generate evaluation plots.

Usage
─────
python src/evaluate.py --model cnn
python src/evaluate.py --model resnet
python src/evaluate.py --model rf
python src/evaluate.py --model svm
python src/evaluate.py --model all        # compare all saved models
"""

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from pathlib import Path
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODELS_DIR, PLOTS_DIR, TEST_FOLD, BATCH_SIZE, IDX_TO_CLASS, NUM_CLASSES,
)
from features import load_feature_cache
from data_utils import AudioFeatureDataset
from models.baseline import load_baseline, flatten
from models.cnn import AudioCNN
from models.resnet import AudioResNet


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def predict_baseline(model_name: str, X_test: np.ndarray) -> np.ndarray:
    model = load_baseline(model_name)
    X_flat = flatten(X_test)
    return model.predict(X_flat)


def predict_dl(model_name: str, X_test: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "cnn":
        model = AudioCNN()
    else:
        model = AudioResNet()

    ckpt_path = MODELS_DIR / f"{model_name}_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    # dummy labels (not used)
    dummy_y = np.zeros(len(X_test), dtype=np.int64)
    ds      = AudioFeatureDataset(X_test, dummy_y, augment=False)
    loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            logits = model(X_batch.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())

    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    model_name: str) -> dict:
    class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'─'*55}")
    print(f"  Model : {model_name.upper()}")
    print(f"{'─'*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=class_names)}")

    return {"model": model_name, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str) -> None:
    class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(f"Confusion Matrix — {model_name.upper()}",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved confusion matrix → {out}")


def plot_training_history(model_name: str) -> None:
    hist_path = MODELS_DIR / f"{model_name}_history.json"
    if not hist_path.exists():
        print(f"  [SKIP] No history file for {model_name}")
        return

    with open(hist_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, h["train_loss"], label="Train", color="#4C72B0", lw=2)
    ax1.plot(epochs, h["val_loss"],   label="Val",   color="#DD8452", lw=2)
    ax1.set_title("Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, [a*100 for a in h["train_acc"]], label="Train",
             color="#4C72B0", lw=2)
    ax2.plot(epochs, [a*100 for a in h["val_acc"]],   label="Val",
             color="#DD8452", lw=2)
    ax2.set_title("Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(f"Training History — {model_name.upper()}",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"training_history_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved training history → {out}")


def plot_comparison(results: list) -> None:
    """Bar chart comparing all models."""
    if len(results) < 2:
        return
    names   = [r["model"].upper() for r in results]
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [r[metric] * 100 for r in results]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                      color=color, edgecolor="white", linewidth=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom",
                    fontsize=8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison on Test Set",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved model comparison → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

AVAILABLE_MODELS = ["rf", "svm", "cnn", "resnet"]


def evaluate_model(model_name: str, X: np.ndarray, y: np.ndarray,
                   folds: np.ndarray) -> dict:
    test_mask = folds == TEST_FOLD
    X_test, y_test = X[test_mask], y[test_mask]

    if model_name in ("rf", "svm"):
        y_pred = predict_baseline(model_name, X_test)
    else:
        y_pred = predict_dl(model_name, X_test)

    metrics = compute_metrics(y_test, y_pred, model_name)
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_training_history(model_name)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Audio Event Detection model")
    parser.add_argument("--model", required=True,
                        choices=AVAILABLE_MODELS + ["all"],
                        help="Model to evaluate (or 'all')")
    parser.add_argument("--feature", default="logmel",
                        choices=["mfcc", "mel", "logmel"])
    args = parser.parse_args()

    X, y, folds = load_feature_cache(args.feature)

    to_eval = AVAILABLE_MODELS if args.model == "all" else [args.model]
    all_results = []

    for name in to_eval:
        ckpt = MODELS_DIR / f"{name}_best.pt"
        pkl  = MODELS_DIR / f"{name}_pipeline.pkl"
        if not ckpt.exists() and not pkl.exists():
            print(f"[SKIP] {name} — no saved model found.")
            continue
        try:
            result = evaluate_model(name, X, y, folds)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    if len(all_results) > 1:
        plot_comparison(all_results)


if __name__ == "__main__":
    main()
