"""
models/baseline.py — Random Forest & SVM baselines.

Both models receive mean+std pooled MFCC / Mel vectors as input.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, TRAIN_FOLDS, VAL_FOLD, TEST_FOLD, SEED


def flatten(X: np.ndarray) -> np.ndarray:
    """
    Convert (N, 1, H, W) → (N, H*2) by mean + std over time axis.
    If already 2-D (N, D) passes through unchanged.
    """
    if X.ndim == 4:
        X = X[:, 0, :, :]          # (N, H, W)
        return np.concatenate([X.mean(axis=2), X.std(axis=2)], axis=1)
    if X.ndim == 3:
        return np.concatenate([X.mean(axis=2), X.std(axis=2)], axis=1)
    return X


def build_rf(n_estimators: int = 300, max_depth: int = None) -> Pipeline:
    """Random Forest pipeline (with StandardScaler)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


def build_svm(C: float = 10.0, kernel: str = "rbf",
              gamma: str = "scale") -> Pipeline:
    """SVM pipeline (with StandardScaler)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            C=C, kernel=kernel, gamma=gamma,
            class_weight="balanced",
            probability=True,
            random_state=SEED,
        )),
    ])


MODEL_BUILDERS = {"rf": build_rf, "svm": build_svm}


def train_baseline(model_name: str, X: np.ndarray, y: np.ndarray,
                   folds: np.ndarray) -> Pipeline:
    """
    Train the chosen baseline on TRAIN_FOLDS and save to disk.

    Returns the fitted pipeline.
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Choose from {list(MODEL_BUILDERS)}")

    X_flat = flatten(X)

    train_mask = np.isin(folds, TRAIN_FOLDS)
    X_train, y_train = X_flat[train_mask], y[train_mask]

    print(f"Training {model_name.upper()} on {X_train.shape[0]} samples …")
    model = MODEL_BUILDERS[model_name]()
    model.fit(X_train, y_train)

    save_path = MODELS_DIR / f"{model_name}_pipeline.pkl"
    joblib.dump(model, save_path)
    print(f"✅ Saved → {save_path}")

    return model


def load_baseline(model_name: str) -> Pipeline:
    """Load a saved baseline pipeline."""
    path = MODELS_DIR / f"{model_name}_pipeline.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}.\n"
            f"Run: python src/train.py --model {model_name}"
        )
    return joblib.load(path)
