"""
data_utils.py — Dataset loading and PyTorch Dataset wrappers.
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    AUDIO_DIR, METADATA_CSV, TARGET_CLASS_IDS, CLASS_TO_IDX,
    SAMPLE_RATE, CLIP_DURATION, TRAIN_FOLDS, VAL_FOLD, TEST_FOLD,
)


# ─────────────────────────────────────────────────────────────────────────────
# Raw waveform helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_waveform(filepath: str, sr: int = SAMPLE_RATE,
                  duration: float = CLIP_DURATION) -> np.ndarray:
    """Load and normalise a WAV clip.  Returns a 1-D float32 array."""
    target_len = int(sr * duration)
    wave, _ = librosa.load(filepath, sr=sr, mono=True, duration=duration)

    # Pad if shorter than target
    if len(wave) < target_len:
        wave = np.pad(wave, (0, target_len - len(wave)))
    else:
        wave = wave[:target_len]

    return wave.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata loading with fold-based splits
# ─────────────────────────────────────────────────────────────────────────────

def load_metadata(filter_classes: bool = True) -> pd.DataFrame:
    """
    Read UrbanSound8K.csv and add a 'filepath' column.
    Optionally filter to only TARGET_CLASS_IDS.
    """
    df = pd.read_csv(METADATA_CSV)
    df["filepath"] = df.apply(
        lambda r: str(AUDIO_DIR / f"fold{r.fold}" / r.slice_file_name), axis=1
    )
    if filter_classes:
        df = df[df["classID"].isin(TARGET_CLASS_IDS)].reset_index(drop=True)
    df["label"] = df["classID"].map(CLASS_TO_IDX)
    return df


def get_split_dfs():
    """Return (train_df, val_df, test_df) based on official fold splits."""
    df = load_metadata()
    train_df = df[df["fold"].isin(TRAIN_FOLDS)].reset_index(drop=True)
    val_df   = df[df["fold"] == VAL_FOLD].reset_index(drop=True)
    test_df  = df[df["fold"] == TEST_FOLD].reset_index(drop=True)
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset (uses pre-computed feature tensors)
# ─────────────────────────────────────────────────────────────────────────────

class AudioFeatureDataset(Dataset):
    """
    Dataset that serves (feature_tensor, label) pairs.
    features: np.ndarray of shape (N, C, H, W)  [image-like for CNN]
    labels  : np.ndarray of shape (N,)
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 augment: bool = False):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = self._time_mask(x)
            x = self._freq_mask(x)
        return x, self.y[idx]

    # Simple SpecAugment-style masks
    @staticmethod
    def _time_mask(spec: torch.Tensor, max_t: int = 20) -> torch.Tensor:
        _, _, T = spec.shape
        t = np.random.randint(0, max_t)
        t0 = np.random.randint(0, max(1, T - t))
        spec = spec.clone()
        spec[:, :, t0:t0 + t] = 0
        return spec

    @staticmethod
    def _freq_mask(spec: torch.Tensor, max_f: int = 15) -> torch.Tensor:
        _, F, _ = spec.shape
        f = np.random.randint(0, max_f)
        f0 = np.random.randint(0, max(1, F - f))
        spec = spec.clone()
        spec[:, f0:f0 + f, :] = 0
        return spec


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_metadata()
    print(f"Total samples (5 classes): {len(df)}")
    print(df["class"].value_counts())
    train, val, test = get_split_dfs()
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
