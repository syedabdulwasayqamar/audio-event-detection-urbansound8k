"""
features.py — Extract and cache Mel Spectrogram, MFCC, and Log-Mel features.

Run as a script to build the feature cache:
    python src/features.py [--feature mel|mfcc|logmel]
"""

import argparse
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    SAMPLE_RATE, N_MELS, N_MFCC, HOP_LENGTH, N_FFT, FMAX,
    CLIP_DURATION, FEATURES_DIR,
)
from data_utils import load_metadata, load_waveform


def extract_mfcc(wave: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Returns MFCCs of shape (N_MFCC, T)."""
    mfcc = librosa.feature.mfcc(
        y=wave, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    return mfcc.astype(np.float32)


def extract_mel(wave: np.ndarray, sr: int = SAMPLE_RATE,
                log: bool = False) -> np.ndarray:
    """Returns Mel spectrogram of shape (N_MELS, T). If log=True, applies log(1 + S)."""
    S = librosa.feature.melspectrogram(
        y=wave, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=FMAX,
    )
    if log:
        S = np.log1p(S)
    return S.astype(np.float32)


def extract_logmel(wave: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Convenience wrapper: log-Mel spectrogram."""
    return extract_mel(wave, sr=sr, log=True)


def flatten_features(feat_2d: np.ndarray) -> np.ndarray:
    """Concat [mean, std] along the time axis → 1-D vector."""
    return np.concatenate([feat_2d.mean(axis=1), feat_2d.std(axis=1)])


FEATURE_FNS = {
    "mfcc":   extract_mfcc,
    "mel":    extract_mel,
    "logmel": extract_logmel,
}


def build_feature_cache(feature_type: str = "logmel",
                        for_cnn: bool = True) -> None:
    """
    Extract features for every clip and save to an .npz file.

    Parameters
    ----------
    feature_type : one of 'mfcc', 'mel', 'logmel'
    for_cnn      : if True, shape is (N, 1, H, W); else (N, H*W) for sklearn
    """
    fn = FEATURE_FNS[feature_type]
    df = load_metadata()

    X_list, y_list, fold_list = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"Extracting {feature_type}"):
        try:
            wave = load_waveform(row["filepath"])
            feat = fn(wave)                     # (H, W)

            if for_cnn:
                feat = feat[np.newaxis, ...]    # (1, H, W)

            X_list.append(feat)
            y_list.append(row["label"])
            fold_list.append(row["fold"])

        except Exception as e:
            print(f"  [WARN] Skipping {row['filepath']}: {e}")

    X = np.stack(X_list)                       # (N, ...)
    y = np.array(y_list, dtype=np.int64)
    folds = np.array(fold_list, dtype=np.int32)

    out_path = FEATURES_DIR / f"{feature_type}_features.npz"
    np.savez_compressed(out_path, X=X, y=y, folds=folds)
    print(f"\n✅ Saved {len(y)} samples → {out_path}")
    print(f"   Feature shape: {X.shape}")


def load_feature_cache(feature_type: str = "logmel"):
    """Load the cached .npz and return (X, y, folds)."""
    path = FEATURES_DIR / f"{feature_type}_features.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found: {path}\n"
            f"Run:  python src/features.py --feature {feature_type}"
        )
    data = np.load(path)
    return data["X"], data["y"], data["folds"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio features")
    parser.add_argument("--feature", choices=["mfcc", "mel", "logmel"],
                        default="logmel",
                        help="Feature type to extract (default: logmel)")
    parser.add_argument("--flat", action="store_true",
                        help="Also save flattened (mean+std) version for sklearn")
    args = parser.parse_args()

    print(f"=== Extracting {args.feature} features (CNN format) ===")
    build_feature_cache(feature_type=args.feature, for_cnn=True)

    if args.flat:
        print(f"=== Extracting {args.feature} features (flat / sklearn format) ===")
        build_feature_cache(feature_type=args.feature, for_cnn=False)
