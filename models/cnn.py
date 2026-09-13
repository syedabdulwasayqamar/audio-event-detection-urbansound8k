"""
models/cnn.py — Convolutional Neural Network for audio classification.

Input:  (B, 1, N_MELS, T)  — single-channel log-Mel spectrogram
Output: (B, NUM_CLASSES)   — class logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NUM_CLASSES


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → MaxPool block."""

    def __init__(self, in_ch: int, out_ch: int,
                 pool: tuple = (2, 2), dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AudioCNN(nn.Module):
    """
    4-block CNN for spectrogram classification.

    Architecture
    ────────────
    Block 1 : 1  →  32 ch  │ pool (2,2)
    Block 2 : 32 →  64 ch  │ pool (2,2)
    Block 3 : 64 → 128 ch  │ pool (2,2)
    Block 4 : 128→ 256 ch  │ pool (2,2)
    Global Average Pool
    FC: 256 → 128 → num_classes
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,   32,  pool=(2, 2), dropout=0.2),
            ConvBlock(32,  64,  pool=(2, 2), dropout=0.2),
            ConvBlock(64,  128, pool=(2, 2), dropout=0.3),
            ConvBlock(128, 256, pool=(2, 2), dropout=0.3),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # → (B, 256, 1, 1)
            nn.Flatten(),                # → (B, 256)
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ─────────────────────────────────────────────────────────────────────────────
# Quick architecture summary
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = AudioCNN()
    dummy = torch.zeros(4, 1, 128, 173)   # (B, C, N_MELS, T)
    out   = model(dummy)
    print(f"AudioCNN output shape: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
