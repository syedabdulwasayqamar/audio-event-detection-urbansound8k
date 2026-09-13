"""
models/resnet.py — ResNet-18 adapted for single-channel spectrogram input.

We adapt torchvision's ResNet-18 by:
  1. Replacing the first conv to accept 1-channel input.
  2. Replacing the final FC layer for NUM_CLASSES outputs.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NUM_CLASSES


class AudioResNet(nn.Module):
    """
    ResNet-18 backbone adapted for grayscale (1-ch) spectrograms.

    Parameters
    ----------
    num_classes : number of output classes
    pretrained  : if True, load ImageNet weights and adapt the conv1 layer
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 pretrained: bool = False):
        super().__init__()

        # Load backbone
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Adapt conv1: 3-ch → 1-ch (sum the weights across channel dim)
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            # Average the 3 input channels
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv

        # Adapt final FC
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# Quick architecture check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = AudioResNet(pretrained=False)
    dummy = torch.zeros(4, 1, 128, 173)
    out   = model(dummy)
    print(f"AudioResNet output shape: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
