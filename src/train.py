"""
train.py — Unified training entry-point for all models.

Usage
─────
# Baseline models (uses logmel features by default)
python src/train.py --model rf
python src/train.py --model svm

# Deep-learning models
python src/train.py --model cnn
python src/train.py --model resnet
python src/train.py --model resnet --pretrained

Optional flags
──────────────
--feature   [mfcc|mel|logmel]  feature type  (default: logmel)
--epochs    int                DL epochs     (default: from config)
--lr        float              learning rate (default: from config)
--batch     int                batch size    (default: from config)
--pretrained                   ResNet only: load ImageNet weights
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODELS_DIR, TRAIN_FOLDS, VAL_FOLD, TEST_FOLD,
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, SEED,
)
from features import load_feature_cache
from data_utils import AudioFeatureDataset
from models.baseline import train_baseline, flatten
from models.cnn import AudioCNN
from models.resnet import AudioResNet


def set_seed(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_dl(model: nn.Module, model_name: str,
             X: np.ndarray, y: np.ndarray, folds: np.ndarray,
             epochs: int, lr: float, batch_size: int) -> dict:
    """
    Train a PyTorch model using TRAIN_FOLDS, evaluate on VAL_FOLD each epoch.
    Returns a history dict with train/val loss and accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device)

    train_mask = np.isin(folds, TRAIN_FOLDS)
    val_mask   = folds == VAL_FOLD

    train_ds = AudioFeatureDataset(X[train_mask], y[train_mask], augment=True)
    val_ds   = AudioFeatureDataset(X[val_mask],   y[val_mask],   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for X_batch, y_batch in tqdm(train_loader,
                                     desc=f"Epoch {epoch:02d}/{epochs} [train]",
                                     leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            t_loss    += loss.item() * len(y_batch)
            t_correct += (logits.argmax(1) == y_batch).sum().item()
            t_total   += len(y_batch)

        t_loss /= t_total
        t_acc   = t_correct / t_total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)

                v_loss    += loss.item() * len(y_batch)
                v_correct += (logits.argmax(1) == y_batch).sum().item()
                v_total   += len(y_batch)

        v_loss /= v_total
        v_acc   = v_correct / v_total
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(f"  Epoch {epoch:02d} | "
              f"train loss {t_loss:.4f}  acc {t_acc:.4f} | "
              f"val loss {v_loss:.4f}  acc {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            ckpt_path = MODELS_DIR / f"{model_name}_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  💾 Saved best checkpoint (val_acc={v_acc:.4f})")

    hist_path = MODELS_DIR / f"{model_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Training complete. Best val_acc = {best_val_acc:.4f}")
    print(f"   History saved → {hist_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Audio Event Detection model")
    parser.add_argument("--model",   required=True,
                        choices=["rf", "svm", "cnn", "resnet"],
                        help="Model to train")
    parser.add_argument("--feature", default="logmel",
                        choices=["mfcc", "mel", "logmel"],
                        help="Feature type (default: logmel)")
    parser.add_argument("--epochs",  type=int,   default=EPOCHS)
    parser.add_argument("--lr",      type=float, default=LR)
    parser.add_argument("--batch",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained ImageNet weights (ResNet only)")
    args = parser.parse_args()

    set_seed()

    print(f"\n{'='*60}")
    print(f"  Model   : {args.model.upper()}")
    print(f"  Feature : {args.feature}")
    print(f"{'='*60}\n")

    X, y, folds = load_feature_cache(args.feature)

    if args.model in ("rf", "svm"):
        train_baseline(args.model, X, y, folds)

    elif args.model == "cnn":
        model = AudioCNN()
        train_dl(model, "cnn", X, y, folds,
                 args.epochs, args.lr, args.batch)

    elif args.model == "resnet":
        model = AudioResNet(pretrained=args.pretrained)
        name  = "resnet_pretrained" if args.pretrained else "resnet"
        train_dl(model, name, X, y, folds,
                 args.epochs, args.lr, args.batch)


if __name__ == "__main__":
    main()
