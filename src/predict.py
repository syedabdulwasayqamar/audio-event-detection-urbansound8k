"""
predict.py — Single-file inference.

Usage
─────
python src/predict.py --wav path/to/dog.wav --model cnn
python src/predict.py --wav path/to/siren.wav --model resnet
python src/predict.py --wav path/to/horn.wav --model rf

Returns the predicted class label and class probabilities.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODELS_DIR, SAMPLE_RATE, CLIP_DURATION,
    IDX_TO_CLASS, NUM_CLASSES,
)
from data_utils import load_waveform
from features import extract_logmel
from models.baseline import load_baseline, flatten
from models.cnn import AudioCNN
from models.resnet import AudioResNet


def predict(wav_path: str, model_name: str = "cnn") -> dict:
    """
    Predict the event class for a WAV file.

    Parameters
    ----------
    wav_path   : path to the audio file
    model_name : one of 'cnn', 'resnet', 'rf', 'svm'

    Returns
    -------
    dict with keys:
        - 'label'       : predicted class name (str)
        - 'class_idx'   : predicted class index (int)
        - 'probs'       : {class_name: probability} (dict)
    """
    wave = load_waveform(str(wav_path))
    feat = extract_logmel(wave)                 # (N_MELS, T)

    if model_name in ("rf", "svm"):
        model = load_baseline(model_name)
        X_flat = flatten(feat[np.newaxis, np.newaxis, ...])
        idx  = int(model.predict(X_flat)[0])
        if hasattr(model, "predict_proba"):
            probs_arr = model.predict_proba(X_flat)[0]
        else:
            probs_arr = np.zeros(NUM_CLASSES)
            probs_arr[idx] = 1.0

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "cnn":
            net = AudioCNN()
        elif model_name in ("resnet", "resnet_pretrained"):
            net = AudioResNet()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        ckpt = MODELS_DIR / f"{model_name}_best.pt"
        net.load_state_dict(torch.load(ckpt, map_location=device))
        net = net.to(device).eval()

        tensor = torch.tensor(feat[np.newaxis, np.newaxis, ...],
                              dtype=torch.float32).to(device)    # (1,1,H,W)
        with torch.no_grad():
            logits = net(tensor)                                 # (1, C)
        probs_arr = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        idx = int(probs_arr.argmax())

    label = IDX_TO_CLASS[idx]
    probs = {IDX_TO_CLASS[i]: float(probs_arr[i]) for i in range(NUM_CLASSES)}

    return {"label": label, "class_idx": idx, "probs": probs}


def main():
    parser = argparse.ArgumentParser(description="Audio Event Detection — Inference")
    parser.add_argument("--wav",   required=True, help="Path to WAV file")
    parser.add_argument("--model", default="cnn",
                        choices=["cnn", "resnet", "resnet_pretrained", "rf", "svm"],
                        help="Model to use for inference (default: cnn)")
    args = parser.parse_args()

    result = predict(args.wav, args.model)

    print(f"\n{'='*50}")
    print(f"  File    : {Path(args.wav).name}")
    print(f"  Model   : {args.model.upper()}")
    print(f"  ➜ Prediction: {result['label'].upper()}")
    print(f"\n  Class Probabilities:")
    for cls, prob in sorted(result["probs"].items(),
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 30)
        print(f"    {cls:<18} {bar:<30} {prob*100:5.1f}%")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
