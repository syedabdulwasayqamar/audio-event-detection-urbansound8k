"""
app/app.py — Gradio web application for Audio Event Detection.

Run:
    python app/app.py
"""

import sys
import os
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import IDX_TO_CLASS, NUM_CLASSES, SAMPLE_RATE
from data_utils import load_waveform
from features import extract_logmel
from predict import predict

CLASS_EMOJIS = {
    "siren":         "🚨",
    "dog_bark":      "🐕",
    "car_horn":      "📯",
    "drilling":      "🔩",
    "engine_idling": "🚗",
}

CLASS_COLORS = {
    "siren":         "#E74C3C",
    "dog_bark":      "#2ECC71",
    "car_horn":      "#F39C12",
    "drilling":      "#3498DB",
    "engine_idling": "#9B59B6",
}

AVAILABLE_MODELS = ["cnn", "resnet", "rf", "svm"]


def make_spectrogram_fig(wav_path: str):
    wave = load_waveform(str(wav_path))
    logmel = extract_logmel(wave)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             facecolor="#1a1a2e")

    t = np.linspace(0, len(wave) / SAMPLE_RATE, len(wave))
    axes[0].plot(t, wave, color="#00d4ff", linewidth=0.6, alpha=0.85)
    axes[0].set_title("Waveform", color="white", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Time (s)", color="#aaa")
    axes[0].set_ylabel("Amplitude", color="#aaa")
    axes[0].set_facecolor("#0d0d1a")
    axes[0].tick_params(colors="#aaa")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#333")

    img = axes[1].imshow(
        logmel, aspect="auto", origin="lower",
        cmap="magma", interpolation="nearest",
    )
    axes[1].set_title("Log-Mel Spectrogram", color="white",
                      fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Time Frames", color="#aaa")
    axes[1].set_ylabel("Mel Bins", color="#aaa")
    axes[1].set_facecolor("#0d0d1a")
    axes[1].tick_params(colors="#aaa")
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#333")
    fig.colorbar(img, ax=axes[1], label="Log Magnitude").ax.yaxis.label.set_color("#aaa")

    plt.tight_layout(pad=1.5)
    return fig


def make_prob_fig(probs: dict, pred_label: str):
    classes = list(probs.keys())
    values  = [probs[c] * 100 for c in classes]
    colors  = [CLASS_COLORS.get(c, "#888") for c in classes]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#1a1a2e")
    bars = ax.barh(classes, values, color=colors,
                   edgecolor="#333", linewidth=0.5, height=0.55)

    for bar, val in zip(bars, values):
        ax.text(min(val + 1, 98), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="white", fontsize=10)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", color="#aaa")
    ax.set_title("Class Probabilities", color="white",
                 fontsize=13, fontweight="bold")
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    return fig


def run_inference(wav_file, model_name: str):
    if wav_file is None:
        return "⚠️ Please upload a WAV file.", None, None

    model_name = model_name.lower()

    try:
        result = predict(wav_file, model_name)
    except FileNotFoundError as e:
        return (f"❌ Model not found!\n\n{e}\n\n"
                "Please train the model first using:\n"
                f"  python src/train.py --model {model_name}"), None, None
    except Exception as e:
        return f"❌ Error during inference:\n{e}", None, None

    label     = result["label"]
    probs     = result["probs"]
    emoji     = CLASS_EMOJIS.get(label, "🔊")
    conf      = probs[label] * 100

    sorted_probs = dict(
        sorted(probs.items(), key=lambda x: x[1], reverse=True)
    )

    pred_text = (
        f"## {emoji} Detected: **{label.replace('_', ' ').upper()}**\n\n"
        f"**Confidence:** {conf:.1f}%\n\n"
        f"**Model used:** {model_name.upper()}\n\n"
        "---\n\n"
        "**All class probabilities:**\n\n"
        + "\n".join([
            f"- {CLASS_EMOJIS.get(c,'🔊')} `{c}`: {p*100:.1f}%"
            for c, p in sorted_probs.items()
        ])
    )

    spec_fig = make_spectrogram_fig(wav_file)
    prob_fig = make_prob_fig(sorted_probs, label)

    return pred_text, spec_fig, prob_fig


CSS = """
body { font-family: 'Inter', sans-serif; background: #0d0d1a; color: #e0e0f0; }
.gradio-container { max-width: 960px; margin: auto; }
#title { text-align: center; padding: 20px 0; }
#title h1 { font-size: 2.2em; background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.prediction-box { border-radius: 12px; padding: 16px;
                  background: rgba(124,58,237,0.08);
                  border: 1px solid rgba(124,58,237,0.3); }
"""

with gr.Blocks(
    css=CSS,
    title="🔊 Audio Event Detection",
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="cyan",
        neutral_hue="slate",
    ),
) as demo:

    gr.HTML("""
        <div id="title">
            <h1>🔊 Audio Event Detection</h1>
            <p style="color:#aaa;font-size:1.05em;">
                Upload a WAV file and detect: 
                🚨 Siren &nbsp;|&nbsp; 🐕 Dog Bark &nbsp;|&nbsp;
                📯 Car Horn &nbsp;|&nbsp; 🔩 Drilling &nbsp;|&nbsp; 🚗 Engine
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Upload Audio File",
                type="filepath",
                sources=["upload", "microphone"],
            )
            model_select = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value="cnn",
                label="Model",
                info="Select the model for inference",
            )
            predict_btn = gr.Button("🔍 Detect Sound", variant="primary",
                                    size="lg")

        with gr.Column(scale=2):
            prediction_out = gr.Markdown(
                value="*Upload a WAV file and click Detect Sound.*",
                elem_classes=["prediction-box"],
            )

    with gr.Row():
        spec_out = gr.Plot(label="Signal Analysis")
        prob_out = gr.Plot(label="Class Probabilities")

    gr.Markdown("### How it works")
    gr.Markdown("""
1. **Upload** any WAV audio file (will be resampled to 22 050 Hz, truncated/padded to 4 s)
2. **Select** a model (CNN is recommended for best accuracy)
3. Click **Detect Sound**
4. See the predicted class, confidence, waveform, and spectrogram

> **Note:** Deep-learning models (CNN / ResNet) require training first:
> ```
> python src/features.py --feature logmel
> python src/train.py --model cnn
> ```
""")

    predict_btn.click(
        fn=run_inference,
        inputs=[audio_input, model_select],
        outputs=[prediction_out, spec_out, prob_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
