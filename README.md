# 🔊 Audio Event Detection — UrbanSound8K

Detect environmental sounds (Siren, Dog Bark, Car Horn, Drilling, Engine) using classical ML and deep learning.

## Project Structure

```
project_audio_event_detection/
├── data/                  # UrbanSound8K dataset (download separately)
├── notebooks/
│   ├── 01_EDA.ipynb       # Exploratory Data Analysis
│   ├── 02_feature_extraction.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_deep_learning.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── config.py          # Global configuration
│   ├── data_utils.py      # Dataset loading utilities
│   ├── features.py        # Feature extraction (MFCC, Mel, Log-Mel)
│   ├── models/
│   │   ├── baseline.py    # Random Forest, SVM
│   │   ├── cnn.py         # CNN model (PyTorch)
│   │   └── resnet.py      # ResNet model (PyTorch)
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Evaluation & metrics
│   └── predict.py         # Inference on a single WAV file
├── app/
│   └── app.py             # Gradio deployment app
├── outputs/
│   ├── models/            # Saved model weights
│   ├── features/          # Cached feature arrays
│   └── plots/             # EDA & evaluation plots
├── requirements.txt
└── setup.py
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
#    https://urbansounddataset.weebly.com/urbansound8k.html
#    Extract to: data/UrbanSound8K/

# 3. Run EDA
jupyter notebook notebooks/01_EDA.ipynb

# 4. Extract features
python src/features.py

# 5. Train baseline models
python src/train.py --model rf
python src/train.py --model svm

# 6. Train deep learning models
python src/train.py --model cnn
python src/train.py --model resnet

# 7. Evaluate
python src/evaluate.py --model cnn

# 8. Run deployment app
python app/app.py
```

## Target Classes (5 of UrbanSound8K's 10)
| Label | Class Name     | US8K ID |
|-------|----------------|---------|
| 0     | dog_bark       | 3       |
| 1     | car_horn       | 1       |
| 2     | drilling       | 4       |
| 3     | engine_idling  | 5       |
| 4     | siren          | 8       |

