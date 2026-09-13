from pathlib import Path

# =========================================================
# PROJECT ROOT
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =========================================================
# DATASET (IMPORTANT: matches your nested folder structure)
# =========================================================
# Your actual path:
# data/raw/UrbanSound8K/UrbanSound8K/audio/...
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "UrbanSound8K" / "UrbanSound8K"

AUDIO_DIR = DATA_DIR / "audio"
METADATA_CSV = DATA_DIR / "metadata" / "UrbanSound8K.csv"

# =========================================================
# TARGET CLASSES (5-class subset)
# =========================================================
# 1 = car_horn
# 3 = dog_bark
# 4 = drilling
# 5 = engine_idling
# 8 = siren

TARGET_CLASS_IDS = [1, 3, 4, 5, 8]

TARGET_LABELS = {
    1: "car_horn",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    8: "siren",
}

# model label mapping (0 → N-1)
CLASS_TO_IDX = {
    cls_id: idx for idx, cls_id in enumerate(sorted(TARGET_CLASS_IDS))
}

IDX_TO_CLASS = {
    idx: TARGET_LABELS[cls_id]
    for cls_id, idx in CLASS_TO_IDX.items()
}

NUM_CLASSES = len(TARGET_CLASS_IDS)

# =========================================================
# AUDIO PARAMETERS
# =========================================================
SAMPLE_RATE = 22050
CLIP_DURATION = 4.0

N_MFCC = 40
N_MELS = 128

HOP_LENGTH = 512
N_FFT = 2048
FMAX = 8000

# =========================================================
# OUTPUT DIRECTORIES
# =========================================================
FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

for d in [FEATURES_DIR, MODELS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =========================================================
# TRAIN/TEST SPLIT (UrbanSound8K standard)
# =========================================================
TRAIN_FOLDS = list(range(1, 9))  # 1–8
VAL_FOLD = 9
TEST_FOLD = 10
