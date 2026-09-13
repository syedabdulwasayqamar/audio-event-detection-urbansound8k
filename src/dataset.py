import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset

from src.config import AUDIO_DIR, METADATA_CSV


class UrbanSoundDataset(Dataset):
    def __init__(self, metadata_csv, fold_list=None, sr=22050, duration=4):

        self.metadata = pd.read_csv(METADATA_CSV)

        if fold_list is not None:
            self.metadata = self.metadata[self.metadata["fold"].isin(fold_list)]

        self.sr = sr
        self.duration = duration
        self.samples = sr * duration

        self.classes = sorted(self.metadata["class"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.metadata)

    def _load_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.sr)

        audio = torch.tensor(audio)

        if len(audio) < self.samples:
            padding = self.samples - len(audio)
            audio = torch.nn.functional.pad(audio, (0, padding))
        else:
            audio = audio[:self.samples]

        return audio

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        fold = row["fold"]
        file_name = row["slice_file_name"]
        label = row["class"]

        file_path = AUDIO_DIR / f"fold{fold}" / file_name

        audio = self._load_audio(file_path)
        label_idx = self.class_to_idx[label]

        return audio.float(), torch.tensor(label_idx)
