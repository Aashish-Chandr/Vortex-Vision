"""
PyTorch dataset for anomaly detection training.
Supports UCF-Crime, ShanghaiTech Campus, and custom frame directories.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AnomalyDataset(Dataset):
    """
    Loads JPEG frames from a directory.
    mode="normal"  -> returns frames only (for autoencoder training)
    mode="labeled" -> returns (frame, label) where label=0 normal, 1 anomaly
    """

    TRANSFORM = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(self, data_dir: str, mode: str = "normal"):
        self.mode = mode
        self.samples = []

        root = Path(data_dir)
        if mode == "normal":
            self.samples = [(p, 0) for p in sorted(root.glob("**/*.jpg"))]
        else:
            for label, subdir in [(0, "normal"), (1, "anomaly")]:
                self.samples += [(p, label) for p in sorted((root / subdir).glob("**/*.jpg"))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frame = cv2.imread(str(path))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.TRANSFORM(frame)

        if self.mode == "normal":
            return tensor
        return tensor, torch.tensor(label, dtype=torch.float32)
