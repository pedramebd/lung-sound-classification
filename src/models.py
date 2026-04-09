"""
Model definitions for respiratory sound classification.

Includes:
  - Traditional ML classifiers (SVM, Random Forest, XGBoost)
  - CNN architecture for mel spectrogram classification
  - PyTorch Dataset class for spectrograms
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ══════════════════════════════════════════════
# PyTorch Dataset
# ══════════════════════════════════════════════

class SpectrogramDataset(Dataset):
    """PyTorch Dataset for mel spectrogram images + labels."""

    def __init__(
        self,
        spectrograms: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ):
        """
        Args:
            spectrograms: (N, n_mels, time_steps) float arrays
            labels: (N,) integer class labels
            augment: whether to apply data augmentation
        """
        self.spectrograms = torch.FloatTensor(spectrograms).unsqueeze(1)  # (N, 1, H, W)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.spectrograms[idx]
        label = self.labels[idx]

        if self.augment:
            spec = self._augment(spec)

        return spec, label

    @staticmethod
    def _augment(spec: torch.Tensor) -> torch.Tensor:
        """Simple spectrogram augmentation: time/frequency masking (SpecAugment-lite)."""
        _, h, w = spec.shape

        # Frequency masking
        if torch.rand(1).item() > 0.5:
            f = int(torch.randint(1, max(2, h // 8), (1,)).item())
            f0 = int(torch.randint(0, h - f, (1,)).item())
            spec[:, f0 : f0 + f, :] = 0

        # Time masking
        if torch.rand(1).item() > 0.5:
            t = int(torch.randint(1, max(2, w // 8), (1,)).item())
            t0 = int(torch.randint(0, w - t, (1,)).item())
            spec[:, :, t0 : t0 + t] = 0

        return spec


# ══════════════════════════════════════════════
# CNN Model
# ══════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class LungSoundCNN(nn.Module):
    """
    ResNet-style CNN for respiratory sound classification.

    Input: (batch, 1, n_mels, time_steps)  — single-channel mel spectrogram
    Output: (batch, num_classes)  — class logits
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, num_blocks: int, stride: int):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ══════════════════════════════════════════════
# Lightweight CNN (faster training alternative)
# ══════════════════════════════════════════════

class LungSoundCNNLight(nn.Module):
    """Simpler CNN for quick experiments."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ══════════════════════════════════════════════
# ML model builders
# ══════════════════════════════════════════════

def get_ml_models() -> dict:
    """Return a dict of scikit-learn / XGBoost classifiers to evaluate."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    return {
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=20, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }
