import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class ClassificationHead(nn.Module):
    # standard VGG-style FC classifier head
    # 512*7*7 -> 4096 -> 4096 -> num_classes

    def __init__(self, num_classes=37, dropout_p=0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(self.pool(x))


class VGG11Classifier(nn.Module):
    """VGG11 encoder + classification head."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.head = ClassificationHead(num_classes=num_classes, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
