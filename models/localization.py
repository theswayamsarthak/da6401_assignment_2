import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

# VGG paper uses 224x224 inputs, hardcoding this as instructed
INPUT_SIZE = 224


class LocalizationHead(nn.Module):
    # regression head that predicts [x_center, y_center, w, h] in pixel space
    # sigmoid at the end to bound the output, then scale by image size

    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(1024, 4),
        )
        self.output_act = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(x)
        return self.output_act(self.regressor(x)) * INPUT_SIZE


class VGG11Localizer(nn.Module):
    """VGG11-based single object localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.head = LocalizationHead(dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns predicted bounding box as [x_center, y_center, width, height]
        in pixel coordinates (not normalised).
        """
        return self.head(self.encoder(x))
