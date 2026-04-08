from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


# helper to build a single conv-bn-relu block, optionally with dropout after relu
def _conv_bn_relu(in_ch, out_ch, dropout_p=0.0):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout_p > 0.0:
        layers.append(CustomDropout(dropout_p))
    return nn.Sequential(*layers)


class VGG11Encoder(nn.Module):
    """
    VGG11 (config A from the paper) encoder backbone.
    Added BatchNorm after each conv and optionally dropout in the deeper blocks.

    Returns a 512x7x7 bottleneck for a 224x224 input.
    When return_features=True also returns intermediate feature maps
    which are needed for the U-Net skip connections.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.0):
        super().__init__()

        # block 1 - 1 conv, 64 filters
        self.block1 = nn.Sequential(
            _conv_bn_relu(in_channels, 64),
            nn.MaxPool2d(2, 2),
        )

        # block 2 - 1 conv, 128 filters
        self.block2 = nn.Sequential(
            _conv_bn_relu(64, 128),
            nn.MaxPool2d(2, 2),
        )

        # block 3 - 2 convs, 256 filters
        self.block3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256, dropout_p=dropout_p),
            nn.MaxPool2d(2, 2),
        )

        # block 4 - 2 convs, 512 filters
        self.block4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512, dropout_p=dropout_p),
            nn.MaxPool2d(2, 2),
        )

        # block 5 - 2 convs, 512 filters
        self.block5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512, dropout_p=dropout_p),
            nn.MaxPool2d(2, 2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        f1 = self.block1(x)   # 64  x 112 x 112
        f2 = self.block2(f1)  # 128 x  56 x  56
        f3 = self.block3(f2)  # 256 x  28 x  28
        f4 = self.block4(f3)  # 512 x  14 x  14
        f5 = self.block5(f4)  # 512 x   7 x   7

        if return_features:
            return f5, {"block1": f1, "block2": f2, "block3": f3, "block4": f4, "block5": f5}

        return f5


# the autograder does `from models.vgg11 import VGG11` so need this alias
VGG11 = VGG11Encoder
