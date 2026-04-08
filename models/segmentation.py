import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class DecoderBlock(nn.Module):
    # one step of the upsampling path:
    # transposed conv to double spatial dims -> concat skip connection -> conv

    def __init__(self, in_ch, skip_ch, out_ch, dropout_p=0.0):
        super().__init__()
        # using ConvTranspose2d for learnable upsampling (no bilinear allowed)
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        if dropout_p > 0.0:
            self.conv.append(CustomDropout(dropout_p))

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FinalUpsample(nn.Module):
    # last upsample block has no skip connection (going from 112 -> 224)
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.upsample(x))


class VGG11UNet(nn.Module):
    """
    U-Net with VGG11 as the encoder/contracting path.

    Decoder mirrors the encoder depth - 4 skip-connected upsampling stages
    bring us from 7x7 back up to 224x224, each using transposed convolutions.
    Skip connections concatenate encoder feature maps at matching resolutions.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # decoder goes 7->14->28->56->112->224
        self.dec4 = DecoderBlock(512, 512, 512, dropout_p=dropout_p)  # fuse with block4
        self.dec3 = DecoderBlock(512, 256, 256, dropout_p=dropout_p)  # fuse with block3
        self.dec2 = DecoderBlock(256, 128, 128, dropout_p=dropout_p)  # fuse with block2
        self.dec1 = DecoderBlock(128, 64, 64, dropout_p=dropout_p)    # fuse with block1
        self.dec0 = FinalUpsample(64, 32)                              # no skip here

        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, feats = self.encoder(x, return_features=True)

        d = self.dec4(bottleneck, feats["block4"])
        d = self.dec3(d, feats["block3"])
        d = self.dec2(d, feats["block2"])
        d = self.dec1(d, feats["block1"])
        d = self.dec0(d)

        return self.seg_head(d)
