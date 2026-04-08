import os

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.classification import ClassificationHead
from models.localization import LocalizationHead
from models.segmentation import DecoderBlock, FinalUpsample


def _load_state_dict(path, device):
    ckpt = torch.load(path, map_location=device)
    # handle both plain state_dict and wrapped {"state_dict": ...} format
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _extract(state_dict, prefix):
    """Pull out weights for a sub-module and strip the prefix."""
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


class MultiTaskPerceptionModel(nn.Module):
    """
    Unified multi-task model that shares a single VGG11 backbone across
    classification, localization and segmentation heads.

    On init it loads the three saved checkpoints and uses those weights to
    initialise the shared encoder and each task head.
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder  = VGG11Encoder(in_channels=in_channels)
        self.cls_head = ClassificationHead(num_classes=num_breeds)
        self.loc_head = LocalizationHead()

        # unet decoder
        self.dec4 = DecoderBlock(512, 512, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.dec0 = FinalUpsample(64, 32)
        self.seg_head = nn.Conv2d(32, seg_classes, kernel_size=1)

        self._load_checkpoints(classifier_path, localizer_path, unet_path, device)
        self.to(device)

    def _load_checkpoints(self, cls_path, loc_path, unet_path, device):
        if os.path.isfile(cls_path):
            sd = _load_state_dict(cls_path, device)
            enc_w = _extract(sd, "encoder.")
            if enc_w:
                self.encoder.load_state_dict(enc_w, strict=True)
            cls_w = _extract(sd, "head.")
            if cls_w:
                self.cls_head.load_state_dict(cls_w, strict=True)

        if os.path.isfile(loc_path):
            sd = _load_state_dict(loc_path, device)
            loc_w = _extract(sd, "head.")
            if loc_w:
                self.loc_head.load_state_dict(loc_w, strict=True)

        if os.path.isfile(unet_path):
            sd = _load_state_dict(unet_path, device)
            for name in ("dec4", "dec3", "dec2", "dec1", "dec0"):
                w = _extract(sd, f"{name}.")
                if w:
                    getattr(self, name).load_state_dict(w, strict=True)
            seg_w = _extract(sd, "seg_head.")
            if seg_w:
                self.seg_head.load_state_dict(seg_w, strict=True)

    def forward(self, x: torch.Tensor):
        """
        Single forward pass through the shared backbone.
        Returns a dict with 'classification', 'localization', 'segmentation'.
        """
        bottleneck, feats = self.encoder(x, return_features=True)

        cls_out = self.cls_head(bottleneck)
        loc_out = self.loc_head(bottleneck)

        d = self.dec4(bottleneck, feats["block4"])
        d = self.dec3(d, feats["block3"])
        d = self.dec2(d, feats["block2"])
        d = self.dec1(d, feats["block1"])
        d = self.dec0(d)
        seg_out = self.seg_head(d)

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }
