import os
import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.classification import ClassificationHead
from models.localization import LocalizationHead
from models.segmentation import DecoderBlock, FinalUpsample


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
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str  = "checkpoints/localizer.pth",
        unet_path: str       = "checkpoints/unet.pth",
    ):
        import gdown

        # force redownload to ensure fresh weights
        for path in [classifier_path, localizer_path, unet_path]:
            if os.path.isfile(path):
                os.remove(path)
        os.makedirs(os.path.dirname(classifier_path), exist_ok=True)

        gdown.download(id="1KwGL5dstmVb_bC7DG7jZTxMp8Wr029hC", output=classifier_path, quiet=False)
        gdown.download(id="195UsDxEByCsMwrvdshZ4uKE6MGM8uRhr",  output=localizer_path,  quiet=False)
        gdown.download(id="17KLfTcFznv_SYLtWvyoEK8RRcn8ZETdD",  output=unet_path,       quiet=False)

        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder  = VGG11Encoder(in_channels=in_channels)
        self.cls_head = ClassificationHead(num_classes=num_breeds)
        self.loc_head = LocalizationHead()

        self.dec4 = DecoderBlock(512, 512, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.dec0 = FinalUpsample(64, 32)
        self.seg_head = nn.Conv2d(32, seg_classes, kernel_size=1)

        self._load_all(classifier_path, localizer_path, unet_path, device)
        self.to(device)

    def _get_sd(self, path, device):
        ckpt = torch.load(path, map_location=device)
        return ckpt.get("state_dict", ckpt)

    def _strip(self, sd, prefix):
        return {k[len(prefix):]: v
                for k, v in sd.items()
                if k.startswith(prefix)}

    def _load_all(self, cls_path, loc_path, unet_path, device):

        # --- classifier.pth ---
        if os.path.isfile(cls_path):
            sd = self._get_sd(cls_path, device)

            enc_sd = self._strip(sd, "encoder.")
            if enc_sd:
                self.encoder.load_state_dict(enc_sd, strict=True)
                print(f"  loaded encoder from {cls_path}")

            # ClassificationHead expects keys like "fc.1.weight" not "head.fc.1.weight"
            cls_sd = self._strip(sd, "head.")
            if cls_sd:
                self.cls_head.load_state_dict(cls_sd, strict=True)
                print(f"  loaded cls_head from {cls_path}")

        # --- localizer.pth ---
        if os.path.isfile(loc_path):
            sd = self._get_sd(loc_path, device)

            # LocalizationHead expects keys like "regressor.1.weight" not "head.regressor.1.weight"
            loc_sd = self._strip(sd, "head.")
            if loc_sd:
                self.loc_head.load_state_dict(loc_sd, strict=True)
                print(f"  loaded loc_head from {loc_path}")

        # --- unet.pth ---
        if os.path.isfile(unet_path):
            sd = self._get_sd(unet_path, device)

            for name in ("dec4", "dec3", "dec2", "dec1", "dec0"):
                part_sd = self._strip(sd, f"{name}.")
                if part_sd:
                    getattr(self, name).load_state_dict(part_sd, strict=True)
            print(f"  loaded decoder from {unet_path}")

            seg_sd = self._strip(sd, "seg_head.")
            if seg_sd:
                self.seg_head.load_state_dict(seg_sd, strict=True)
                print(f"  loaded seg_head from {unet_path}")

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
