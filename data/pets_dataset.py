import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

# imagenet stats since the autograder will feed normalised images
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 224  # fixed for VGG11


class OxfordIIITPetDataset(Dataset):
    """
    Loads the Oxford-IIIT Pet dataset for one of three tasks:
    classification, localization, segmentation, or all three (multitask).

    Expects the standard dataset layout:
        root/images/*.jpg
        root/annotations/xmls/*.xml
        root/annotations/trimaps/*.png
        root/annotations/trainval.txt
        root/annotations/test.txt
    """

    def __init__(self, root, split="trainval", task="multitask", transform=None, augment=False):
        self.root    = Path(root)
        self.task    = task
        self.transform = transform
        self.augment = augment

        list_file = "test.txt" if split == "test" else "trainval.txt"
        ann_file  = self.root / "annotations" / list_file

        self.samples = []
        with open(ann_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                self.samples.append({
                    "name":     parts[0],
                    "class_id": int(parts[1]) - 1,  # file is 1-indexed
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        name = info["name"]

        img = Image.open(self.root / "images" / f"{name}.jpg").convert("RGB")
        orig_w, orig_h = img.size

        if self.transform is not None:
            img = self.transform(img)

        # simple flip augmentation
        flip = self.augment and np.random.rand() > 0.5
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = (img_np - _MEAN) / _STD
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))  # CHW

        out = {"image": img_tensor}

        if self.task in ("classification", "multitask"):
            out["label"] = torch.tensor(info["class_id"], dtype=torch.long)

        if self.task in ("localization", "multitask"):
            bbox = self._load_bbox(name, orig_w, orig_h)
            if flip:
                bbox[0] = IMG_SIZE - bbox[0]  # mirror x_center
            out["bbox"] = torch.tensor(bbox, dtype=torch.float32)

        if self.task in ("segmentation", "multitask"):
            mask = self._load_mask(name)
            mask_img = Image.fromarray(mask.astype(np.uint8))
            mask_img = mask_img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            if flip:
                mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)
            out["mask"] = torch.from_numpy(np.array(mask_img, dtype=np.int64))

        return out

    def _load_bbox(self, name, orig_w, orig_h):
        xml_path = self.root / "annotations" / "xmls" / f"{name}.xml"
        tree = ET.parse(xml_path)
        bb = tree.find(".//bndbox")
        xmin = float(bb.find("xmin").text) * IMG_SIZE / orig_w
        xmax = float(bb.find("xmax").text) * IMG_SIZE / orig_w
        ymin = float(bb.find("ymin").text) * IMG_SIZE / orig_h
        ymax = float(bb.find("ymax").text) * IMG_SIZE / orig_h
        return [
            (xmin + xmax) / 2,
            (ymin + ymax) / 2,
            xmax - xmin,
            ymax - ymin,
        ]

    def _load_mask(self, name):
        # trimap values: 1=pet, 2=background, 3=border -> remap to 0,1,2
        mask = np.array(Image.open(self.root / "annotations" / "trimaps" / f"{name}.png"))
        out = np.zeros_like(mask, dtype=np.int64)
        out[mask == 1] = 0
        out[mask == 2] = 1
        out[mask == 3] = 2
        return out
