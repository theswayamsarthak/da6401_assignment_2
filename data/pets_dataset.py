import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import albumentations as A

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 224


class OxfordIIITPetDataset(Dataset):

    def __init__(self, root, split="trainval", task="multitask", transform=None, augment=False):
        self.root      = Path(root)
        self.task      = task
        self.split     = split
        self.transform = transform
        self.augment   = augment
        self.cache_dir = self.root / ".cache"
        self.cache_dir.mkdir(exist_ok=True)

        list_file = "test.txt" if split == "test" else "trainval.txt"
        ann_file  = self.root / "annotations" / list_file

        all_samples = []
        with open(ann_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                all_samples.append({
                    "name":     parts[0],
                    "class_id": int(parts[1]) - 1,
                })

        # filter out samples missing any required annotation files
        self.samples = []
        skipped = 0
        for s in all_samples:
            xml_path  = self.root / "annotations" / "xmls"    / f"{s['name']}.xml"
            mask_path = self.root / "annotations" / "trimaps"  / f"{s['name']}.png"
            img_path  = self.root / "images"                   / f"{s['name']}.jpg"

            if not img_path.exists():
                skipped += 1
                continue
            # XML annotations only exist for trainval, not test split
            if task in ("localization", "multitask") and split != "test" and not xml_path.exists():
                skipped += 1
                continue
            if task in ("segmentation", "multitask") and not mask_path.exists():
                skipped += 1
                continue
            self.samples.append(s)

        if skipped:
            print(f"  [{split}/{task}] skipped {skipped} samples with missing annotations, "
                  f"{len(self.samples)} remaining")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        name = info["name"]

        # --- load image (from cache if available) ---
        cache_path = self.cache_dir / f"{name}.pt"
        if cache_path.exists():
            img_tensor = torch.load(cache_path, weights_only=True)  # uint8 CHW
        else:
            img = Image.open(self.root / "images" / f"{name}.jpg").convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            img_np     = np.array(img, dtype=np.uint8)
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))  # CHW uint8
            torch.save(img_tensor, cache_path)

        # --- augmentation ---
        if self.augment:
            img_np = img_tensor.permute(1, 2, 0).numpy()
            aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
            ])
            img_np     = aug(image=img_np)["image"]
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))

        # --- normalize to float ---
        img_f = img_tensor.float() / 255.0
        mean  = torch.tensor(_MEAN).view(3, 1, 1)
        std   = torch.tensor(_STD).view(3, 1, 1)
        img_f = (img_f - mean) / std

        out = {"image": img_f}

        if self.task in ("classification", "multitask"):
            out["label"] = torch.tensor(info["class_id"], dtype=torch.long)

        if self.task in ("localization", "multitask"):
            img_orig       = Image.open(self.root / "images" / f"{name}.jpg").convert("RGB")
            orig_w, orig_h = img_orig.size
            bbox           = self._load_bbox(name, orig_w, orig_h)
            out["bbox"]    = torch.tensor(bbox, dtype=torch.float32)

        if self.task in ("segmentation", "multitask"):
            mask     = self._load_mask(name)
            mask_img = Image.fromarray(mask.astype(np.uint8))
            mask_img = mask_img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            out["mask"] = torch.from_numpy(np.array(mask_img, dtype=np.int64))

        return out

    def _load_bbox(self, name, orig_w, orig_h):
        xml_path = self.root / "annotations" / "xmls" / f"{name}.xml"
        if not xml_path.exists():
            # return a default center box if XML missing (test split has no XMLs)
            return [IMG_SIZE / 2, IMG_SIZE / 2, IMG_SIZE * 0.5, IMG_SIZE * 0.5]
        tree = ET.parse(xml_path)
        bb   = tree.find(".//bndbox")
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
        mask = np.array(Image.open(
            self.root / "annotations" / "trimaps" / f"{name}.png"
        ))
        out = np.zeros_like(mask, dtype=np.int64)
        out[mask == 1] = 0
        out[mask == 2] = 1
        out[mask == 3] = 2
        return out
