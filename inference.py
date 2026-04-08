import torch
import numpy as np
from PIL import Image

from multitask import MultiTaskPerceptionModel

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def run_inference(image_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskPerceptionModel().to(device)
    model.eval()

    x = preprocess(image_path).to(device)
    with torch.no_grad():
        out = model(x)

    return {
        "class_id": out["classification"].argmax(1).item(),
        "bbox":     out["localization"].squeeze(0).tolist(),
        "seg_mask": out["segmentation"].argmax(1).squeeze(0).cpu().numpy(),
    }
