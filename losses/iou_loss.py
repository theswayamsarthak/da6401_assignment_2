import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    IoU loss for bounding box regression.
    Loss = 1 - IoU, so it's always in [0, 1].

    Boxes are expected in (x_center, y_center, width, height) format.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be 'none', 'mean' or 'sum', got '{reduction}'")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # convert from cx,cy,w,h to x1,y1,x2,y2 for easier area calculation
        pred  = self._to_xyxy(pred_boxes)
        target = self._to_xyxy(target_boxes)

        # intersection
        ix1 = torch.max(pred[:, 0], target[:, 0])
        iy1 = torch.max(pred[:, 1], target[:, 1])
        ix2 = torch.min(pred[:, 2], target[:, 2])
        iy2 = torch.min(pred[:, 3], target[:, 3])

        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

        pred_area   = (pred[:, 2]   - pred[:, 0]).clamp(min=0)   * (pred[:, 3]   - pred[:, 1]).clamp(min=0)
        target_area = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)

        union = pred_area + target_area - inter + self.eps
        iou   = inter / union

        loss = 1.0 - iou  # [B], in [0,1]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    @staticmethod
    def _to_xyxy(boxes):
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)
