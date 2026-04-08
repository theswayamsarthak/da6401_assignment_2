import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    # implementing dropout from scratch since we can't use nn.Dropout
    # basic idea: randomly zero out elements during training and scale up
    # the remaining ones so the expected value stays the same (inverted dropout)

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout prob should be between 0 and 1, got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # during eval just pass through unchanged
        if not self.training or self.p == 0.0:
            return x

        # sample a binary mask - each element stays with prob (1-p)
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full(x.shape, keep_prob, dtype=x.dtype, device=x.device))

        # scale up by 1/keep_prob so the expected value is preserved at test time
        return x * mask / keep_prob

    def extra_repr(self):
        return f"p={self.p}"
