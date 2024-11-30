from typing import Optional, Tuple

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self, layer_shape: Tuple[int, int], eps: Optional[float] = None
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(layer_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.norm(x)

        return output
