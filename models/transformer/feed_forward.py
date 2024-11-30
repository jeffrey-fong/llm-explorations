import torch
from torch import nn


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, ffn_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear1(x)
        output = self.relu(output)
        output = self.linear2(output)
        return output


class SwiGLUFeedForwardLayer(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, ffn_size)
        self.swish = nn.SiLU()
        self.linear = nn.Linear(ffn_size, ffn_size)
        self.linear2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        swish = self.swish(x)
        output = swish * self.linear(x)
        output = self.linear2(output)
        return output
