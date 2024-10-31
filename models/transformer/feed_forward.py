import torch
from torch import nn


class FeedForwardLayer(nn.Module):
    def __init__(
        self, hidden_size: int, ffn_size: int, activation: str = "relu"
    ) -> None:
        super().__init__()
        self.activation = activation

        self.linear1 = nn.Linear(hidden_size, ffn_size)
        self.linear2 = nn.Linear(ffn_size, hidden_size)
        if self.activation == "relu":
            self.act_fn = nn.ReLU()
        elif self.activation == "swiglu":
            self.act_fn = nn.SiLU()
            self.linear3 = nn.Linear(hidden_size, ffn_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            output = self.linear1(x)
            output = self.act_fn(output)
            output = self.linear2(output)
        elif self.activation == "swiglu":
            output1 = self.act_fn(self.linear1(x))
            output3 = self.linear3(x)
            output = self.linear2(output1 * output3)

        return output
