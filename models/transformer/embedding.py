from typing import Literal

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden_size: int,
        vocab_size: int,
        device: Literal["cpu", "cuda"],
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device

        self.pos_emb = self._get_pos_embedding()
        self.token_emb = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=1)

    def _get_pos_embedding(self) -> torch.Tensor:
        pos = torch.arange(0, self.seq_len).float().unsqueeze(1)
        i = torch.arange(0, self.hidden_size, step=2).float()
        pos_emb = torch.zeros(self.seq_len, self.hidden_size).float()
        pos_emb[:, 0::2] = torch.sin(pos / 10000 ** (i / self.hidden_size))
        pos_emb[:, 1::2] = torch.cos(pos / 10000 ** (i / self.hidden_size))

        return pos_emb.to(self.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        pos_emb = self.pos_emb[:seq_len, :]
        token_emb = self.token_emb(x)
        return token_emb + pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
