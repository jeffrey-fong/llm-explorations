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


class RopeEmbedding(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden_size: int,
        vocab_size: int,
        base: float,
        device: Literal["cpu", "cuda"],
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.base = base
        self.device = device

        self.cos_pos_emb, self.sin_pos_emb = self._get_rope_embedding()
        self.token_emb = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=1)

    def _get_rope_embedding(self) -> torch.Tensor:
        # Get theta
        i = torch.arange(0, self.hidden_size, step=2).float()
        theta = self.base ** (-2 * i / self.hidden_size).unsqueeze(0)
        # Get tensor of positions x theta
        pos = torch.arange(0, self.seq_len).float().unsqueeze(1)
        m_theta = pos @ theta  # (seq_len, hidden_size/2)
        # Get cos and sin
        cos_m_theta = torch.cos(m_theta)  # (seq_len, hidden_size/2)
        sin_m_theta = torch.sin(m_theta)  # (seq_len, hidden_size/2)
        # Repeat each value to get (cos1,cos1,cos2,cos2,...) and (sin1,sin1,sin2,sin2,...)
        cos_m_theta = cos_m_theta.repeat_interleave(2, dim=1)  # (seq_len, hidden_size)
        sin_m_theta = sin_m_theta.repeat_interleave(2, dim=1)  # (seq_len, hidden_size)

        return cos_m_theta.to(self.device), sin_m_theta.to(self.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        cos_pos_emb = self.cos_pos_emb[:seq_len, :].unsqueeze(0)
        sin_pos_emb = self.sin_pos_emb[:seq_len, :].unsqueeze(0)
        token_emb = self.token_emb(x)
        inv_token_emb = torch.stack(
            (-token_emb[..., 1::2], token_emb[..., ::2]), dim=-1
        ).flatten(-2)
        return token_emb * cos_pos_emb + inv_token_emb * sin_pos_emb
