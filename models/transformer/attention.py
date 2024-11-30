import math
from typing import Optional

import torch
from torch import nn

from config import TransformerConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig, is_rope: bool = False) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_heads
        self.is_rope = is_rope
        if self.is_rope:
            self.cos_pos_emb, self.sin_pos_emb = self._get_rope_embedding()

        hidden_size = config.hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _get_rope_embedding(self) -> torch.Tensor:
        # Get theta
        i = torch.arange(0, self.head_dim, step=2).float()
        theta = self.config.base ** (-i / self.head_dim)
        # Get tensor of positions x theta
        pos = torch.arange(0, self.config.seq_len).float()
        m_theta = torch.outer(pos, theta)  # (seq_len, hidden_size/2)
        # Get cos and sin
        cos_m_theta = torch.cos(m_theta)  # (seq_len, hidden_size/2)
        sin_m_theta = torch.sin(m_theta)  # (seq_len, hidden_size/2)
        # Repeat each value to get (cos1,cos1,cos2,cos2,...) and (sin1,sin1,sin2,sin2,...)
        cos_m_theta = cos_m_theta.repeat_interleave(2, dim=1)  # (seq_len, hidden_size)
        sin_m_theta = sin_m_theta.repeat_interleave(2, dim=1)  # (seq_len, hidden_size)

        return cos_m_theta.to(self.config.device), sin_m_theta.to(self.config.device)

    def apply_rope_embedding(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        cos_pos_emb = self.cos_pos_emb[:seq_len, :].unsqueeze(0)
        sin_pos_emb = self.sin_pos_emb[:seq_len, :].unsqueeze(0)
        inv_x = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
        return x * cos_pos_emb + inv_x * sin_pos_emb

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        # Reshape q, k, v to match the number of heads
        # (b_size, n_heads, seq_len, head_dim)
        q = q.view(
            q.size(0), q.size(1), self.config.num_heads, self.head_dim
        ).transpose(1, 2)
        k = k.view(
            k.size(0), k.size(1), self.config.num_heads, self.head_dim
        ).transpose(1, 2)
        v = v.view(
            v.size(0), v.size(1), self.config.num_heads, self.head_dim
        ).transpose(1, 2)
        # Apply rope embedding if provided
        if self.is_rope:
            for i in range(self.config.num_heads):
                q[:, i, :, :] = self.apply_rope_embedding(q[:, i, :, :])
                k[:, i, :, :] = self.apply_rope_embedding(k[:, i, :, :])
        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Apply mask by setting masked positions to -inf
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, torch.finfo(torch.float).min)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        # Join back the heads
        batch_size, _, seq_len, _ = output.size()
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        return output
