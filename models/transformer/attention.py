import math
from typing import Optional

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Apply mask by setting masked positions to -inf
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, torch.finfo(torch.float).min)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.attention = Attention(self.head_dim)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        enc: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.q_proj(x) if enc is None else self.q_proj(enc)
        k = self.k_proj(x) if enc is None else self.k_proj(enc)
        v = self.v_proj(x)
        # Reshape q, k, v to match the number of heads
        # (b_size, n_heads, seq_len, head_dim)
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        # Compute attention
        output = self.attention(q, k, v, mask=mask)
        # Join back the heads
        batch_size, _, seq_len, _ = output.size()
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        return output
