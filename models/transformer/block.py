from typing import Optional

import torch
from torch import nn

from models.transformer.attention import MultiHeadAttention
from models.transformer.feed_forward import FeedForwardLayer
from models.transformer.layer_norm import LayerNorm, RMSNorm


class EncoderBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, ffn_size: int, dropout_rate: float
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.ffn = FeedForwardLayer(hidden_size, ffn_size)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.attn_norm = LayerNorm(hidden_size)
        self.ffn_norm = LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pre_x = x
        x = self.attn(x, mask=mask)
        x = self.attn_dropout(x)
        x = self.attn_norm(pre_x + x)
        pre_x = x
        x = self.ffn(x)
        x = self.ffn_dropout(x)
        x = self.ffn_norm(pre_x + x)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, ffn_size: int, dropout_rate: float
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.cross_attn = MultiHeadAttention(hidden_size, num_heads)
        self.ffn = FeedForwardLayer(hidden_size, ffn_size)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.cross_attn_dropout = nn.Dropout(dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.attn_norm = LayerNorm(hidden_size)
        self.ffn_norm = LayerNorm(hidden_size)

    def forward(
        self, x: torch.Tensor, from_enc: torch.Tensor, causal_mask: torch.Tensor
    ) -> torch.Tensor:
        # Masked self-attention
        pre_x = x
        x = self.attn(x, mask=causal_mask)
        x = self.attn_dropout(x)
        x = self.attn_norm(pre_x + x)
        # Cross-attention
        pre_x = x
        x = self.cross_attn(x, enc=from_enc)
        x = self.cross_attn_dropout(x)
        x = self.attn_norm(pre_x + x)
        # Feed-forward
        pre_x = x
        x = self.ffn(x)
        x = self.ffn_dropout(x)
        x = self.ffn_norm(pre_x + x)

        return x


class DecoderOnlyBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.ffn_norm = RMSNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.ffn = FeedForwardLayer(hidden_size, ffn_size)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        # Attention with causal mask
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, mask=causal_mask, freqs_cis=freqs_cis)
        x = residual + x
        # Feed-forward
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x
