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
        self.attn_norm = LayerNorm(hidden_size)
        self.ffn_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.attn_norm(x + self.dropout(self.attn(x, x, x, mask=mask)))
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, ffn_size: int, dropout_rate: float
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.cross_attn = MultiHeadAttention(hidden_size, num_heads)
        self.ffn = FeedForwardLayer(hidden_size, ffn_size)
        self.attn_norm = LayerNorm(hidden_size)
        self.cross_attn_norm = LayerNorm(hidden_size)
        self.ffn_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Masked self-attention
        x = self.attn_norm(x + self.dropout(self.attn(x, x, x, mask=tgt_mask)))
        # Cross-attention
        x = self.cross_attn_norm(
            x + self.dropout(self.cross_attn(x, enc, enc, mask=src_mask))
        )
        # Feed-forward
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))

        return x


class DecoderOnlyBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, ffn_size: int, dropout_rate: float
    ) -> None:
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size)
        self.post_norm = RMSNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.ffn = FeedForwardLayer(hidden_size, ffn_size)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # Attention with causal mask
        residual = x
        x = self.pre_norm(x)
        x = self.attn(x, mask=causal_mask)
        x = residual + x
        # Feed-forward
        residual = x
        x = self.post_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x
