import torch
from torch import nn

from config import TransformerConfig
from models.transformer.attention import (
    MultiHeadAttention,
    MultiHeadDifferentialAttention,
)
from models.transformer.feed_forward import SwiGLUFeedForwardLayer
from models.transformer.layer_norm import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.attn = MultiHeadAttention(config, is_rope=True)
        self.ffn = SwiGLUFeedForwardLayer(config.hidden_size, config.ffn_size)
        self.rms_norm = RMSNorm((config.seq_len, config.hidden_size))

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # Attention with causal mask
        x = self.rms_norm(x)
        x = x + self.attn(x, x, x, mask=causal_mask)
        # Feed-forward
        x = self.rms_norm(x)
        x = x + self.ffn(x)

        return x


class DifferentialDecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.attn = MultiHeadDifferentialAttention(config, layer_id, is_rope=True)
        self.ffn = SwiGLUFeedForwardLayer(config.hidden_size, config.ffn_size)
        self.rms_norm = RMSNorm((config.seq_len, config.hidden_size))

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # Attention with causal mask
        x = self.rms_norm(x)
        x = x + self.attn(x, x, x, mask=causal_mask)
        # Feed-forward
        x = self.rms_norm(x)
        x = x + self.ffn(x)

        return x
