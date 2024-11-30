from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """Transformer model configuration."""

    seq_len: int
    vocab_size: int
    device: str
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    ffn_size: Optional[int] = None
    num_heads: Optional[int] = None
    dropout_rate: Optional[float] = None
    base: Optional[float] = None  # RoPE
