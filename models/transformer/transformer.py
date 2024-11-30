"""Module containing Transformer and Attention implementations."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from config import TransformerConfig
from models.transformer.block import DecoderBlock


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_layers)]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def _make_padded_causal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a causal mask which prevents attending to padding tokens."""
        # Create padding mask (b_size, 1, seq_len, 1)
        # No padding in this training so pad_mask is always False
        pad_mask = torch.zeros(x.size(0), 1, x.size(1), 1, dtype=torch.bool).to(
            x.device
        )
        # Create causal mask
        batch_size, seq_len = x.size(0), x.size(1)
        # Create upper triangular matrix
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        # Expand mask for batch size and number of heads
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len).to(x.device)

        return pad_mask | causal_mask

    def forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self._make_padded_causal_mask(input_ids)

        inputs = self.token_emb(input_ids)
        for block in self.decoder:
            inputs = block(inputs, causal_mask)
        logits = self.lm_head(inputs)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        return logits, loss
