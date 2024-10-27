"""Module containing Transformer and Attention implementations."""

import math
from typing import Literal, Optional

import torch
from torch import nn

from models.transformer.block import DecoderBlock, EncoderBlock
from models.transformer.embedding import SinusoidalEmbedding


class Transformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        device: Literal["cpu", "cuda"],
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.input_emb = SinusoidalEmbedding(seq_len, 512, self.vocab_size, device)
        self.target_emb = SinusoidalEmbedding(seq_len, 512, self.vocab_size, device)
        self.encoder = nn.ModuleList(
            [EncoderBlock(512, 8, 2048, 0.1) for _ in range(6)]
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(512, 8, 2048, 0.1) for _ in range(6)]
        )
        self.lm_head = nn.Linear(512, self.vocab_size)

    def _make_causal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a causal mask to prevent attending to future tokens."""
        batch_size, seq_len = x.size(0), x.size(1)
        # Create lower triangular matrix
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        # Expand mask for batch size and number of heads
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch_size, 1, seq_len, seq_len)
        return mask.to(x.device)

    def forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        inputs = self.input_emb(input_ids)
        for block in self.encoder:
            inputs = block(inputs)

        # TODO: Implement padding masking to prevent attending to padding tokens
        # This should create a mask that is True for padding tokens (usually token_id = 0)
        # and False for regular tokens, then expand it for the attention heads
        # The mask should be applied in both encoder and decoder attention layers

        targets = self.target_emb(labels)
        # Create causal mask to prevent attending to future tokens
        causal_mask = self._make_causal_mask(targets)
        for block in self.decoder:
            targets = block(targets, inputs, causal_mask)
        logits = self.lm_head(targets)

        # Assuming we have labels for computing the loss
        # If labels are not provided in the forward method, you might need to modify the method signature
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
    ):
        # TODO: Implement generation
        pass
