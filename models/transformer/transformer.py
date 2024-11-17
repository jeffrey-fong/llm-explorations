"""Module containing Transformer and Attention implementations."""

import math
from typing import Literal, Optional

import torch
from torch import nn

from models.transformer.block import DecoderBlock, DecoderOnlyBlock, EncoderBlock
from models.transformer.embedding import RopeEmbedding, SinusoidalEmbedding


class ClassicalTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        pad_id: int,
        device: Literal["cpu", "cuda"],
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.input_emb = SinusoidalEmbedding(seq_len, 512, self.vocab_size, device)
        self.target_emb = SinusoidalEmbedding(seq_len, 512, self.vocab_size, device)
        self.encoder = nn.ModuleList(
            [EncoderBlock(512, 8, 2048, 0.1) for _ in range(6)]
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(512, 8, 2048, 0.1) for _ in range(6)]
        )
        self.lm_head = nn.Linear(512, self.vocab_size)

    def _make_encoder_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Creates encoder mask which prevents attending to padding tokens."""
        mask = (x == self.pad_id).unsqueeze(1).unsqueeze(1)
        # mask = mask.unsqueeze(-1) | mask.unsqueeze(-2)
        return mask.to(x.device)

    def _make_decoder_mask(self, x: torch.Tensor, x_emb: torch.Tensor) -> torch.Tensor:
        """Creates decoder mask which prevents attending to future and padding tokens."""
        # Create padding mask
        pad_mask = (
            (x == self.pad_id).unsqueeze(1).unsqueeze(3).to(x.device)
        )  # (b, 1, s, 1)

        # Create causal mask
        batch_size, seq_len = x_emb.size(0), x_emb.size(1)
        # Create lower triangular matrix
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        # Expand mask for batch size and number of heads
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len).to(x.device)

        return pad_mask | causal_mask

    def forward(
        self, input_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        inputs = self.input_emb(input_ids)
        encoder_mask = self._make_encoder_mask(input_ids)
        for block in self.encoder:
            inputs = block(inputs, encoder_mask)

        # Shift right to get the labels
        labels = target_ids[:, 1:]
        target_ids = target_ids[:, :-1]

        targets = self.target_emb(target_ids)
        decoder_mask = self._make_decoder_mask(target_ids, targets)
        for block in self.decoder:
            targets = block(targets, inputs, encoder_mask, decoder_mask)
        logits = self.lm_head(targets)

        # Assuming we have labels for computing the loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        loss = loss_fct(
            logits.contiguous().view(-1, self.vocab_size),
            labels.contiguous().view(-1),
        )

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
    ):
        # TODO: Implement generation
        pass


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        pad_id: int,
        device: Literal["cpu", "cuda"],
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.input_emb = RopeEmbedding(seq_len, 512, self.vocab_size, 10000, device)
        self.decoder = nn.ModuleList(
            [DecoderOnlyBlock(512, 8, 2048, 0.1) for _ in range(6)]
        )
        self.lm_head = nn.Linear(512, self.vocab_size)

    def _make_padded_causal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a causal mask which prevents attending to padding tokens."""
        # Create padding mask (b_size, 1, seq_len, 1)
        pad_mask = (x == self.pad_id).unsqueeze(1).unsqueeze(3).to(x.device)
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
        inputs = self.input_emb(input_ids)

        causal_mask = self._make_padded_causal_mask(input_ids)
        for block in self.decoder:
            inputs = block(inputs, causal_mask)
        logits = self.lm_head(inputs)

        # Assuming we have labels for computing the loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        return logits, loss
