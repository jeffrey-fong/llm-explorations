"""Module containing Transformer and Attention implementations."""

import math
from typing import Literal, Optional

import torch
from torch import nn

from models.transformer.block import DecoderBlock, DecoderOnlyBlock, EncoderBlock
from models.transformer.embedding import SinusoidalEmbedding, get_rope_freqs


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

        self.hidden_size = 512
        self.num_heads = 8
        self.ffn_size = 2048
        self.dropout_rate = 0.1
        self.num_layers = 6
        self.activation = "relu"

        self.input_emb = SinusoidalEmbedding(
            seq_len, self.hidden_size, self.vocab_size, device
        )
        self.target_emb = SinusoidalEmbedding(
            seq_len, self.hidden_size, self.vocab_size, device
        )
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    self.hidden_size,
                    self.num_heads,
                    self.ffn_size,
                    self.dropout_rate,
                    self.activation,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    self.hidden_size,
                    self.num_heads,
                    self.ffn_size,
                    self.dropout_rate,
                    self.activation,
                )
                for _ in range(self.num_layers)
            ]
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
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        inputs = self.input_emb(input_ids)
        encoder_mask = self._make_encoder_mask(input_ids)
        for block in self.encoder:
            inputs = block(inputs, encoder_mask)

        targets = self.target_emb(labels)
        decoder_mask = self._make_decoder_mask(labels, targets)
        for block in self.decoder:
            targets = block(targets, inputs, decoder_mask)
        logits = self.lm_head(targets)

        # Assuming we have labels for computing the loss
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

        self.hidden_size = 512
        self.num_heads = 8
        self.ffn_size = 2048
        self.num_layers = 12
        self.theta = 10000
        self.activation = "swiglu"

        self.freqs_cis = get_rope_freqs(
            self.theta, seq_len, self.hidden_size // self.num_heads, device
        )

        self.input_emb = nn.Embedding(
            self.vocab_size, self.hidden_size, padding_idx=pad_id
        )
        self.decoder = nn.ModuleList(
            [
                DecoderOnlyBlock(
                    self.hidden_size, self.num_heads, self.ffn_size, self.activation
                )
                for _ in range(self.num_layers)
            ]
        )
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)

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
            inputs = block(inputs, causal_mask, self.freqs_cis)
        logits = self.lm_head(inputs)

        # Assuming we have labels for computing the loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        return logits, loss
