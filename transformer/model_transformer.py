"""
Transformer model for the plateau region, matching PlateauGRU's design
and interface exactly, so it drops into the existing train.py without
any changes there -- just swap the import and the model constructor.

Same design choice as PlateauGRU:
  - The model does NOT predict absolute SoC.
  - It predicts, at every timestep, the delta-SoC since the session's
    anchor (the OCV-table value at plateau entry).
  - Absolute SoC is reconstructed as anchor + delta at inference time.

One difference from the GRU that matters here: a vanilla transformer
encoder attends bidirectionally across the whole sequence by default,
which would let it use *future* timesteps to predict the current one.
The GRU can't do that -- it's recurrent, so it only ever sees the past.
Since this model is meant for real-time SoC estimation (the BMS doesn't
have future voltage samples at inference time), a causal mask is applied
so each timestep can only attend to itself and earlier timesteps.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, added to the input embeddings
    so the transformer has a notion of timestep order (it has none natively,
    unlike the GRU which is inherently sequential)."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1), :]


class PlateauTransformer(nn.Module):
    """
    Causal transformer encoder over a sequence of per-timestep features
    (e.g. smoothed voltage, dV/dt, dV/dQ, temperature).

    Input:  x       (B, T, input_size) padded feature sequences
            lengths (B,) true sequence lengths before padding
    Output: (B, T) predicted delta-SoC at every timestep (padding included,
             caller is responsible for masking with `lengths` -- same
             contract as PlateauGRU).
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 32, num_layers: int = 1,
                 num_heads: int = 4, dropout: float = 0.0, max_len: int = 2000):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device

        # Padding positions get -inf (ignored in attention entirely), same
        # float dtype as the causal mask below -- PyTorch expects both
        # masks in matching dtype when both are passed together.
        arange = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        is_pad = arange >= lengths.to(device).unsqueeze(1)  # (B, T), True = pad
        key_padding_mask = torch.zeros(B, T, device=device)
        key_padding_mask.masked_fill_(is_pad, float("-inf"))

        # Causal mask: position i can only attend to positions <= i.
        # Matches the GRU's past-only behavior for real-time inference.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        h = self.input_proj(x)
        h = self.pos_encoding(h)
        out = self.encoder(h, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        delta = self.head(out).squeeze(-1)  # (B, T)
        return delta