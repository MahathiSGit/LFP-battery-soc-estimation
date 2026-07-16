"""
GRU model for the plateau region.
Design choice, matching the workflow diagram:
  - The model does NOT predict absolute SoC.
  - It predicts, at every timestep, the delta-SoC since the session's
    anchor (the OCV-table value at plateau entry).
  - Absolute SoC is reconstructed as anchor + delta at inference time.
This keeps the network's job well-posed (learn a bounded correction)
instead of ill-posed (reconstruct SoC from a flat voltage curve).
"""
import torch
import torch.nn as nn


class PlateauGRU(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 32, num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, T, input_size) padded feature sequences
        lengths: (B,) true sequence lengths before padding
        returns: (B, T) predicted delta-SoC at every timestep (padding included,
                 caller is responsible for masking with `lengths`)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        delta = self.head(out).squeeze(-1)  # (B, T)
        return delta