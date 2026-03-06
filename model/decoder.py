"""
decoder.py — Mel-spectrogram Acoustic Decoder.

Architecture:
  Transformer-based, frame-level (one token per mel frame).
  Uses the same Pre-LN Transformer block as the encoder.
  Receives frame-level hidden states from the Variance Adaptor.
"""

import math
import torch
import torch.nn as nn

from .config import DecoderConfig
from .text_encoder import SinusoidalPositionalEncoding, TransformerEncoderLayer


class AcousticDecoder(nn.Module):
    """
    Frame-level transformer decoder → mel spectrogram.

    Input:
        x        : (B, T_frame, hidden_dim)   from VarianceAdaptor
        mel_mask : (B, T_frame)               True = padding

    Output:
        mel: (B, T_frame, n_mels)  — predicted mel spectrogram
    """

    def __init__(self, cfg: DecoderConfig, n_mels: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Optional learned linear to map hidden_dim → cfg.hidden_dim
        # (allows encoder and decoder to have different widths)
        self.input_proj = (
            nn.Identity()
            if hidden_dim == cfg.hidden_dim
            else nn.Linear(hidden_dim, cfg.hidden_dim)
        )

        self.pos_enc = SinusoidalPositionalEncoding(
            cfg.hidden_dim, cfg.max_seq_len, cfg.dropout
        )

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    cfg.hidden_dim, cfg.num_heads, cfg.ff_dim, cfg.dropout
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)

        # Linear projection to mel dimensions
        self.mel_proj = nn.Linear(cfg.hidden_dim, n_mels)

    def forward(
        self,
        x: torch.Tensor,
        mel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)       # (B, T, cfg.hidden_dim)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mel_mask)

        x = self.norm(x)
        mel = self.mel_proj(x)       # (B, T, n_mels)
        return mel
