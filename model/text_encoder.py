"""
text_encoder.py — Phoneme/character encoder with positional encoding.

Architecture:
  Embedding → Positional Encoding → N × Transformer Encoder Layers
  → output shape: (B, T_text, hidden_dim)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TextEncoderConfig


# ---------------------------------------------------------------------------
# Positional Encoding (sinusoidal, non-learned)
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Classic sine/cosine positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Feed-Forward sublayer (position-wise)
# ---------------------------------------------------------------------------
class PositionWiseFF(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


# ---------------------------------------------------------------------------
# Transformer Encoder Layer
# ---------------------------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = PositionWiseFF(d_model, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, T, d_model)
        src_key_padding_mask: (B, T) — True where padded
        """
        attn_out, _ = self.self_attn(
            x, x, x, key_padding_mask=src_key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        x = self.ff(x)
        return x


# ---------------------------------------------------------------------------
# Text Encoder
# ---------------------------------------------------------------------------
class TextEncoder(nn.Module):
    """
    Maps a phoneme/character sequence → contextual hidden states.

    Input:
        tokens      : (B, T_text)   LongTensor of phoneme indices
        padding_mask: (B, T_text)   BoolTensor, True = padding position

    Output:
        (B, T_text, hidden_dim)
    """

    def __init__(self, cfg: TextEncoderConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim

        self.embedding = nn.Embedding(cfg.vocab_size, d, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(d, cfg.max_seq_len, cfg.dropout)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d, cfg.num_heads, cfg.ff_dim, cfg.dropout)
                for _ in range(cfg.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(tokens)  # (B, T, d)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        return self.norm(x)  # (B, T, d)
