"""
emotion_encoder.py — Emotion & expression conditioning module.

Produces a conditioning vector that is broadcast-added to the hidden states
of both the text encoder output and the acoustic decoder input.

Supports:
  • Categorical emotion (one of N classes)
  • Continuous intensity scalar [0, 1] (how strong the emotion is)
  • Style token lookup (Global Style Tokens, Wang et al. 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EmotionConfig


# ---------------------------------------------------------------------------
# Global Style Token (GST) bank — optional learnable reference embeddings
# ---------------------------------------------------------------------------
class GlobalStyleTokens(nn.Module):
    """
    A bank of K learnable style tokens attended-to by a reference embedding.
    At training time the reference can come from the target audio's embedding;
    at inference time it comes from the emotion label directly.
    """

    def __init__(self, num_tokens: int = 10, token_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, token_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        query : (B, 1, token_dim)  — emotion embedding used as query
        returns : (B, 1, token_dim)
        """
        tokens = self.tokens.unsqueeze(0).expand(query.size(0), -1, -1)  # (B, K, D)
        out, _ = self.attn(query, tokens, tokens)
        return out  # (B, 1, token_dim)


# ---------------------------------------------------------------------------
# Main Emotion Encoder
# ---------------------------------------------------------------------------
class EmotionEncoder(nn.Module):
    """
    Maps (emotion_id, intensity) → conditioning vector of shape (B, hidden_dim).

    The conditioning vector is later broadcast-added to text-encoder hidden
    states (element-wise), making emotion influence every phoneme.
    """

    def __init__(self, cfg: EmotionConfig, hidden_dim: int):
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dim

        # Learnable emotion embedding table
        self.emotion_emb = nn.Embedding(cfg.num_emotions, D)

        # Optional intensity MLP  (scalar → D-dim vector)
        if cfg.use_intensity:
            self.intensity_proj = nn.Sequential(
                nn.Linear(1, D // 2),
                nn.Tanh(),
                nn.Linear(D // 2, D),
            )

        # Global Style Tokens for finer stylistic control
        self.gst = GlobalStyleTokens(num_tokens=cfg.num_emotions, token_dim=D)

        # Final linear to project to model's hidden_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, hidden_dim),
            nn.Tanh(),          # Keeps conditioning bounded
        )

    def forward(
        self,
        emotion_ids: torch.Tensor,          # (B,)  LongTensor
        intensity: torch.Tensor | None = None,  # (B,)  FloatTensor in [0,1]
    ) -> torch.Tensor:
        """
        Returns emotion conditioning vector: (B, hidden_dim).
        Add this (unsqueezed) to hidden states: hidden + emotion_vec.unsqueeze(1)
        """
        # Base emotion embedding
        e = self.emotion_emb(emotion_ids)                   # (B, D)

        # Blend with intensity if provided
        if self.cfg.use_intensity and intensity is not None:
            i_vec = self.intensity_proj(intensity.unsqueeze(-1))  # (B, D)
            e = e + i_vec

        # Global Style Token refinement
        gst_out = self.gst(e.unsqueeze(1))                  # (B, 1, D)
        e = e + gst_out.squeeze(1)                          # (B, D)

        return self.output_proj(e)                          # (B, hidden_dim)
