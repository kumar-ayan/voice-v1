"""
prosody.py — Variance Adaptor (Duration, Pitch, Energy predictor).

This is the heart of FastSpeech2's expressiveness.
It predicts:
  1. Duration  → how long each phoneme lasts (length regulator)
  2. Pitch (F0) → fundamental frequency contour (melody / intonation)
  3. Energy     → amplitude / loudness contour

All three are conditioned on the emotion vector so that e.g.
"excited" speech becomes faster, higher-pitched, and louder automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ProsodyConfig


# ---------------------------------------------------------------------------
# Shared conv predictor backbone
# ---------------------------------------------------------------------------
class ConvPredictor(nn.Module):
    """
    2-layer 1-D CNN predictor used for duration, pitch, and energy.
    Input : (B, T, d_model)
    Output: (B, T, 1)
    """

    def __init__(self, d_model: int, hidden: int, kernel: int, dropout: float):
        super().__init__()
        padding = (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel, padding=padding),
            nn.ReLU(),
            nn.LayerNorm(hidden),       # Applied after transpose below
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel, padding=padding),
            nn.ReLU(),
        )
        # Per-layer norms applied in forward with transposes
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → conv expects (B, C, T)
        h = x.transpose(1, 2)
        h = F.relu(h)

        # Layer 1
        h = self.net[0](h)              # Conv1d
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = F.relu(h)
        h = self.net[3](h)              # Dropout (applied after norm)

        # Layer 2
        h = self.net[4](h)              # Conv1d
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = F.relu(h)

        out = self.linear(h.transpose(1, 2))  # (B, T, 1)
        return out.squeeze(-1)               # (B, T)


# ---------------------------------------------------------------------------
# Length Regulator — expand encoder output to frame-level
# ---------------------------------------------------------------------------
class LengthRegulator(nn.Module):
    """
    Expands phoneme-level features to frame-level using predicted durations.
    During training, ground-truth durations are used.
    During inference, predicted durations (rounded) are used.
    """

    def forward(
        self,
        x: torch.Tensor,            # (B, T_phoneme, D)
        durations: torch.Tensor,    # (B, T_phoneme) — integer frame counts
        max_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          output: (B, T_frame, D)  upsampled sequence
          mel_len: (B,)            actual frame lengths per item
        """
        outputs = []
        mel_lens = []
        for i in range(x.size(0)):
            # Repeat each phoneme embedding by its duration
            expanded = torch.repeat_interleave(x[i], durations[i].long(), dim=0)
            outputs.append(expanded)
            mel_lens.append(expanded.size(0))

        mel_lens_t = torch.tensor(mel_lens, dtype=torch.long, device=x.device)

        # Pad all sequences to same length
        if max_len is None:
            max_len = max(mel_lens)
        output = torch.zeros(x.size(0), max_len, x.size(-1), device=x.device)
        for i, seq in enumerate(outputs):
            L = min(seq.size(0), max_len)
            output[i, :L] = seq[:L]

        return output, mel_lens_t


# ---------------------------------------------------------------------------
# Main Variance Adaptor
# ---------------------------------------------------------------------------
class VarianceAdaptor(nn.Module):
    """
    Complete prosody control block.

    Forward inputs:
        x            : (B, T_phoneme, D)   — encoder hidden states
        emotion_vec  : (B, D)              — from EmotionEncoder
        gt_durations : (B, T_phoneme) | None  — teacher-forced at training
        gt_pitch     : (B, T_phoneme) | None
        gt_energy    : (B, T_phoneme) | None
        duration_scale : float — multiply predicted durations (speed control, inference)
        pitch_scale    : float — multiply predicted pitch     (voice height control)
        energy_scale   : float — multiply predicted energy    (loudness control)

    Returns:
        output   : (B, T_frame, D)   frame-level features
        mel_lens : (B,)
        pred     : dict with {'duration', 'pitch', 'energy'}  (for loss computation)
    """

    def __init__(self, cfg: ProsodyConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim
        K = cfg.kernel_size
        drop = cfg.dropout

        # Predictors (all share the same architecture, different weights)
        self.duration_predictor = ConvPredictor(d_model, H, K, drop)
        self.pitch_predictor = ConvPredictor(d_model, H, K, drop)
        self.energy_predictor = ConvPredictor(d_model, H, K, drop)

        # Pitch / energy quantisation embeddings (like FastSpeech2)
        # The predicted scalar is bucketed, then embedded and added to x.
        self.pitch_bins = nn.Parameter(
            torch.linspace(-3, 3, cfg.n_pitch_bins - 1), requires_grad=False
        )
        self.energy_bins = nn.Parameter(
            torch.linspace(-3, 3, cfg.n_energy_bins - 1), requires_grad=False
        )
        self.pitch_embedding = nn.Embedding(cfg.n_pitch_bins, d_model)
        self.energy_embedding = nn.Embedding(cfg.n_energy_bins, d_model)

        self.length_regulator = LengthRegulator()

    # ------------------------------------------------------------------
    # Helper: standardise a raw value using dataset statistics
    # ------------------------------------------------------------------
    def _normalize_pitch(self, p: torch.Tensor) -> torch.Tensor:
        return (p - self.cfg.pitch_mean) / (self.cfg.pitch_std + 1e-8)

    def _normalize_energy(self, e: torch.Tensor) -> torch.Tensor:
        return (e - self.cfg.energy_mean) / (self.cfg.energy_std + 1e-8)

    def _embed_pitch(self, p: torch.Tensor) -> torch.Tensor:
        """p: (B, T) normalised log-F0 → (B, T, D)"""
        buckets = torch.bucketize(p, self.pitch_bins)
        return self.pitch_embedding(buckets)

    def _embed_energy(self, e: torch.Tensor) -> torch.Tensor:
        buckets = torch.bucketize(e, self.energy_bins)
        return self.energy_embedding(buckets)

    def forward(
        self,
        x: torch.Tensor,
        emotion_vec: torch.Tensor,
        gt_durations: torch.Tensor | None = None,
        gt_pitch: torch.Tensor | None = None,
        gt_energy: torch.Tensor | None = None,
        duration_scale: float = 1.0,
        pitch_scale: float = 1.0,
        energy_scale: float = 1.0,
        max_mel_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # Inject emotion into predictor inputs (broadcast add)
        ex = x + emotion_vec.unsqueeze(1)   # (B, T, D)

        # --- Duration ---
        log_dur_pred = self.duration_predictor(ex)           # (B, T)
        dur_pred = torch.clamp(
            torch.exp(log_dur_pred) - 1, min=0              # softplus-like
        ) * duration_scale

        if gt_durations is not None:
            durations_for_lr = gt_durations.float()
        else:
            durations_for_lr = torch.round(dur_pred).long()

        # --- Length Regulate (phoneme→frame) ---
        x_frame, mel_lens = self.length_regulator(x, durations_for_lr, max_mel_len)
        ex_frame = x_frame + emotion_vec.unsqueeze(1)       # (B, T_frame, D)

        # --- Pitch ---
        pitch_pred = self.pitch_predictor(ex_frame)          # (B, T_frame)
        if gt_pitch is not None:
            pitch_norm = self._normalize_pitch(gt_pitch)
            x_frame = x_frame + self._embed_pitch(pitch_norm)
        else:
            pitch_val = pitch_pred * pitch_scale
            x_frame = x_frame + self._embed_pitch(self._normalize_pitch(pitch_val))

        # --- Energy ---
        energy_pred = self.energy_predictor(ex_frame)        # (B, T_frame)
        if gt_energy is not None:
            energy_norm = self._normalize_energy(gt_energy)
            x_frame = x_frame + self._embed_energy(energy_norm)
        else:
            energy_val = energy_pred * energy_scale
            x_frame = x_frame + self._embed_energy(self._normalize_energy(energy_val))

        predictions = {
            "duration": log_dur_pred,   # log-space, for MSE loss vs log(gt+1)
            "pitch": pitch_pred,
            "energy": energy_pred,
        }

        return x_frame, mel_lens, predictions
