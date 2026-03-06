"""
losses.py — All training loss functions for VoiceAI.

Stage 1 (Acoustic Model):
  - Mel Loss (MAE + MSE)
  - Duration Loss (MSE on log-scale)
  - Pitch Loss (MSE)
  - Energy Loss (MSE)

Stage 2 (Vocoder - Adversarial):
  - Generator GAN Loss
  - Discriminator Loss
  - Feature Matching Loss
  - Mel Reconstruction Loss (L1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Masking utility
# ---------------------------------------------------------------------------
def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    MSE averaged over non-padded positions.
    pred, target: (B, T) or (B, T, C)
    mask: (B, T), True = valid position (NOT padded)
    """
    loss = F.mse_loss(pred, target, reduction="none")
    if mask is not None:
        if loss.dim() == 3:
            mask = mask.unsqueeze(-1)
        loss = loss * mask.float()
        return loss.sum() / (mask.float().sum() * (loss.size(-1) if loss.dim() == 3 else 1) + 1e-8)
    return loss.mean()


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """MAE averaged over non-padded positions."""
    loss = F.l1_loss(pred, target, reduction="none")
    if mask is not None:
        if loss.dim() == 3:
            mask = mask.unsqueeze(-1)
        loss = loss * mask.float()
        return loss.sum() / (mask.float().sum() * (loss.size(-1) if loss.dim() == 3 else 1) + 1e-8)
    return loss.mean()


# ---------------------------------------------------------------------------
# Stage 1: Acoustic Loss
# ---------------------------------------------------------------------------
class AcousticLoss(nn.Module):
    """
    Combined loss for the acoustic model.

    mel_weight, dur_weight, pitch_weight, energy_weight control the
    contribution of each term.
    """

    def __init__(
        self,
        mel_weight: float = 1.0,
        dur_weight: float = 1.0,
        pitch_weight: float = 1.0,
        energy_weight: float = 1.0,
    ):
        super().__init__()
        self.mel_w = mel_weight
        self.dur_w = dur_weight
        self.pit_w = pitch_weight
        self.ene_w = energy_weight

    def forward(
        self,
        # Predictions
        mel_pred: torch.Tensor,           # (B, T_frame, n_mels)
        dur_pred: torch.Tensor,           # (B, T_text)  log-scale
        pitch_pred: torch.Tensor,         # (B, T_frame)
        energy_pred: torch.Tensor,        # (B, T_frame)
        # Ground truth
        mel_gt: torch.Tensor,             # (B, T_frame, n_mels)
        dur_gt: torch.Tensor,             # (B, T_text)  raw integer durations
        pitch_gt: torch.Tensor,           # (B, T_frame)
        energy_gt: torch.Tensor,          # (B, T_frame)
        # Masks (True = valid / not padded)
        src_mask: torch.Tensor | None = None,  # (B, T_text)
        mel_mask: torch.Tensor | None = None,  # (B, T_frame)
    ) -> dict[str, torch.Tensor]:
        # Mel loss: MAE + MSE (PostNet-style combined objective)
        mel_mae = masked_mae(mel_pred, mel_gt, ~mel_mask if mel_mask is not None else None)
        mel_mse = masked_mse(mel_pred, mel_gt, ~mel_mask if mel_mask is not None else None)
        mel_loss = (mel_mae + mel_mse) * 0.5

        # Duration: predict log(dur + 1), supervise with log(gt + 1)
        log_dur_gt = torch.log(dur_gt.float() + 1.0)
        dur_loss = masked_mse(dur_pred, log_dur_gt, ~src_mask if src_mask is not None else None)

        # Pitch & Energy
        pitch_loss = masked_mse(pitch_pred, pitch_gt, ~mel_mask if mel_mask is not None else None)
        energy_loss = masked_mse(energy_pred, energy_gt, ~mel_mask if mel_mask is not None else None)

        total = (
            self.mel_w * mel_loss
            + self.dur_w * dur_loss
            + self.pit_w * pitch_loss
            + self.ene_w * energy_loss
        )

        return {
            "total":  total,
            "mel":    mel_loss,
            "dur":    dur_loss,
            "pitch":  pitch_loss,
            "energy": energy_loss,
        }


# ---------------------------------------------------------------------------
# Stage 2: Vocoder Adversarial Losses
# ---------------------------------------------------------------------------

def discriminator_loss(
    real_outs: list[torch.Tensor],
    fake_outs: list[torch.Tensor],
) -> torch.Tensor:
    """
    Hinge GAN discriminator loss.
    D wants real → 1, fake → -1.
    """
    loss = 0.0
    for r, f in zip(real_outs, fake_outs):
        loss += torch.mean(F.relu(1.0 - r)) + torch.mean(F.relu(1.0 + f))
    return loss / len(real_outs)


def generator_loss(fake_outs: list[torch.Tensor]) -> torch.Tensor:
    """
    Generator tries to fool D → fake should have high (positive) scores.
    """
    loss = 0.0
    for f in fake_outs:
        loss += torch.mean(F.relu(1.0 - f))
    return loss / len(fake_outs)


def feature_matching_loss(
    real_fmaps: list[list[torch.Tensor]],
    fake_fmaps: list[list[torch.Tensor]],
) -> torch.Tensor:
    """
    L1 distance between discriminator intermediate feature maps.
    Stabilises GAN training significantly (Larsen et al. 2016).
    """
    loss = 0.0
    n = 0
    for real_disc_fmaps, fake_disc_fmaps in zip(real_fmaps, fake_fmaps):
        for r, f in zip(real_disc_fmaps, fake_disc_fmaps):
            loss += F.l1_loss(f, r.detach())
            n += 1
    return loss / max(n, 1)


def mel_reconstruction_loss(
    real_wav: torch.Tensor,
    fake_wav: torch.Tensor,
    mel_fn,                       # callable: wav → mel (e.g. torchaudio transform)
) -> torch.Tensor:
    """
    L1 loss in mel space between real and generated waveforms.
    Provides low-frequency guidance to the vocoder generator.
    """
    real_mel = mel_fn(real_wav.squeeze(1))
    fake_mel = mel_fn(fake_wav.squeeze(1))
    return F.l1_loss(fake_mel, real_mel)
