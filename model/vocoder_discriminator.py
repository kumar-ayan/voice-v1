"""
vocoder_discriminator.py — HiFi-GAN Discriminators (MPD + MSD).

Used ONLY during adversarial training of the vocoder.
Not required for acoustic model training or inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from .config import VocoderConfig


# ---------------------------------------------------------------------------
# Multi-Period Discriminator (MPD)
# ---------------------------------------------------------------------------
class PeriodDiscriminator(nn.Module):
    """Discriminates real vs generated audio across a fixed period p."""

    def __init__(self, period: int, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm = spectral_norm if use_spectral_norm else weight_norm

        channels = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            stride = (3, 1) if i < 4 else (1, 1)
            padding = (2, 0) if i < 4 else (1, 0)
            self.convs.append(
                norm(
                    nn.Conv2d(channels[i], channels[i + 1], (5, 1), stride=stride, padding=padding)
                )
            )
        self.conv_post = norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """x: (B, 1, T)"""
        fmaps = []
        B, C, T = x.shape
        # Pad to be divisible by period
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            x = F.pad(x, (0, pad_len), "reflect")
            T = x.shape[-1]
        x = x.view(B, C, T // self.period, self.period)  # (B, 1, T/p, p)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmaps.append(x)

        x = self.conv_post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(p) for p in periods]
        )

    def forward(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> tuple[list, list, list, list]:
        real_outs, fake_outs, real_fmaps, fake_fmaps = [], [], [], []
        for d in self.discriminators:
            r_out, r_fmap = d(real)
            f_out, f_fmap = d(fake)
            real_outs.append(r_out)
            fake_outs.append(f_out)
            real_fmaps.append(r_fmap)
            fake_fmaps.append(f_fmap)
        return real_outs, fake_outs, real_fmaps, fake_fmaps


# ---------------------------------------------------------------------------
# Multi-Scale Discriminator (MSD)
# ---------------------------------------------------------------------------
class ScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1,   128,  15, stride=1, padding=7)),
            norm(nn.Conv1d(128, 128,  41, stride=2, padding=20, groups=4)),
            norm(nn.Conv1d(128, 256,  41, stride=2, padding=20, groups=16)),
            norm(nn.Conv1d(256, 512,  41, stride=4, padding=20, groups=16)),
            norm(nn.Conv1d(512, 1024, 41, stride=4, padding=20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 41, stride=1, padding=20, groups=16)),
            norm(nn.Conv1d(1024, 1024,  5, stride=1, padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list]:
        fmaps = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),  # first one uses SN
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> tuple[list, list, list, list]:
        real_outs, fake_outs, real_fmaps, fake_fmaps = [], [], [], []
        for pool, d in zip(self.pools, self.discriminators):
            r = pool(real)
            f = pool(fake)
            r_out, r_fmap = d(r)
            f_out, f_fmap = d(f)
            real_outs.append(r_out)
            fake_outs.append(f_out)
            real_fmaps.append(r_fmap)
            fake_fmaps.append(f_fmap)
        return real_outs, fake_outs, real_fmaps, fake_fmaps
