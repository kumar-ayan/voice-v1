"""
vocoder.py — HiFi-GAN Neural Vocoder (from scratch).

Converts mel spectrograms → raw waveform audio.

Reference: Kong et al., "HiFi-GAN: Generative Adversarial Networks for
           Efficient and High Fidelity Speech Synthesis" (2020).

This is the Generator only. Discriminator networks (MPD + MSD) used during
adversarial training are in vocoder_discriminator.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VocoderConfig


# ---------------------------------------------------------------------------
# Multi-Receptive Field Fusion (MRF) residual block
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """
    Stacked dilated convolutions with different dilation rates.
    Applied separately for each kernel size in the MRF bank.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation_sizes: list[int] | None = None,
        leaky_slope: float = 0.1,
    ):
        super().__init__()
        if dilation_sizes is None:
            dilation_sizes = [1, 3, 5]
        self.slope = leaky_slope

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilation_sizes:
            pad = (kernel_size * d - d) // 2
            self.convs1.append(
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=pad)
            )
            self.convs2.append(
                nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)
            )

        self.convs1.apply(self._init_weights)
        self.convs2.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, 0.0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            # Residual path 1
            xt = F.leaky_relu(x, self.slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.slope)
            xt = c2(xt)
            x = x + xt
        return x


# ---------------------------------------------------------------------------
# HiFi-GAN Generator
# ---------------------------------------------------------------------------
class HiFiGANGenerator(nn.Module):
    """
    Mel → waveform.

    Example upsample factors: [8, 8, 2, 2] → 256× total
    With hop_length=256 and sr=22050 → correct time resolution.
    """

    def __init__(self, cfg: VocoderConfig):
        super().__init__()
        self.cfg = cfg
        slope = cfg.leaky_relu_slope
        ch = cfg.upsample_initial_channel

        # Pre-conv
        self.conv_pre = nn.Conv1d(cfg.in_channels, ch, 7, padding=3)

        # Upsampling blocks
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
            out_ch = ch // (2 ** (i + 1))
            self.ups.append(
                nn.ConvTranspose1d(
                    ch // (2 ** i),
                    out_ch,
                    k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )
            # MRF block: one ResBlock per resblock_kernel_size
            mrf = nn.ModuleList()
            for rk, rd in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
                mrf.append(ResBlock(out_ch, rk, rd, slope))
            self.resblocks.append(mrf)

        # Post-conv → 1 output channel (waveform)
        final_ch = ch // (2 ** len(cfg.upsample_rates))
        self.conv_post = nn.Conv1d(final_ch, 1, 7, padding=3)
        self.slope = slope

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel : (B, n_mels, T_mel)   — note: channel-first
        returns : (B, 1, T_wav)    — raw waveform
        """
        x = self.conv_pre(mel)

        for i, (up, mrf_list) in enumerate(zip(self.ups, self.resblocks)):
            x = F.leaky_relu(x, self.slope)
            x = up(x)

            # Average outputs of MRF residual blocks
            xs = None
            for mrf in mrf_list:
                xs = mrf(x) if xs is None else xs + mrf(x)
            x = xs / len(mrf_list)

        x = F.leaky_relu(x, self.slope)
        x = self.conv_post(x)
        x = torch.tanh(x)          # Output in [-1, 1]
        return x
