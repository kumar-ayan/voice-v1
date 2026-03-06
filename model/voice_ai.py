"""
voice_ai.py — Full VoiceAI Model (end-to-end TTS with emotion control).

Pipeline:
  [Text Tokens] ──► TextEncoder
                         │
  [Emotion ID]  ──► EmotionEncoder ──► conditioning vector
                         │
              ┌──────────▼──────────┐
              │   VarianceAdaptor   │  (Duration, Pitch, Energy)
              └──────────┬──────────┘
                         │ frame-level hidden states
              ┌──────────▼──────────┐
              │   AcousticDecoder   │
              └──────────┬──────────┘
                         │ mel spectrogram
              ┌──────────▼──────────┐
              │  HiFiGAN Generator  │  (separate training stage)
              └──────────┬──────────┘
                         │ raw waveform
"""

import torch
import torch.nn as nn

from .config import ModelConfig
from .text_encoder import TextEncoder
from .emotion_encoder import EmotionEncoder
from .prosody import VarianceAdaptor
from .decoder import AcousticDecoder
from .vocoder import HiFiGANGenerator


class VoiceAI(nn.Module):
    """
    End-to-end emotionally expressive TTS model.

    Typical training workflow:
      Stage 1: Train TextEncoder + EmotionEncoder + VarianceAdaptor + AcousticDecoder
               jointly with mel loss + prosody losses. (acoustic_model_parameters())
      Stage 2: Freeze Stage 1. Train HiFiGANGenerator adversarially with
               MPD + MSD discriminators. (vocoder_parameters())

    At inference, call model.synthesize() for a full pipeline forward pass.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim

        self.text_encoder = TextEncoder(cfg.text_encoder)
        self.emotion_encoder = EmotionEncoder(cfg.emotion, D)
        self.variance_adaptor = VarianceAdaptor(cfg.prosody, D)
        self.decoder = AcousticDecoder(cfg.decoder, cfg.mel.n_mels, D)
        self.vocoder = HiFiGANGenerator(cfg.vocoder)

    # ------------------------------------------------------------------
    # Parameter groups for staged training
    # ------------------------------------------------------------------
    def acoustic_model_parameters(self):
        """Returns params for Stage-1 (acoustic model) training."""
        return (
            list(self.text_encoder.parameters())
            + list(self.emotion_encoder.parameters())
            + list(self.variance_adaptor.parameters())
            + list(self.decoder.parameters())
        )

    def vocoder_parameters(self):
        """Returns params for Stage-2 (vocoder only) GAN training."""
        return list(self.vocoder.parameters())

    # ------------------------------------------------------------------
    # Forward — acoustic model only (Stage 1)
    # ------------------------------------------------------------------
    def forward_acoustic(
        self,
        tokens: torch.Tensor,               # (B, T_text)
        emotion_ids: torch.Tensor,           # (B,)
        intensity: torch.Tensor | None = None,   # (B,) float in [0,1]
        src_padding_mask: torch.Tensor | None = None,  # (B, T_text)
        gt_durations: torch.Tensor | None = None,
        gt_pitch: torch.Tensor | None = None,
        gt_energy: torch.Tensor | None = None,
        max_mel_len: int | None = None,
    ) -> dict:
        """
        Returns a dict with:
          'mel'       : (B, T_frame, n_mels)  predicted mel spectrogram
          'mel_lens'  : (B,)                  frame lengths (for masking loss)
          'duration'  : (B, T_text)           log-duration predictions
          'pitch'     : (B, T_frame)          pitch predictions
          'energy'    : (B, T_frame)          energy predictions
        """
        # 1. Encode phoneme sequence
        enc_out = self.text_encoder(tokens, src_padding_mask)   # (B, T, D)

        # 2. Encode emotion
        emo_vec = self.emotion_encoder(emotion_ids, intensity)  # (B, D)

        # 3. Variance Adaptor (length-regulate + pitch/energy conditioning)
        va_out, mel_lens, preds = self.variance_adaptor(
            enc_out,
            emo_vec,
            gt_durations=gt_durations,
            gt_pitch=gt_pitch,
            gt_energy=gt_energy,
            max_mel_len=max_mel_len,
        )  # (B, T_frame, D)

        # 4. Build mel padding mask for decoder
        B, T_frame, _ = va_out.shape
        mel_mask = (
            torch.arange(T_frame, device=va_out.device).unsqueeze(0)
            >= mel_lens.unsqueeze(1)
        )  # (B, T_frame), True = pad

        # 5. Decode → mel
        mel = self.decoder(va_out, mel_mask)   # (B, T_frame, n_mels)

        return {
            "mel": mel,
            "mel_lens": mel_lens,
            "mel_mask": mel_mask,
            **preds,
        }

    # ------------------------------------------------------------------
    # Forward — vocoder only (Stage 2)
    # ------------------------------------------------------------------
    def forward_vocoder(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel  : (B, n_mels, T)  — NOTE: channel-first for conv
        wav  : (B, 1, T_wav)
        """
        return self.vocoder(mel)

    # ------------------------------------------------------------------
    # Full inference pipeline (no GT needed)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def synthesize(
        self,
        tokens: torch.Tensor,                    # (1, T_text)
        emotion_id: int,
        intensity: float = 1.0,
        speed: float = 1.0,
        pitch_scale: float = 1.0,
        energy_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Full TTS inference. Returns raw waveform (1, 1, T_wav).

        Args:
            tokens      : phoneme token tensor (batch of 1)
            emotion_id  : integer index into EMOTION_LABELS
            intensity   : 0.0 (neutral) → 1.0 (full emotion)
            speed       : < 1 slower, > 1 faster (duration scaling)
            pitch_scale : changes voice pitch contour
            energy_scale: changes loudness contour
        """
        self.eval()
        device = next(self.parameters()).device
        tokens = tokens.to(device)

        emo_ids = torch.tensor([emotion_id], device=device)
        emo_int = torch.tensor([intensity], device=device)

        enc_out = self.text_encoder(tokens)
        emo_vec = self.emotion_encoder(emo_ids, emo_int)

        va_out, mel_lens, _ = self.variance_adaptor(
            enc_out,
            emo_vec,
            duration_scale=speed,
            pitch_scale=pitch_scale,
            energy_scale=energy_scale,
        )

        mel = self.decoder(va_out)              # (1, T, n_mels)
        wav = self.vocoder(mel.transpose(1, 2)) # vocoder: channel-first
        return wav                              # (1, 1, T_wav)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def count_parameters(self) -> dict:
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {
            "text_encoder":     count(self.text_encoder),
            "emotion_encoder":  count(self.emotion_encoder),
            "variance_adaptor": count(self.variance_adaptor),
            "decoder":          count(self.decoder),
            "vocoder":          count(self.vocoder),
            "total":            count(self),
        }
