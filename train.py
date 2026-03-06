"""
train.py — Two-stage training script for VoiceAI.

Usage:
  # Stage 1: Train acoustic model
  python train.py --stage 1 --data_dir ./data --config_preset base

  # Stage 2: Train vocoder (GAN)
  python train.py --stage 2 --data_dir ./data --acoustic_ckpt checkpoints/acoustic_best.pt
"""

import os
import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.cuda.amp import GradScaler, autocast

from model.config import ModelConfig, MelConfig
from model.voice_ai import VoiceAI
from model.losses import AcousticLoss, discriminator_loss, generator_loss, feature_matching_loss, mel_reconstruction_loss
from model.vocoder_discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from model.dataset import build_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learning-rate scheduler: Noam / Transformer schedule
# ---------------------------------------------------------------------------
def noam_lr(step: int, d_model: int, warmup: int) -> float:
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: ModelConfig):
    d = cfg.hidden_dim
    w = cfg.training.warmup_steps
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: noam_lr(step, d, w)
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(model: nn.Module, optimizer, scheduler, step: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    log.info(f"Saved checkpoint → {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    log.info(f"Loaded checkpoint ← {path} (step {ckpt.get('step', '?')})")
    return ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# Stage 1: Acoustic Model Training
# ---------------------------------------------------------------------------
def train_acoustic(cfg: ModelConfig, data_dir: str, resume: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Stage 1 — Acoustic model training on {device}")

    model = VoiceAI(cfg).to(device)
    params = model.acoustic_model_parameters()

    tcfg = cfg.training
    optimizer = torch.optim.AdamW(
        params, lr=tcfg.learning_rate, betas=tcfg.betas, weight_decay=tcfg.weight_decay
    )
    scheduler = build_scheduler(optimizer, cfg)
    criterion = AcousticLoss(
        mel_weight=tcfg.mel_loss_weight,
        dur_weight=tcfg.duration_loss_weight,
        pitch_weight=tcfg.pitch_loss_weight,
        energy_weight=tcfg.energy_loss_weight,
    )
    scaler = GradScaler(enabled=tcfg.fp16)

    step = 0
    if resume:
        step = load_checkpoint(resume, model, optimizer, scheduler, device)

    train_dl = build_dataloader(data_dir, cfg, split="train")
    val_dl   = build_dataloader(data_dir, cfg, split="val")

    log.info(f"Model parameters: {model.count_parameters()}")

    model.train()
    while step < tcfg.max_steps:
        for batch in train_dl:
            if step >= tcfg.max_steps:
                break

            # Move to device
            tokens     = batch["tokens"].to(device)
            mel_gt     = batch["mel_t"].to(device)       # (B, T, n_mels)
            duration   = batch["duration"].to(device)
            pitch      = batch["pitch"].to(device)
            energy     = batch["energy"].to(device)
            emotion_id = batch["emotion_id"].to(device)
            intensity  = batch["intensity"].to(device)
            src_mask   = batch["src_mask"].to(device)
            mel_mask   = batch["mel_mask"].to(device)
            mel_lens   = batch["mel_lens"].to(device)

            max_mel = mel_gt.size(1)

            optimizer.zero_grad()

            with autocast(enabled=tcfg.fp16):
                out = model.forward_acoustic(
                    tokens, emotion_id, intensity,
                    src_padding_mask=src_mask,
                    gt_durations=duration,
                    gt_pitch=pitch,
                    gt_energy=energy,
                    max_mel_len=max_mel,
                )

                losses = criterion(
                    mel_pred=out["mel"],
                    dur_pred=out["duration"],
                    pitch_pred=out["pitch"],
                    energy_pred=out["energy"],
                    mel_gt=mel_gt,
                    dur_gt=duration,
                    pitch_gt=pitch,
                    energy_gt=energy,
                    src_mask=src_mask,
                    mel_mask=mel_mask,
                )

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, tcfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            if step % tcfg.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                log.info(
                    f"[Step {step:>7d}] "
                    f"loss={losses['total']:.4f} "
                    f"mel={losses['mel']:.4f} "
                    f"dur={losses['dur']:.4f} "
                    f"pitch={losses['pitch']:.4f} "
                    f"energy={losses['energy']:.4f} "
                    f"lr={lr:.2e}"
                )

            if step % tcfg.save_every == 0:
                path = os.path.join(tcfg.checkpoint_dir, f"acoustic_step{step}.pt")
                save_checkpoint(model, optimizer, scheduler, step, path)

    # Final save
    save_checkpoint(
        model, optimizer, scheduler, step,
        os.path.join(tcfg.checkpoint_dir, "acoustic_final.pt")
    )
    log.info("Stage 1 training complete.")


# ---------------------------------------------------------------------------
# Stage 2: Vocoder (HiFi-GAN) + Discriminator Training
# ---------------------------------------------------------------------------
def train_vocoder(
    cfg: ModelConfig,
    data_dir: str,
    acoustic_ckpt: str,
    resume_voc: str | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Stage 2 — Vocoder (GAN) training on {device}")

    model = VoiceAI(cfg).to(device)
    # Load frozen acoustic model
    load_checkpoint(acoustic_ckpt, model, device=device)
    # Freeze acoustic sub-modules
    for p in model.acoustic_model_parameters():
        p.requires_grad = False

    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    tcfg = cfg.training
    gen_opt = torch.optim.AdamW(
        model.vocoder_parameters(), lr=2e-4, betas=(0.8, 0.99)
    )
    disc_opt = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()), lr=2e-4, betas=(0.8, 0.99)
    )
    gen_sched  = torch.optim.lr_scheduler.ExponentialLR(gen_opt,  gamma=0.999)
    disc_sched = torch.optim.lr_scheduler.ExponentialLR(disc_opt, gamma=0.999)

    # Build mel transform for reconstruction loss evaluation
    mel_cfg = cfg.mel
    mel_fn = T.MelSpectrogram(
        sample_rate=mel_cfg.sample_rate,
        n_fft=mel_cfg.n_fft, hop_length=mel_cfg.hop_length,
        n_mels=mel_cfg.n_mels, f_min=mel_cfg.f_min, f_max=mel_cfg.f_max,
    ).to(device)

    scaler = GradScaler(enabled=tcfg.fp16)
    train_dl = build_dataloader(data_dir, cfg, split="train")
    step = 0

    model.eval()   # Acoustic part frozen, only vocoder trains
    mpd.train(); msd.train()
    model.vocoder.train()

    while step < tcfg.max_steps:
        for batch in train_dl:
            if step >= tcfg.max_steps:
                break

            real_wav = batch["mel"].to(device)   # reusing "mel" slot—see dataset note
            tokens     = batch["tokens"].to(device)
            emotion_id = batch["emotion_id"].to(device)
            intensity  = batch["intensity"].to(device)
            src_mask   = batch["src_mask"].to(device)
            duration   = batch["duration"].to(device)
            pitch      = batch["pitch"].to(device)
            energy     = batch["energy"].to(device)

            # Generate mel (frozen acoustic model, no grad needed)
            with torch.no_grad():
                out = model.forward_acoustic(
                    tokens, emotion_id, intensity,
                    src_padding_mask=src_mask,
                    gt_durations=duration, gt_pitch=pitch, gt_energy=energy,
                )
                mel = out["mel"].transpose(1, 2)  # (B, n_mels, T)

            # Fake waveform from vocoder
            with autocast(enabled=tcfg.fp16):
                fake_wav = model.forward_vocoder(mel)

            # ---- Discriminator update ----
            disc_opt.zero_grad()
            with autocast(enabled=tcfg.fp16):
                r_mpd, f_mpd, _, _ = mpd(real_wav, fake_wav.detach())
                r_msd, f_msd, _, _ = msd(real_wav, fake_wav.detach())
                d_loss = discriminator_loss(r_mpd + r_msd, f_mpd + f_msd)

            scaler.scale(d_loss).backward()
            scaler.step(disc_opt)
            scaler.update()

            # ---- Generator update ----
            gen_opt.zero_grad()
            with autocast(enabled=tcfg.fp16):
                _, f_mpd, r_fmaps_mpd, f_fmaps_mpd = mpd(real_wav, fake_wav)
                _, f_msd, r_fmaps_msd, f_fmaps_msd = msd(real_wav, fake_wav)
                g_loss  = generator_loss(f_mpd + f_msd)
                fm_loss = feature_matching_loss(r_fmaps_mpd + r_fmaps_msd, f_fmaps_mpd + f_fmaps_msd)
                mr_loss = mel_reconstruction_loss(real_wav, fake_wav, lambda w: torch.log(mel_fn(w).clamp(1e-5)))
                total_g = g_loss + 2.0 * fm_loss + 45.0 * mr_loss

            scaler.scale(total_g).backward()
            scaler.step(gen_opt)
            scaler.update()

            gen_sched.step()
            disc_sched.step()
            step += 1

            if step % tcfg.log_every == 0:
                log.info(
                    f"[Step {step:>7d}] "
                    f"G={total_g:.4f} D={d_loss:.4f} "
                    f"FM={fm_loss:.4f} MR={mr_loss:.4f}"
                )

            if step % tcfg.save_every == 0:
                path = os.path.join(tcfg.checkpoint_dir, f"vocoder_step{step}.pt")
                save_checkpoint(model, gen_opt, gen_sched, step, path)

    log.info("Stage 2 training complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceAI Training")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (Stage 1)")
    parser.add_argument("--acoustic_ckpt", type=str, default=None,
                        help="Frozen acoustic checkpoint for Stage 2")
    args = parser.parse_args()

    cfg = ModelConfig()   # Use defaults; override fields as needed

    if args.stage == 1:
        train_acoustic(cfg, args.data_dir, resume=args.resume)
    elif args.stage == 2:
        if not args.acoustic_ckpt:
            parser.error("--acoustic_ckpt is required for Stage 2")
        train_vocoder(cfg, args.data_dir, args.acoustic_ckpt)
