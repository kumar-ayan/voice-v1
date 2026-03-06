"""
train.py — Two-stage training script for VoiceAI.

Configuration is driven entirely by ModelConfig (model/config.py).
Secrets (HF token, paths) are read from .env via utils/env.py.
Dataset source is set via cfg.dataset:
  • HuggingFace streaming (default): cfg.dataset.hf_repo, cfg.dataset.hf_config
  • Local disk fallback:             cfg.dataset.local_data_dir = "./data"

Usage:
  # Stage 1 — streams People's Speech from HuggingFace
  python train.py --stage 1

  # Stage 1 with a specific HF subset
  python train.py --stage 1 --hf_config clean

  # Stage 1 with local pre-processed data
  python train.py --stage 1 --local_data ./data

  # Stage 2 — vocoder GAN (needs Stage-1 checkpoint)
  python train.py --stage 2 --acoustic_ckpt checkpoints/acoustic_final.pt
"""

import os
import argparse
import logging

import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.cuda.amp import GradScaler, autocast

from model.config import ModelConfig
from model.voice_ai import VoiceAI
from model.losses import (AcousticLoss, discriminator_loss, generator_loss,
                           feature_matching_loss, mel_reconstruction_loss)
from model.vocoder_discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from model.dataset import build_dataloader
from utils.env import apply_env_to_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── LR scheduler ─────────────────────────────────────────────────────────────
def noam_lr(step: int, d_model: int, warmup: int) -> float:
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))


def build_scheduler(optimizer, cfg: ModelConfig):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: noam_lr(s, cfg.hidden_dim, cfg.training.warmup_steps),
    )


# ── Checkpointing ────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, step: int, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    log.info(f"Saved → {path}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    # weights_only=True prevents arbitrary code execution from untrusted checkpoints
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    log.info(f"Loaded ← {path}  (step {ckpt.get('step', '?')})")
    return ckpt.get("step", 0)


# ── Stage 1: Acoustic model ──────────────────────────────────────────────────
def train_acoustic(cfg: ModelConfig, resume: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Stage 1 — Acoustic model  |  device={device}")
    log.info(f"Dataset: {cfg.dataset.hf_repo!r} / {cfg.dataset.hf_config!r}"
             if not cfg.dataset.local_data_dir
             else f"Dataset: local  {cfg.dataset.local_data_dir!r}")

    model = VoiceAI(cfg).to(device)
    params = model.acoustic_model_parameters()
    log.info(f"Parameters: {model.count_parameters()}")

    tcfg = cfg.training
    optimizer = torch.optim.AdamW(
        params, lr=tcfg.learning_rate,
        betas=tcfg.betas, weight_decay=tcfg.weight_decay,
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

    # ── DataLoaders (auto HF or local) ──────────────────────────────────────
    train_dl = build_dataloader(cfg, split="train")
    val_dl   = build_dataloader(cfg, split="val")

    model.train()
    while step < tcfg.max_steps:
        for batch in train_dl:
            if step >= tcfg.max_steps:
                break

            tokens     = batch["tokens"].to(device)
            mel_gt     = batch["mel_t"].to(device)       # (B, T, n_mels)
            duration   = batch["duration"].to(device)
            pitch      = batch["pitch"].to(device)
            energy     = batch["energy"].to(device)
            emotion_id = batch["emotion_id"].to(device)
            intensity  = batch["intensity"].to(device)
            src_mask   = batch["src_mask"].to(device)
            mel_mask   = batch["mel_mask"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=tcfg.fp16):
                out = model.forward_acoustic(
                    tokens, emotion_id, intensity,
                    src_padding_mask=src_mask,
                    gt_durations=duration,
                    gt_pitch=pitch,
                    gt_energy=energy,
                    max_mel_len=mel_gt.size(1),
                )
                losses = criterion(
                    mel_pred=out["mel"],   dur_pred=out["duration"],
                    pitch_pred=out["pitch"], energy_pred=out["energy"],
                    mel_gt=mel_gt,        dur_gt=duration,
                    pitch_gt=pitch,       energy_gt=energy,
                    src_mask=src_mask,    mel_mask=mel_mask,
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
                    f"[{step:>7d}] loss={losses['total']:.4f}  "
                    f"mel={losses['mel']:.4f}  dur={losses['dur']:.4f}  "
                    f"pitch={losses['pitch']:.4f}  energy={losses['energy']:.4f}  "
                    f"lr={lr:.2e}"
                )

            if step % tcfg.save_every == 0:
                save_checkpoint(model, optimizer, scheduler, step,
                                f"{tcfg.checkpoint_dir}/acoustic_step{step}.pt")

    save_checkpoint(model, optimizer, scheduler, step,
                    f"{tcfg.checkpoint_dir}/acoustic_final.pt")
    log.info("Stage 1 complete ✅")


# ── Stage 2: HiFi-GAN vocoder ────────────────────────────────────────────────
def train_vocoder(cfg: ModelConfig, acoustic_ckpt: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Stage 2 — Vocoder GAN  |  device={device}")

    model = VoiceAI(cfg).to(device)
    load_checkpoint(acoustic_ckpt, model, device=device)
    for p in model.acoustic_model_parameters():
        p.requires_grad = False

    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    tcfg     = cfg.training
    gen_opt  = torch.optim.AdamW(model.vocoder_parameters(), lr=2e-4, betas=(0.8, 0.99))
    disc_opt = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()), lr=2e-4, betas=(0.8, 0.99)
    )
    gen_sched  = torch.optim.lr_scheduler.ExponentialLR(gen_opt,  gamma=0.999)
    disc_sched = torch.optim.lr_scheduler.ExponentialLR(disc_opt, gamma=0.999)

    mel_cfg = cfg.mel
    mel_fn  = T.MelSpectrogram(
        sample_rate=mel_cfg.sample_rate, n_fft=mel_cfg.n_fft,
        hop_length=mel_cfg.hop_length,   n_mels=mel_cfg.n_mels,
        f_min=mel_cfg.f_min, f_max=mel_cfg.f_max,
    ).to(device)

    scaler   = GradScaler(enabled=tcfg.fp16)
    train_dl = build_dataloader(cfg, split="train")
    step     = 0

    model.eval(); model.vocoder.train()
    mpd.train(); msd.train()

    while step < tcfg.max_steps:
        for batch in train_dl:
            if step >= tcfg.max_steps:
                break

            real_wav   = batch["mel"].to(device)
            tokens     = batch["tokens"].to(device)
            emotion_id = batch["emotion_id"].to(device)
            intensity  = batch["intensity"].to(device)
            src_mask   = batch["src_mask"].to(device)
            duration   = batch["duration"].to(device)
            pitch      = batch["pitch"].to(device)
            energy     = batch["energy"].to(device)

            with torch.no_grad():
                out = model.forward_acoustic(
                    tokens, emotion_id, intensity,
                    src_padding_mask=src_mask,
                    gt_durations=duration, gt_pitch=pitch, gt_energy=energy,
                )
                mel = out["mel"].transpose(1, 2)   # (B, n_mels, T)

            with autocast(enabled=tcfg.fp16):
                fake_wav = model.forward_vocoder(mel)

            # Discriminator step
            disc_opt.zero_grad()
            with autocast(enabled=tcfg.fp16):
                r_mpd, f_mpd, _, _ = mpd(real_wav, fake_wav.detach())
                r_msd, f_msd, _, _ = msd(real_wav, fake_wav.detach())
                d_loss = discriminator_loss(r_mpd + r_msd, f_mpd + f_msd)
            scaler.scale(d_loss).backward()
            scaler.step(disc_opt); scaler.update()

            # Generator step
            gen_opt.zero_grad()
            with autocast(enabled=tcfg.fp16):
                _, f_mpd, r_fm_mpd, f_fm_mpd = mpd(real_wav, fake_wav)
                _, f_msd, r_fm_msd, f_fm_msd = msd(real_wav, fake_wav)
                g_loss  = generator_loss(f_mpd + f_msd)
                fm_loss = feature_matching_loss(
                    r_fm_mpd + r_fm_msd, f_fm_mpd + f_fm_msd)
                mr_loss = mel_reconstruction_loss(
                    real_wav, fake_wav,
                    lambda w: torch.log(mel_fn(w).clamp(1e-5)))
                total_g = g_loss + 2.0 * fm_loss + 45.0 * mr_loss
            scaler.scale(total_g).backward()
            scaler.step(gen_opt); scaler.update()

            gen_sched.step(); disc_sched.step(); step += 1

            if step % tcfg.log_every == 0:
                log.info(f"[{step:>7d}] G={total_g:.4f}  D={d_loss:.4f}  "
                         f"FM={fm_loss:.4f}  MR={mr_loss:.4f}")

            if step % tcfg.save_every == 0:
                save_checkpoint(model, gen_opt, gen_sched, step,
                                f"{tcfg.checkpoint_dir}/vocoder_step{step}.pt")

    log.info("Stage 2 complete ✅")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceAI Training")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--acoustic_ckpt", type=str, default=None,
                        help="Frozen acoustic checkpoint for Stage 2")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume Stage 1 from")
    # Dataset overrides
    parser.add_argument("--hf_config", type=str, default=None,
                        help="HF dataset config: microset|clean|dirty|clean_sa|dirty_sa")
    parser.add_argument("--local_data", type=str, default=None,
                        help="Local data directory (overrides HF streaming)")
    parser.add_argument("--no_stream", action="store_true",
                        help="Disable HF streaming (download dataset fully)")
    args = parser.parse_args()

    # Build config and apply CLI overrides
    cfg = ModelConfig()
    apply_env_to_config(cfg)      # load .env / environment variables first
    # CLI args override env vars
    if args.hf_config:
        cfg.dataset.hf_config = args.hf_config
    if args.local_data:
        cfg.dataset.local_data_dir = args.local_data
    if args.no_stream:
        cfg.dataset.streaming = False

    if args.stage == 1:
        train_acoustic(cfg, resume=args.resume)
    elif args.stage == 2:
        if not args.acoustic_ckpt:
            parser.error("--acoustic_ckpt required for Stage 2")
        train_vocoder(cfg, args.acoustic_ckpt)
