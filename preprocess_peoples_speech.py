"""
preprocess_peoples_speech.py
────────────────────────────
Downloads & preprocesses the MLCommons People's Speech dataset
(https://huggingface.co/datasets/MLCommons/peoples_speech) into the
directory layout expected by VoiceAI's dataset.py:

  OUTPUT_DIR/
    metadata.csv          ← fileid|text|phonemes|emotion|intensity
    wavs/   fileid.wav    ← 22 050 Hz, mono
    mels/   fileid.pt     ← (n_mels, T) log-mel FloatTensor
    pitch/  fileid.pt     ← (T,) log-F0 FloatTensor  (0 = unvoiced)
    energy/ fileid.pt     ← (T,) RMS energy FloatTensor
    durations/ fileid.pt  ← (T_phoneme,) frame-count LongTensor
                             NOTE: durations are populated via MFA
                             (Montreal Forced Aligner) in a separate step.
                             Here we write placeholder zeros.

People's Speech fields used:
  id            → fileid
  audio.array   → waveform (16 kHz raw)
  audio.sampling_rate → 16 000
  text          → transcript
  (no emotion / phoneme fields – we assign neutral & run G2P)

Usage:
  # Install extras first:
  #   pip install datasets phonemizer librosa pyworld soundfile tqdm
  python preprocess_peoples_speech.py \\
      --config    microset \\          # or clean / dirty / clean_sa etc.
      --split     train \\
      --output    ./data \\
      --max_items 5000 \\              # omit to process everything
      --workers   4

Requirements:
  - eSpeak-NG installed on PATH  (used by phonemizer for G2P)
  - ffmpeg on PATH               (used by datasets for audio decoding)
"""

import os
import re
import csv
import logging
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional imports — give friendly errors
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset, Audio
except ImportError:
    raise SystemExit("❌  Run:  pip install datasets")

try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
    ESPEAK_OK = True
except ImportError:
    ESPEAK_OK = False
    print("⚠️  phonemizer not found — phoneme column will be left empty. "
          "Install with: pip install phonemizer")

try:
    import pyworld as pw
    PYWORLD_OK = True
except ImportError:
    PYWORLD_OK = False
    print("⚠️  pyworld not found — pitch will be zeros. "
          "Install with: pip install pyworld")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants matching model/config.py MelConfig defaults ──────────────────
TARGET_SR   = 22_050
N_FFT       = 1_024
HOP_LENGTH  = 256
WIN_LENGTH  = 1_024
N_MELS      = 80
F_MIN       = 0.0
F_MAX       = 8_000.0

# Emotion assigned to all People's Speech samples (no emotion labels in dataset)
DEFAULT_EMOTION   = "neutral"
DEFAULT_INTENSITY = "1.0"


# ── Audio helpers ────────────────────────────────────────────────────────────

def resample_to_target(array: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample numpy audio array to TARGET_SR."""
    if orig_sr == TARGET_SR:
        return array.astype(np.float32)
    wav = torch.from_numpy(array).float().unsqueeze(0)  # (1, T)
    resampler = T.Resample(orig_sr, TARGET_SR)
    return resampler(wav).squeeze(0).numpy()


def compute_mel(wav: np.ndarray) -> torch.Tensor:
    """wav: (T,) float32 → log-mel (n_mels, T_mel)."""
    wav_t = torch.from_numpy(wav).unsqueeze(0)          # (1, T)
    mel_fn = T.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, power=1.0,
    )
    mel = mel_fn(wav_t).squeeze(0)                      # (n_mels, T_mel)
    return torch.log(mel.clamp(min=1e-5))


def compute_pitch(wav: np.ndarray) -> torch.Tensor:
    """
    Extract log-F0 per frame using WORLD vocoder.
    Returns (T_mel,) tensor  — 0 for unvoiced frames.
    """
    if not PYWORLD_OK:
        # Rough fallback: all zeros
        n_frames = 1 + len(wav) // HOP_LENGTH
        return torch.zeros(n_frames)

    wav_d = wav.astype(np.float64)
    _f0, t = pw.dio(wav_d, TARGET_SR,
                    frame_period=HOP_LENGTH / TARGET_SR * 1000)
    f0 = pw.stonemask(wav_d, _f0, t, TARGET_SR)        # refined F0 (Hz)
    # Convert to log scale; 0 stays 0 (unvoiced)
    log_f0 = np.where(f0 > 0, np.log(f0 + 1e-8), 0.0)
    return torch.from_numpy(log_f0.astype(np.float32))


def compute_energy(wav: np.ndarray) -> torch.Tensor:
    """
    Frame-level RMS energy aligned with mel frames.
    Returns (T_mel,) tensor.
    """
    wav_t  = torch.from_numpy(wav)
    # Pad to round number of frames
    pad    = (HOP_LENGTH - len(wav) % HOP_LENGTH) % HOP_LENGTH
    wav_t  = torch.nn.functional.pad(wav_t, (0, pad))
    frames = wav_t.unfold(0, WIN_LENGTH, HOP_LENGTH)    # (T, WIN_LENGTH)
    rms    = frames.pow(2).mean(dim=-1).sqrt()           # (T,)
    return rms


def grapheme_to_phoneme(text: str) -> str:
    """
    Convert raw text → space-separated IPA phoneme string.
    Falls back to empty string if phonemizer is unavailable.
    """
    if not ESPEAK_OK:
        return ""
    try:
        phonemes = phonemize(
            text,
            backend="espeak",
            language="en-us",
            with_stress=False,
            preserve_punctuation=True,
            njobs=1,
        )
        # Collapse whitespace; return space-separated phonemes
        return re.sub(r"\s+", " ", phonemes).strip()
    except Exception as e:
        log.warning(f"G2P failed: {e}")
        return ""


# ── Main preprocessing ───────────────────────────────────────────────────────

def process_item(item: dict, dirs: dict) -> dict | None:
    """
    Process a single HF dataset item.
    Returns a metadata row dict, or None on failure.
    """
    fileid   = re.sub(r"[\\/]", "_", item["id"])       # safe filename
    text     = item["text"].strip()
    if not text:
        return None

    audio_array = np.array(item["audio"]["array"], dtype=np.float32)
    orig_sr     = item["audio"]["sampling_rate"]        # should be 16 000

    # 1. Resample → TARGET_SR
    wav = resample_to_target(audio_array, orig_sr)

    # 2. Save wav
    wav_path = dirs["wavs"] / f"{fileid}.wav"
    sf.write(str(wav_path), wav, TARGET_SR)

    # 3. Mel
    mel = compute_mel(wav)                              # (n_mels, T_mel)
    torch.save(mel, dirs["mels"] / f"{fileid}.pt")

    # 4. Pitch
    pitch = compute_pitch(wav)
    # Align to mel length
    T_mel = mel.shape[1]
    if pitch.shape[0] > T_mel:
        pitch = pitch[:T_mel]
    elif pitch.shape[0] < T_mel:
        pitch = torch.nn.functional.pad(pitch, (0, T_mel - pitch.shape[0]))
    torch.save(pitch, dirs["pitch"] / f"{fileid}.pt")

    # 5. Energy
    energy = compute_energy(wav)
    if energy.shape[0] > T_mel:
        energy = energy[:T_mel]
    elif energy.shape[0] < T_mel:
        energy = torch.nn.functional.pad(energy, (0, T_mel - energy.shape[0]))
    torch.save(energy, dirs["energy"] / f"{fileid}.pt")

    # 6. G2P
    phonemes = grapheme_to_phoneme(text)

    # 7. Duration placeholder (zeros — replace later with MFA output)
    #    We put one token for each word as a rough stand-in.
    n_words = max(1, len(text.split()))
    torch.save(torch.zeros(n_words, dtype=torch.long),
               dirs["durations"] / f"{fileid}.pt")

    return {
        "fileid":    fileid,
        "text":      text,
        "phonemes":  phonemes,
        "emotion":   DEFAULT_EMOTION,
        "intensity": DEFAULT_INTENSITY,
    }


def run(args):
    log.info(f"Loading People's Speech  config={args.config}  split={args.split}")
    hf_token = os.environ.get("HF_TOKEN") or None
    if hf_token:
        log.info("Using HF_TOKEN from environment")
    ds = load_dataset(
        "MLCommons/peoples_speech",
        args.config,
        split=args.split,
        streaming=True,          # avoids downloading 30 000 h upfront
        trust_remote_code=True,
        token=hf_token,
    ).cast_column("audio", Audio(sampling_rate=16_000))

    out = Path(args.output)
    dirs = {
        "wavs":      out / "wavs",
        "mels":      out / "mels",
        "pitch":     out / "pitch",
        "energy":    out / "energy",
        "durations": out / "durations",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    meta_path = out / "metadata.csv"
    skipped = 0
    written = 0

    with open(meta_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")

        for i, item in enumerate(tqdm(ds, desc="Processing")):
            if args.max_items and i >= args.max_items:
                break
            try:
                row = process_item(item, dirs)
                if row is None:
                    skipped += 1
                    continue
                writer.writerow([
                    row["fileid"], row["text"],
                    row["phonemes"], row["emotion"], row["intensity"],
                ])
                csvfile.flush()
                written += 1
            except Exception as e:
                log.warning(f"Skipped item {i}: {e}")
                skipped += 1

    log.info(f"Done.  Written={written}  Skipped={skipped}")
    log.info(f"Output: {out.resolve()}")
    log.info("")
    log.info("Next steps:")
    log.info(" 1. Run Montreal Forced Aligner (MFA) on wavs/ + metadata.csv")
    log.info("    to produce real per-phoneme durations in durations/")
    log.info(" 2. (Optional) Add emotion labels using a classifier")
    log.info("    e.g. speechbrain/emotion-recognition-wav2vec2-IEMOCAP")
    log.info(" 3. Train:  python train.py --stage 1 --data_dir ./data")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess People's Speech → VoiceAI training format"
    )
    parser.add_argument(
        "--config", default="microset",
        choices=["microset", "clean", "dirty", "clean_sa", "dirty_sa"],
        help="Dataset configuration (default: microset for quick testing)"
    )
    parser.add_argument(
        "--split", default="train",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--output", default="./data",
        help="Output directory (default: ./data)"
    )
    parser.add_argument(
        "--max_items", type=int, default=None,
        help="Cap on number of items to process (omit = all)"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="CPU workers (currently unused; placeholder for parallel G2P)"
    )
    args = parser.parse_args()
    run(args)
