"""
dataset.py — Dataset & DataLoader for VoiceAI.

Two modes, controlled by ModelConfig.dataset:
  ① HuggingFace streaming (default)
     Streams directly from HF Hub — no preprocessing step required.
     People's Speech fields used: id, audio.array, audio.sampling_rate, text.
     Mel / pitch / energy → computed on-the-fly.
     Phonemes → G2P via phonemizer (eSpeak-NG backend).
     Emotion → always "neutral" / 1.0  (PS has no emotion labels).

  ② Local disk (set cfg.dataset.local_data_dir to a path)
     Reads metadata.csv + pre-computed .pt files produced by
     preprocess_peoples_speech.py or any other pipeline.
     Falls back to on-the-fly computation if .pt files are absent.
"""

import os
import re
import csv
import logging
import math
from typing import Iterator

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence

from .config import ModelConfig, MelConfig, PHONEME_VOCAB, EMOTION_LABELS

log = logging.getLogger(__name__)

# ── Optional dependencies ────────────────────────────────────────────────────
try:
    from datasets import load_dataset, Audio as HFAudio
    HF_OK = True
except ImportError:
    HF_OK = False
    log.warning("'datasets' not installed. HF streaming unavailable. "
                "pip install datasets")

# HF token — always read from environment, never hardcode
try:
    from utils.env import get_hf_token as _get_hf_token
except ImportError:
    import os as _os
    def _get_hf_token(): return _os.environ.get("HF_TOKEN") or None

try:
    from phonemizer import phonemize
    PHONEMIZER_OK = True
except ImportError:
    PHONEMIZER_OK = False
    log.warning("'phonemizer' not installed. Using character-level fallback. "
                "pip install phonemizer")

try:
    import pyworld as pw
    PYWORLD_OK = True
except ImportError:
    PYWORLD_OK = False


# ── Constants (kept in sync with MelConfig defaults) ────────────────────────
_TARGET_SR  = 22_050
_N_FFT      = 1_024
_HOP        = 256
_WIN        = 1_024
_N_MELS     = 80


# ────────────────────────────────────────────────────────────────────────────
# Phoneme Tokenizer
# ────────────────────────────────────────────────────────────────────────────
class PhonemeTokenizer:
    def __init__(self):
        self.vocab = {p: i for i, p in enumerate(PHONEME_VOCAB)}
        self.pad_id = self.vocab["<pad>"]
        self.unk_id = self.vocab["<unk>"]
        self.bos_id = self.vocab["<bos>"]
        self.eos_id = self.vocab["<eos>"]

    def encode(self, phoneme_str: str) -> list[int]:
        """Space-separated IPA string → list of ids."""
        ids = [self.bos_id]
        for p in phoneme_str.strip().split():
            ids.append(self.vocab.get(p, self.unk_id))
        ids.append(self.eos_id)
        return ids


# ────────────────────────────────────────────────────────────────────────────
# Audio feature helpers
# ────────────────────────────────────────────────────────────────────────────
def _resample(array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return array.astype(np.float32)
    wav = torch.from_numpy(array).float().unsqueeze(0)
    return T.Resample(orig_sr, target_sr)(wav).squeeze(0).numpy()


def _mel_transform(mel_cfg: MelConfig):
    return T.MelSpectrogram(
        sample_rate=mel_cfg.sample_rate,
        n_fft=mel_cfg.n_fft, hop_length=mel_cfg.hop_length,
        win_length=mel_cfg.win_length, n_mels=mel_cfg.n_mels,
        f_min=mel_cfg.f_min, f_max=mel_cfg.f_max, power=1.0,
    )


def compute_mel(wav: np.ndarray, mel_cfg: MelConfig) -> torch.Tensor:
    """(T,) numpy → log-mel (n_mels, T_mel)."""
    wav_t = torch.from_numpy(wav).unsqueeze(0)
    mel = _mel_transform(mel_cfg)(wav_t).squeeze(0)
    return torch.log(mel.clamp(min=1e-5))


def compute_pitch(wav: np.ndarray, hop: int, sr: int, T_mel: int) -> torch.Tensor:
    """log-F0 per frame, aligned to mel length. Returns zeros if pyworld absent."""
    if PYWORLD_OK:
        try:
            wav_d = wav.astype(np.float64)
            frame_ms = hop / sr * 1000.0
            _f0, t = pw.dio(wav_d, sr, frame_period=frame_ms)
            f0 = pw.stonemask(wav_d, _f0, t, sr)
            log_f0 = np.where(f0 > 0, np.log(f0 + 1e-8), 0.0).astype(np.float32)
            pitch = torch.from_numpy(log_f0)
        except Exception:
            pitch = torch.zeros(T_mel)
    else:
        pitch = torch.zeros(T_mel)

    # Align length to T_mel
    if pitch.shape[0] > T_mel:
        return pitch[:T_mel]
    return torch.nn.functional.pad(pitch, (0, T_mel - pitch.shape[0]))


def compute_energy(wav: np.ndarray, hop: int, win: int, T_mel: int) -> torch.Tensor:
    """RMS energy per mel-aligned frame."""
    wav_t = torch.from_numpy(wav.astype(np.float32))
    pad   = (hop - len(wav_t) % hop) % hop
    wav_t = torch.nn.functional.pad(wav_t, (0, pad))
    frames = wav_t.unfold(0, win, hop)       # (T, win)
    rms    = frames.pow(2).mean(-1).sqrt()   # (T,)
    if rms.shape[0] > T_mel:
        return rms[:T_mel]
    return torch.nn.functional.pad(rms, (0, T_mel - rms.shape[0]))


def _g2p(text: str) -> str:
    """Text → space-separated IPA phonemes."""
    if PHONEMIZER_OK:
        try:
            out = phonemize(text, backend="espeak", language="en-us",
                            with_stress=False, preserve_punctuation=True,
                            njobs=1)
            return re.sub(r"\s+", " ", out).strip()
        except Exception:
            pass
    # Character-level fallback (maps printable chars into vocab)
    return " ".join(c for c in text.lower() if c in PHONEME_VOCAB)


def _uniform_durations(n_phonemes: int, T_mel: int) -> torch.Tensor:
    """Distribute T_mel frames evenly across phonemes when MFA is unavailable."""
    base = T_mel // max(n_phonemes, 1)
    rem  = T_mel % max(n_phonemes, 1)
    durs = [base + (1 if i < rem else 0) for i in range(n_phonemes)]
    return torch.tensor(durs, dtype=torch.long)


# ────────────────────────────────────────────────────────────────────────────
# Core item builder
# ────────────────────────────────────────────────────────────────────────────
def _build_item(
    wav_array: np.ndarray,
    orig_sr: int,
    text: str,
    mel_cfg: MelConfig,
    phonemes: str | None = None,
    emotion: str = "neutral",
    intensity: float = 1.0,
    gt_duration: torch.Tensor | None = None,
    gt_pitch: torch.Tensor | None = None,
    gt_energy: torch.Tensor | None = None,
) -> dict:
    """
    Convert a raw audio array + transcript into a VoiceAI training item.
    Missing GT tensors are estimated on-the-fly.
    """
    wav = _resample(wav_array, orig_sr, mel_cfg.sample_rate)
    mel = compute_mel(wav, mel_cfg)        # (n_mels, T_mel)
    T_mel = mel.shape[1]

    if gt_pitch is None:
        gt_pitch = compute_pitch(wav, mel_cfg.hop_length, mel_cfg.sample_rate, T_mel)
    if gt_energy is None:
        gt_energy = compute_energy(wav, mel_cfg.hop_length, mel_cfg.win_length, T_mel)

    if phonemes is None:
        phonemes = _g2p(text)

    tokenizer = PhonemeTokenizer()
    token_ids = tokenizer.encode(phonemes)
    tokens    = torch.tensor(token_ids, dtype=torch.long)

    if gt_duration is None:
        gt_duration = _uniform_durations(len(token_ids), T_mel)

    emotion_map = {e: i for i, e in enumerate(EMOTION_LABELS)}
    emotion_id  = emotion_map.get(emotion, 0)

    return {
        "tokens":     tokens,                                      # (T_text,)
        "mel":        mel,                                         # (n_mels, T_mel)
        "duration":   gt_duration,                                 # (T_text,)
        "pitch":      gt_pitch,                                    # (T_mel,)
        "energy":     gt_energy,                                   # (T_mel,)
        "emotion_id": torch.tensor(emotion_id,  dtype=torch.long),
        "intensity":  torch.tensor(intensity,   dtype=torch.float),
    }


# ────────────────────────────────────────────────────────────────────────────
# ① HuggingFace Streaming Dataset  (IterableDataset)
# ────────────────────────────────────────────────────────────────────────────
class HFVoiceDataset(IterableDataset):
    """
    Streams People's Speech (or any compatible HF dataset) and yields
    training items compatible with VoiceAI's collate_fn.
    """

    def __init__(self, cfg: ModelConfig, split: str = "train"):
        super().__init__()
        if not HF_OK:
            raise RuntimeError("Install 'datasets':  pip install datasets")

        dcfg = cfg.dataset
        self.mel_cfg  = cfg.mel
        self.split    = split
        self.max_items = dcfg.max_items

        hf_split = dcfg.train_split if split == "train" else dcfg.val_split
        log.info(f"Loading HF dataset {dcfg.hf_repo!r} "
                 f"config={dcfg.hf_config!r} split={hf_split!r} "
                 f"streaming={dcfg.streaming}")

        hf_token = _get_hf_token()
        if hf_token:
            log.info("HF token found — authenticated access enabled")

        # Some configs (e.g. microset) only have a 'train' split.
        # For validation we fall back to a tail slice of 'train'.
        available_splits = self._get_available_splits(dcfg)
        if hf_split not in available_splits:
            log.warning(
                f"Split {hf_split!r} not available for config {dcfg.hf_config!r}. "
                f"Available: {available_splits}. Using 'train' tail as validation."
            )
            self._val_fallback = True
            hf_split = dcfg.train_split
        else:
            self._val_fallback = (split != "train")

        self._split = split
        self._hf_ds = load_dataset(
            dcfg.hf_repo,
            dcfg.hf_config,
            split=hf_split,
            streaming=dcfg.streaming,
            token=hf_token,
        ).cast_column("audio", HFAudio(sampling_rate=16_000))

    @staticmethod
    def _get_available_splits(dcfg) -> list[str]:
        """Query available splits without downloading data."""
        try:
            from datasets import get_dataset_split_names
            return get_dataset_split_names(dcfg.hf_repo, dcfg.hf_config)
        except Exception:
            return ["train"]   # safe fallback

    def __iter__(self) -> Iterator[dict]:
        count = 0
        # When there's no real val split, use every 20th item as validation (~5%)
        is_val_fallback = getattr(self, "_val_fallback", False)
        for i, item in enumerate(self._hf_ds):
            if self.max_items and count >= self.max_items:
                break
            # Val-fallback filtering: train=skip 20th; val=keep only 20th
            if is_val_fallback:
                if self._split == "train" and i % 20 == 0:
                    continue   # reserve these for validation
                if self._split != "train" and i % 20 != 0:
                    continue   # skip training items
            try:
                audio  = item["audio"]
                array  = np.array(audio["array"], dtype=np.float32)
                sr     = audio["sampling_rate"]
                text   = item.get("text", "").strip()
                if not text or len(array) < sr * 0.3:
                    continue
                yield _build_item(array, sr, text, self.mel_cfg)
                count += 1
            except Exception as e:
                log.debug(f"Skipping item: {e}")
                continue


# ────────────────────────────────────────────────────────────────────────────
# ② Local Disk Dataset  (map-style Dataset, backward-compatible)
# ────────────────────────────────────────────────────────────────────────────
class LocalVoiceDataset(Dataset):
    """
    Reads pre-processed data from disk (produced by preprocess_peoples_speech.py
    or any other pipeline). Falls back to on-the-fly computation for missing files.
    """

    def __init__(self, data_dir: str, cfg: ModelConfig, split: str = "train"):
        self.data_dir = data_dir
        self.mel_cfg  = cfg.mel

        meta_path = os.path.join(data_dir, "metadata.csv")
        all_rows  = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|"):
                if len(row) >= 5:
                    all_rows.append(row)

        n_train = int(len(all_rows) * cfg.dataset.train_ratio)
        self.rows = all_rows[:n_train] if split == "train" else all_rows[n_train:]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        fileid, text, phonemes, emotion, intensity = self.rows[idx][:5]

        def load_pt(subdir, fallback=None):
            p = os.path.join(self.data_dir, subdir, f"{fileid}.pt")
            # weights_only=True prevents arbitrary code execution
            return torch.load(p, weights_only=True) if os.path.isfile(p) else fallback

        mel = load_pt("mels")
        if mel is None:
            wav_path = os.path.join(self.data_dir, "wavs", f"{fileid}.wav")
            wav, sr  = torchaudio.load(wav_path)
            wav_np   = wav.mean(0).numpy()
            if sr != self.mel_cfg.sample_rate:
                wav_np = _resample(wav_np, sr, self.mel_cfg.sample_rate)
            mel      = compute_mel(wav_np, self.mel_cfg)
        T_mel = mel.shape[1]

        pitch    = load_pt("pitch",  compute_pitch(
            *((lambda w, s: (w, s))(
                torchaudio.load(os.path.join(self.data_dir, "wavs", f"{fileid}.wav"))[0]
                    .mean(0).numpy(),
                self.mel_cfg.sample_rate,
            )),
            self.mel_cfg.hop_length, self.mel_cfg.sample_rate, T_mel
        ) if not os.path.isfile(os.path.join(self.data_dir, "pitch", f"{fileid}.pt")) else None)
        if pitch is None:
            pitch = load_pt("pitch", torch.zeros(T_mel))

        energy   = load_pt("energy",   torch.zeros(T_mel))
        tokenizer = PhonemeTokenizer()
        tokens    = torch.tensor(tokenizer.encode(phonemes), dtype=torch.long)
        duration  = load_pt("durations", _uniform_durations(len(tokens), T_mel))

        return {
            "tokens":     tokens,
            "mel":        mel,
            "duration":   duration,
            "pitch":      pitch,
            "energy":     energy,
            "emotion_id": torch.tensor(EMOTION_LABELS.index(emotion.strip())
                                       if emotion.strip() in EMOTION_LABELS else 0,
                                       dtype=torch.long),
            "intensity":  torch.tensor(float(intensity), dtype=torch.float),
        }


# ────────────────────────────────────────────────────────────────────────────
# Collate function  (shared by both dataset types)
# ────────────────────────────────────────────────────────────────────────────
def collate_fn(batch: list[dict]) -> dict:
    """Pad a batch and build padding masks."""
    token_list  = [b["tokens"]   for b in batch]
    mel_list    = [b["mel"].T    for b in batch]   # (T, n_mels)
    dur_list    = [b["duration"] for b in batch]
    pitch_list  = [b["pitch"]    for b in batch]
    energy_list = [b["energy"]   for b in batch]

    tokens   = pad_sequence(token_list,  batch_first=True, padding_value=0)
    mels     = pad_sequence(mel_list,    batch_first=True, padding_value=0.0)
    durs     = pad_sequence(dur_list,    batch_first=True, padding_value=0)
    pitches  = pad_sequence(pitch_list,  batch_first=True, padding_value=0.0)
    energies = pad_sequence(energy_list, batch_first=True, padding_value=0.0)

    src_lens = torch.tensor([t.size(0) for t in token_list])
    mel_lens = torch.tensor([m.size(0) for m in mel_list])
    src_mask = torch.arange(tokens.size(1)).unsqueeze(0) >= src_lens.unsqueeze(1)
    mel_mask = torch.arange(mels.size(1)).unsqueeze(0) >= mel_lens.unsqueeze(1)

    return {
        "tokens":     tokens,
        "mel":        mels.transpose(1, 2),   # (B, n_mels, T)
        "mel_t":      mels,                   # (B, T, n_mels) — for decoder loss
        "duration":   durs,
        "pitch":      pitches,
        "energy":     energies,
        "emotion_id": torch.stack([b["emotion_id"] for b in batch]),
        "intensity":  torch.stack([b["intensity"]  for b in batch]),
        "src_mask":   src_mask,
        "mel_mask":   mel_mask,
        "mel_lens":   mel_lens,
        "src_lens":   src_lens,
    }


# ────────────────────────────────────────────────────────────────────────────
# Public factory — picks HF or local automatically
# ────────────────────────────────────────────────────────────────────────────
def build_dataloader(cfg: ModelConfig, split: str = "train") -> DataLoader:
    """
    Returns a DataLoader.
    • If cfg.dataset.local_data_dir is set → LocalVoiceDataset (map-style)
    • Otherwise                            → HFVoiceDataset   (iterable/streaming)
    """
    dcfg = cfg.dataset

    if dcfg.local_data_dir:
        ds = LocalVoiceDataset(dcfg.local_data_dir, cfg, split=split)
        shuffle = (split == "train")
    else:
        ds = HFVoiceDataset(cfg, split=split)
        shuffle = False    # IterableDataset doesn't support shuffle in DataLoader

    return DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=dcfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )
