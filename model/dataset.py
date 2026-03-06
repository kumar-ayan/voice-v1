"""
dataset.py — PyTorch Dataset for expressive TTS training.

Expected directory structure (LJSpeech-compatible, with emotion labels):

  data/
    metadata.csv         ← pipe-separated: fileid|text|phonemes|emotion|intensity
    wavs/
      fileid.wav
    mels/                ← pre-computed (optional, speeds up training)
      fileid.pt
    durations/           ← forced-aligned durations (one int per phoneme)
      fileid.pt
    pitch/               ← extracted log-F0 per frame
      fileid.pt
    energy/              ← RMS energy per frame
      fileid.pt

If mels/durations/pitch/energy are absent, they are computed on-the-fly
(slow but requires no preprocessing).
"""

import os
import csv
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .config import ModelConfig, MelConfig, PHONEME_VOCAB, EMOTION_LABELS


# ---------------------------------------------------------------------------
# Phoneme tokenizer (lookup table)
# ---------------------------------------------------------------------------
class PhonemeTokenizer:
    def __init__(self):
        self.vocab = {p: i for i, p in enumerate(PHONEME_VOCAB)}
        self.pad_id = self.vocab["<pad>"]
        self.unk_id = self.vocab["<unk>"]
        self.bos_id = self.vocab["<bos>"]
        self.eos_id = self.vocab["<eos>"]

    def encode(self, phoneme_str: str) -> list[int]:
        """
        Tokenise a space-separated phoneme string.
        E.g. "h ɛ l oʊ" → [idx, idx, ...]
        """
        tokens = [self.bos_id]
        for p in phoneme_str.strip().split():
            tokens.append(self.vocab.get(p, self.unk_id))
        tokens.append(self.eos_id)
        return tokens


# ---------------------------------------------------------------------------
# MelSpectrogram transform (for on-the-fly computation)
# ---------------------------------------------------------------------------
def build_mel_transform(cfg: MelConfig) -> T.MelSpectrogram:
    return T.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        power=1.0,           # amplitude mel
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VoiceDataset(Dataset):
    """
    Each item returns:
      tokens    : (T_text,)    LongTensor of phoneme IDs
      mel       : (n_mels, T_mel) FloatTensor  (channel-first for conv)
      duration  : (T_text,)    LongTensor of per-phoneme frame counts
      pitch     : (T_mel,)     FloatTensor log-F0
      energy    : (T_mel,)     FloatTensor RMS energy
      emotion_id: ()           LongTensor scalar
      intensity : ()           FloatTensor scalar in [0,1]
    """

    def __init__(
        self,
        data_dir: str,
        cfg: ModelConfig,
        split: str = "train",
        train_ratio: float = 0.95,
    ):
        self.data_dir = data_dir
        self.mel_cfg = cfg.mel
        self.tokenizer = PhonemeTokenizer()
        self.mel_transform = build_mel_transform(cfg.mel)

        self.emotion_map = {e: i for i, e in enumerate(EMOTION_LABELS)}

        # Read metadata
        meta_path = os.path.join(data_dir, "metadata.csv")
        all_rows = []
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) >= 5:
                    all_rows.append(row)

        # Train / val split
        n_train = int(len(all_rows) * train_ratio)
        self.rows = all_rows[:n_train] if split == "train" else all_rows[n_train:]

    def __len__(self) -> int:
        return len(self.rows)

    def _load_or_compute_mel(self, fileid: str) -> torch.Tensor:
        mel_path = os.path.join(self.data_dir, "mels", f"{fileid}.pt")
        if os.path.isfile(mel_path):
            return torch.load(mel_path)
        # Compute on-the-fly
        wav_path = os.path.join(self.data_dir, "wavs", f"{fileid}.wav")
        wav, sr = torchaudio.load(wav_path)
        if sr != self.mel_cfg.sample_rate:
            resampler = T.Resample(sr, self.mel_cfg.sample_rate)
            wav = resampler(wav)
        wav = wav.mean(0, keepdim=True)  # mono
        mel = self.mel_transform(wav).squeeze(0)  # (n_mels, T)
        mel = torch.log(mel.clamp(min=1e-5))       # log-mel
        return mel

    def _load_or_zeros(self, subdir: str, fileid: str, length: int) -> torch.Tensor:
        path = os.path.join(self.data_dir, subdir, f"{fileid}.pt")
        if os.path.isfile(path):
            return torch.load(path)
        return torch.zeros(length)

    def __getitem__(self, idx: int) -> dict:
        fileid, text, phonemes, emotion, intensity = self.rows[idx][:5]

        tokens = torch.tensor(self.tokenizer.encode(phonemes), dtype=torch.long)
        mel = self._load_or_compute_mel(fileid)               # (n_mels, T_mel)
        T_mel = mel.size(1)
        T_text = tokens.size(0)

        duration = self._load_or_zeros("durations", fileid, T_text).long()
        pitch    = self._load_or_zeros("pitch",     fileid, T_mel)
        energy   = self._load_or_zeros("energy",    fileid, T_mel)

        emotion_id = self.emotion_map.get(emotion.strip(), 0)
        intensity_val = float(intensity)

        return {
            "tokens":     tokens,
            "mel":        mel,
            "duration":   duration,
            "pitch":      pitch,
            "energy":     energy,
            "emotion_id": torch.tensor(emotion_id, dtype=torch.long),
            "intensity":  torch.tensor(intensity_val, dtype=torch.float),
            "fileid":     fileid,
        }


# ---------------------------------------------------------------------------
# Collate function (for DataLoader)
# ---------------------------------------------------------------------------
def collate_fn(batch: list[dict]) -> dict:
    """Pad a list of dataset items into a batch with masks."""
    token_list  = [b["tokens"]   for b in batch]
    mel_list    = [b["mel"].T    for b in batch]   # (T, n_mels)
    dur_list    = [b["duration"] for b in batch]
    pitch_list  = [b["pitch"]    for b in batch]
    energy_list = [b["energy"]   for b in batch]

    tokens  = pad_sequence(token_list,  batch_first=True, padding_value=0)
    mels    = pad_sequence(mel_list,    batch_first=True, padding_value=0.0)  # (B, T, n_mels)
    durs    = pad_sequence(dur_list,    batch_first=True, padding_value=0)
    pitches = pad_sequence(pitch_list,  batch_first=True, padding_value=0.0)
    energies= pad_sequence(energy_list, batch_first=True, padding_value=0.0)

    # Padding masks (True = padded position)
    src_lens = torch.tensor([t.size(0) for t in token_list])
    mel_lens = torch.tensor([m.size(0) for m in mel_list])
    src_mask = torch.arange(tokens.size(1)).unsqueeze(0) >= src_lens.unsqueeze(1)
    mel_mask = torch.arange(mels.size(1)).unsqueeze(0) >= mel_lens.unsqueeze(1)

    return {
        "tokens":     tokens,
        "mel":        mels.transpose(1, 2),   # (B, n_mels, T) for vocoder compat
        "mel_t":      mels,                   # (B, T, n_mels) for decoder loss
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


def build_dataloader(
    data_dir: str,
    cfg: ModelConfig,
    split: str = "train",
    num_workers: int = 4,
) -> DataLoader:
    ds = VoiceDataset(data_dir, cfg, split=split)
    return DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )
