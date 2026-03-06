"""
config.py — Central configuration for the Voice AI model.
All hyperparameters live here. Edit this file to reshape the model.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Emotion / Expression labels
# ---------------------------------------------------------------------------
EMOTION_LABELS: List[str] = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgusted",
    "surprised",
    "excited",
    "whisper",
    "calm",
]

# ---------------------------------------------------------------------------
# Text / Phoneme vocabulary
# ---------------------------------------------------------------------------
# A minimal IPA-style phoneme set. Replace with your full set.
PHONEME_VOCAB: List[str] = [
    "<pad>", "<unk>", "<bos>", "<eos>", " ",
    "p", "b", "t", "d", "k", "g",
    "f", "v", "s", "z", "ʃ", "ʒ", "h",
    "m", "n", "ŋ",
    "l", "r", "w", "j",
    "æ", "ɑ", "ɒ", "ɔ", "ə", "ɛ", "ɪ", "ʊ", "ʌ",
    "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ",
    "i", "u", "e", "o", "a",
    ",", ".", "?", "!", ";", ":",
]


@dataclass
class TextEncoderConfig:
    vocab_size: int = len(PHONEME_VOCAB)
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 1024


@dataclass
class EmotionConfig:
    num_emotions: int = len(EMOTION_LABELS)
    embed_dim: int = 128            # Projects → model hidden dim
    use_intensity: bool = True       # Float 0-1 intensity scalar per emotion


@dataclass
class ProsodyConfig:
    """Variance adaptor: predicts duration, pitch (F0), and energy."""
    hidden_dim: int = 256
    kernel_size: int = 3
    dropout: float = 0.1
    # Pitch stats (log-Hz); set from dataset at preprocessing.
    pitch_mean: float = 0.0
    pitch_std: float = 1.0
    # Energy stats (dB); set from dataset at preprocessing.
    energy_mean: float = 0.0
    energy_std: float = 1.0
    # Number of buckets for pitch / energy quantisation (for supervision)
    n_pitch_bins: int = 256
    n_energy_bins: int = 256


@dataclass
class DecoderConfig:
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 6
    ff_dim: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 4096         # Frame-level sequence


@dataclass
class MelConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0


@dataclass
class VocoderConfig:
    """HiFi-GAN style vocoder config."""
    in_channels: int = 80           # = n_mels
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    leaky_relu_slope: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    betas: tuple = (0.9, 0.98)
    weight_decay: float = 1e-6
    warmup_steps: int = 4000
    max_steps: int = 300_000
    grad_clip: float = 1.0
    fp16: bool = True               # Mixed precision

    # Loss weights
    mel_loss_weight: float = 1.0
    duration_loss_weight: float = 1.0
    pitch_loss_weight: float = 1.0
    energy_loss_weight: float = 1.0

    # Logging / checkpointing
    log_every: int = 100
    save_every: int = 5_000
    eval_every: int = 5_000
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"


@dataclass
class DatasetConfig:
    """HuggingFace dataset settings used by dataset.py and train.py."""
    # HuggingFace repo id
    hf_repo: str = "MLCommons/peoples_speech"
    # Dataset config / subset name
    # Choices: "microset" | "clean" | "dirty" | "clean_sa" | "dirty_sa"
    hf_config: str = "microset"
    # HF split names
    train_split: str = "train"
    val_split: str = "validation"
    # Stream from HF instead of loading everything into RAM
    streaming: bool = True
    # Number of DataLoader workers
    num_workers: int = 0          # 0 = main process (safe for streaming)
    # Fallback to local disk data (set to a path string to override HF)
    # e.g. local_data_dir = "./data"   → reads metadata.csv + .pt files
    local_data_dir: str = ""
    # Fraction of data used for training (rest = validation)
    train_ratio: float = 0.95
    # Cap number of items per epoch (None = unlimited)
    max_items: int = 0            # 0 = no cap


@dataclass
class ModelConfig:
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    mel: MelConfig = field(default_factory=MelConfig)
    vocoder: VocoderConfig = field(default_factory=VocoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    # Shared hidden dimension across encoder / decoder
    hidden_dim: int = 256
