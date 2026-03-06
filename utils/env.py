"""
utils/env.py — Secure environment & configuration loader for VoiceAI.

Loads .env automatically (via python-dotenv if installed), then exposes
typed helpers to read each setting. All secrets come from environment
variables — NEVER from hardcoded strings in source code.

Usage:
    from utils.env import get_hf_token, get_dataset_config, apply_env_to_config
"""

import os
import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auto-load .env if python-dotenv is available
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _loaded = load_dotenv(override=False)   # don't override already-set vars
    if _loaded:
        log.info("Loaded .env")
except ImportError:
    pass   # dotenv not installed; rely on shell-exported env vars


# ---------------------------------------------------------------------------
# Typed accessors
# ---------------------------------------------------------------------------

def get_hf_token() -> str | None:
    """Return the HuggingFace access token, or None if not set."""
    token = os.environ.get("HF_TOKEN", "").strip()
    return token if token else None


def get_dataset_config_name() -> str:
    """HF dataset subset name (microset | clean | dirty | …)."""
    return os.environ.get("HF_DATASET_CONFIG", "microset").strip()


def get_local_data_dir() -> str:
    """Local pre-processed data dir; empty string → use HF streaming."""
    return os.environ.get("LOCAL_DATA_DIR", "").strip()


def get_checkpoint_dir() -> str:
    return os.environ.get("CHECKPOINT_DIR", "checkpoints").strip()


def get_log_dir() -> str:
    return os.environ.get("LOG_DIR", "logs").strip()


# ---------------------------------------------------------------------------
# Apply env vars to a ModelConfig instance
# ---------------------------------------------------------------------------
def apply_env_to_config(cfg) -> None:
    """
    Mutate a ModelConfig in-place using values from environment variables.
    Env vars take precedence over code defaults but NOT over explicit CLI args.

    Call this right after constructing ModelConfig():
        cfg = ModelConfig()
        apply_env_to_config(cfg)
    """
    local_dir = get_local_data_dir()
    if local_dir:
        cfg.dataset.local_data_dir = local_dir
    else:
        cfg.dataset.hf_config = get_dataset_config_name()

    cfg.training.checkpoint_dir = get_checkpoint_dir()
    cfg.training.log_dir        = get_log_dir()
