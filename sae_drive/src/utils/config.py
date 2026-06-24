"""Configuration loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load YAML config into an OmegaConf DictConfig."""
    cfg = OmegaConf.load(str(path))
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Top-level config must be a mapping, got {type(cfg)}")
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Persist a config snapshot next to checkpoints."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(path))


def merge_overrides(cfg: DictConfig, overrides: list[str]) -> DictConfig:
    """Merge CLI-style dotted overrides (e.g. ['training.epochs=10'])."""
    if not overrides:
        return cfg
    override_cfg = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(cfg, override_cfg)  # type: ignore[return-value]


def to_container(cfg: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf to a pure-Python dict (e.g. for wandb config)."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
