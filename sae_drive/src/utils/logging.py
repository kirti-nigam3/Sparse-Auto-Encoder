"""Unified logging: TensorBoard + Weights & Biases."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


class ExperimentLogger:
    """Dual TensorBoard + W&B logger with a uniform API."""

    def __init__(self, cfg: DictConfig, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tb_writer: SummaryWriter | None = None
        if cfg.experiment.get("log_to_tensorboard", True):
            self.tb_writer = SummaryWriter(log_dir=str(self.output_dir / "tb"))

        self.use_wandb = bool(cfg.experiment.get("log_to_wandb", False)) and _HAS_WANDB
        if self.use_wandb:
            from .config import to_container

            wandb.init(
                project=cfg.experiment.get("wandb_project", "sae-drive"),
                entity=cfg.experiment.get("wandb_entity", None),
                name=cfg.experiment.get("name", "sae_drive"),
                dir=str(self.output_dir),
                config=to_container(cfg),
            )

    def log_scalars(self, scalars: dict[str, float], step: int, prefix: str = "") -> None:
        for k, v in scalars.items():
            key = f"{prefix}/{k}" if prefix else k
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(key, v, step)
            if self.use_wandb:
                wandb.log({key: v}, step=step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        if self.tb_writer is not None:
            self.tb_writer.add_histogram(tag, values, step)
        if self.use_wandb:
            wandb.log({tag: wandb.Histogram(values.detach().cpu().numpy())}, step=step)

    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        if self.tb_writer is not None:
            self.tb_writer.add_image(tag, image, step)
        if self.use_wandb:
            wandb.log({tag: wandb.Image(image.detach().cpu())}, step=step)

    def log_artifact(self, name: str, path: str | Path) -> None:
        if self.use_wandb:
            artifact = wandb.Artifact(name=name, type="model")
            artifact.add_file(str(path))
            wandb.log_artifact(artifact)

    def watch_model(self, model: torch.nn.Module) -> None:
        if self.use_wandb:
            wandb.watch(model, log="gradients", log_freq=500)

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()

    def log_dict(self, payload: dict[str, Any], step: int) -> None:
        """Best-effort logging of arbitrary keyed payload."""
        for k, v in payload.items():
            if isinstance(v, (int, float)):
                self.log_scalars({k: float(v)}, step)
