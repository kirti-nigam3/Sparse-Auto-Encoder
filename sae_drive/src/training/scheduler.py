"""LR schedulers."""
from __future__ import annotations

import math

import torch


class WarmupCosineLR:
    """Linear warmup followed by cosine annealing to zero. Step-based."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step = 0

    def _scale(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self) -> None:
        self._step += 1
        scale = self._scale(self._step)
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * scale

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, state: dict) -> None:
        self._step = int(state["step"])

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
