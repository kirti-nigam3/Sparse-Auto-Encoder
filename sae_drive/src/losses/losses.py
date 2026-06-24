"""Multi-objective SAE-Drive loss.

L_total = λ1 L_recon + λ2 L_sparse + λ3 L_future + λ4 L_temporal + λ5 L_dead

All terms operate on the dict returned by SAEDrive.forward().
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


@dataclass
class LossWeights:
    recon: float = 1.0
    sparse: float = 0.04
    future: float = 1.0
    temporal: float = 0.1
    dead: float = 0.01


class SparseDriveLoss(nn.Module):
    """Composite loss for SAE-Drive training."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        lc = cfg.loss
        self.sparsity_type = str(lc.sparsity_type)
        self.lambda_flow = float(lc.lambda_flow)
        self.lambda_ego = float(lc.lambda_ego)

        # Linear warmup schedule for sparsity and future weights
        self.sparse_initial = float(lc.lambda_sparse_initial)
        self.sparse_final = float(lc.lambda_sparse_final)
        self.sparse_warmup = int(lc.lambda_sparse_warmup_steps)
        self.future_initial = float(lc.lambda_future_initial)
        self.future_final = float(lc.lambda_future_final)
        self.future_warmup = int(lc.lambda_future_warmup_steps)

        self.lambda_recon = float(lc.lambda_recon)
        self.lambda_temporal = float(lc.lambda_temporal)
        self.lambda_dead = float(lc.lambda_dead)

    # ------------------------------------------------------------------
    def current_weights(self, step: int) -> LossWeights:
        sparse = self._anneal(self.sparse_initial, self.sparse_final, self.sparse_warmup, step)
        future = self._anneal(self.future_initial, self.future_final, self.future_warmup, step)
        return LossWeights(
            recon=self.lambda_recon,
            sparse=sparse,
            future=future,
            temporal=self.lambda_temporal,
            dead=self.lambda_dead,
        )

    @staticmethod
    def _anneal(v0: float, v1: float, warmup: int, step: int) -> float:
        if warmup <= 0:
            return v1
        if step >= warmup:
            return v1
        t = step / max(1, warmup)
        return v0 + (v1 - v0) * t

    # ------------------------------------------------------------------
    # Individual term implementations
    # ------------------------------------------------------------------
    def reconstruction_loss(self, e: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, e)

    def sparsity_loss(self, z: torch.Tensor, activation_ema: torch.Tensor | None = None) -> torch.Tensor:
        if self.sparsity_type == "l1":
            return z.abs().mean()
        if self.sparsity_type == "tanh":
            if activation_ema is None:
                scale = z.abs().mean().detach().clamp(min=1e-6)
            else:
                scale = activation_ema.clamp(min=1e-6).unsqueeze(0)
            return torch.tanh(8.0 * z / scale).mean()
        raise ValueError(f"Unknown sparsity_type: {self.sparsity_type}")

    def future_loss(
        self,
        flow_pred: torch.Tensor | None,
        flow_target: torch.Tensor | None,
        ego_pred: torch.Tensor | None,
        ego_target: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=(flow_pred.device if flow_pred is not None else ego_pred.device))
        if flow_pred is not None and flow_target is not None:
            l_flow = F.l1_loss(flow_pred, flow_target)
            components["flow"] = l_flow.detach()
            total = total + self.lambda_flow * l_flow
        if ego_pred is not None and ego_target is not None:
            l_ego = F.l1_loss(ego_pred, ego_target)
            components["ego"] = l_ego.detach()
            total = total + self.lambda_ego * l_ego
        return total, components

    def temporal_consistency_loss(self, z: torch.Tensor, neighbor_z: torch.Tensor | None, alpha: torch.Tensor | None) -> torch.Tensor:
        """L2 between consecutive overlapping clip codes weighted by overlap fraction α."""
        if neighbor_z is None or alpha is None:
            return torch.zeros((), device=z.device)
        diff = (z - neighbor_z).pow(2).mean(dim=-1)              # [B]
        return (alpha * diff).mean()

    def dead_neuron_loss(self, activation_ema: torch.Tensor, threshold: float, target: float = 0.01) -> torch.Tensor:
        """Push dormant neurons toward a target activation frequency."""
        dead_mask = (activation_ema < threshold).float()
        if dead_mask.sum() == 0:
            return torch.zeros((), device=activation_ema.device)
        gap = (1.0 - activation_ema / max(target, 1e-6)).clamp(min=0.0)
        return (dead_mask * gap.pow(2)).sum() / dead_mask.sum().clamp(min=1.0)

    # ------------------------------------------------------------------
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor | None],
        step: int,
        activation_ema: torch.Tensor | None = None,
        dead_threshold: float = 0.001,
    ) -> dict[str, torch.Tensor]:
        weights = self.current_weights(step)
        e = outputs["embedding"]
        recon = outputs["recon"]
        z = outputs["z"]
        flow_pred = outputs.get("flow_pred")
        ego_pred = outputs.get("ego_pred")

        l_recon = self.reconstruction_loss(e, recon)
        l_sparse = self.sparsity_loss(z, activation_ema)
        l_future, future_components = self.future_loss(
            flow_pred=flow_pred,
            flow_target=targets.get("future_flow"),
            ego_pred=ego_pred,
            ego_target=targets.get("future_ego"),
        )
        l_temporal = self.temporal_consistency_loss(
            z,
            neighbor_z=targets.get("neighbor_z"),
            alpha=targets.get("alpha"),
        )
        l_dead = (
            self.dead_neuron_loss(activation_ema, dead_threshold)
            if activation_ema is not None
            else torch.zeros((), device=z.device)
        )

        total = (
            weights.recon * l_recon
            + weights.sparse * l_sparse
            + weights.future * l_future
            + weights.temporal * l_temporal
            + weights.dead * l_dead
        )

        report = {
            "loss/total": total.detach(),
            "loss/recon": l_recon.detach(),
            "loss/sparse": l_sparse.detach(),
            "loss/future": l_future.detach(),
            "loss/temporal": l_temporal.detach(),
            "loss/dead": l_dead.detach(),
            "weights/recon": torch.tensor(weights.recon),
            "weights/sparse": torch.tensor(weights.sparse),
            "weights/future": torch.tensor(weights.future),
            "weights/temporal": torch.tensor(weights.temporal),
            "weights/dead": torch.tensor(weights.dead),
        }
        for k, v in future_components.items():
            report[f"loss/future_{k}"] = v
        return {"total": total, "report": report}
