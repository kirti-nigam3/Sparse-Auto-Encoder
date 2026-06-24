"""Decoder heads: feature reconstruction, future optical flow, future ego trajectory."""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class FeatureReconstructionHead(nn.Module):
    """Identity-shaped head: the SAE itself already produces ê.

    This wrapper allows attaching an optional MLP refinement on top of the SAE
    reconstruction (used if a richer decoder is desired). For phase-1 training
    this is just a pass-through.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, recon: torch.Tensor) -> torch.Tensor:
        return recon


class FlowPredictionHead(nn.Module):
    """Predict future optical flow [B, ΔT, 2, h, w] from sparse code z + temporal context."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        latent_dim = int(cfg.model.bottleneck.latent_dim)
        d = cfg.model.decoder
        self.flow_horizon = int(d.flow_horizon)
        self.flow_channels = int(d.flow_channels)
        self.hidden = int(d.flow_hidden_dim)

        H = int(cfg.data.image_height) // int(cfg.data.flow_downscale)
        W = int(cfg.data.image_width) // int(cfg.data.flow_downscale)
        self.out_h = H
        self.out_w = W

        # Start from low-res 7x12 grid (factor 32 below H,W) then upsample 4x to (H, W).
        # 224/32=7, 384/32=12; 56/8=7, 96/8=12 — works for H/4 flow.
        self.start_h = max(1, self.out_h // 8)
        self.start_w = max(1, self.out_w // 8)
        start_dim = self.start_h * self.start_w * self.hidden

        self.proj = nn.Sequential(
            nn.Linear(latent_dim, self.hidden * 2),
            nn.GELU(),
            nn.Linear(self.hidden * 2, self.flow_horizon * start_dim),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(self.hidden, self.hidden, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.GroupNorm(8, self.hidden),
            nn.GELU(),
            nn.ConvTranspose3d(self.hidden, self.hidden // 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.GroupNorm(8, self.hidden // 2),
            nn.GELU(),
            nn.ConvTranspose3d(self.hidden // 2, self.hidden // 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.GroupNorm(4, self.hidden // 4),
            nn.GELU(),
            nn.Conv3d(self.hidden // 4, self.flow_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        feat = self.proj(z)                                                       # [B, dT*S*h]
        feat = feat.view(b, self.flow_horizon, self.hidden, self.start_h, self.start_w)
        feat = feat.permute(0, 2, 1, 3, 4)                                        # [B, C, dT, h, w]
        flow = self.upsample(feat)                                                # [B, 2, dT, H, W]
        if flow.shape[-2] != self.out_h or flow.shape[-1] != self.out_w:
            flow = torch.nn.functional.interpolate(
                flow,
                size=(self.flow_horizon, self.out_h, self.out_w),
                mode="trilinear",
                align_corners=False,
            )
        flow = flow.permute(0, 2, 1, 3, 4)                                        # [B, dT, 2, H, W]
        return flow


class EgoTrajectoryHead(nn.Module):
    """Predict future ego displacement sequence (dx, dy, dtheta) from sparse code."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        latent_dim = int(cfg.model.bottleneck.latent_dim)
        d = cfg.model.decoder
        self.horizon = int(d.flow_horizon)
        self.ego_dim = int(d.ego_dim)
        hidden = int(d.ego_hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.horizon * self.ego_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        out = self.mlp(z).view(b, self.horizon, self.ego_dim)
        return out
