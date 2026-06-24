"""SAE-Drive: top-level model composing encoder + SAE bottleneck + decoders."""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .encoder import HybridVideoEncoder
from .sae import TopKSparseAutoencoder
from .decoder import EgoTrajectoryHead, FeatureReconstructionHead, FlowPredictionHead


class SAEDrive(nn.Module):
    """Full SAE-Drive model.

    Returns a dict with all intermediate tensors required by the multi-objective loss.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = HybridVideoEncoder(cfg)
        self.bottleneck = TopKSparseAutoencoder(cfg)
        self.recon_head = FeatureReconstructionHead(self.encoder.embed_dim)
        self.flow_head = FlowPredictionHead(cfg)
        self.ego_head = EgoTrajectoryHead(cfg)

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        enc_out = self.encoder(clip)
        e = enc_out["embedding"]
        sae_out = self.bottleneck(e)
        z = sae_out["z"]
        recon = self.recon_head(sae_out["recon"])
        flow_pred = self.flow_head(z)
        ego_pred = self.ego_head(z)
        return {
            "embedding": e,
            "z": z,
            "pre_activation": sae_out["pre_activation"],
            "recon": recon,
            "flow_pred": flow_pred,
            "ego_pred": ego_pred,
        }

    # ------------------------------------------------------------------
    # Convenience: encode-only path for analysis
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        enc_out = self.encoder(clip)
        e = enc_out["embedding"]
        pre = self.bottleneck.encode_preactivations(e)
        z = self.bottleneck.topk_relu(pre, self.bottleneck.topk)
        return {"embedding": e, "pre_activation": pre, "z": z}

    def maintain_constraints(self) -> None:
        """Call after every optimizer step."""
        self.bottleneck.maintain_decoder_norm()
