"""Overcomplete TopK sparse autoencoder bottleneck.

Implements:
  * Overcomplete dictionary (M >> d).
  * TopK gating (Anthropic 2024).
  * Optional L1 / tanh sparsity penalty (added externally).
  * Decoder weight column normalization.
  * Dead neuron tracking (EMA of activation frequency).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class TopKSparseAutoencoder(nn.Module):
    """Sparse bottleneck z = TopK(ReLU(W_enc e + b_enc)); ê = W_dec z + b_dec.

    Attributes:
        latent_dim (M): number of dictionary atoms.
        topk (k): number of active neurons per clip.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        b = cfg.model.bottleneck
        e_dim = int(cfg.model.encoder.embed_dim)
        self.input_dim = e_dim
        self.latent_dim = int(b.latent_dim)
        self.topk = int(b.topk)
        self.normalize_decoder = bool(b.normalize_decoder)
        self.dead_window = int(b.dead_neuron_window)
        self.dead_threshold = float(b.dead_neuron_threshold)

        self.encoder = nn.Linear(self.input_dim, self.latent_dim, bias=True)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim, bias=True)
        self.pre_bias = nn.Parameter(torch.zeros(self.input_dim))

        # Initialization: tied (Anthropic recipe). W_dec ~ unit norm; W_enc = W_dec^T.
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
        with torch.no_grad():
            self._normalize_decoder_weights_()
            self.encoder.weight.copy_(self.decoder.weight.t())
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

        # Activation frequency tracking (EMA)
        self.register_buffer("activation_ema", torch.zeros(self.latent_dim))
        self.register_buffer("steps_since_active", torch.zeros(self.latent_dim, dtype=torch.long))
        self.ema_decay = 0.99

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _normalize_decoder_weights_(self) -> None:
        w = self.decoder.weight                  # [d, M]
        norms = w.norm(dim=0, keepdim=True).clamp(min=1e-8)
        w.div_(norms)

    @torch.no_grad()
    def maintain_decoder_norm(self) -> None:
        """Call after every optimizer step if `normalize_decoder` is True."""
        if self.normalize_decoder:
            self._normalize_decoder_weights_()

    # ------------------------------------------------------------------
    def encode_preactivations(self, e: torch.Tensor) -> torch.Tensor:
        return self.encoder(e - self.pre_bias)

    @staticmethod
    def topk_relu(pre: torch.Tensor, k: int) -> torch.Tensor:
        """ReLU then keep top-k per sample; zero the rest."""
        relu = torch.relu(pre)
        if k <= 0 or k >= relu.shape[-1]:
            return relu
        topk_vals, topk_idx = torch.topk(relu, k=k, dim=-1)
        out = torch.zeros_like(relu)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    # ------------------------------------------------------------------
    def forward(self, e: torch.Tensor) -> dict[str, torch.Tensor]:
        """e: [B, d] → z: [B, M], ê: [B, d]."""
        pre = self.encode_preactivations(e)
        z = self.topk_relu(pre, self.topk)
        recon = self.decoder(z) + self.pre_bias

        # Update activation tracking (no grad)
        with torch.no_grad():
            active = (z > 0).float().mean(dim=0)                  # fraction of batch each neuron fired in
            self.activation_ema.mul_(self.ema_decay).add_(active, alpha=1.0 - self.ema_decay)
            fired_now = (active > 0).long()
            self.steps_since_active = self.steps_since_active + 1
            self.steps_since_active[fired_now.bool()] = 0

        return {
            "z": z,
            "pre_activation": pre,
            "recon": recon,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def dead_neuron_mask(self) -> torch.Tensor:
        """Return a 0/1 mask over neurons that count as 'dead'."""
        return (self.activation_ema < self.dead_threshold).float()

    @torch.no_grad()
    def neuron_utilization(self) -> torch.Tensor:
        return self.activation_ema.clone()

    @torch.no_grad()
    def reset_dead_neurons(self, generator: torch.Generator | None = None) -> int:
        """Optionally reset dead-neuron weights to fresh random vectors (Anthropic neuron-resampling).

        Returns number of neurons reset.
        """
        dead = (self.steps_since_active > self.dead_window).nonzero(as_tuple=False).squeeze(-1)
        if dead.numel() == 0:
            return 0
        device = self.decoder.weight.device
        new_dirs = torch.randn(self.input_dim, dead.numel(), device=device, generator=generator)
        new_dirs = new_dirs / new_dirs.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight[:, dead] = new_dirs
        self.encoder.weight[dead] = new_dirs.t() * 0.2           # smaller scale on encoder side
        self.encoder.bias[dead] = 0.0
        self.activation_ema[dead] = self.dead_threshold * 2
        self.steps_since_active[dead] = 0
        return int(dead.numel())
