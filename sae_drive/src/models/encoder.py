"""Hybrid 3D CNN stem + factorized space-time Transformer encoder."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# 3D ResNet stem (lightweight)
# ---------------------------------------------------------------------------
class Conv3DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int, int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()
        if stride != (1, 1, 1) or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class Conv3DStem(nn.Module):
    """ResNet3D-style stem.

    Input: [B, 3, T, H, W]
    Output: [B, C, T/4, H/16, W/16]
    """

    def __init__(self, in_channels: int = 3, stem_channels: int = 64, feature_channels: int = 512) -> None:
        super().__init__()
        c1, c2, c3, c4 = stem_channels, stem_channels * 2, stem_channels * 4, feature_channels
        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(c1),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.layer1 = Conv3DBlock(c1, c2, stride=(2, 2, 2))
        self.layer2 = Conv3DBlock(c2, c3, stride=(2, 2, 2))
        self.layer3 = Conv3DBlock(c3, c4, stride=(1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# ---------------------------------------------------------------------------
# Factorized space-time Transformer block
# ---------------------------------------------------------------------------
class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.dropout(self.proj(out))


class TransformerBlock(nn.Module):
    """Factorized (spatial then temporal) attention block.

    Input expected as [B, T, S, C]; spatial axis S = H'*W'.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = MultiheadSelfAttention(dim, heads, dropout)
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = MultiheadSelfAttention(dim, heads, dropout)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, S, C]
        b, t, s, c = x.shape
        # Spatial attention: over S, for each (b, t)
        xs = rearrange(x, "b t s c -> (b t) s c")
        xs = xs + self.attn_s(self.norm_s(xs))
        x = rearrange(xs, "(b t) s c -> b t s c", b=b)
        # Temporal attention: over T, for each (b, s)
        xt = rearrange(x, "b t s c -> (b s) t c")
        xt = xt + self.attn_t(self.norm_t(xt))
        x = rearrange(xt, "(b s) t c -> b t s c", b=b)
        # Channel MLP
        x = x + self.mlp(self.norm_m(x))
        return x


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
class LearnedPosEmb(nn.Module):
    def __init__(self, max_t: int, max_s: int, dim: int) -> None:
        super().__init__()
        self.t_pos = nn.Parameter(torch.zeros(1, max_t, 1, dim))
        self.s_pos = nn.Parameter(torch.zeros(1, 1, max_s, dim))
        nn.init.trunc_normal_(self.t_pos, std=0.02)
        nn.init.trunc_normal_(self.s_pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, S, C]
        _, t, s, _ = x.shape
        return x + self.t_pos[:, :t] + self.s_pos[:, :, :s]


# ---------------------------------------------------------------------------
# Full hybrid encoder
# ---------------------------------------------------------------------------
class HybridVideoEncoder(nn.Module):
    """Hybrid 3D CNN + factorized Transformer encoder producing a clip embedding.

    Forward returns a dict:
      * `tokens`: [B, T', S, d]    — full token sequence after transformer
      * `temporal`: [B, T', d]     — spatially pooled per temporal token
      * `embedding`: [B, d]        — clip-level pooled embedding (fed to SAE)
      * `stem_features`: [B, C, T', H', W']  — for decoder skip connections
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        m = cfg.model.encoder
        self.embed_dim = int(m.embed_dim)
        self.feature_channels = int(m.feature_channels)

        self.stem = Conv3DStem(
            in_channels=int(m.in_channels),
            stem_channels=int(m.stem_channels),
            feature_channels=self.feature_channels,
        )
        self.token_proj = nn.Conv3d(self.feature_channels, self.embed_dim, kernel_size=1)

        # We pre-compute max grid size from data config to size positional embedding
        T = int(cfg.data.clip_length)
        H = int(cfg.data.image_height)
        W = int(cfg.data.image_width)
        # Stem reduces: T → T/4 (after layer1+layer2 = stride 4), H,W → /16
        self.t_grid = max(1, T // 4)
        self.s_grid = (H // 16) * (W // 16)

        self.pos_emb = LearnedPosEmb(self.t_grid, self.s_grid, self.embed_dim)
        self.dropout = nn.Dropout(float(m.dropout))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.embed_dim,
                    heads=int(m.transformer_heads),
                    mlp_ratio=float(m.transformer_mlp_ratio),
                    dropout=float(m.dropout),
                )
                for _ in range(int(m.transformer_layers))
            ]
        )

        self.norm = nn.LayerNorm(self.embed_dim)
        self.clip_query = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.clip_query, std=0.02)
        self.clip_pool_attn = nn.MultiheadAttention(self.embed_dim, num_heads=int(m.transformer_heads), batch_first=True)

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        # clip: [B, T, 3, H, W] → permute to [B, 3, T, H, W]
        x = clip.permute(0, 2, 1, 3, 4).contiguous()
        feat = self.stem(x)                                    # [B, C, T', H', W']
        b, _, tp, hp, wp = feat.shape
        x = self.token_proj(feat)                              # [B, d, T', H', W']
        x = rearrange(x, "b c t h w -> b t (h w) c")           # [B, T', S, d]
        x = self.pos_emb(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)                                       # [B, T', S, d]

        temporal = x.mean(dim=2)                                # [B, T', d]
        # Learned attention pool over T' for clip embedding
        query = self.clip_query.expand(b, -1, -1)               # [B, 1, d]
        pooled, _ = self.clip_pool_attn(query, temporal, temporal, need_weights=False)
        embedding = pooled.squeeze(1)                           # [B, d]

        return {
            "tokens": x,
            "temporal": temporal,
            "embedding": embedding,
            "stem_features": feat,
            "grid": torch.tensor([tp, hp, wp]),
        }
