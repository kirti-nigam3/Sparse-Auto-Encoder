"""Spatiotemporal transforms for driving video clips."""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize(clip: torch.Tensor) -> torch.Tensor:
    """Normalize a clip [T, 3, H, W] in [0, 1] with ImageNet stats."""
    return (clip - IMAGENET_MEAN.to(clip.device)) / IMAGENET_STD.to(clip.device)


def denormalize(clip: torch.Tensor) -> torch.Tensor:
    return clip * IMAGENET_STD.to(clip.device) + IMAGENET_MEAN.to(clip.device)


def resize_clip(clip: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize clip [T, 3, H, W] to (height, width)."""
    return F.interpolate(clip, size=(height, width), mode="bilinear", align_corners=False)


def center_crop_16_9(clip: torch.Tensor) -> torch.Tensor:
    """Center crop to 16:9 aspect ratio."""
    _, _, h, w = clip.shape
    target_ratio = 16.0 / 9.0
    cur_ratio = w / h
    if cur_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return clip[..., left : left + new_w]
    new_h = int(w / target_ratio)
    top = (h - new_h) // 2
    return clip[..., top : top + new_h, :]


def horizontal_flip(clip: torch.Tensor, flow: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Flip clip and (optionally) the corresponding optical flow horizontally.

    For flow [T, 2, H, W] in (u, v) format, the u channel sign must invert.
    """
    clip = torch.flip(clip, dims=[-1])
    if flow is not None:
        flow = torch.flip(flow, dims=[-1])
        flow[..., 0, :, :] = -flow[..., 0, :, :]
    return clip, flow


def color_jitter(clip: torch.Tensor, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.1) -> torch.Tensor:
    """Apply per-clip color jitter (consistent across frames). Expects clip in [0, 1]."""
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        clip = (clip * factor).clamp(0.0, 1.0)
    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        mean = clip.mean(dim=[1, 2, 3], keepdim=True)
        clip = ((clip - mean) * factor + mean).clamp(0.0, 1.0)
    if saturation > 0:
        factor = 1.0 + random.uniform(-saturation, saturation)
        gray = clip.mean(dim=1, keepdim=True)
        clip = (gray + (clip - gray) * factor).clamp(0.0, 1.0)
    return clip


def temporal_reverse(clip: torch.Tensor, flow: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reverse clip in time. Flow vectors must invert direction."""
    clip = torch.flip(clip, dims=[0])
    if flow is not None:
        flow = -torch.flip(flow, dims=[0])
    return clip, flow
