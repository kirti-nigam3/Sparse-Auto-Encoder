"""Offline preprocessing utilities for RAFT optical flow + ego pose extraction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_raft_flow(
    frames: torch.Tensor,
    raft_model: torch.nn.Module,
    iters: int = 20,
    downscale: int = 4,
) -> torch.Tensor:
    """Compute forward optical flow for consecutive frame pairs.

    Args:
        frames: [N, 3, H, W] uint8 or float in [0, 255].
        raft_model: torchvision RAFT model or equivalent.
        iters: RAFT refinement iterations.
        downscale: output flow spatial downscale factor.

    Returns:
        flow: [N-1, 2, H/downscale, W/downscale] float16.
    """
    if frames.dtype != torch.float32:
        frames = frames.float()
    # torchvision RAFT expects [-1, 1]
    frames = frames / 127.5 - 1.0
    flows: list[torch.Tensor] = []
    for i in range(frames.shape[0] - 1):
        f0 = frames[i : i + 1]
        f1 = frames[i + 1 : i + 2]
        flow_predictions = raft_model(f0, f1, num_flow_updates=iters)
        flow = flow_predictions[-1]                # [1, 2, H, W]
        if downscale > 1:
            h, w = flow.shape[-2:]
            flow = F.interpolate(
                flow,
                size=(h // downscale, w // downscale),
                mode="bilinear",
                align_corners=False,
            ) / downscale
        flows.append(flow.squeeze(0).half().cpu())
    return torch.stack(flows, dim=0)


def build_manifest(
    video_root: str | Path,
    flow_root: str | Path | None,
    ego_root: str | Path | None,
    train_ids: Iterable[str],
    val_ids: Iterable[str],
    output_path: str | Path,
    source_fps: int = 30,
) -> None:
    """Generate a manifest.json describing all sequences and per-sequence metadata."""
    video_root = Path(video_root)
    train_ids = list(train_ids)
    val_ids = list(val_ids)

    def _entry(seq_id: str) -> dict[str, object]:
        frames_dir = video_root / seq_id
        if frames_dir.is_dir():
            num_frames = len(sorted(frames_dir.glob("*.jpg")))
            entry: dict[str, object] = {
                "id": seq_id,
                "frames_dir": seq_id,
                "fps": source_fps,
                "num_frames": num_frames,
            }
        else:
            video_path = video_root / f"{seq_id}.mp4"
            import av  # type: ignore
            with av.open(str(video_path)) as c:
                stream = c.streams.video[0]
                num_frames = int(stream.frames)
            entry = {
                "id": seq_id,
                "video_path": f"{seq_id}.mp4",
                "fps": source_fps,
                "num_frames": num_frames,
            }
        if flow_root is not None:
            entry["flow_path"] = f"{seq_id}.npy"
        if ego_root is not None:
            entry["ego_path"] = f"{seq_id}.npy"
        return entry

    manifest = {
        "train": [_entry(sid) for sid in train_ids],
        "val": [_entry(sid) for sid in val_ids],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def save_flow(flow: np.ndarray | torch.Tensor, path: str | Path) -> None:
    arr = flow.detach().cpu().numpy() if isinstance(flow, torch.Tensor) else flow
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr.astype(np.float16))
