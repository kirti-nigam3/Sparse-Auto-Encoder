"""Collect sparse-code activations across a dataset for downstream analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import DrivingClipDataset, collate_clips
from ..models import SAEDrive


@torch.no_grad()
def collect_activations(
    model: SAEDrive,
    loader: DataLoader,
    device: torch.device,
    max_clips: int | None = None,
) -> dict[str, np.ndarray]:
    """Iterate `loader` and gather per-clip metadata + sparse codes.

    Returns a dict with:
      * `z`: [N, M] float32 sparse codes
      * `embedding`: [N, d] dense embeddings
      * `sequence_idx`: [N] int64
      * `start_frame`: [N] int64
      * `stride`: [N] int64
    """
    model.eval()
    z_chunks: list[np.ndarray] = []
    e_chunks: list[np.ndarray] = []
    seq_chunks: list[np.ndarray] = []
    start_chunks: list[np.ndarray] = []
    stride_chunks: list[np.ndarray] = []
    seen = 0
    for batch in tqdm(loader, desc="collect"):
        clip = batch["clip"].to(device, non_blocking=True)
        out = model.encode(clip)
        z_chunks.append(out["z"].detach().cpu().float().numpy())
        e_chunks.append(out["embedding"].detach().cpu().float().numpy())
        seq_chunks.append(batch["sequence_idx"].numpy())
        start_chunks.append(batch["start_frame"].numpy())
        stride_chunks.append(batch["stride"].numpy())
        seen += clip.shape[0]
        if max_clips is not None and seen >= max_clips:
            break
    return {
        "z": np.concatenate(z_chunks, axis=0),
        "embedding": np.concatenate(e_chunks, axis=0),
        "sequence_idx": np.concatenate(seq_chunks, axis=0),
        "start_frame": np.concatenate(start_chunks, axis=0),
        "stride": np.concatenate(stride_chunks, axis=0),
    }


def build_eval_loader(cfg: DictConfig, split: str | None = None) -> DataLoader:
    if split is None:
        split = cfg.data.get("split_val", "val")
    dataset = DrivingClipDataset(cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=bool(cfg.training.pin_memory),
        collate_fn=collate_clips,
    )


def save_activations(data: dict[str, np.ndarray], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **data)


def load_activations(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(str(path)) as data:
        return {k: data[k] for k in data.files}
