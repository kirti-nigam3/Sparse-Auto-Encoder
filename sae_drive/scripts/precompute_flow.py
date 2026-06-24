"""Precompute RAFT optical flow for the training set.

Usage:
    python -m scripts.precompute_flow \
        --config configs/default.yaml \
        --split train \
        --device cuda

The script reads the manifest defined in the config, decodes each sequence with PyAV,
runs torchvision RAFT-Large for consecutive frame pairs, and writes flow arrays of
shape [N-1, 2, H/d, W/d] (float16) under `data.flow_root/{seq_id}.npy`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.preprocessing import compute_raft_flow, save_flow
from src.utils import load_config, merge_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute optical flow with RAFT")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-frames", type=int, default=64, help="Frame chunk size per RAFT call")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _load_raft(device: torch.device) -> torch.nn.Module:
    try:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "torchvision.models.optical_flow not available; install torchvision >= 0.16."
        ) from exc
    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights, progress=True).to(device).eval()
    return model


def _decode_sequence(video_root: Path, seq: dict) -> np.ndarray:
    if "frames_dir" in seq:
        from PIL import Image

        dir_path = video_root / seq["frames_dir"]
        files = sorted(dir_path.glob("*.jpg"))
        frames = [np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8) for f in files]
        return np.stack(frames, axis=0)
    if "video_path" in seq:
        import av  # type: ignore

        frames = []
        with av.open(str(video_root / seq["video_path"])) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
        return np.stack(frames, axis=0)
    raise RuntimeError(f"Sequence {seq.get('id')} has no readable frame source")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_overrides(cfg, args.overrides)

    manifest_path = Path(cfg.data.manifest)
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    sequences = manifest[args.split]

    video_root = Path(cfg.data.video_root)
    flow_root = Path(cfg.data.flow_root)
    flow_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model = _load_raft(device)
    downscale = int(cfg.data.flow_downscale)

    for seq in tqdm(sequences, desc=f"RAFT {args.split}"):
        out_path = flow_root / f"{seq['id']}.npy"
        if out_path.exists():
            continue
        try:
            frames = _decode_sequence(video_root, seq)
        except Exception as exc:
            print(f"[skip] {seq.get('id')}: {exc}", file=sys.stderr)
            continue

        # Center crop to 16:9 and resize before flow to match dataloader spatial layout
        H = int(cfg.data.image_height)
        W = int(cfg.data.image_width)
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()                    # [N, 3, H0, W0]

        # Center 16:9 crop
        h0, w0 = frames_t.shape[-2:]
        target_ratio = 16.0 / 9.0
        cur_ratio = w0 / h0
        if cur_ratio > target_ratio:
            new_w = int(h0 * target_ratio)
            left = (w0 - new_w) // 2
            frames_t = frames_t[..., left : left + new_w]
        else:
            new_h = int(w0 / target_ratio)
            top = (h0 - new_h) // 2
            frames_t = frames_t[..., top : top + new_h, :]
        frames_t = torch.nn.functional.interpolate(frames_t, size=(H, W), mode="bilinear", align_corners=False)

        # Run RAFT in chunks
        chunks: list[torch.Tensor] = []
        bs = max(2, int(args.batch_frames))
        for i in range(0, frames_t.shape[0] - 1, bs - 1):
            chunk = frames_t[i : i + bs].to(device)
            flow = compute_raft_flow(chunk, model, iters=20, downscale=downscale)
            chunks.append(flow.cpu())
        flow_all = torch.cat(chunks, dim=0)
        save_flow(flow_all, out_path)

    print(f"Flow precomputed for {len(sequences)} sequences → {flow_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
