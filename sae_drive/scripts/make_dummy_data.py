"""Generate a tiny synthetic dataset for end-to-end pipeline validation.

Creates:
  - data_demo/frames/<seq>/000000.jpg ... (random-noise frames)
  - data_demo/flow/<seq>.npy            (random flow tensor, [N-1, 2, h, w] float16)
  - data_demo/ego/<seq>.npy             (random per-frame displacement, [N, 3])
  - data_demo/manifest.json

Used to verify the train loop, scheduler, logger, and analysis scripts without
real driving footage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def make_sequence(out_root: Path, seq_id: str, num_frames: int, H: int, W: int, flow_h: int, flow_w: int) -> dict:
    frames_dir = out_root / "frames" / seq_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(seq_id) & 0xFFFF)

    # Generate frames with smooth time-varying noise so flow targets are non-trivial
    base = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    for i in range(num_frames):
        shift = (i * 2) % W
        frame = np.roll(base, shift=shift, axis=1)
        frame = (frame.astype(np.int16) + rng.integers(-10, 10, size=frame.shape, dtype=np.int16)).clip(0, 255).astype(np.uint8)
        Image.fromarray(frame, mode="RGB").save(frames_dir / f"{i:06d}.jpg", quality=85)

    # Flow
    flow = rng.standard_normal(size=(num_frames - 1, 2, flow_h, flow_w)).astype(np.float16)
    flow_dir = out_root / "flow"
    flow_dir.mkdir(parents=True, exist_ok=True)
    np.save(flow_dir / f"{seq_id}.npy", flow)

    # Ego (dx, dy, dtheta per source frame; small magnitude in meters & radians)
    ego = rng.standard_normal(size=(num_frames, 3)).astype(np.float32) * 0.5
    ego_dir = out_root / "ego"
    ego_dir.mkdir(parents=True, exist_ok=True)
    np.save(ego_dir / f"{seq_id}.npy", ego)

    return {
        "id": seq_id,
        "frames_dir": f"frames/{seq_id}",
        "fps": 30,
        "num_frames": num_frames,
        "flow_path": f"{seq_id}.npy",
        "ego_path": f"{seq_id}.npy",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Make tiny synthetic dataset")
    parser.add_argument("--root", type=str, default="data_demo")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--flow-downscale", type=int, default=4)
    parser.add_argument("--frames-per-seq", type=int, default=160)
    parser.add_argument("--num-train", type=int, default=4)
    parser.add_argument("--num-val", type=int, default=2)
    args = parser.parse_args()

    out_root = Path(args.root)
    out_root.mkdir(parents=True, exist_ok=True)
    flow_h = args.height // args.flow_downscale
    flow_w = args.width // args.flow_downscale

    train_entries = [
        make_sequence(out_root, f"train_seq_{i:03d}", args.frames_per_seq, args.height, args.width, flow_h, flow_w)
        for i in range(args.num_train)
    ]
    val_entries = [
        make_sequence(out_root, f"val_seq_{i:03d}", args.frames_per_seq, args.height, args.width, flow_h, flow_w)
        for i in range(args.num_val)
    ]

    manifest = {"train": train_entries, "val": val_entries}
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote dummy dataset to {out_root}/")
    print(f"  train: {len(train_entries)} sequences")
    print(f"  val  : {len(val_entries)} sequences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
