"""Video clip dataset for driving footage.

The dataset expects a manifest JSON of the form:

```
{
  "train": [
    {
      "id": "scene_001",
      "video_path": "scene_001.mp4",
      "fps": 30,
      "num_frames": 600,
      "flow_path": "flow/scene_001.npy",        # optional, [N, 2, h, w] float16
      "ego_path": "ego/scene_001.npy"           # optional, [N, 3]   (dx, dy, dtheta) per frame
    },
    ...
  ],
  "val": [ ... ]
}
```

Each `__getitem__` yields a temporal window suitable for SAE-Drive training.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from . import transforms as T


@dataclass
class ClipSpec:
    """A single training clip's location inside a sequence."""

    sequence_idx: int
    start_frame: int          # frame index in the source video (source FPS)
    stride: int               # frame stride to reach target FPS
    clip_length: int          # T
    future_horizon: int       # ΔT


class DrivingClipDataset(Dataset):
    """Loads driving video clips with synchronized flow + ego targets.

    Decoding strategy: by default uses PyAV (`av`) which is robust across codecs.
    Optionally substitute with `decord` for higher throughput.
    """

    def __init__(self, cfg: DictConfig, split: str = "train") -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.is_train = split == cfg.data.get("split_train", "train")

        self.video_root = Path(cfg.data.video_root)
        self.flow_root = Path(cfg.data.flow_root) if cfg.data.get("flow_root") else None
        self.ego_root = Path(cfg.data.ego_root) if cfg.data.get("ego_root") else None

        manifest_path = Path(cfg.data.manifest)
        with manifest_path.open("r") as f:
            manifest = json.load(f)
        self.sequences: list[dict[str, Any]] = manifest[split]

        self.source_fps = int(cfg.data.source_fps)
        self.target_fps = int(cfg.data.fps)
        self.base_stride = max(1, self.source_fps // self.target_fps)
        self.clip_length = int(cfg.data.clip_length)
        self.future_horizon = int(cfg.data.future_horizon)
        self.clip_stride = int(cfg.data.clip_stride)

        self.height = int(cfg.data.image_height)
        self.width = int(cfg.data.image_width)
        self.flow_downscale = int(cfg.data.flow_downscale)
        self.min_ego_speed = float(cfg.data.min_ego_speed_kmh)

        aug = cfg.data.augment
        self.aug_hflip = bool(aug.horizontal_flip) and self.is_train
        self.aug_color = bool(aug.color_jitter) and self.is_train
        self.aug_speed = bool(aug.speed_jitter) and self.is_train
        self.aug_temp_jitter = int(aug.temporal_jitter_frames) if self.is_train else 0
        self.aug_temp_reverse = float(aug.temporal_reverse_prob) if self.is_train else 0.0

        self.clip_specs: list[ClipSpec] = []
        self._index_clips()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _index_clips(self) -> None:
        for seq_idx, seq in enumerate(self.sequences):
            num_frames = int(seq["num_frames"])
            stride = self.base_stride
            window_frames = (self.clip_length + self.future_horizon) * stride
            if num_frames <= window_frames:
                continue
            # Drop near-static segments if ego data available
            ego_mask = self._load_ego_speed_mask(seq, num_frames)
            for start in range(0, num_frames - window_frames, self.clip_stride * stride):
                if ego_mask is not None:
                    end = start + window_frames
                    if not ego_mask[start:end].any():
                        continue
                self.clip_specs.append(
                    ClipSpec(
                        sequence_idx=seq_idx,
                        start_frame=start,
                        stride=stride,
                        clip_length=self.clip_length,
                        future_horizon=self.future_horizon,
                    )
                )

    def _load_ego_speed_mask(self, seq: dict[str, Any], num_frames: int) -> np.ndarray | None:
        if self.ego_root is None or seq.get("ego_path") is None:
            return None
        ego_path = self.ego_root / seq["ego_path"]
        if not ego_path.exists():
            return None
        ego = np.load(ego_path).astype(np.float32)
        # ego is (N, 3) frame-to-frame displacement in meters/radians
        disp = np.linalg.norm(ego[:, :2], axis=1)
        speed_mps = disp * self.source_fps
        speed_kmh = speed_mps * 3.6
        mask = speed_kmh > self.min_ego_speed
        if mask.shape[0] < num_frames:
            pad = np.zeros(num_frames - mask.shape[0], dtype=bool)
            mask = np.concatenate([mask, pad])
        return mask

    def __len__(self) -> int:
        return len(self.clip_specs)

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        spec = self.clip_specs[idx]
        seq = self.sequences[spec.sequence_idx]

        stride = spec.stride
        if self.aug_speed:
            stride = random.choices([stride - 1, stride, stride + 1], weights=[0.2, 0.6, 0.2])[0]
            stride = max(1, stride)

        start = spec.start_frame
        if self.aug_temp_jitter > 0:
            start = max(0, start + random.randint(-self.aug_temp_jitter, self.aug_temp_jitter))

        # frame indices for the input clip and the future window
        total_frames = self.clip_length + self.future_horizon
        frame_indices = [start + i * stride for i in range(total_frames)]
        # Safety clamp
        max_idx = int(seq["num_frames"]) - 1
        frame_indices = [min(fi, max_idx) for fi in frame_indices]

        clip_full = self._read_frames(seq, frame_indices)  # [T+ΔT, 3, H, W] in [0,1]
        clip = clip_full[: self.clip_length]
        future_indices_in_clip = list(range(self.clip_length, self.clip_length + self.future_horizon))

        flow = self._load_flow(seq, frame_indices, future_indices_in_clip)
        ego = self._load_ego(seq, frame_indices, future_indices_in_clip)

        # Spatial preprocessing
        clip = T.center_crop_16_9(clip)
        clip = T.resize_clip(clip, self.height, self.width)
        if flow is not None:
            fh = self.height // self.flow_downscale
            fw = self.width // self.flow_downscale
            flow = T.resize_clip(flow, fh, fw)

        # Augmentation
        if self.aug_color:
            clip = T.color_jitter(clip)
        if self.aug_hflip and random.random() < 0.5:
            clip, flow = T.horizontal_flip(clip, flow)
            if ego is not None:
                ego = ego.clone()
                ego[:, 0] = -ego[:, 0]
                ego[:, 2] = -ego[:, 2]
        if self.aug_temp_reverse > 0 and random.random() < self.aug_temp_reverse:
            clip, _ = T.temporal_reverse(clip, None)
            # Note: temporal reverse invalidates future prediction targets; we mask them.
            if flow is not None:
                flow = torch.zeros_like(flow)
            if ego is not None:
                ego = torch.zeros_like(ego)

        clip = T.normalize(clip)

        out: dict[str, torch.Tensor] = {
            "clip": clip,                           # [T, 3, H, W]
            "sequence_idx": torch.tensor(spec.sequence_idx, dtype=torch.long),
            "start_frame": torch.tensor(start, dtype=torch.long),
            "stride": torch.tensor(stride, dtype=torch.long),
        }
        if flow is not None:
            out["future_flow"] = flow              # [ΔT, 2, h, w]
        if ego is not None:
            out["future_ego"] = ego                # [ΔT, 3]
        return out

    # ------------------------------------------------------------------
    # Backend readers
    # ------------------------------------------------------------------
    def _read_frames(self, seq: dict[str, Any], frame_indices: list[int]) -> torch.Tensor:
        """Decode the requested frame indices from disk.

        Backends:
          * `frames_dir`: directory with `%06d.jpg` images (preferred for randomness)
          * `video_path`: encoded video; uses PyAV
        """
        if "frames_dir" in seq:
            return self._read_frames_dir(seq["frames_dir"], frame_indices)
        if "video_path" in seq:
            return self._read_frames_pyav(seq["video_path"], frame_indices)
        raise RuntimeError(f"Sequence {seq.get('id')} has no readable frame source")

    def _read_frames_dir(self, frames_dir: str, indices: list[int]) -> torch.Tensor:
        from PIL import Image

        dir_path = self.video_root / frames_dir
        frames = []
        for idx in indices:
            img_path = dir_path / f"{idx:06d}.jpg"
            img = Image.open(img_path).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
            frames.append(arr)
        stacked = np.stack(frames, axis=0)                          # [N, H, W, 3]
        tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0
        return tensor

    def _read_frames_pyav(self, video_path: str, indices: list[int]) -> torch.Tensor:
        import av  # type: ignore

        target = sorted(set(indices))
        target_set = set(target)
        decoded: dict[int, np.ndarray] = {}
        with av.open(str(self.video_root / video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx in target_set:
                    decoded[frame_idx] = frame.to_ndarray(format="rgb24")
                    if len(decoded) == len(target_set):
                        break
        # If decoded is missing some frames, fall back to nearest available
        last_arr: np.ndarray | None = None
        frames = []
        for idx in indices:
            if idx in decoded:
                last_arr = decoded[idx]
            if last_arr is None:
                raise RuntimeError(f"Failed to decode any frame from {video_path}")
            frames.append(last_arr)
        stacked = np.stack(frames, axis=0)
        return torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------
    def _load_flow(self, seq: dict[str, Any], frame_indices: list[int], future_idx_in_clip: list[int]) -> torch.Tensor | None:
        if self.flow_root is None or seq.get("flow_path") is None:
            return None
        flow_path = self.flow_root / seq["flow_path"]
        if not flow_path.exists():
            return None
        # Flow is precomputed as [N-1, 2, h, w] (forward flow at frame t → t+1)
        flow_all = np.load(flow_path, mmap_mode="r")
        future_global = [frame_indices[i] for i in future_idx_in_clip]
        flow_slice = np.stack([flow_all[min(fi, flow_all.shape[0] - 1)] for fi in future_global], axis=0)
        return torch.from_numpy(flow_slice.astype(np.float32))

    def _load_ego(self, seq: dict[str, Any], frame_indices: list[int], future_idx_in_clip: list[int]) -> torch.Tensor | None:
        if self.ego_root is None or seq.get("ego_path") is None:
            return None
        ego_path = self.ego_root / seq["ego_path"]
        if not ego_path.exists():
            return None
        ego_all = np.load(ego_path)                          # [N, 3]
        future_global = [frame_indices[i] for i in future_idx_in_clip]
        ego_slice = np.stack([ego_all[min(fi, ego_all.shape[0] - 1)] for fi in future_global], axis=0)
        return torch.from_numpy(ego_slice.astype(np.float32))


def collate_clips(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Robust collate that drops keys missing in any sample (defensive against partial targets)."""
    keys = set(batch[0].keys())
    for sample in batch[1:]:
        keys &= set(sample.keys())
    out = {}
    for k in keys:
        out[k] = torch.stack([s[k] for s in batch], dim=0)
    return out
