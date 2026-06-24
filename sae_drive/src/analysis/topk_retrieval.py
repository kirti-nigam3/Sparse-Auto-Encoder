"""Top-K clip retrieval helpers (clip identifier lookup + thumbnail export)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def gather_neuron_clip_metadata(
    activations: dict[str, np.ndarray],
    neuron_top_idx: np.ndarray,
    manifest_sequences: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """For each neuron, resolve top-K clip indices to sequence + frame range metadata.

    Args:
        activations: dict with `sequence_idx`, `start_frame`, `stride`.
        neuron_top_idx: [M, K] clip indices.
        manifest_sequences: list of sequence dicts loaded from the manifest.

    Returns:
        Nested list of length M; each item is a list of K metadata dicts.
    """
    seq_idx = activations["sequence_idx"]
    start = activations["start_frame"]
    stride = activations["stride"]
    out: list[list[dict[str, Any]]] = []
    for clips in neuron_top_idx:
        per_neuron = []
        for ci in clips:
            si = int(seq_idx[ci])
            seq = manifest_sequences[si]
            per_neuron.append(
                {
                    "clip_index": int(ci),
                    "sequence_id": seq.get("id"),
                    "video_path": seq.get("video_path"),
                    "frames_dir": seq.get("frames_dir"),
                    "start_frame": int(start[ci]),
                    "stride": int(stride[ci]),
                }
            )
        out.append(per_neuron)
    return out


def save_neuron_index(
    path: str | Path,
    neuron_metadata: list[list[dict[str, Any]]],
    extras: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {"neurons": neuron_metadata}
    if extras:
        payload.update(extras)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
