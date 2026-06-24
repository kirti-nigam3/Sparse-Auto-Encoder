"""Neuron-level interpretability utilities.

Provided utilities:
  * Top-K activating clips per neuron.
  * Neuron Selectivity Score (NSS) given evaluation-time action labels.
  * Action Separation (AS) metric.
  * Latent traversal generator for the future-flow head.
"""
from __future__ import annotations

import numpy as np
import torch

from ..models import SAEDrive


# ---------------------------------------------------------------------------
# Top-K activating clips
# ---------------------------------------------------------------------------
def topk_clips_per_neuron(z: np.ndarray, k: int = 100) -> np.ndarray:
    """For each neuron i, return indices of the top-k clips by activation.

    Args:
        z: [N, M] sparse code matrix.
        k: number of top clips per neuron.

    Returns:
        idx: [M, k] int64 matrix of clip indices.
    """
    n, m = z.shape
    k = min(k, n)
    # Argpartition is faster than full sort
    part = np.argpartition(-z, kth=k - 1, axis=0)[:k]                  # [k, M]
    top_vals = np.take_along_axis(z, part, axis=0)                     # [k, M]
    order = np.argsort(-top_vals, axis=0)                              # [k, M]
    sorted_idx = np.take_along_axis(part, order, axis=0)               # [k, M]
    return sorted_idx.T                                                # [M, k]


def neuron_active_clips(z: np.ndarray, neuron_id: int, top_k: int = 100) -> np.ndarray:
    activations = z[:, neuron_id]
    if (activations > 0).sum() == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(-activations)
    keep = order[: top_k]
    return keep[activations[keep] > 0]


# ---------------------------------------------------------------------------
# Selectivity / separation metrics (require evaluation-time labels)
# ---------------------------------------------------------------------------
def neuron_selectivity_score(z: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """KL(P_i || U) per neuron, using each clip's most-active neuron.

    Args:
        z: [N, M] activations.
        labels: [N] integer class labels in {0, ..., num_classes-1} (or -1 for unlabeled).
        num_classes: A.

    Returns:
        nss: [M] selectivity score per neuron.
    """
    mask = labels >= 0
    z_l = z[mask]
    y = labels[mask]
    n, m = z_l.shape
    most_active = np.argmax(z_l, axis=1)                  # [n]
    # P_i(a) = P(most_active == i | label = a) / P(label = a) — but we use the formulation in the plan.
    p = np.zeros((m, num_classes), dtype=np.float64)
    for a in range(num_classes):
        ya = y == a
        if ya.sum() == 0:
            continue
        active_for_a = most_active[ya]
        counts = np.bincount(active_for_a, minlength=m)
        p[:, a] = counts / max(1, ya.sum())
    # Normalize per neuron row to a distribution over classes
    row_sums = np.maximum(p.sum(axis=1, keepdims=True), 1e-9)
    p_norm = p / row_sums
    uniform = 1.0 / num_classes
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p_norm > 0, p_norm / uniform, 1.0)
        log_ratio = np.where(p_norm > 0, np.log(ratio), 0.0)
        kl = (p_norm * log_ratio).sum(axis=1)
    return kl


def action_separation(z: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Mean pairwise cosine distance between class centroids in z-space."""
    mask = labels >= 0
    z_l = z[mask]
    y = labels[mask]
    centroids: list[np.ndarray] = []
    for a in range(num_classes):
        ya = z_l[y == a]
        if ya.shape[0] == 0:
            continue
        c = ya.mean(axis=0)
        n = np.linalg.norm(c)
        if n > 0:
            c = c / n
        centroids.append(c)
    if len(centroids) < 2:
        return float("nan")
    C = np.stack(centroids, axis=0)
    sims = C @ C.T
    iu = np.triu_indices(C.shape[0], k=1)
    return float(1.0 - sims[iu].mean())


def action_purity(labels_per_cluster: dict[int, np.ndarray]) -> float:
    """Compute mean cluster purity given per-cluster label arrays."""
    purities = []
    for _, lab in labels_per_cluster.items():
        if lab.size == 0:
            continue
        counts = np.bincount(lab[lab >= 0])
        if counts.size == 0:
            continue
        purities.append(counts.max() / counts.sum())
    return float(np.mean(purities)) if purities else float("nan")


# ---------------------------------------------------------------------------
# Latent traversal for the future-flow head
# ---------------------------------------------------------------------------
@torch.no_grad()
def latent_traversal(
    model: SAEDrive,
    z: torch.Tensor,
    neuron_id: int,
    alphas: torch.Tensor,
) -> torch.Tensor:
    """Sweep neuron `neuron_id` over `alphas` and decode the future flow at each value.

    Args:
        model: SAE-Drive model.
        z: [1, M] reference sparse code.
        neuron_id: which neuron to perturb.
        alphas: [K] additive scales.

    Returns:
        flows: [K, ΔT, 2, H, W] predicted future flows.
    """
    flows = []
    for a in alphas:
        z_perturbed = z.clone()
        z_perturbed[0, neuron_id] = z_perturbed[0, neuron_id] + a
        flow = model.flow_head(z_perturbed)
        flows.append(flow.squeeze(0))
    return torch.stack(flows, dim=0)


# ---------------------------------------------------------------------------
# Spatiotemporal Grad-CAM
# ---------------------------------------------------------------------------
def grad_cam_3d(model: SAEDrive, clip: torch.Tensor, neuron_id: int) -> torch.Tensor:
    """Compute a 3D Grad-CAM activation map for `neuron_id`.

    Args:
        clip: [1, T, 3, H, W] preprocessed clip.
        neuron_id: which sparse neuron to analyze.

    Returns:
        cam: [T', H', W'] heatmap (raw feature-grid resolution).
    """
    model.eval()
    clip = clip.requires_grad_(True)

    # Hook the stem feature map for gradients
    activations: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []

    def fwd_hook(_module, _inp, out):
        activations.append(out)
        out.register_hook(lambda g: grads.append(g))

    handle = model.encoder.stem.register_forward_hook(fwd_hook)
    try:
        out = model(clip)
        score = out["z"][0, neuron_id]
        model.zero_grad()
        score.backward(retain_graph=False)
    finally:
        handle.remove()

    feat = activations[0]                  # [1, C, T', H', W']
    grad = grads[0]                        # [1, C, T', H', W']
    alpha = grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = torch.relu((alpha * feat).sum(dim=1)).squeeze(0)
    cam = cam / (cam.max().clamp(min=1e-8))
    return cam.detach()
