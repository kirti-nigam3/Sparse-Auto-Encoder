"""Neuron interpretability + latent space analysis.

Pipeline:
  1. Load checkpoint
  2. Collect sparse codes across validation set
  3. Compute top-K activating clips per neuron
  4. Run UMAP + HDBSCAN
  5. Save artifacts (activations.npz, neurons.json, umap.png)

Usage:
    python -m scripts.analyze_neurons \
        --config configs/default.yaml \
        --ckpt runs/exp/best.pt \
        --output-dir runs/exp/analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.analysis import (
    build_eval_loader,
    cluster_metrics,
    collect_activations,
    gather_neuron_clip_metadata,
    run_hdbscan,
    run_umap,
    save_activations,
    save_neuron_index,
    save_scatter,
    topk_clips_per_neuron,
)
from src.models import SAEDrive
from src.utils import load_checkpoint, load_config, merge_overrides, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SAE-Drive neurons")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_overrides(cfg, args.overrides)
    set_seed(int(cfg.experiment.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.experiment.device)
    model = SAEDrive(cfg).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.eval()

    # 1. Collect activations
    loader = build_eval_loader(cfg)
    max_clips = args.max_clips if args.max_clips is not None else int(cfg.evaluation.num_clips_for_umap)
    print(f"[1/4] Collecting activations for up to {max_clips} clips ...")
    activations = collect_activations(model, loader, device=device, max_clips=max_clips)
    save_activations(activations, out_dir / "activations.npz")

    z = activations["z"].astype(np.float32)
    print(f"      z matrix: {z.shape}, mean L0 = {(z > 0).sum(axis=1).mean():.2f}")

    # 2. Top-K activating clips per neuron
    print("[2/4] Computing top-K activating clips per neuron ...")
    k = int(cfg.evaluation.topk_clips_per_neuron)
    top_idx = topk_clips_per_neuron(z, k=k)

    # Resolve clip metadata using the manifest
    with open(cfg.data.manifest, "r") as f:
        manifest = json.load(f)
    sequences = manifest[cfg.data.get("split_val", "val")]
    neuron_meta = gather_neuron_clip_metadata(activations, top_idx, sequences)

    # 3. UMAP + HDBSCAN
    print("[3/4] Running UMAP ...")
    umap_emb = run_umap(
        z,
        n_neighbors=int(cfg.evaluation.umap_n_neighbors),
        min_dist=float(cfg.evaluation.umap_min_dist),
        metric=str(cfg.evaluation.umap_metric),
    )
    np.save(out_dir / "umap_embedding.npy", umap_emb)
    save_scatter(umap_emb, None, out_dir / "umap_unlabeled.png", title="UMAP of sparse codes")

    print("      Running HDBSCAN ...")
    labels, probs = run_hdbscan(
        umap_emb,
        min_cluster_size=int(cfg.evaluation.hdbscan_min_cluster_size),
        min_samples=int(cfg.evaluation.hdbscan_min_samples),
    )
    np.save(out_dir / "hdbscan_labels.npy", labels)
    np.save(out_dir / "hdbscan_probs.npy", probs)
    save_scatter(umap_emb, labels, out_dir / "umap_hdbscan.png", title="UMAP + HDBSCAN")

    metrics = cluster_metrics(z, labels)
    metrics["num_clusters"] = int(len(set(labels)) - (1 if -1 in labels else 0))
    metrics["noise_fraction"] = float((labels == -1).mean())
    print(f"      Cluster metrics: {metrics}")

    # 4. Save neuron index + summary
    print("[4/4] Saving neuron index ...")
    save_neuron_index(
        out_dir / "neurons.json",
        neuron_meta,
        extras={
            "metrics": metrics,
            "config_ckpt": args.ckpt,
            "z_shape": list(z.shape),
            "topk": k,
        },
    )

    print(f"Artifacts written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
