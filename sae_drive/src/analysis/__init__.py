"""Analysis utilities."""
from .activations import build_eval_loader, collect_activations, load_activations, save_activations
from .neuron_analysis import (
    action_purity,
    action_separation,
    grad_cam_3d,
    latent_traversal,
    neuron_active_clips,
    neuron_selectivity_score,
    topk_clips_per_neuron,
)
from .topk_retrieval import gather_neuron_clip_metadata, save_neuron_index
from .umap_vis import cluster_metrics, run_hdbscan, run_tsne, run_umap, save_scatter

__all__ = [
    "build_eval_loader",
    "collect_activations",
    "load_activations",
    "save_activations",
    "topk_clips_per_neuron",
    "neuron_active_clips",
    "neuron_selectivity_score",
    "action_separation",
    "action_purity",
    "latent_traversal",
    "grad_cam_3d",
    "gather_neuron_clip_metadata",
    "save_neuron_index",
    "run_umap",
    "run_tsne",
    "run_hdbscan",
    "cluster_metrics",
    "save_scatter",
]
