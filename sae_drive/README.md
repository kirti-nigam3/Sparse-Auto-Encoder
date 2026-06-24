# SAE-Drive: Sparse Action Encoder for Driving Video Representation Learning

PyTorch implementation of a sparse autoencoder for discovering interpretable action-centric latent representations from driving videos without action label supervision.

See [`../SAE_Drive_Research_Plan.md`](../SAE_Drive_Research_Plan.md) for the complete research design.

## Installation

```bash
conda create -n sae_drive python=3.10 -y
conda activate sae_drive
pip install -r requirements.txt
```

## Project Structure

```
sae_drive/
├── configs/            # YAML experiment configurations
├── src/
│   ├── data/           # Dataset loader, transforms, preprocessing
│   ├── models/         # Encoder, SAE bottleneck, decoders
│   ├── losses/         # Multi-objective loss
│   ├── training/       # Trainer + LR scheduling
│   ├── analysis/       # UMAP, top-K retrieval, neuron analysis
│   └── utils/          # Config loading, logging helpers
└── scripts/
    ├── precompute_flow.py   # RAFT flow precomputation
    ├── train.py              # End-to-end training
    ├── evaluate.py           # Validation pipeline
    └── analyze_neurons.py    # Neuron interpretability
```

## Quick Start

### 1. Preprocess data (compute RAFT flow once)

```bash
python -m scripts.precompute_flow \
  --video-root /path/to/nuscenes \
  --output-dir ./data/flow \
  --config configs/default.yaml
```

### 2. Train

```bash
python -m scripts.train --config configs/default.yaml
```

### 3. Evaluate

```bash
python -m scripts.evaluate --config configs/default.yaml --ckpt runs/exp1/best.pt
```

### 4. Neuron interpretability analysis

```bash
python -m scripts.analyze_neurons \
  --config configs/default.yaml \
  --ckpt runs/exp1/best.pt \
  --output-dir runs/exp1/analysis
```

## Reproducibility

- All seeds are fixed via `experiment.seed` in YAML config.
- Configuration is logged to TensorBoard and Weights & Biases.
- Model checkpoints store config + git SHA + RNG state.

## Citation

If you build on this work, please cite the accompanying research plan.
