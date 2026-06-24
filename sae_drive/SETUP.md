# SAE-Drive — Setup & Experiment Walkthrough

This guide walks you from a clean machine to a fully trained SAE-Drive run with
neuron interpretability artifacts.

> Verified on this workstation: Ubuntu 20.04, Python 3.8.10, NVIDIA RTX 3090
> (driver 470, CUDA 11.8 wheels), 24 GB VRAM.

---

## 0. System prerequisites (already satisfied here)

| Component | Detected on this machine | Notes |
|---|---|---|
| OS | Linux Ubuntu 20.04 (kernel 5.15) | OK |
| Python | 3.8.10 (`/usr/bin/python3`) | 3.8 supported; 3.10+ recommended for new projects |
| `venv` module | present | `python3 -m venv` works |
| GPU | NVIDIA RTX 3090, 24 GB, driver 470.239 | Uses CUDA 11.8 wheels (compatible with driver 470) |
| `ffmpeg` | 7.0.2 static, `/home/kig7kor/bin/ffmpeg` | Used implicitly by PyAV/torchvision |
| `pip` | upgraded to 25.x in the venv | Done by setup commands below |
| `conda` | not installed | Not required; venv is sufficient |

If you move this project to another machine, install missing pieces with:

```bash
sudo apt update
sudo apt install -y python3-venv ffmpeg
# Optional but recommended (newer Python):
# sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.10 python3.10-venv
```

---

## 1. Create the virtual environment

```bash
cd ~/Desktop/Bosch/Sparse\ Encoder/sae_drive
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

---

## 2. Install dependencies (verified working set)

PyTorch 2.2.2 + CUDA 11.8 is the latest combo compatible with NVIDIA driver
470. Newer PyTorch versions (≥ 2.4) require driver ≥ 525.

```bash
# A. PyTorch + torchvision (CUDA 11.8 wheels)
pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.2.2 torchvision==0.17.2

# B. Rest of the runtime stack
pip install \
  "einops>=0.7.0" "av>=10.0.0" "opencv-python>=4.8.0" \
  "pyyaml>=6.0" "omegaconf>=2.3.0" "tqdm>=4.65.0" \
  "tensorboard>=2.14.0" "wandb>=0.16.0" \
  "h5py>=3.9.0" "scipy>=1.10.0" \
  "matplotlib>=3.7.0" "seaborn>=0.13.0" "scikit-learn>=1.3.0"

# C. Interpretability libs
pip install "umap-learn>=0.5.4" "hdbscan>=0.8.33"
```

> The `requirements.txt` in this repo lists looser lower-bounds. The exact
> versions above are what was verified end-to-end on this box.

---

## 3. Verify the environment

```bash
source .venv/bin/activate
python scripts/check_env.py
```

Expected output (last block):

```
[3] CUDA / GPU
  [ OK ] torch.cuda.is_available()
        GPU 0: NVIDIA GeForce RTX 3090 (cap (8, 6), mem 25.4 GB)
[4] RAFT weights
  [ OK ] download torchvision RAFT-Large weights
[5] Project import + dummy forward pass
  [ OK ] import src.*
  [ OK ] dummy forward+backward
All checks passed.
```

This actually downloads RAFT weights and runs a model forward/backward on GPU.

---

## 4. Smoke-test the full training + analysis pipeline (5 minutes, no real data)

```bash
# Generate a small synthetic dataset
python scripts/make_dummy_data.py --root data_demo --frames-per-seq 120 --num-train 3 --num-val 2

# Train 2 epochs end-to-end (downscaled config)
PYTHONPATH=. python -u scripts/train.py --config configs/smoke.yaml

# Validate from the best checkpoint
PYTHONPATH=. python -u scripts/evaluate.py --config configs/smoke.yaml --ckpt runs/smoke/best.pt

# Analysis: collect activations → top-K → UMAP → HDBSCAN
PYTHONPATH=. python -u scripts/analyze_neurons.py \
    --config configs/smoke.yaml \
    --ckpt runs/smoke/best.pt \
    --output-dir runs/smoke/analysis
```

After this you should see:

```
runs/smoke/
├── best.pt
├── epoch_0001.pt
├── epoch_0002.pt
├── last.pt
├── config.yaml
├── tb/                       # tensorboard event files
└── analysis/
    ├── activations.npz
    ├── neurons.json
    ├── umap_embedding.npy
    ├── umap_unlabeled.png
    ├── umap_hdbscan.png
    ├── hdbscan_labels.npy
    └── hdbscan_probs.npy
```

> If you see `noise_fraction: 1.0` in the analysis output, that is expected
> for the toy 5-sequence dummy data — HDBSCAN needs hundreds of clips to form
> clusters.

---

## 5. Real experiment on driving video

### 5.1 Prepare data

Layout:

```
/path/to/driving_dataset/
├── scene_001/000000.jpg, 000001.jpg, ...
├── scene_002/000000.jpg, 000001.jpg, ...
├── ego/scene_001.npy        # [N, 3] (dx, dy, dtheta), per-source-frame
├── ego/scene_002.npy
├── flow/                    # (populated by precompute_flow.py)
└── manifest.json            # built by build_manifest()
```

Build the manifest in Python:

```python
from src.data.preprocessing import build_manifest
build_manifest(
    video_root="/path/to/driving_dataset",
    flow_root="/path/to/driving_dataset/flow",
    ego_root="/path/to/driving_dataset/ego",
    train_ids=[...],
    val_ids=[...],
    output_path="/path/to/driving_dataset/manifest.json",
    source_fps=30,
)
```

### 5.2 Configure

Copy and edit `configs/default.yaml`, pointing the `data.*` paths at your dataset.

Critical knobs for a 24 GB GPU:

| Knob | Recommended start |
|---|---|
| `data.clip_length` | 32 (3.2 s @ 10 Hz) |
| `data.future_horizon` | 8 |
| `training.batch_size` | 8 (lower to 4 if OOM) |
| `model.bottleneck.latent_dim` | 4096 |
| `model.bottleneck.topk` | 32 |

### 5.3 Precompute optical flow (one-time, GPU)

```bash
PYTHONPATH=. python -u scripts/precompute_flow.py \
    --config configs/default.yaml --split train --device cuda
PYTHONPATH=. python -u scripts/precompute_flow.py \
    --config configs/default.yaml --split val   --device cuda
```

This writes `data.flow_root/<seq_id>.npy` (float16) for every sequence.

### 5.4 Train

```bash
# Login once if you want W&B logging:
wandb login

PYTHONPATH=. python -u scripts/train.py --config configs/default.yaml
```

Override anything from the CLI without editing YAML:

```bash
PYTHONPATH=. python -u scripts/train.py --config configs/default.yaml \
    training.batch_size=4 model.bottleneck.topk=16 experiment.log_to_wandb=false
```

### 5.5 Monitor

```bash
tensorboard --logdir runs/sae_drive_default/tb --port 6006
```

W&B run will appear under your configured project automatically (set
`experiment.log_to_wandb=false` to disable).

### 5.6 Evaluate

```bash
PYTHONPATH=. python -u scripts/evaluate.py \
    --config configs/default.yaml --ckpt runs/sae_drive_default/best.pt
```

### 5.7 Neuron interpretability + UMAP

```bash
PYTHONPATH=. python -u scripts/analyze_neurons.py \
    --config configs/default.yaml \
    --ckpt runs/sae_drive_default/best.pt \
    --output-dir runs/sae_drive_default/analysis
```

Produced artifacts (see Section 9–11 of the research plan for interpretation):

- `activations.npz` — N × M sparse codes + clip metadata
- `neurons.json` — top-K activating clips per neuron (sequence + frame range)
- `umap_*.png` — UMAP scatter (unlabeled and HDBSCAN-colored)
- `hdbscan_labels.npy` — per-clip cluster id (−1 = noise)

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available() == False` | Wrong PyTorch wheel (CPU-only) | Reinstall with `--index-url https://download.pytorch.org/whl/cu118` |
| `CUDA driver version is insufficient` | Driver < 520 with PyTorch 2.4+ | Stay on the verified torch 2.2.2 + cu118 combo |
| `RuntimeError: CUDA out of memory` | Batch / clip too big | Lower `training.batch_size`, `data.clip_length`, or `model.bottleneck.latent_dim` |
| `ModuleNotFoundError: src` | Forgot `PYTHONPATH=.` | Use `PYTHONPATH=. python -u scripts/<x>.py ...` |
| Analysis shows `noise_fraction: 1.0` | Too few clips for HDBSCAN | Lower `evaluation.hdbscan_min_cluster_size` or use more validation data |
| RAFT download fails | Network / firewall | Pre-download `raft_large_C_T_SKHT_V2-ff5fadd5.pth` into `~/.cache/torch/hub/checkpoints/` |
| `wandb` interactive login | First-time W&B | Run `wandb login` and paste API key, OR set `experiment.log_to_wandb: false` |
