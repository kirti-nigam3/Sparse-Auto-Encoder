"""Standalone validation pipeline (re-runs the validation loop from a checkpoint).

Usage:
    python -m scripts.evaluate --config configs/default.yaml --ckpt runs/exp/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.losses import SparseDriveLoss
from src.models import SAEDrive
from src.training import SAEDriveTrainer
from src.utils import load_checkpoint, load_config, merge_overrides, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAE-Drive")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_overrides(cfg, args.overrides)
    set_seed(int(cfg.experiment.seed))

    # Reuse trainer's data loaders + loss but bypass training
    trainer = SAEDriveTrainer(cfg)
    load_checkpoint(args.ckpt, trainer.model, map_location=trainer.device)
    val_loss = trainer.validate(epoch=-1)
    print(f"Validation total loss: {val_loss:.6f}")

    # Persist per-key val summary alongside checkpoint
    out_path = Path(args.ckpt).with_suffix(".eval.txt")
    out_path.write_text(f"val_loss_total={val_loss:.6f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
