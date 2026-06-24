"""Train SAE-Drive end-to-end.

Usage:
    python -m scripts.train --config configs/default.yaml [overrides ...]

Examples:
    python -m scripts.train --config configs/default.yaml training.batch_size=4
"""
from __future__ import annotations

import argparse
import sys

from src.training import SAEDriveTrainer
from src.utils import load_config, merge_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAE-Drive")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="OmegaConf dotlist overrides")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_overrides(cfg, args.overrides)
    trainer = SAEDriveTrainer(cfg)
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
