"""Environment + dependency sanity check.

Run with:
    python scripts/check_env.py

Verifies:
  * Python version and platform
  * Core deps importable
  * CUDA / GPU visibility
  * RAFT weights downloadable
  * Project package importable end-to-end
  * Dummy forward pass through SAEDrive
"""
from __future__ import annotations

import importlib
import platform
import sys
import traceback
from typing import Callable

REQUIRED = [
    ("torch", "2.0.0"),
    ("torchvision", "0.15.0"),
    ("numpy", "1.20.0"),
    ("einops", "0.6.0"),
    ("av", None),                 # PyAV
    ("cv2", None),                # opencv-python
    ("PIL", None),                # Pillow
    ("yaml", None),
    ("omegaconf", "2.3.0"),
    ("tqdm", None),
    ("tensorboard", None),
    ("wandb", None),
    ("umap", None),               # umap-learn
    ("hdbscan", None),
    ("sklearn", None),
    ("matplotlib", None),
    ("h5py", None),
    ("scipy", None),
]


def _check(label: str, fn: Callable[[], None]) -> bool:
    sys.stdout.write(f"  [....] {label}")
    sys.stdout.flush()
    try:
        fn()
        sys.stdout.write(f"\r  [ OK ] {label}\n")
        return True
    except Exception as e:                   # noqa: BLE001
        sys.stdout.write(f"\r  [FAIL] {label}: {e}\n")
        return False


def check_python() -> bool:
    print("[1] Python runtime")
    ok = True
    ok &= _check(
        f"python >= 3.8 (got {platform.python_version()})",
        lambda: (_ for _ in ()).throw(RuntimeError("python too old"))
        if sys.version_info < (3, 8)
        else None,
    )
    return ok


def check_imports() -> bool:
    print("[2] Required packages")
    ok = True
    for name, min_version in REQUIRED:
        def _imp(n=name, mv=min_version):
            mod = importlib.import_module(n)
            ver = getattr(mod, "__version__", "?")
            if mv and ver != "?":
                # naive comparison
                cur = tuple(int(x) for x in ver.split(".")[:3] if x.isdigit())
                req = tuple(int(x) for x in mv.split(".")[:3] if x.isdigit())
                if cur < req:
                    raise RuntimeError(f"version {ver} < required {mv}")
            return ver
        ok &= _check(f"import {name}", _imp)
    return ok


def check_cuda() -> bool:
    print("[3] CUDA / GPU")
    ok = True

    def _torch_cuda():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() == False")
        n = torch.cuda.device_count()
        for i in range(n):
            print(
                f"        GPU {i}: {torch.cuda.get_device_name(i)} "
                f"(cap {torch.cuda.get_device_capability(i)}, "
                f"mem {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)"
            )

    ok &= _check("torch.cuda.is_available()", _torch_cuda)
    return ok


def check_raft() -> bool:
    print("[4] RAFT weights")

    def _raft():
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        _ = Raft_Large_Weights.C_T_SKHT_V2          # downloads on first access
        m = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2)
        del m

    return _check("download torchvision RAFT-Large weights", _raft)


def check_project() -> bool:
    print("[5] Project import + dummy forward pass")
    ok = True

    def _import_pkg():
        sys.path.insert(0, ".")
        import src                                  # noqa: F401
        from src.models import SAEDrive             # noqa: F401
        from src.losses import SparseDriveLoss      # noqa: F401
        from src.data import DrivingClipDataset     # noqa: F401
        from src.training import SAEDriveTrainer    # noqa: F401
        from src.analysis import run_umap           # noqa: F401

    ok &= _check("import src.*", _import_pkg)

    def _dummy_forward():
        import torch
        from omegaconf import OmegaConf

        from src.losses import SparseDriveLoss
        from src.models import SAEDrive

        cfg = OmegaConf.load("configs/default.yaml")
        # Shrink for CPU smoke test
        cfg.data.clip_length = 4
        cfg.data.future_horizon = 2
        cfg.data.image_height = 96
        cfg.data.image_width = 160
        cfg.model.bottleneck.latent_dim = 256
        cfg.model.bottleneck.topk = 8
        cfg.model.decoder.flow_horizon = 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SAEDrive(cfg).to(device)
        loss_fn = SparseDriveLoss(cfg).to(device)

        clip = torch.randn(2, cfg.data.clip_length, 3, cfg.data.image_height, cfg.data.image_width, device=device)
        flow_h = cfg.data.image_height // cfg.data.flow_downscale
        flow_w = cfg.data.image_width // cfg.data.flow_downscale
        targets = {
            "future_flow": torch.randn(2, cfg.data.future_horizon, 2, flow_h, flow_w, device=device),
            "future_ego": torch.randn(2, cfg.data.future_horizon, 3, device=device),
        }
        out = model(clip)
        loss = loss_fn(out, targets, step=0, activation_ema=model.bottleneck.activation_ema)
        loss["total"].backward()
        print(f"        forward OK on {device.type}, total loss = {float(loss['total']):.4f}")

    ok &= _check("dummy forward+backward", _dummy_forward)
    return ok


def main() -> int:
    print(f"Platform : {platform.platform()}")
    print(f"Python   : {sys.executable}")
    print(f"Version  : {platform.python_version()}")
    print()

    results = [
        check_python(),
        check_imports(),
        check_cuda(),
        check_raft(),
        check_project(),
    ]
    print()
    if all(results):
        print("All checks passed.")
        return 0
    print("Some checks failed — see lines marked [FAIL] above.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
