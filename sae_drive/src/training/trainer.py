"""End-to-end trainer for SAE-Drive."""
from __future__ import annotations

import time
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import DrivingClipDataset, collate_clips
from ..losses import SparseDriveLoss
from ..models import SAEDrive
from ..utils import (
    ExperimentLogger,
    load_checkpoint,
    save_checkpoint,
    save_config,
    set_seed,
    worker_init_fn,
)
from .scheduler import WarmupCosineLR


class SAEDriveTrainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        set_seed(int(cfg.experiment.seed))
        self.device = torch.device(cfg.experiment.device)

        self.output_dir = Path(cfg.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.output_dir / "config.yaml")

        # Data
        train_set = DrivingClipDataset(cfg, split=cfg.data.get("split_train", "train"))
        val_set = DrivingClipDataset(cfg, split=cfg.data.get("split_val", "val"))
        self.train_loader = DataLoader(
            train_set,
            batch_size=int(cfg.training.batch_size),
            shuffle=True,
            num_workers=int(cfg.training.num_workers),
            pin_memory=bool(cfg.training.pin_memory),
            prefetch_factor=int(cfg.training.prefetch_factor),
            collate_fn=collate_clips,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=int(cfg.training.num_workers) > 0,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=int(cfg.training.batch_size),
            shuffle=False,
            num_workers=max(1, int(cfg.training.num_workers) // 2),
            pin_memory=bool(cfg.training.pin_memory),
            collate_fn=collate_clips,
            worker_init_fn=worker_init_fn,
            persistent_workers=False,
        )

        # Model
        self.model = SAEDrive(cfg).to(self.device)
        self.loss_fn = SparseDriveLoss(cfg).to(self.device)

        # Optimizer
        self.optimizer = self._build_optimizer()
        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = steps_per_epoch * int(cfg.training.epochs)
        self.scheduler = WarmupCosineLR(
            self.optimizer,
            warmup_steps=int(cfg.training.lr_warmup_steps),
            total_steps=total_steps,
        )

        self.use_amp = bool(cfg.experiment.mixed_precision) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Logging
        self.logger = ExperimentLogger(cfg, self.output_dir)
        self.logger.watch_model(self.model)

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")

        if cfg.training.get("resume_from"):
            state = load_checkpoint(
                cfg.training.resume_from,
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                map_location=self.device,
            )
            self.start_epoch = int(state.get("epoch", 0))
            self.global_step = int(state.get("global_step", 0))

    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or name.endswith(".bias") or "pos_emb" in name or "clip_query" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "weight_decay": float(self.cfg.training.weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        opt = self.cfg.training.optimizer.lower()
        if opt == "adamw":
            return torch.optim.AdamW(
                groups,
                lr=float(self.cfg.training.learning_rate),
                betas=tuple(self.cfg.training.betas),
            )
        if opt == "adam":
            return torch.optim.Adam(groups, lr=float(self.cfg.training.learning_rate))
        raise ValueError(f"Unknown optimizer: {opt}")

    # ------------------------------------------------------------------
    def _forward_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        clip = batch["clip"].to(self.device, non_blocking=True)
        outputs = self.model(clip)
        targets: dict[str, torch.Tensor | None] = {}
        if "future_flow" in batch:
            ff = batch["future_flow"].to(self.device, non_blocking=True)
            targets["future_flow"] = ff
        if "future_ego" in batch:
            targets["future_ego"] = batch["future_ego"].to(self.device, non_blocking=True)
        return {"outputs": outputs, "targets": targets}

    # ------------------------------------------------------------------
    def train(self) -> None:
        cfg_train = self.cfg.training
        log_every = int(cfg_train.log_every_n_steps)
        save_every = int(cfg_train.save_every_n_epochs)
        val_every = int(cfg_train.val_every_n_epochs)
        for epoch in range(self.start_epoch, int(cfg_train.epochs)):
            self._train_one_epoch(epoch, log_every)
            if (epoch + 1) % val_every == 0:
                val_loss = self.validate(epoch)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    ckpt_path = self.output_dir / "best.pt"
                    save_checkpoint(
                        ckpt_path,
                        self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch + 1,
                        global_step=self.global_step,
                        cfg=self.cfg,
                        extra={"val_loss": val_loss},
                    )
                    self.logger.log_artifact("best", ckpt_path)
            if (epoch + 1) % save_every == 0:
                save_checkpoint(
                    self.output_dir / f"epoch_{epoch + 1:04d}.pt",
                    self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch + 1,
                    global_step=self.global_step,
                    cfg=self.cfg,
                )

        save_checkpoint(
            self.output_dir / "last.pt",
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=int(cfg_train.epochs),
            global_step=self.global_step,
            cfg=self.cfg,
        )
        self.logger.close()

    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int, log_every: int) -> None:
        self.model.train()
        grad_clip = float(self.cfg.training.grad_clip_norm)
        dead_threshold = float(self.cfg.model.bottleneck.dead_neuron_threshold)

        pbar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
        running: dict[str, float] = {}
        running_n = 0
        t0 = time.time()
        for batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                fwd = self._forward_batch(batch)
                loss_out = self.loss_fn(
                    fwd["outputs"],
                    fwd["targets"],
                    step=self.global_step,
                    activation_ema=self.model.bottleneck.activation_ema,
                    dead_threshold=dead_threshold,
                )
                total = loss_out["total"]

            self.scaler.scale(total).backward()
            if grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.model.maintain_constraints()

            # Periodic dead-neuron resampling
            if self.global_step > 0 and self.global_step % int(self.cfg.model.bottleneck.dead_neuron_window) == 0:
                reset = self.model.bottleneck.reset_dead_neurons()
                if reset:
                    self.logger.log_scalars({"sae/reset_neurons": float(reset)}, self.global_step)

            # Accumulate logging
            for k, v in loss_out["report"].items():
                if isinstance(v, torch.Tensor):
                    running[k] = running.get(k, 0.0) + float(v.item())
            running_n += 1

            if self.global_step % log_every == 0:
                with torch.no_grad():
                    z = fwd["outputs"]["z"]
                    l0 = (z > 0).float().sum(dim=-1).mean().item()
                    util = self.model.bottleneck.neuron_utilization()
                    dead_frac = float((util < dead_threshold).float().mean().item())
                payload = {k: v / max(1, running_n) for k, v in running.items()}
                payload["sae/l0"] = l0
                payload["sae/dead_fraction"] = dead_frac
                payload["lr"] = self.scheduler.current_lr
                self.logger.log_scalars(payload, self.global_step, prefix="train")
                running, running_n = {}, 0
                pbar.set_postfix(loss=f"{payload.get('loss/total', 0.0):.4f}", l0=f"{l0:.1f}")

            self.global_step += 1

        # Per-epoch histograms
        with torch.no_grad():
            self.logger.log_histogram("sae/activation_ema", self.model.bottleneck.activation_ema, self.global_step)
        self.logger.log_scalars({"epoch_time_sec": time.time() - t0}, self.global_step, prefix="train")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        dead_threshold = float(self.cfg.model.bottleneck.dead_neuron_threshold)
        agg: dict[str, float] = {}
        n = 0
        for batch in tqdm(self.val_loader, desc=f"val {epoch}", leave=False):
            fwd = self._forward_batch(batch)
            loss_out = self.loss_fn(
                fwd["outputs"],
                fwd["targets"],
                step=self.global_step,
                activation_ema=self.model.bottleneck.activation_ema,
                dead_threshold=dead_threshold,
            )
            for k, v in loss_out["report"].items():
                if isinstance(v, torch.Tensor):
                    agg[k] = agg.get(k, 0.0) + float(v.item())
            n += 1
        n = max(1, n)
        means = {f"val/{k}": v / n for k, v in agg.items()}
        means["val/epoch"] = float(epoch)
        self.logger.log_scalars(means, self.global_step)
        return means.get("val/loss/total", float("inf"))
