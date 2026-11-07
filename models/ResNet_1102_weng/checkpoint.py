"""Save best models and resume training safely after interruptions."""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from config import CFG, Config


def _config_text(cfg: Config) -> str:
    return json.dumps(asdict(cfg), sort_keys=True, default=str)


def _weights(model) -> dict:
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def _rng_state() -> dict:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (name, keys.tolist(), pos, has_gauss, cached),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32),
                         pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Resume requires the same number of CUDA devices")
        torch.cuda.set_rng_state_all(state["cuda"])


def _save(data: dict, path: Path) -> None:
    """Atomically replace a checkpoint to avoid a partially written .pt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        torch.save(data, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


class CheckpointManager:
    def __init__(self, output_dir: Path, cfg: Config = CFG):
        self.output_dir = Path(output_dir)
        self.cfg = cfg
        self.best_mae = float("inf")
        self.best_epoch = 0
        self.best_weights = None
        self.period_best_mae = float("inf")
        self.period_best_epoch = 0
        self.period_weights = None
        self.bad_epochs = 0

    def _period_path(self, epoch: int) -> Path:
        size = self.cfg.periodic_save
        first = ((epoch - 1) // size) * size + 1
        last = first + size - 1
        return self.output_dir / f"best_{first:03d}_{last:03d}.pt"

    def save_epoch(self, epoch: int, model, optimizer, scheduler,
                   val_mae: float) -> bool:
        """Call once after validation. Returns True when validation improved."""
        if not np.isfinite(val_mae) or epoch < 1:
            raise ValueError("Expected a finite validation MAE and epoch >= 1")

        period_start = ((epoch - 1) // self.cfg.periodic_save) * self.cfg.periodic_save + 1
        if epoch == period_start:
            self.period_best_mae = float("inf")
            self.period_best_epoch = 0
            self.period_weights = None

        improved = val_mae < self.best_mae - self.cfg.min_delta
        if improved:
            self.best_mae, self.best_epoch = float(val_mae), epoch
            self.best_weights = _weights(model)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if val_mae < self.period_best_mae:
            self.period_best_mae, self.period_best_epoch = float(val_mae), epoch
            self.period_weights = _weights(model)

        state = {
            "epoch": epoch,  # completed epoch; resume from epoch + 1
            "config": _config_text(self.cfg),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": _rng_state(),
            "best_mae": self.best_mae,
            "best_epoch": self.best_epoch,
            "best_weights": self.best_weights,
            "period_best_mae": self.period_best_mae,
            "period_best_epoch": self.period_best_epoch,
            "period_weights": self.period_weights,
            "bad_epochs": self.bad_epochs,
        }
        _save(state, self.output_dir / "last.pt")
        self._write_best_files()
        return improved

    def _write_best_files(self) -> None:
        _save({"epoch": self.best_epoch, "val_mae": self.best_mae,
               "model": self.best_weights}, self.output_dir / "best.pt")
        _save({"epoch": self.period_best_epoch, "val_mae": self.period_best_mae,
               "model": self.period_weights},
              self._period_path(self.period_best_epoch))

    def resume(self, model, optimizer, scheduler=None) -> int:
        """Restore last completed epoch and RNG. Returns next epoch (1-based)."""
        path = self.output_dir / "last.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["config"] != _config_text(self.cfg):
            raise ValueError("Current config differs from saved training config")
        if (scheduler is None) != (state["scheduler"] is None):
            raise ValueError("Scheduler availability differs from checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        self.best_mae, self.best_epoch = state["best_mae"], state["best_epoch"]
        self.best_weights = state["best_weights"]
        self.period_best_mae = state["period_best_mae"]
        self.period_best_epoch = state["period_best_epoch"]
        self.period_weights = state["period_weights"]
        self.bad_epochs = state["bad_epochs"]
        _restore_rng(state["rng"])
        self._write_best_files()  # Repair files if power failed during saving.
        return state["epoch"] + 1

    def should_stop(self) -> bool:
        return self.cfg.early_stopping and self.bad_epochs >= self.cfg.patience
