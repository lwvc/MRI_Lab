"""Train the 3D brain-age model; resume safely from last.pt."""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from checkpoint import CheckpointManager
from config import CFG
from dataset import BrainAgeDataset, make_loader
from model import BrainAgeResNet


class InverseTimeDecay:
    """Keras-style inverse-time learning-rate decay per optimizer update."""

    def __init__(self, optimizer, decay: float):
        self.optimizer = optimizer
        self.decay = decay
        self.initial_lr = optimizer.param_groups[0]["lr"]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        self.optimizer.param_groups[0]["lr"] = (
            self.initial_lr / (1.0 + self.decay * self.step_count)
        )

    def state_dict(self):
        return {"decay": self.decay, "initial_lr": self.initial_lr,
                "step_count": self.step_count}

    def load_state_dict(self, state):
        if state["decay"] != self.decay or state["initial_lr"] != self.initial_lr:
            raise ValueError("Learning-rate schedule differs from checkpoint")
        self.step_count = state["step_count"]
        self.optimizer.param_groups[0]["lr"] = (
            self.initial_lr / (1.0 + self.decay * self.step_count)
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_penalty(model: BrainAgeResNet, coefficient: float):
    """Match Keras kernel_regularizer on conv1/conv2, fc1 and output only."""
    return coefficient * sum(weight.square().sum()
                             for weight in model.regularized_weights())


def run_epoch(model, loader, device, l2_coefficient, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    absolute_error, count, l2_sum = 0.0, 0, 0.0

    with torch.enable_grad() if training else torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            age = batch["age"].to(device, non_blocking=True).view(-1, 1)
            sex = batch["sex"].to(device, non_blocking=True).view(-1, 1)
            scanner = batch["scanner"].to(device, non_blocking=True).view(-1, 1)

            prediction = model(image, sex, scanner)
            mae = F.l1_loss(prediction, age)
            l2 = l2_penalty(model, l2_coefficient)
            loss = mae + l2
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered")

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

            n = age.size(0)
            absolute_error += F.l1_loss(prediction.detach(), age, reduction="sum").item()
            l2_sum += l2.detach().item() * n
            count += n

    if count == 0:
        raise RuntimeError("Empty dataloader")
    mae_value = absolute_error / count
    l2_value = l2_sum / count
    return mae_value, l2_value, mae_value + l2_value


def repair_history(path: Path, completed_epoch: int):
    """Discard log rows written beyond last.pt after an interrupted save."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = [row for row in reader if int(row["epoch"]) <= completed_epoch]
    if len(rows) != completed_epoch or [int(row["epoch"]) for row in rows] != list(range(1, completed_epoch + 1)):
        raise ValueError("Training log does not agree with completed checkpoint epochs")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_history(path: Path, record: dict, resume: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("epoch", "train_mae", "train_l2", "train_total_loss",
               "val_mae", "val_l2", "val_total_loss", "lr", "best_epoch")
    with path.open("a" if resume else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if not resume:
            writer.writeheader()
        writer.writerow(record)


def main():
    parser = argparse.ArgumentParser(description="Train 3D brain-age ResNet")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=CFG.data_root)
    parser.add_argument("--registration", choices=("affine", "rigid", "nonlinear"),
                        default=CFG.registration)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = replace(CFG, train_csv=args.train_csv.resolve(),
                  val_csv=args.val_csv.resolve(), output_dir=args.output_dir.resolve(),
                  data_root=args.data_root, registration=args.registration)
    checkpoint_path = cfg.output_dir / "last.pt"
    if args.resume and not checkpoint_path.is_file():
        raise FileNotFoundError(f"No checkpoint to resume: {checkpoint_path}")
    if not args.resume and checkpoint_path.exists():
        raise FileExistsError("last.pt exists; use --resume or a different output directory")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = BrainAgeDataset(cfg.train_csv, cfg, split="train")
    val_data = BrainAgeDataset(cfg.val_csv, cfg, split="val")
    if set(train_data.full_df.Subject_ID) & set(val_data.full_df.Subject_ID):
        raise ValueError("Train and validation contain overlapping Subject_ID")
    val_loader, _ = make_loader(val_data, cfg)

    model = BrainAgeResNet(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                 betas=cfg.adam_betas, eps=cfg.adam_eps, weight_decay=0)
    scheduler = InverseTimeDecay(optimizer, cfg.lr_decay)
    manager = CheckpointManager(cfg.output_dir, cfg)
    start_epoch = manager.resume(model, optimizer, scheduler) if args.resume else 1

    history_path = cfg.output_dir / "training_log.csv"
    if args.resume and not history_path.exists():
        raise FileNotFoundError("training_log.csv missing; inspect checkpoint before resuming")
    if args.resume:
        repair_history(history_path, start_epoch - 1)
    if not args.resume and history_path.exists():
        raise FileExistsError("training_log.csv exists; use a fresh output directory")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if manager.should_stop():
        print("Checkpoint already meets EarlyStopping criterion; no more epochs")
        return
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_data.set_epoch(epoch - 1)
        train_loader, train_sampler = make_loader(train_data, cfg)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)

        lr_used = optimizer.param_groups[0]["lr"]
        train_mae, train_l2, train_total = run_epoch(
            model, train_loader, device, cfg.l2_coefficient, optimizer, scheduler)
        val_mae, val_l2, val_total = run_epoch(
            model, val_loader, device, cfg.l2_coefficient)

        improved = manager.save_epoch(epoch, model, optimizer, scheduler, val_mae)
        write_history(history_path, {
            "epoch": epoch, "train_mae": train_mae, "train_l2": train_l2,
            "train_total_loss": train_total, "val_mae": val_mae,
            "val_l2": val_l2, "val_total_loss": val_total,
            "lr": lr_used, "best_epoch": manager.best_epoch,
        }, resume=args.resume or epoch != 1)
        print(f"Epoch {epoch:04d} | train MAE {train_mae:.4f} | "
              f"val MAE {val_mae:.4f} | best {manager.best_epoch} "
              f"{'*' if improved else ''}", flush=True)
        if manager.should_stop():
            print(f"Early stopping at epoch {epoch}")
            break


if __name__ == "__main__":
    main()
