from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataloader import (AGE_SAMPLE_PLAN, BrainAgeDataset, SUPPORTED_REGISTRATIONS,
                        make_balanced_loader, read_subject_csv)
from model import BrainAgeResNet
from training_state import (FORMAT_VERSION, INPUT_ENCODING, SAMPLER_VERSION,
                            atomic_save, atomic_text, best_checkpoint,
                            capture_rng, file_hash, load_checkpoint, restore_rng, write_history)

def parse_args():
    parser = argparse.ArgumentParser(description="Train the five-split brain-age study.")
    for name in ("data-root", "train-csv", "val-csv", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--registration", choices=SUPPORTED_REGISTRATIONS, default="rigid")  # 改 default="affine"
    parser.add_argument("--metadata", choices=("none", "sex", "scanner", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4) # 改 default=1e-3
    parser.add_argument("--lr-step-size", type=int, default=100)
    parser.add_argument("--lr-gamma", type=float, default=0.7)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--periodic-save", type=int, default=100)
    parser.add_argument("--resume", type=Path, help="Resume last.pt; use a new output directory to branch from epoch_N.pt.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def select_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                        else "cpu" if requested == "auto" else requested)


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Match the defaults of the 20261005 script in a fresh PyTorch process.
    # Keep RNG seeds without imposing additional deterministic GPU algorithms.
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    absolute_error_sum = 0.0
    sample_count = 0
    lr_last = None
    with torch.enable_grad() if training else torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            age = batch["age"].view(-1, 1).to(device, non_blocking=True)
            sex = batch["sex"].view(-1, 1).to(device, non_blocking=True)
            scanner = batch["scanner"].view(-1, 1).to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(image, sex, scanner)
            loss = F.l1_loss(prediction, age)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; last completed epoch remains available")
            if training:
                loss.backward()
                lr_last = optimizer.param_groups[0]["lr"]
                optimizer.step()
            absolute_error_sum += F.l1_loss(prediction.detach(), age, reduction="sum").item()
            sample_count += len(age)
    if not sample_count:
        raise RuntimeError("Dataloader produced no samples")
    return absolute_error_sum / sample_count, sample_count, lr_last


def training_config(args):
    return {"execution_mode": "seed_only_v1", 
            "metadata": args.metadata,
            "registration": args.registration, 
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, 
                    "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "scheduler": "StepLR",

            "weight_decay": args.weight_decay, 
            "seed": args.seed,
            "num_workers": args.num_workers, 
            "periodic_save": args.periodic_save,
            "device_type": select_device(args.device).type,
            "train_sha256": file_hash(args.train_csv), 
            "val_sha256": file_hash(args.val_csv),
            "code_sha256": {name: file_hash(Path(__file__).with_name(name))
                            for name in ("model.py", "dataloader.py", "train.py", "training_state.py")}}

def validate_resume_config(checkpoint, config):
    if checkpoint.get("kind") != "resume":
        raise ValueError("Resume requires last.pt")

    if checkpoint["training_config"] != config:
        raise ValueError("Training configuration differs")

    scheduler_epoch = checkpoint["scheduler_state_dict"]["last_epoch"]

    if scheduler_epoch != checkpoint["epoch"]:
        raise ValueError("Scheduler epoch does not match checkpoint")


def write_batch_manifest(dataset, sampler, path):
    import pandas as pd
    records = [{"batch_id": number, "batch_position": position,
                "Subject_ID": str(dataset.df.iloc[index]["Subject_ID"]),
                "AGE": float(dataset.df.iloc[index]["AGE"])}
               for number, batch in enumerate(sampler) for position, index in enumerate(batch)]
    pd.DataFrame(records).to_csv(path, index=False)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size != 4 or args.num_workers < 0 or args.periodic_save < 1:
        raise ValueError("Require epochs>=1, batch_size=4, num_workers>=0, periodic_save>=1")
    if (not all(math.isfinite(v) for v in (args.learning_rate, args.lr_gamma, args.weight_decay))
            or args.learning_rate <= 0 or args.lr_step_size <= 0
            or not 0 < args.lr_gamma <= 1 or args.weight_decay < 0):
        raise ValueError("Invalid optimizer or scheduler settings")
    if not 0 <= args.seed < 2**32 - args.epochs * 1009 - len(AGE_SAMPLE_PLAN):
        raise ValueError("Seed/epoch combination exceeds the sampling RNG range")
    train_frame, val_frame = read_subject_csv(args.train_csv), read_subject_csv(args.val_csv)
    if len(train_frame) != 3543 or len(val_frame) != 464:
        raise ValueError("This study expects 3543 training and 464 validation subjects")
    if set(train_frame.Subject_ID) & set(val_frame.Subject_ID):
        raise ValueError("Training and validation subjects overlap")
    existing = args.output_dir.exists() and any(args.output_dir.iterdir())
    if existing and args.resume is None:
        raise FileExistsError("Output directory is not empty; choose a new directory or use --resume")
    if existing and args.resume.resolve() != (args.output_dir / "last.pt").resolve():
        raise ValueError("Resume in place only from last.pt; use an empty directory for a historical checkpoint")
    set_reproducibility(args.seed)
    device = select_device(args.device)
    config = training_config(args)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        validate_resume_config(checkpoint, config)
    start_epoch = checkpoint["epoch"] if checkpoint else 0
    if start_epoch > args.epochs:
        raise ValueError(f"Checkpoint already completed epoch {start_epoch}")

    train_dataset = BrainAgeDataset(args.train_csv, args.data_root, "train", args.registration, args.seed)
    val_dataset = BrainAgeDataset(args.val_csv, args.data_root, "val", args.registration, args.seed)
    val_loader, val_sampler = make_balanced_loader(val_dataset, args.batch_size, args.num_workers,
                                                  False, 0, device.type == "cuda", drop_last=False)
    model = BrainAgeResNet(metadata=args.metadata).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )
    history = []
    best_val_mae, best_epoch, best_weights = float("inf"), 0, None
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint["history"]
        best_val_mae, best_epoch = checkpoint["best_val_mae"], checkpoint["best_epoch"]
        best_weights = checkpoint["best_model_state_dict"]
        restore_rng(checkpoint["rng_state"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source, name in ((args.train_csv, "train_datalist.csv"), (args.val_csv, "val_datalist.csv")):
        shutil.copy2(source, args.output_dir / name)
    source_directory = args.output_dir / "code_snapshot"
    source_directory.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, source_directory / source.name)
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "numpy": str(np.__version__), "cuda": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version(), "platform": platform.platform(),
                   "cudnn_benchmark": torch.backends.cudnn.benchmark,
                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    atomic_text(json.dumps({"training_config": config, "input_encoding": INPUT_ENCODING,
                           "sampler_version": SAMPLER_VERSION, "requested_epochs": args.epochs,
                           "environment": environment}, indent=2), args.output_dir / "run_config.json")
    write_batch_manifest(val_dataset, val_sampler, args.output_dir / "val_batches.csv")
    write_history(history, args.output_dir / "training_log.csv")
    if checkpoint:
        atomic_save(best_checkpoint(checkpoint), args.output_dir / "best.pt")
        if start_epoch % args.periodic_save == 0:
            atomic_save(checkpoint, args.output_dir / f"epoch_{start_epoch}.pt")
        if not existing:
            atomic_save(checkpoint, args.output_dir / "last.pt")
    print(f"Device {device} | {args.registration} | metadata {args.metadata} | seed {args.seed} | GPU mode seed_only | resume epoch {start_epoch}", flush=True)
    for epoch_index in range(start_epoch, args.epochs):
        train_dataset.resample_for_epoch(epoch_index)
        train_loader, train_sampler = make_balanced_loader(train_dataset, args.batch_size, args.num_workers,
                                                          True, args.seed, device.type == "cuda", drop_last=False)
        train_sampler.set_epoch(epoch_index)
        if len(train_dataset) != 2160 or len(train_sampler) != 540 or len(val_sampler) != 116:
            raise RuntimeError("Unexpected training/validation sample or batch count")
        lr_first = optimizer.param_groups[0]["lr"]
        train_mae, train_count, lr_last = run_epoch(model, train_loader, device, optimizer)
        val_mae, val_count, _ = run_epoch(model, val_loader, device)
        if train_count != 2160 or val_count != 464:
            raise RuntimeError("Some subjects were skipped")
        scheduler.step()

        epoch = epoch_index + 1
        improved = val_mae < best_val_mae
        if improved:
            best_val_mae, best_epoch = val_mae, epoch
            best_weights = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        history.append({"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae,
                        "train_samples": train_count, "val_samples": val_count,
                        "lr_first": lr_first, "lr_last": lr_last,
                        "lr_next": optimizer.param_groups[0]["lr"],
                        "global_step": epoch * len(train_loader), "best_epoch": best_epoch})
        checkpoint = {"format_version": FORMAT_VERSION, "kind": "resume", "epoch": epoch,
                      "best_val_mae": best_val_mae, "best_epoch": best_epoch,
                      "model_state_dict": model.state_dict(), "best_model_state_dict": best_weights,
                      "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                      "global_step": epoch * len(train_loader),
                      "rng_state": capture_rng(), "history": history,
                      "model_config": {"metadata": args.metadata, "dropout": 0.2},
                      "training_config": config, "input_encoding": INPUT_ENCODING, "sampler_version": SAMPLER_VERSION}
        # last.pt is authoritative. Recover best.pt and CSV from it on resume
        # if power failed between the separate file replacements.
        atomic_save(checkpoint, args.output_dir / "last.pt")
        if improved:
            atomic_save(best_checkpoint(checkpoint), args.output_dir / "best.pt")
        if epoch % args.periodic_save == 0:
            atomic_save(checkpoint, args.output_dir / f"epoch_{epoch}.pt")
        write_history(history, args.output_dir / "training_log.csv")
        print(f"Epoch {epoch:04d} | train MAE {train_mae:.4f} | val MAE {val_mae:.4f} | "
              f"LR {lr_first:.9g}->{optimizer.param_groups[0]['lr']:.9g} | best {best_epoch}", flush=True)

        


if __name__ == "__main__":
    main()
