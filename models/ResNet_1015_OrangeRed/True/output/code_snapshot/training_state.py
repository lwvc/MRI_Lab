"""Versioned checkpoints and training history utilities."""
from __future__ import annotations

import csv
import hashlib
import io
import os
import random
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 3
INPUT_ENCODING = {"SEX": "1=male->0; 2=female->1", "Scanner": "1.5T->0; 3T->1"}
SAMPLER_VERSION = "age_rank_equal_count_v1"
LOG_FIELDS = ("epoch", "train_mae", "val_mae", "train_samples", "val_samples",
              "lr_first", "lr_last", "lr_next", "global_step", "best_epoch")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"kind": state[0], "keys": state[1].tolist(),
                      "position": state[2], "has_gauss": state[3], "cached_gaussian": state[4]},
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value["kind"], np.asarray(value["keys"], dtype=np.uint32),
                         value["position"], value["has_gauss"], value["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def atomic_save(checkpoint, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(text: str, path: Path):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_history(history, path: Path):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=LOG_FIELDS)
    writer.writeheader()
    writer.writerows(history)
    atomic_text(buffer.getvalue(), path)


def load_checkpoint(path: Path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Use the old code for old checkpoints; this study uses binary scanner encoding")
    if checkpoint.get("input_encoding") != INPUT_ENCODING or checkpoint.get("sampler_version") != SAMPLER_VERSION:
        raise ValueError("Checkpoint encoding/sampler does not match this study")
    return checkpoint


def best_checkpoint(checkpoint):
    return {"format_version": FORMAT_VERSION, "kind": "inference",
            "input_encoding": INPUT_ENCODING, "sampler_version": SAMPLER_VERSION,
            "epoch": checkpoint["best_epoch"], "best_val_mae": checkpoint["best_val_mae"],
            "model_state_dict": checkpoint["best_model_state_dict"],
            "model_config": checkpoint["model_config"], "training_config": checkpoint["training_config"]}
