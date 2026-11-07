"""MRI dataset and reproducible age-balanced training batches."""
from __future__ import annotations

import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from config import CFG, Config

# Keep the directory conventions of the previous PyTorch dataloader.py.
DATASET_DIRS = {
    "IXI": "IXI", "ABIDE1": "ABIDE 1", "ABIDE2": "ABIDE 2",
    "FCON": "FCP1000", "NKI-RS": "NKI-RS", "HCA": "HCP-Aging",
    "CamCAN": "CamCAN",
}
REGISTRATION_DIRS = {
    "affine": ("affine", "rmi"),
    "rigid": ("ndf", "rmi"),
    "nonlinear": ("df", "wmi"),
}


def read_subjects(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ("Subject_ID", "SEX", "AGE", "Scanner")
    if any(col not in df.columns for col in required):
        raise ValueError(f"CSV must contain {required}")
    df = df.copy()
    df["Subject_ID"] = df["Subject_ID"].astype(str).str.strip()
    for col in ("SEX", "AGE", "Scanner"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df[list(required)].isna().any().any() or not np.isfinite(df[["SEX", "AGE", "Scanner"]]).all().all():
        raise ValueError("CSV contains missing or non-finite values")
    if df["Subject_ID"].duplicated().any():
        raise ValueError("Duplicate Subject_ID in CSV")
    if not df["SEX"].isin([1, 2]).all() or not df["Scanner"].isin([1.5, 3.0]).all():
        raise ValueError("Expected SEX=1/2 and Scanner=1.5/3.0")
    df["sex_model"] = df["SEX"].map({1: 0.0, 2: 1.0})
    df["scanner_model"] = df["Scanner"].map({1.5: 0.0, 3.0: 1.0})
    return df.reset_index(drop=True)


def image_path(subject_id: str, cfg: Config) -> Path:
    dataset = next((key for key in DATASET_DIRS if subject_id.startswith(key)), None)
    # HCA / CamCAN have different identifier prefixes in the source CSV.
    if dataset is None:
        dataset = "CamCAN" if subject_id.startswith("cc") else "HCA" if subject_id.startswith("HCA") else None
    if dataset is None:
        raise ValueError(f"Unknown Subject_ID prefix: {subject_id}")
    if cfg.registration not in REGISTRATION_DIRS:
        raise ValueError(f"Unknown registration: {cfg.registration}")

    reg_dir, image_dir = REGISTRATION_DIRS[cfg.registration]
    directory = cfg.data_root / DATASET_DIRS[dataset] / reg_dir / image_dir
    if cfg.registration == "rigid":
        patterns = [f"rmi{subject_id}_raw_tbet_rigid.nii*"]
    elif cfg.registration == "nonlinear":
        patterns = [f"wmi{subject_id}_raw_tbet.nii*"]
    else:
        if dataset == "IXI":
            patterns = [f"rmi{subject_id.replace('_', '')}-*_tbet_affine.nii*"]
        elif dataset == "ABIDE1":
            patterns = [f"rmi{subject_id.split('_', 1)[1].zfill(7)}_tbet_affine.nii*"]
        elif dataset == "ABIDE2":
            patterns = [f"rmi{subject_id.split('_', 1)[1]}_tbet_affine.nii*"]
        elif dataset == "NKI-RS":
            patterns = [f"rmis{subject_id.split('_', 1)[1]}-*_tbet_affine.nii*"]
        elif dataset == "CamCAN":
            patterns = [f"rmisub-{subject_id}_t1w_tbet_affine.nii*"]
        else:
            patterns = [f"rmi{subject_id}_tbet_affine.nii*"]
    matches = sorted({p for pattern in patterns for p in directory.glob(pattern) if p.is_file()})
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one MRI for {subject_id}, found {len(matches)} in {directory} "
            f"with {patterns}"
        )
    return matches[0]


class BrainAgeDataset(Dataset):
    def __init__(self, csv_path: Path, cfg: Config = CFG, split: str = "train"):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        self.cfg, self.split = cfg, split
        self.full_df = read_subjects(csv_path)
        self.df = self.full_df
        self._paths: dict[str, Path] = {}
        if split == "train":
            self.set_epoch(0)
        else:
            self._check_paths()

    def _check_paths(self):
        for subject_id in self.df["Subject_ID"]:
            if subject_id not in self._paths:
                self._paths[subject_id] = image_path(subject_id, self.cfg)

    def set_epoch(self, epoch: int):
        """Rebuild the same sample for a given seed and epoch, including after resume."""
        if self.split != "train":
            raise ValueError("Only training data are resampled")
        parts = []
        for bin_id, (low, high, count) in enumerate(self.cfg.age_sample_plan):
            group = self.full_df.loc[self.full_df["AGE"].between(low, high, inclusive="left")]
            if len(group) < count:
                raise ValueError(f"Age [{low}, {high}) has {len(group)} samples; need {count}")
            parts.append(group.sample(n=count, replace=False,
                                      random_state=self.cfg.seed + epoch * 1009 + bin_id))
        self.df = pd.concat(parts, ignore_index=True)
        self._check_paths()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = self._paths[row["Subject_ID"]]
        image = nib.load(str(path)).get_fdata(dtype=np.float32)
        if image.shape != self.cfg.input_shape or not np.isfinite(image).all():
            raise ValueError(f"Invalid MRI shape/values: {path} ({image.shape})")
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0),
            "age": torch.tensor(row["AGE"], dtype=torch.float32),
            "sex": torch.tensor(row["sex_model"], dtype=torch.float32),
            "scanner": torch.tensor(row["scanner_model"], dtype=torch.float32),
            "subject_id": row["Subject_ID"],
        }


class BalancedAgeBatchSampler(Sampler[list[int]]):
    """Form each training batch from equal-sized age-ranked groups."""
    def __init__(self, dataset: BrainAgeDataset, batch_size: int, seed: int):
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        ordered = self.dataset.df.sort_values(["AGE", "Subject_ID"], kind="stable").index.to_numpy()
        groups = [part.tolist() for part in np.array_split(ordered, self.batch_size)]
        rng = random.Random(self.seed + self.epoch)
        for group in groups:
            rng.shuffle(group)
        batches = [list(batch) for batch in zip(*groups)]  # drop incomplete batches
        rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return len(self.dataset) // self.batch_size


def make_loader(dataset: BrainAgeDataset, cfg: Config = CFG):
    """Train: balanced batches. Validation/test: ordinary fixed-order batches."""
    options = dict(num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    if dataset.split == "train" and cfg.balanced_train_batches:
        sampler = BalancedAgeBatchSampler(dataset, cfg.batch_size, cfg.seed)
        return DataLoader(dataset, batch_sampler=sampler, **options), sampler
    return DataLoader(dataset, batch_size=cfg.batch_size,
                      shuffle=(dataset.split == "train"), drop_last=(dataset.split == "train"),
                      **options), None
