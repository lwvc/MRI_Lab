from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


EXPECTED_IMAGE_SHAPE = (193, 229, 193)
REQUIRED_COLUMNS = ("Subject_ID", "SEX", "AGE", "Scanner")

# Expected layout:
# All registrations: DATA_ROOT / dataset folder / registration folder / image folder.
# The shared default root is C:/OpenData; affine now has its own subdirectory.
SUPPORTED_REGISTRATIONS = ("affine", "rigid", "nonlinear")
REGISTRATION_FOLDERS = {
    "affine": "affine",
    "rigid": "ndf",
    # This computer's nonlinear/deformation outputs live under df/wmi.
    "nonlinear": "df",
}
IMAGE_FOLDERS = {
    "affine": "rmi",
    "rigid": "rmi",
    "nonlinear": "wmi",
}

DATASET_FOLDERS = {
    "IXI": "IXI",
    "ABIDE1": "ABIDE 1",
    "ABIDE2": "ABIDE 2",
    "FCON": "FCP1000",
    "NKI-RS": "NKI-RS",
    "HCA": "HCP-Aging",
    "CamCAN": "CamCAN",
}

# Per-epoch quotas agreed with the supervisor on 2026-10-08.
# These strata select the training subset; batch formation then ranks ALL
# selected subjects by age and splits them into four equal-count groups.
AGE_SAMPLE_PLAN = (
    (5.0, 15.0, 280),
    (15.0, 25.0, 280),
    (25.0, 35.0, 280),
    (35.0, 45.0, 280),
    (45.0, 55.0, 280),
    (55.0, 65.0, 280),
    (65.0, 75.0, 268),
    (75.0, 85.0, 172),
    (85.0, 105.0, 40),
)


def dataset_name(subject_id: str) -> str:
    if subject_id.startswith("IXI_"):
        return "IXI"
    if subject_id.startswith("ABIDE1_"):
        return "ABIDE1"
    if subject_id.startswith("ABIDE2_"):
        return "ABIDE2"
    if subject_id.startswith("FCON_"):
        return "FCON"
    if subject_id.startswith("NKI-RS_"):
        return "NKI-RS"
    if subject_id.startswith("HCA"):
        return "HCA"
    if subject_id.startswith("cc"):
        return "CamCAN"
    raise ValueError(f"Unsupported Subject_ID: {subject_id}")


def normalize_registration(registration: str) -> str:
    normalized = registration.lower()
    if normalized not in SUPPORTED_REGISTRATIONS:
        raise ValueError(
            f"registration must be one of {SUPPORTED_REGISTRATIONS}, "
            f"got {registration!r}"
        )
    return normalized


def _registered_t1_patterns(subject_id: str, prefix: str, suffix: str) -> list[str]:
    if subject_id.startswith("IXI_"):
        core = subject_id.replace("_", "")
        return [
            f"{prefix}{core}-*{suffix}",
            f"{prefix}{subject_id}{suffix}",
        ]

    if subject_id.startswith("ABIDE1_"):
        numeric_id = subject_id.split("_", maxsplit=1)[1]
        return [
            f"{prefix}{numeric_id.zfill(7)}{suffix}",
            f"{prefix}{subject_id}{suffix}",
        ]

    if subject_id.startswith("ABIDE2_"):
        numeric_id = subject_id.split("_", maxsplit=1)[1]
        return [
            f"{prefix}{numeric_id}{suffix}",
            f"{prefix}{subject_id}{suffix}",
        ]

    if subject_id.startswith("FCON_"):
        return [f"{prefix}{subject_id}{suffix}"]

    if subject_id.startswith("NKI-RS_"):
        core = subject_id.split("_", maxsplit=1)[1]
        return [
            f"{prefix}s{core}-*{suffix}",
            f"{prefix}{subject_id}{suffix}",
        ]

    if subject_id.startswith("HCA"):
        return [f"{prefix}{subject_id}{suffix}"]

    if subject_id.startswith("cc"):
        return [
            f"{prefix}sub-{subject_id}_t1w{suffix}",
            f"{prefix}{subject_id}{suffix}",
        ]

    raise ValueError(f"Unsupported Subject_ID: {subject_id}")


def filename_patterns(subject_id: str, registration: str = "affine") -> list[str]:
    registration = normalize_registration(registration)
    if registration == "rigid":
        return [f"rmi{subject_id}_raw_tbet_rigid.nii*"]

    if registration == "affine":
        patterns = _registered_t1_patterns(
            subject_id,
            prefix="rmi",
            suffix="_tbet_affine.nii*",
        )
        # Keep the previously validated affine pattern exact.
        return patterns[:1]

    return [f"wmi{subject_id}_raw_tbet.nii*"]


def filename_pattern(subject_id: str, registration: str = "affine") -> str:
    return filename_patterns(subject_id, registration)[0]


@dataclass(frozen=True)
class RegistrationPathResolver:
    data_root: Path
    registration: str = "affine"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "registration",
            normalize_registration(self.registration),
        )

    def image_directory(self, subject_id: str) -> Path:
        name = dataset_name(subject_id)
        return (
            self.data_root
            / DATASET_FOLDERS[name]
            / REGISTRATION_FOLDERS[self.registration]
            / IMAGE_FOLDERS[self.registration]
        )

    def resolve(self, subject_id: str) -> Path:
        image_dir = self.image_directory(subject_id)
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory does not exist for {subject_id}: {image_dir}"
            )

        patterns = filename_patterns(subject_id, self.registration)
        matches = []
        for pattern in patterns:
            matches.extend(path for path in image_dir.glob(pattern) if path.is_file())
        matches = sorted(set(matches), key=str)
        if not matches:
            raise FileNotFoundError(
                f"No image matched Subject_ID={subject_id}, "
                f"directory={image_dir}, patterns={patterns}"
            )
        if len(matches) > 1:
            joined = "\n  ".join(str(path) for path in matches)
            raise RuntimeError(
                f"Multiple images matched Subject_ID={subject_id}. "
                f"Do not select the first file silently:\n  {joined}"
            )
        return matches[0]


# Backward-compatible name for older affine-only scripts/imports.
AffinePathResolver = RegistrationPathResolver


def read_subject_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    dataframe = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    dataframe = dataframe.copy()
    dataframe["Subject_ID"] = dataframe["Subject_ID"].astype(str).str.strip()
    for column in ("SEX", "AGE", "Scanner"):
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")

    if dataframe[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(f"{csv_path} contains missing required values")
    if dataframe["Subject_ID"].duplicated().any():
        duplicates = dataframe.loc[
            dataframe["Subject_ID"].duplicated(keep=False), "Subject_ID"
        ].tolist()
        raise ValueError(f"{csv_path} contains duplicate Subject_ID values: {duplicates}")

    if len(dataframe) == 0:
        raise ValueError(f"{csv_path} is empty")
    if not np.isfinite(dataframe[["SEX", "AGE", "Scanner"]].to_numpy()).all():
        raise ValueError(f"{csv_path} contains non-finite values")
    # CSV values retain their original meaning. Do not infer encoding from a
    # single-sex subset: SEX=1 always means male in this study's source files.
    sex_values = set(dataframe["SEX"].unique())
    if not sex_values.issubset({1, 2}):
        raise ValueError(
            f"{csv_path}: SEX must use 1=male, 2=female; got {sex_values}"
        )
    dataframe["sex_model"] = dataframe["SEX"].map({1: 0.0, 2: 1.0})
    scanner_values = set(dataframe["Scanner"].unique())
    if not scanner_values.issubset({1.5, 3.0}):
        raise ValueError(f"{csv_path}: Scanner must use 1.5/3.0 T; got {scanner_values}")
    dataframe["scanner_model"] = dataframe["Scanner"].map({1.5: 0.0, 3.0: 1.0})

    dataframe["source_row"] = np.arange(len(dataframe), dtype=np.int64)
    return dataframe


def load_mri(
    path: str | Path,
    expected_shape: Sequence[int] = EXPECTED_IMAGE_SHAPE,
) -> torch.Tensor:
    image = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
    if image.ndim == 4 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D MRI, got shape {image.shape}: {path}")
    if tuple(image.shape) != tuple(expected_shape):
        raise ValueError(
            f"Expected MRI shape {tuple(expected_shape)}, got {image.shape}: {path}"
        )
    if not np.isfinite(image).all():
        raise ValueError(f"MRI contains NaN or infinity: {path}")

    return torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)


class BrainAgeDataset(Dataset):
    def __init__(
        self,
        csv_file: str | Path,
        data_root: str | Path,
        purpose: str,
        registration: str = "affine",
        seed: int = 1,
    ) -> None:
        if purpose not in {"train", "val", "test"}:
            raise ValueError("purpose must be train, val, or test")

        # self.full_df = read_subject_csv(csv_file)
        # self.registration = normalize_registration(registration)
        # self.resolver = RegistrationPathResolver(
        #     Path(data_root),
        #     registration=self.registration,
        # )
        # self.purpose = purpose
        # self.seed = seed
        # self._path_cache: dict[str, str] = {}

        # 改
        self.full_df = read_subject_csv(csv_file)

        if "File_Path" not in self.full_df.columns:
            raise ValueError("CSV missing File_Path column")

        self.registration = normalize_registration(registration)
        self.purpose = purpose
        self.seed = seed
        # 改

        if purpose == "train":
            self.resample_for_epoch(epoch=0)
        else:
            self.df = self.full_df.reset_index(drop=True)
            self._add_image_paths()

    # def _resolve_cached(self, subject_id: str) -> str:
    #     if subject_id not in self._path_cache:
    #         self._path_cache[subject_id] = str(self.resolver.resolve(subject_id))
    #     return self._path_cache[subject_id]

    # def _add_image_paths(self) -> None:
    #     self.df = self.df.copy()
    #     self.df["image_path"] = self.df["Subject_ID"].map(self._resolve_cached)

    # 改
    def _add_image_paths(self) -> None:
        self.df = self.df.copy()

        paths = self.df["File_Path"]

        if paths.isna().any() or paths.astype(str).str.strip().eq("").any():
            raise ValueError("CSV contains empty File_Path")

        self.df["image_path"] = paths.astype(str).str.strip()

        missing = [
            path for path in self.df["image_path"]
            if not Path(path).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} MRI files not found. "
                f"First missing file: {missing[0]}"
            )
        # 改

    def resample_for_epoch(self, epoch: int) -> None:
        if self.purpose != "train":
            raise RuntimeError("Only the training dataset can be resampled")

        sampled_parts = []
        for bin_index, (age_min, age_max, requested_count) in enumerate(
            AGE_SAMPLE_PLAN
        ):
            subset = self.full_df[
                (self.full_df["AGE"] >= age_min)
                & (self.full_df["AGE"] < age_max)
            ]
            sample_count = requested_count
            if len(subset) < requested_count:
                raise ValueError(
                    f"Need {requested_count} subjects in [{age_min}, {age_max}), "
                    f"found {len(subset)}"
                )
            sampled = subset.sample(
                n=sample_count,
                replace=False,
                random_state=self.seed + epoch * 1009 + bin_index,
            )
            sampled_parts.append(sampled)

        self.df = pd.concat(sampled_parts, ignore_index=True)
        self._add_image_paths()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.df.iloc[index]
        image_path = row["image_path"]
        image = load_mri(image_path)
        return {
            "image": image,
            "age": torch.tensor(float(row["AGE"]), dtype=torch.float32),
            "sex": torch.tensor(float(row["sex_model"]), dtype=torch.float32),
            "scanner": torch.tensor(float(row["scanner_model"]), dtype=torch.float32),
            "subject_id": str(row["Subject_ID"]),
            "image_path": str(image_path),
            "source_row": torch.tensor(int(row["source_row"]), dtype=torch.int64),
        }


class BalancedAgeBatchSampler(Sampler[list[int]]):
    """Age-ranked equal-count groups; one subject per group in each batch.

    Ties are broken by Subject_ID, so evaluation batches are independent of
    CSV row order, training seed, and registration. Training shuffles each
    group and batch order deterministically using seed + epoch.
    """

    def __init__(
        self,
        dataset: BrainAgeDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        seed: int = 1,
        drop_last: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _generate_balanced_batches(self) -> list[list[int]]:
        frame = self.dataset.df.reset_index(drop=True)
        ordered = frame.sort_values(["AGE", "Subject_ID"], kind="stable").index.to_numpy()
        groups = [part.tolist() for part in np.array_split(ordered, self.batch_size)]
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            for group in groups:
                rng.shuffle(group)
        batches = [[group[i] for group in groups if i < len(group)]
                   for i in range(max(map(len, groups), default=0))]
        return [batch for batch in batches
                if not self.drop_last or len(batch) == self.batch_size]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches = self._generate_balanced_batches()
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self._generate_balanced_batches())


def make_balanced_loader(
    dataset: BrainAgeDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    drop_last: bool = True,
) -> tuple[DataLoader, BalancedAgeBatchSampler]:
    sampler = BalancedAgeBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    return loader, sampler
