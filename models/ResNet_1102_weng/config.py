"""Configuration for PyTorch brain-age training."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Paths: CSVs and output_dir are supplied when launching train.py.
    data_root: Path = Path("C:/OpenData")
    train_csv: Path | None = None
    val_csv: Path | None = None
    output_dir: Path | None = None
    registration: str = "affine"  # affine / rigid (ndf) / nonlinear (df)

    # Model
    input_shape: tuple[int, int, int] = (193, 229, 193)
    filters: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
    fc_units: int = 256
    metadata: str = "both"  # none / sex / scanner / both
    dropout: float = 0.2
    norm_type: str = "bn"  # Temporary; replace with BRN later
    norm_eps: float = 1e-3
    norm_momentum: float = 0.01  # PyTorch convention; Keras momentum=0.99
    renorm_momentum: float = 0.99  # For custom BRN implementation

    # Training
    epochs: int = 1000
    batch_size: int = 4
    learning_rate: float = 1e-3
    lr_decay: float = 1e-6  # Inverse-time decay per optimizer step
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-7
    l2_coefficient: float = 5e-5  # Apply to legacy kernel weights only
    seed: int = 1
    num_workers: int = 4

    # Training-only age-stratified sampling; validation/test use fixed CSVs.
    age_sample_plan: tuple[tuple[int, int, int], ...] = (
        (5, 15, 280), (15, 25, 280), (25, 35, 280),
        (35, 45, 280), (45, 55, 280), (55, 65, 280),
        (65, 75, 268), (75, 85, 172), (85, 105, 40),
    )
    balanced_train_batches: bool = True

    # Checkpoints and stopping
    monitor: str = "val_mae"
    early_stopping: bool = True
    patience: int = 100
    min_delta: float = 0.0
    periodic_save: int = 100


CFG = Config()
