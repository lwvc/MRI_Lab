from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataloader import BrainAgeDataset, SUPPORTED_REGISTRATIONS, make_balanced_loader
from model import BrainAgeResNet
from training_state import load_checkpoint, file_hash, INPUT_ENCODING, SAMPLER_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained registered-MRI brain-age checkpoint."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split label; all choices evaluate the full CSV without resampling.",
    )
    parser.add_argument(
        "--registration",
        choices=SUPPORTED_REGISTRATIONS,
        default=None,
        help="MRI registration folder and filename suffix to use.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def regression_metrics(
    true_age: np.ndarray,
    predicted_age: np.ndarray,
) -> dict[str, float]:
    residual = predicted_age - true_age
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    denominator = float(np.sum((true_age - true_age.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator)
    correlation = float(np.corrcoef(true_age, predicted_age)[0, 1])
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson_r": correlation}


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["training_config"]
    registration = config["registration"]
    if args.registration is not None and args.registration != registration:
        raise ValueError("Registration does not match the checkpoint")
    # if args.batch_size != config["batch_size"]:
    #     raise ValueError("Keep checkpoint batch size: BatchNorm uses batch statistics")
    if args.split in ("train", "val") and file_hash(args.csv) != config[f"{args.split}_sha256"]:
        raise ValueError("CSV does not match the checkpoint's training/validation split")
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError("Prediction CSV exists; use --overwrite to replace it")
    model = BrainAgeResNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = BrainAgeDataset(
        csv_file=args.csv,
        data_root=args.data_root,
        # Training-set evaluation must not invoke per-epoch training resampling.
        purpose="test" if args.split == "train" else args.split,
        registration=registration,
        seed=args.seed,
    )
    loader, sampler = make_balanced_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        seed=args.seed,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    sampler.set_epoch(0)

    records: list[dict[str, object]] = []
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True)
            sex = batch["sex"].view(-1, 1).to(device, non_blocking=True)
            scanner = batch["scanner"].view(-1, 1).to(device, non_blocking=True)
            prediction = model(image, sex, scanner).cpu().numpy().reshape(-1)

            for batch_index, predicted_age in enumerate(prediction):
                records.append(
                    {
                        "source_row": int(batch["source_row"][batch_index]),
                        "Subject_ID": batch["subject_id"][batch_index],
                        "image_path": batch["image_path"][batch_index],
                        "AGE": float(batch["age"][batch_index]),
                        "Predicted_Age": float(predicted_age),
                        "batch_id": batch_id,
                        "batch_position": batch_index,
                        "SEX_model": float(batch["sex"][batch_index]),
                        "Scanner_model": float(batch["scanner"][batch_index]),
                        "registration": registration,
                        "training_seed": config["seed"],
                        "checkpoint_epoch": checkpoint["epoch"],
                    }
                )

    if len(records) != len(dataset):
        raise RuntimeError(
            f"Expected {len(dataset)} predictions but received {len(records)}"
        )

    prediction_df = pd.DataFrame(records).sort_values("source_row")
    if not np.isfinite(prediction_df["Predicted_Age"]).all():
        raise RuntimeError("Non-finite predictions")
    prediction_df["PAD_uncorrected"] = (
        prediction_df["Predicted_Age"] - prediction_df["AGE"]
    )
    metrics = regression_metrics(
        prediction_df["AGE"].to_numpy(dtype=np.float64),
        prediction_df["Predicted_Age"].to_numpy(dtype=np.float64),
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.drop(columns="source_row").to_csv(args.output_csv, index=False)
    args.output_csv.with_suffix(".json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_hash(args.checkpoint),
        "csv_sha256": file_hash(args.csv), "split": args.split, "registration": registration,
        "training_seed": config["seed"], "checkpoint_epoch": checkpoint["epoch"],
        "batch_size": args.batch_size, "input_encoding": INPUT_ENCODING,
        "sampler_version": SAMPLER_VERSION, "metrics": metrics,
    }, indent=2), encoding="utf-8")
    print(
        " | ".join(
            (
                f"MAE {metrics['mae']:.4f}",
                f"RMSE {metrics['rmse']:.4f}",
                f"R2 {metrics['r2']:.4f}",
                f"Pearson r {metrics['pearson_r']:.4f}",
            )
        )
    )
    print(f"Split: {args.split} | Samples: {len(prediction_df)}")
    print(f"Predictions: {args.output_csv}")


if __name__ == "__main__":
    main()
