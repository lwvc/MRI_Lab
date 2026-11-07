"""
Independent test for Regional-FOV DGM SFCN.

Supported datasets:
- MACL
- OAS1
- MMRR
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Define_Dataset_SM_Global import MRIDataset_SM
from SFCN_DGM import SFCN


# =========================================================
# 1. Configuration
# =========================================================

DATASETS = ["MACL", "MMRR", "OAS1"]

BASE_DIR = r"C:\Lab\test_retest_dataset"

final_min_coords = np.array([52, 77, 50])
final_max_coords = np.array([140, 168, 122])

OUTPUT_DIM = 95
BATCH_SIZE = 4

PRETRAIN_PT_PATH = (
    r"C:\Lab\models\SFCN\sm\pretrained"
    r"\pretrain_regionalFOV_SM_DGM_withcc_v2_20260305.pt"
)

DATASET_CONFIG = {
    "MACL": {
        "image_dir": r"C:\Lab\test_retest_dataset\Maclaren\ndf\dgm16",
        "excel": r"C:\Lab\test_retest_dataset\Maclaren\Maclaren.xlsx",
    },

    "MMRR": {
        "image_dir": r"C:\Lab\test_retest_dataset\MMRR\ndf\dgm16",
        "excel": r"C:\Lab\test_retest_dataset\MMRR\MMRR.xlsx",
    },

    "OAS1": {
            "image_dir": r"C:\Lab\test_retest_dataset\OAS1\ndf\dgm16",
            "excel": r"C:\Lab\test_retest_dataset\OAS1\OAS1.xlsx",
        },
}


# =========================================================
# 2. Device / Model
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"目前使用的裝置是: {device}")


model = SFCN(
    output_dim=OUTPUT_DIM
).to(device)


state_dict = torch.load(
    PRETRAIN_PT_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict, strict=True)
model.eval()


# =========================================================
# 3. Age bins
# =========================================================

bins = np.linspace(5, 100, 96)

bin_centers = [
    (bins[i] + bins[i + 1]) / 2
    for i in range(len(bins) - 1)
]

age_bins_center = torch.tensor(
    bin_centers,
    dtype=torch.float32,
    device=device,
)


# =========================================================
# 4. Test function
# =========================================================

def run_test(dataset_name):

    print("\n" + "=" * 70)
    print(f"開始測試：{dataset_name}")
    print("=" * 70)

    config = DATASET_CONFIG[dataset_name]

    image_dir = config["image_dir"]
    excel_path = config["excel"]

    output_name = os.path.join(
        BASE_DIR,
        f"IndependentTest_{dataset_name}_DGM.xlsx",
    )

    print(f"影像資料夾: {image_dir}")
    print(f"Excel: {excel_path}")
    print(f"Model: {PRETRAIN_PT_PATH}")
    print(f"Crop min: {final_min_coords.tolist()}")
    print(f"Crop max: {final_max_coords.tolist()}")


    # -----------------------------------------------------
    # Read Excel
    # -----------------------------------------------------

    df = pd.read_excel(excel_path)

    required_columns = {
        "Subject_ID",
        "Age",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{dataset_name} Excel 缺少欄位: "
            f"{sorted(missing_columns)}"
        )


    test_df = df[
        ["Subject_ID", "Age"]
    ].copy()

    test_df["Subject_ID"] = (
        test_df["Subject_ID"]
        .astype(str)
        .str.strip()
    )

    test_df["Original_Subject_ID"] = (
        test_df["Subject_ID"]
    )


    # -----------------------------------------------------
    # Construct filename
    # -----------------------------------------------------

    test_df["Filename"] = (
        test_df["Subject_ID"].apply(
            lambda x: f"{x}_dgm16.nii"
        )
    )


    dataset_df = test_df[
        ["Filename", "Age"]
    ].copy()


    # -----------------------------------------------------
    # Check files
    # -----------------------------------------------------

    missing_files = []

    for filename in dataset_df["Filename"]:

        path = os.path.join(
            image_dir,
            filename,
        )

        if not os.path.exists(path):
            missing_files.append(filename)


    if missing_files:

        print(
            f"[警告] {dataset_name} "
            f"有 {len(missing_files)} 個影像不存在"
        )

        for name in missing_files[:10]:
            print(f"  找不到: {name}")

        raise FileNotFoundError(
            f"{dataset_name} 有影像缺失，停止測試。"
        )


    print(
        f"Excel 樣本數: {len(dataset_df)}"
    )


    # -----------------------------------------------------
    # Dataset / DataLoader
    # -----------------------------------------------------

    test_dataset = MRIDataset_SM(
        data_folders_set=[image_dir],
        labels_df=dataset_df,
        bin_range=(5, 100),
        crop_coords=(
            final_min_coords,
            final_max_coords,
        ),
    )


    print(
        f"成功載入影像數: "
        f"{len(test_dataset)}"
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    test_mae_total = 0.0
    test_samples = 0

    true_ages = []
    pred_ages = []


    with torch.no_grad():

        for images, labels_hard, _ in tqdm(
            test_loader,
            desc=dataset_name,
        ):

            images = images.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            labels_hard = labels_hard.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )


            outputs = (
                model(images)[0]
                .squeeze(-1)
                .squeeze(-1)
                .squeeze(-1)
            )


            if outputs.ndim != 2 or outputs.shape[1] != OUTPUT_DIM:
                raise RuntimeError(
                    f"模型輸出尺寸錯誤: {tuple(outputs.shape)}"
                )
            probs = outputs.exp()


            pred_age = torch.sum(
                probs * age_bins_center,
                dim=1,
            )


            true_ages.extend(
                labels_hard
                .cpu()
                .numpy()
                .tolist()
            )


            pred_ages.extend(
                pred_age
                .cpu()
                .numpy()
                .tolist()
            )


            test_mae_total += (
                torch.abs(
                    pred_age - labels_hard
                )
                .sum()
                .item()
            )


            test_samples += images.size(0)


    if test_samples == 0:

        raise RuntimeError(
            f"{dataset_name} 沒有載入任何影像"
        )


    test_mae = (
        test_mae_total / test_samples
    )


    print(
        f"\n{dataset_name} Test MAE: "
        f"{test_mae:.4f}"
    )


    # -----------------------------------------------------
    # Save result
    # -----------------------------------------------------

    result_df = pd.DataFrame({
        "Subject_ID":
            test_df[
                "Original_Subject_ID"
            ].tolist(),

        "Filename":
            dataset_df[
                "Filename"
            ].tolist(),

        "Test TrueAge":
            true_ages,

        "Test PredictedAge":
            pred_ages,
    })


    result_df.to_excel(
        output_name,
        index=False,
    )


    print(
        f"結果已儲存：{output_name}"
    )


# =========================================================
# 5. Run all datasets
# =========================================================

if __name__ == "__main__":

    for dataset_name in DATASETS:

        run_test(dataset_name)

    print("\n三個資料集全部測試完成。")