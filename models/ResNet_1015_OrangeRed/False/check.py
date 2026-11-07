# from pathlib import Path
# from dataloader import BrainAgeDataset

# BASE_DIR = Path(__file__).resolve().parent.parent

# dataset = BrainAgeDataset(
#     csv_file=BASE_DIR / "train_datalist.csv",
#     data_root=BASE_DIR,  # 保留介面相容，影像路徑直接取自 CSV
#     purpose="train",
#     registration="rigid",
#     seed=1,
# )

# sample = dataset[0]

# print("Subject:", sample["subject_id"])
# print("MRI path:", sample["image_path"])
# print("MRI shape:", sample["image"].shape)
# print("MRI dtype:", sample["image"].dtype)
# print("Age:", sample["age"].item())
# print("Sex:", sample["sex"].item())
# print("Scanner:", sample["scanner"].item())

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def inspect(csv_file: Path, split: str):
    df = pd.read_csv(csv_file)
    for col in ('Subject_ID', 'File_Path'):
        if col not in df:
            raise ValueError(f'{csv_file} missing column {col}')
    rows = []
    for i, row in df.iterrows():
        path = Path(str(row['File_Path']).strip())
        result = {'split': split, 'Subject_ID': row['Subject_ID'], 'File_Path': str(path)}
        try:
            nii = nib.load(str(path))
            img = np.asarray(nii.get_fdata(dtype=np.float32))
            if img.ndim == 4 and img.shape[-1] == 1:
                img = img[..., 0]
            finite = np.isfinite(img)
            values = img[finite]
            nonzero = values[values != 0]
            zooms = nii.header.get_zooms()[:3]
            result.update({
                'status': 'ok' if finite.all() and img.shape == (193, 229, 193) else 'check',
                'shape': str(img.shape),
                'orientation': ''.join(nib.aff2axcodes(nii.affine)),
                'voxel_spacing': ','.join(f'{x:.4f}' for x in zooms),
                'affine': ','.join(f'{x:.4f}' for x in nii.affine.ravel()),
                'finite_fraction': float(finite.mean()),
                'zero_fraction': float(np.mean(img == 0)),
                'min': float(values.min()) if values.size else np.nan,
                'max': float(values.max()) if values.size else np.nan,
                'mean': float(values.mean()) if values.size else np.nan,
                'std': float(values.std()) if values.size else np.nan,
                'nz_p01': float(np.percentile(nonzero, 1)) if nonzero.size else np.nan,
                'nz_p50': float(np.percentile(nonzero, 50)) if nonzero.size else np.nan,
                'nz_p99': float(np.percentile(nonzero, 99)) if nonzero.size else np.nan,
            })
        except Exception as exc:
            result['status'] = f'error: {exc}'
        rows.append(result)
        if (i + 1) % 100 == 0:
            print(f'{split}: {i + 1}/{len(df)}', flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=Path, required=True)
    parser.add_argument('--val-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('mri_input_check.csv'))
    args = parser.parse_args()
    report = pd.concat([inspect(args.train_csv, 'train'), inspect(args.val_csv, 'val')], ignore_index=True)
    report.to_csv(args.output, index=False, encoding='utf-8-sig')
    cols = ['min', 'max', 'mean', 'std', 'zero_fraction', 'nz_p01', 'nz_p50', 'nz_p99']
    print('\nNumerical summary by split:')
    print(report.groupby('split')[cols].agg(['min', 'median', 'max']).to_string())
    print('\nStatus counts:')
    print(report.groupby(['split', 'status']).size().to_string())
    for field in ('shape', 'orientation', 'voxel_spacing'):
        print(f'\n{field}:')
        print(report.groupby(['split', field]).size().to_string())
    print(f'\nSaved report: {args.output.resolve()}')


if __name__ == '__main__':
    main()