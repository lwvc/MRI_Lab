
from pathlib import Path
import numpy as np
import pandas as pd
import pingouin as pg

DATA_DIR = Path(__file__).resolve().parent / "data"

MODELS = (
    ("SFCN-reg T1", "SFCN_reg.xlsx", "T1WI"),
    ("SFCN-reg GM", "SFCN_reg.xlsx", "GMP"),
    ("SFCN-reg WM", "SFCN_reg.xlsx", "WMP"),
    ("SFCN-sm T1", "SFCN_sm.xlsx", "T1WI"),
    ("SFCN-sm GM", "SFCN_sm.xlsx", "GMP"),
    ("SFCN-sm WM", "SFCN_sm.xlsx", "WMP"),
    ("ResNet T1", "ResNet.xlsx", "T1"),
    ("ResNet T1 B0", "ResNet.xlsx", "T1_B0"),
    ("ResNet T1 SEX", "ResNet.xlsx", "T1_SEX"),
    ("ResNet T1 SEX B0", "ResNet.xlsx", "T1_SEX_B0"),
)

OAS1_OVERLAP = {
    "OAS1_0061", "OAS1_0092", "OAS1_0111",
    "OAS1_0150", "OAS1_0156", "OAS1_0202",
    "OAS1_0230", "OAS1_0353", "OAS1_0368",
    "OAS1_0379",
}

MMRR_EXCLUDE = ("sub_502", "sub_742")


def load_data(dataset, filename, suffix):
    df = pd.read_excel(
        DATA_DIR / filename,
        sheet_name=f"{dataset}_{suffix}",
        usecols=["Subject ID", "True Age", "Predicted Age", "pair"],
    )

    df = df.dropna(
        subset=["Subject ID", "True Age", "Predicted Age", "pair"]
    ).copy()

    df["Pair"] = (
        df["pair"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    # 同一 pair 中的兩筆資料分別標記為 Scan 1、Scan 2
    df["Scan"] = df.groupby("Pair", sort=False).cumcount() + 1

    if not df.groupby("Pair").size().eq(2).all():
        raise ValueError(f"配對異常：{dataset} / {filename} / {suffix}")

    return df


def calculate_mae(df):
    # MAE = (1/n) * sum(|True Age - Predicted Age|)
    return np.mean(np.abs(df["True Age"] - df["Predicted Age"]))


def get_pairs(df):
    return df.pivot(
        index="Pair",
        columns="Scan",
        values="Predicted Age",
    )


def calculate_mad(df):
    # MAD = (1/N) * sum(|Predicted Age 1 - Predicted Age 2|)
    pairs = get_pairs(df)
    return np.mean(np.abs(pairs[1] - pairs[2]))


def calculate_icc(df):
    # ICC(2,1) = (MSB - MSE) /
    #            [MSB + (k-1)*MSE + (k/n)*(MSW-MSE)]
    #
    # MSB：受試者間均方
    # MSW：掃描次數間均方
    # MSE：殘差均方
    # k = 2，n = 配對受試者數
    #
    # Pingouin ICC(A,1)：two-way absolute agreement,
    # single measurement，對應此 ICC 點估計。

    result = pg.intraclass_corr(
        data=df.rename(columns={"Predicted Age": "Pred"}),
        targets="Pair",
        raters="Scan",
        ratings="Pred",
    )

    row = result.loc[result["Type"] == "ICC(A,1)"].iloc[0]
    return row["ICC"], row["CI95"]


def calculate_maclaren_sd(df):
    df = df.copy()
    df["Subject"] = df["Subject ID"].str.extract(
        r"^(sub-\d+)_run-"
    )

    result = {}
    intra_values = []
    total_values = []

    for s in range(1, 4):
        sub = df[df["Subject"] == f"sub-{s:02d}"]
        pairs = get_pairs(sub)

        # SD_intra = sqrt(sum((Pred1 - Pred2)^2) / (2*m))
        # m = 20 sessions
        diff = pairs[1] - pairs[2]
        sd_intra = np.sqrt(np.sum(diff**2) / (2 * len(diff)))

        # SD_total = sqrt(sum((Pred - Mean Pred)^2) / (N-1))
        # N = 40 scans
        sd_total = sub["Predicted Age"].std(ddof=1)

        result[f"Sub{s} intra/total"] = (
            f"{sd_intra:.3f} / {sd_total:.3f}"
        )

        intra_values.append(sd_intra)
        total_values.append(sd_total)

    # 分別平均三位受試者的 SD
    mean_intra = np.mean(intra_values)
    mean_total = np.mean(total_values)

    # Delta SD = Mean SD_total - Mean SD_intra
    result["Mean SD intra"] = mean_intra
    result["Mean SD total"] = mean_total
    result["Delta SD"] = mean_total - mean_intra

    return result


def fmt(value):
    return f"{value:.3f}"


def fmt_pair(full, filtered):
    # 括號外：全部資料；括號內：排除後
    return f"{full:.3f} ({filtered:.3f})"


def fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def calculate_metrics(dataset, model, filename, suffix):
    df = load_data(dataset, filename, suffix)

    mae = calculate_mae(df)
    mad = calculate_mad(df)

    row = {
        "Model": model,
        "N scans": len(df),
        "N pairs": len(get_pairs(df)),
        "MAE": fmt(mae),
        "MAD": fmt(mad),
    }

    if dataset == "Maclaren":
        row.update(calculate_maclaren_sd(df))
        return row

    icc, ci = calculate_icc(df)
    row["ICC(2,1)"] = fmt(icc)
    row["95% CI"] = fmt_ci(ci)

    if dataset == "OAS1" and model.startswith("ResNet"):
        # MAE：只排除 overlap 受試者的 MR1，共 10 scans
        subject = df["Subject ID"].str.replace(r"_MR\d+$", "", regex=True)
        is_mr1 = df["Subject ID"].str.endswith("_MR1")

        mae_df = df[~(subject.isin(OAS1_OVERLAP) & is_mr1)]

        # MAD、ICC：排除 overlap 受試者的完整 MR1/MR2 配對
        pair_df = df[~df["Pair"].isin(OAS1_OVERLAP)]

        clean_icc, clean_ci = calculate_icc(pair_df)

        row["N scans"] = f"{len(df)} ({len(mae_df)})"
        row["N pairs"] = f"{len(get_pairs(df))} ({len(get_pairs(pair_df))})"

        row["MAE"] = fmt_pair(mae, calculate_mae(mae_df))
        row["MAD"] = fmt_pair(mad, calculate_mad(pair_df))
        row["ICC(2,1)"] = fmt_pair(icc, clean_icc)
        row["95% CI"] = f"{fmt_ci(ci)} ({fmt_ci(clean_ci)})"

    elif dataset == "MMRR":
        # MAE 排除 sub_502、sub_742，共 4 scans
        # MAD 與 ICC 不排除
        subject = df["Subject ID"].str.extract(r"^(sub_\d+)")[0]
        clean = df[~subject.isin(MMRR_EXCLUDE)]

        row["N scans"] = f"{len(df)} ({len(clean)})"
        row["MAE"] = fmt_pair(mae, calculate_mae(clean))

    return row


def main():
    for dataset in ("Maclaren", "OAS1", "MMRR"):
        rows = [
            calculate_metrics(dataset, model, filename, suffix)
            for model, filename, suffix in MODELS
        ]

        table = pd.DataFrame(rows)

        print(f"\n========== {dataset} ==========")
        print(
            table.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )


if __name__ == "__main__":
    main()
