
"""Brain Age Test-Retest Dashboard."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ============================================================
# 1. Configuration
# ============================================================

DATA = Path(__file__).resolve().parent / "data"

DATASETS = ("Maclaren", "OAS1", "MMRR")

MODELS = (
    "SFCN-reg T1",
    "SFCN-reg GM",
    "SFCN-reg WM",
    "SFCN-reg DGM",
    "SFCN-sm T1",
    "SFCN-sm GM",
    "SFCN-sm WM",
    "SFCN-sm DGM",
)

MODEL_RGB = (
    (.13, .30, .52),  # SFCN-reg T1
    (.20, .49, .72),  # SFCN-reg GM
    (.35, .67, .65),  # SFCN-reg WM
    (.41, .40, .71),  # SFCN-reg DGM

    (.65, .18, .20),  # SFCN-sm T1
    (.87, .40, .30),  # SFCN-sm GM
    (.90, .61, .26),  # SFCN-sm WM
    (.71, .36, .64),  # SFCN-sm DGM
)
SUB_RGB = (
    (.35, .80, .45),
    (.35, .60, 1.00),
    (.95, .45, .90),
)

OPACITY = 0.45
SPECIAL_OPACITY = 0.60

REG_COLOR = "rgb(184,173,102)"
SPECIAL_COLOR = "rgb(255,3,26)"
MOTION_COLOR = "rgb(255,217,0)"

MMRR_SPECIAL = {"sub_502", "sub_742"}
MMRR_MOTION = "sub_742"


def rgb(color):
    return "rgb(" + ",".join(
        str(round(value * 255)) for value in color
    ) + ")"


COLORS = dict(zip(MODELS, map(rgb, MODEL_RGB)))

SUB_COLORS = dict(zip(
    ("sub-01", "sub-02", "sub-03"),
    map(rgb, SUB_RGB)
))


# ============================================================
# 2. Load Data
# ============================================================

def subject_id(dataset, scan_id):

    scan_id = str(scan_id)

    if dataset == "Maclaren":
        return scan_id.split("_run-")[0]

    if dataset == "OAS1":
        return re.sub(r"_MR\d+$", "", scan_id)

    match = re.match(r"sub_\d+", scan_id)
    return match.group(0) if match else scan_id


def model_file_and_sheet(model):

    method, image = model.split(" ")

    if method == "SFCN-sm":
        file = (
            "SFCN_sm_dgm.xlsx"
            if image == "DGM"
            else "SFCN_sm.xlsx"
        )
    else:
        file = (
            "SFCN_reg_dgm.xlsx"
            if image == "DGM"
            else "SFCN_reg.xlsx"
        )

    suffix = {
        "T1": "T1WI",
        "GM": "GMP",
        "WM": "WMP",
        "DGM": "dgm" if method == "SFCN-sm" else "DGM",
    }[image]

    return file, suffix


@st.cache_data
def load_data():

    frames = []

    for model in MODELS:

        file, suffix = model_file_and_sheet(model)

        for dataset in DATASETS:

            # Read Excel
            df = pd.read_excel(
                DATA / file,
                sheet_name=f"{dataset}_{suffix}"
            )

            required = [
                "Subject ID",
                "True Age",
                "Predicted Age",
                "pair",
                "Scanner",
                "Sex (1=Male, 2=Female)",
            ]

            missing = [
                col for col in required
                if col not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"{file} / {dataset}_{suffix} "
                    f"缺少欄位：{missing}"
                )

            df = df.dropna(subset=[
                "Subject ID",
                "True Age",
                "Predicted Age",
                "pair",
            ]).copy()

            # Basic information
            df["Scan ID"] = df["Subject ID"].astype(str)

            df["Subject"] = df["Scan ID"].map(
                lambda s: subject_id(dataset, s)
            )

            df["Dataset"] = dataset
            df["Model"] = model

            df["Pair"] = (
                df["pair"].astype(str)
                .str.replace(r"\.0$", "", regex=True)
                .str.strip()
            )

            df["True Age"] = pd.to_numeric(
                df["True Age"], errors="coerce"
            )

            df["Predicted Age"] = pd.to_numeric(
                df["Predicted Age"], errors="coerce"
            )

            df = df.dropna(
                subset=["True Age", "Predicted Age"]
            ).copy()

            df["AE"] = (
                df["True Age"] - df["Predicted Age"]
            ).abs()

            df = df.rename(columns={
                "True Age": "True",
                "Predicted Age": "Pred",
                "Sex (1=Male, 2=Female)": "Sex",
            })

            frames.append(df[[
                "Dataset", "Model", "Subject",
                "Scan ID", "Pair", "True", "Pred",
                "AE", "Sex", "Scanner"
            ]])

    # Combine all models and datasets
    scans = pd.concat(frames, ignore_index=True)

    # Test-Retest pairs
    paired = []

    for (dataset, model, subject, pair), group in scans.groupby(
        ["Dataset", "Model", "Subject", "Pair"],
        sort=False
    ):

        if len(group) != 2:
            continue

        a, b = group.iloc[0], group.iloc[1]

        paired.append({
            "Dataset": dataset,
            "Model": model,
            "Subject": subject,
            "Pair": pair,
            "True": a["True"],
            "Scan1": a["Scan ID"],
            "Scan2": b["Scan ID"],
            "Pred1": a["Pred"],
            "Pred2": b["Pred"],
            "MAD": abs(a["Pred"] - b["Pred"]),
        })

    pairs = pd.DataFrame(paired, columns=[
        "Dataset", "Model", "Subject", "Pair",
        "True", "Scan1", "Scan2",
        "Pred1", "Pred2", "MAD"
    ])

    return scans, pairs


# ============================================================
# 3. Point Details
# ============================================================

def details(fig, event, kind):

    points = event.selection.points if event else []

    if not points:
        st.caption("點擊資料點可查看詳細資訊。")
        return

    point = points[0]
    trace = fig.data[point["curve_number"]]

    if trace.customdata is None:
        return

    values = trace.customdata[point["point_index"]]

    if kind == "MAD":
        names = [
            "Subject", "Pair", "True Age",
            "Scan 1", "Scan 2",
            "Pred 1", "Pred 2", "Difference"
        ]
    else:
        names = [
            "Scan ID", "Subject", "True Age",
            "Predicted Age", "Absolute Error",
            "Sex", "Scanner"
        ]

    st.dataframe(
        pd.DataFrame({
            "Field": names,
            "Value": values
        }),
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# 4. MAD Distribution
# ============================================================

def mad_chart(pairs, dataset):

    data = pairs[pairs["Dataset"].eq(dataset)]

    fig = go.Figure()

    for i, model in enumerate(MODELS):

        part = data[data["Model"].eq(model)]

        if dataset == "Maclaren":

            groups = [
                (subject, part[part["Subject"].eq(subject)])
                for subject in SUB_COLORS
            ]

        else:
            groups = [(model, part)]

        for name, group in groups:

            if group.empty:
                continue

            if dataset == "Maclaren":

                s = list(SUB_COLORS).index(name)
                center = i + (s - 1) * 0.15

                x = center + np.linspace(
                    -0.025, 0.025, len(group)
                )

                color = SUB_COLORS[name]

            else:

                x = i + np.linspace(
                    -0.16, 0.16, len(group)
                )

                color = COLORS[model]

            custom = group[[
                "Subject", "Pair", "True",
                "Scan1", "Scan2",
                "Pred1", "Pred2", "MAD"
            ]].to_numpy()

            # MMRR motion
            if dataset == "MMRR":

                colors = [
                    MOTION_COLOR
                    if subject == MMRR_MOTION
                    else color
                    for subject in group["Subject"]
                ]

                symbols = [
                    "star"
                    if subject == MMRR_MOTION
                    else "circle"
                    for subject in group["Subject"]
                ]

                color = colors

            else:
                symbols = "circle"

            fig.add_trace(go.Scatter(
                x=x,
                y=group["MAD"],
                mode="markers",
                name=name,
                legendgroup=name,
                showlegend=(
                    i == 0 if dataset == "Maclaren"
                    else False
                ),
                customdata=custom,
                marker=dict(
                    color=color,
                    symbol=symbols,
                    size=9,
                    opacity=OPACITY,
                    line=dict(
                        color="rgb(120,120,120)",
                        width=1.2
                    )
                ),
                hovertemplate=(
                    f"{model}<br>"
                    "Subject: %{customdata[0]}<br>"
                    "Pair: %{customdata[1]}<br>"
                    "Scan 1: %{customdata[3]}<br>"
                    "Scan 2: %{customdata[4]}<br>"
                    "Difference: %{y:.2f} yr"
                    "<extra></extra>"
                )
            ))

        # Mean MAD
        mean = part["MAD"].mean()

        if pd.notna(mean):

            fig.add_shape(
                type="line",
                x0=i - 0.25,
                x1=i + 0.25,
                y0=mean,
                y1=mean,
                line=dict(
                    color="rgb(102,102,102)",
                    dash="dash",
                    width=2
                )
            )

            fig.add_annotation(
                x=i,
                y=mean + 0.45,
                text=f"{mean:.2f}",
                showarrow=False,
                font=dict(
                    size=11,
                    color="#444444"
                )
            )

    fig.update_layout(
        title=f"MAD - {dataset}",
        template="plotly_white",
        height=600,
        margin=dict(b=120),
        dragmode="select",
        clickmode="event+select",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(MODELS))),
            ticktext=MODELS,
            tickangle=-35,
            range=[-0.6, len(MODELS) - 0.4]
        ),
        yaxis=dict(
            title="|Scan 1 - Scan 2| (years)",
            range=[0, 12],
            dtick=2
        )
    )

    return fig


# ============================================================
# 5. MAE - 8 Models (2 x 4)
# ============================================================

def mae_chart(scans, dataset):

    data = scans[scans["Dataset"].eq(dataset)]

    lo = min(data["True"].min(), data["Pred"].min()) - 1
    hi = max(data["True"].max(), data["Pred"].max()) + 1

    ticks = np.arange(
        np.ceil(lo / 5) * 5,
        np.floor(hi / 5) * 5 + 1,
        5
    )

    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=MODELS,
        horizontal_spacing=0.07,
        vertical_spacing=0.16
    )

    for i, model in enumerate(MODELS):

        row = i // 4 + 1
        col = i % 4 + 1

        part = data[data["Model"].eq(model)]

        if dataset == "Maclaren":

            groups = [
                (
                    subject,
                    part[part["Subject"].eq(subject)],
                    SUB_COLORS[subject]
                )
                for subject in SUB_COLORS
            ]

        else:

            if dataset == "MMRR":

                special = part["Subject"].isin(
                    MMRR_SPECIAL
                )

            else:

                special = pd.Series(
                    False,
                    index=part.index
                )

            groups = [
                (model, part[~special], COLORS[model]),
                ("Special", part[special], SPECIAL_COLOR)
            ]

        for name, group, color in groups:

            if group.empty:
                continue

            custom = group[[
                "Scan ID", "Subject", "True",
                "Pred", "AE", "Sex", "Scanner"
            ]].to_numpy()

            opacity = (
                SPECIAL_OPACITY
                if name == "Special"
                else OPACITY
            )

            fig.add_trace(
                go.Scatter(
                    x=group["True"],
                    y=group["Pred"],
                    mode="markers",
                    name=name,
                    showlegend=False,
                    customdata=custom,
                    marker=dict(
                        color=color,
                        size=7,
                        opacity=opacity
                    ),
                    hovertemplate=(
                        "Scan: %{customdata[0]}<br>"
                        "Subject: %{customdata[1]}<br>"
                        "True Age: %{x:.2f}<br>"
                        "Predicted Age: %{y:.2f}<br>"
                        "AE: %{customdata[4]:.2f}<br>"
                        "Sex: %{customdata[5]}<br>"
                        "Scanner: %{customdata[6]}"
                        "<extra></extra>"
                    )
                ),
                row=row,
                col=col
            )

        # y = x
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
                line=dict(
                    color="gray",
                    dash="dash",
                    width=1.2
                )
            ),
            row=row,
            col=col
        )

        # Regression line for MMRR
        if dataset == "MMRR" and part["True"].nunique() > 1:

            slope, intercept = np.polyfit(
                part["True"],
                part["Pred"],
                1
            )

            fig.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[
                        slope * lo + intercept,
                        slope * hi + intercept
                    ],
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                    line=dict(
                        color=REG_COLOR,
                        width=1.5
                    )
                ),
                row=row,
                col=col
            )

        # Axis settings
        fig.update_xaxes(
            range=[lo, hi],
            tickvals=ticks,
            showgrid=True,
            title_text="True Age (yr)",
            row=row,
            col=col
        )

        fig.update_yaxes(
            range=[lo, hi],
            tickvals=ticks,
            showgrid=True,
            title_text="Predicted Age (yr)",
            row=row,
            col=col
        )

        # MAE and sample size
        mae = part["AE"].mean()
        n = len(part)

        fig.layout.annotations[i].text = (
            f"{model}<br>MAE: {mae:.2f} | N={n}"
        )

    fig.update_layout(
        title=f"MAE - {dataset}",
        template="plotly_white",
        width=1600,
        height=780,
        autosize=False,
        showlegend=False,
        dragmode="select",
        clickmode="event+select",
        margin=dict(
            l=55,
            r=30,
            t=110,
            b=50
        )
    )

    return fig


# ============================================================
# 6. Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Brain Age Test-Retest",
    layout="wide"
)

st.title("Brain Age Test-Retest Dashboard")

try:
    scans, pairs = load_data()

except (
    ValueError,
    KeyError,
    FileNotFoundError
) as exc:

    st.error(f"Data loading failed: {exc}")
    st.stop()


mad_tab, mae_tab = st.tabs([
    "MAD - Three Datasets",
    "MAE - 8 Models"
])


# MAD
with mad_tab:

    for dataset in DATASETS:

        fig = mad_chart(pairs, dataset)

        event = st.plotly_chart(
            fig,
            width="stretch",
            key=f"mad_{dataset}",
            on_select="rerun",
            selection_mode="points"
        )

        details(fig, event, "MAD")

        summary = (
            pairs[pairs["Dataset"].eq(dataset)]
            .groupby("Model")["MAD"]
            .agg(N_Pairs="count", MAD_yr="mean")
            .reindex(MODELS)
            .reset_index()
        )

        st.dataframe(
            summary.round(3),
            hide_index=True,
            use_container_width=True
        )

        st.divider()


# MAE
with mae_tab:

    dataset = st.selectbox(
        "Dataset",
        DATASETS,
        key="mae_dataset"
    )

    fig = mae_chart(scans, dataset)

    event = st.plotly_chart(
        fig,
        width="stretch",
        key=f"mae_{dataset}",
        on_select="rerun",
        selection_mode="points"
    )

    details(fig, event, "MAE")

    summary = (
        scans[scans["Dataset"].eq(dataset)]
        .groupby("Model")["AE"]
        .agg(N="count", MAE_yr="mean")
        .reindex(MODELS)
        .reset_index()
    )

    st.dataframe(
        summary.round(3),
        hide_index=True,
        use_container_width=True
    )
