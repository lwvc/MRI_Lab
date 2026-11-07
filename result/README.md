# Brain Age Test-Retest Dashboard

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The three Excel inputs live in `data/`. You may replace them with new results while keeping the same sheet names and fields.

## Rules

- MAD: `abs(Scan1 predicted age - Scan2 predicted age)`, mean across scan pairs for each dataset/model.
- Maclaren: subject + pair (20 consecutive two-scan pairs for each subject).
- MMRR: subject + pair; OAS1: subject + pair.
- MAE: mean across all available scan-level absolute errors.
- Transparency: 35% transparent (Plotly opacity 0.65).
- All points preserve ID, age, and metadata in their hover and click details.
- MAE axes use the same range for every model within a dataset, computed across true/predicted ages ±1 year.
- Regression line only for MMRR. Flagged subjects are highlighted but are **not excluded** from mean or regression.

Click a data point to populate the details below each plot. Depending on the Streamlit version, this uses Plotly point selection (`on_select='rerun'`).

**Privacy:** Repository includes subject-level data. Only make it public if the sharing terms and consent permit publication; otherwise use a private repository/deployment.
