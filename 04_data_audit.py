"""Run the dataset checks used before feature ablation.

The goal here is not model selection. It checks panel integrity, split-level
prevalence, missingness, temporal drift, and strong feature redundancy. The
2023-2025 period is used only for descriptive dataset diagnostics in this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from wildfire_config import ARTIFACTS_DIR, DATASET_PATH, MISSING_RATE_COLUMNS

OUTPUT_DIR = ARTIFACTS_DIR / "audit"


def calculate_psi(reference: pd.Series, comparison: pd.Series, n_bins: int = 10) -> float:
    """Population Stability Index using reference-defined quantile bins."""
    reference = reference.dropna()
    comparison = comparison.dropna()
    if reference.empty or comparison.empty:
        return np.nan

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cmp_counts, _ = np.histogram(comparison, bins=edges)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-4, None)
    cmp_pct = np.clip(cmp_counts / cmp_counts.sum(), 1e-4, None)
    return float(np.sum((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)))


def split_summary(name: str, part: pd.DataFrame) -> dict:
    return {
        "split": name,
        "rows": len(part),
        "fires": int(part["fire_occurred"].sum()),
        "prevalence": float(part["fire_occurred"].mean()),
        "start": str(part["date"].min().date()),
        "end": str(part["date"].max().date()),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    duplicate_rows = int(df.duplicated().sum())
    duplicate_panel_keys = int(df.duplicated(["date", "US_L3NAME"]).sum())
    if duplicate_panel_keys:
        raise ValueError(f"Found {duplicate_panel_keys} duplicate ecoregion-date rows.")

    rows_per_date = df.groupby("date").size()
    if not rows_per_date.eq(13).all():
        raise ValueError("The dataset does not contain exactly 13 ecoregions for every date.")

    train = df.loc[df["year"].between(2013, 2020)].copy()
    validation = df.loc[df["year"].between(2021, 2022)].copy()
    test = df.loc[df["year"].between(2023, 2025)].copy()

    pd.DataFrame([
        split_summary("train", train),
        split_summary("validation", validation),
        split_summary("test", test),
    ]).to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    missingness = pd.DataFrame({
        "train": train.isna().mean(),
        "validation": validation.isna().mean(),
        "test": test.isna().mean(),
    }).sort_values("train", ascending=False)
    missingness.to_csv(OUTPUT_DIR / "missingness_by_split.csv")

    numeric = train.select_dtypes(include=[np.number]).columns
    drift_rows = []
    for column in numeric:
        if column in {"fire_occurred", "fire_count"}:
            continue
        drift_rows.append({
            "feature": column,
            "psi_train_to_validation": calculate_psi(train[column], validation[column]),
            "psi_train_to_test": calculate_psi(train[column], test[column]),
        })
    pd.DataFrame(drift_rows).sort_values(
        "psi_train_to_validation", ascending=False
    ).to_csv(OUTPUT_DIR / "feature_drift_psi.csv", index=False)

    features = train.select_dtypes(include=[np.number]).drop(
        columns=["fire_occurred", "fire_count", "US_L3CODE"], errors="ignore"
    )
    corr = features.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "abs_spearman"})
        .query("abs_spearman >= 0.85")
        .sort_values("abs_spearman", ascending=False)
    )
    high_corr.to_csv(OUTPUT_DIR / "high_spearman_pairs.csv", index=False)

    available_missing = [c for c in MISSING_RATE_COLUMNS if c in train.columns]
    if available_missing:
        row_missing = train[available_missing].mean(axis=1)
        pd.DataFrame({
            "row_weather_missing_rate": row_missing,
            "fire_occurred": train["fire_occurred"].to_numpy(),
        }).to_csv(OUTPUT_DIR / "training_row_missingness.csv", index=False)

    print("Dataset audit complete")
    print(f"Rows: {len(df):,}")
    print(f"Duplicate rows: {duplicate_rows}")
    print("Duplicate ecoregion-date rows: 0")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Saved audit tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
