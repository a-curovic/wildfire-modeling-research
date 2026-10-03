"""Paired temporal block bootstrap for the preprocessing finalists.

The comparison uses out-of-fold probabilities from the imputer screen so every
model is evaluated on the same rows.
"""

import pandas as pd
import numpy as np
from math import ceil
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from wildfire_config import ARTIFACTS_DIR, RANDOM_STATE

OUTPUT_DIR = ARTIFACTS_DIR / "screening"

def calculate_paired_metric_differences(sample: pd.DataFrame, model_a: str, model_b: str) -> dict[str, float]:

    y_true = sample["y_true"].to_numpy()

    probability_a = sample[model_a].to_numpy()

    probability_b = sample[model_b].to_numpy()

    if np.unique(y_true).size < 2:
        raise ValueError("Bootstrap sample contains one target class.")

    return {
        "ap_difference": float(average_precision_score(y_true, probability_a) - average_precision_score(y_true, probability_b)),

        "brier_difference": float(brier_score_loss(y_true, probability_a) - brier_score_loss(y_true, probability_b)),

        "log_loss_difference": float(log_loss(y_true, probability_a, labels=[0, 1]) - log_loss(y_true, probability_b, labels=[0, 1])),
    }


def sample_date_blocks(fold_data: pd.DataFrame, block_size_days: int, rng: np.random.Generator) -> pd.DataFrame:

    fold_data = fold_data.sort_values(["date", "US_L3NAME"]).reset_index(drop=True).copy()
    
    fold_data["date"] = fold_data["date"].dt.normalize()

    date_values = fold_data["date"].to_numpy(dtype="datetime64[ns]")

    dates = np.unique(date_values)
    n_dates = len(dates)

    if n_dates == 0:
        raise ValueError("Validation fold has no dates.")

    positions_by_date = {
        date: np.flatnonzero(date_values == date) for date in dates
    }

    assert all(len(positions) == 13 for positions in positions_by_date.values()), "Every sampled date must contain 13 ecoregions."

    block_size = min(block_size_days, n_dates)

    n_blocks = ceil(n_dates / block_size)

    maximum_start = n_dates - block_size

    starts = rng.integers(low=0, high=maximum_start + 1, size=n_blocks)

    sampled_dates = np.concatenate([dates[start:start + block_size] for start in starts])[:n_dates]

    sampled_positions = np.concatenate([positions_by_date[date] for date in sampled_dates])

    return fold_data.iloc[sampled_positions].reset_index(drop=True)


def paired_temporal_block_bootstrap(
    paired_predictions: pd.DataFrame,
    model_a: str,
    model_b: str,
    block_size_days: int = 30,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(
        random_state
    )

    metrics = [
        "ap_difference",
        "brier_difference",
        "log_loss_difference",
    ]

    fold_groups = {
        fold: fold_data.copy()
        for fold, fold_data
        in paired_predictions.groupby(
            "fold",
            sort=True,
        )
    }

    # Original paired estimates, calculated within fold.
    original_fold_results = []

    for fold, fold_data in fold_groups.items():
        differences = (
            calculate_paired_metric_differences(
                fold_data,
                model_a,
                model_b,
            )
        )

        differences["fold"] = fold
        differences["weight"] = len(fold_data)

        original_fold_results.append(
            differences
        )

    original_fold_results = pd.DataFrame(
        original_fold_results
    )

    point_estimates = {
        metric: float(
            np.average(
                original_fold_results[metric],
                weights=original_fold_results[
                    "weight"
                ],
            )
        )
        for metric in metrics
    }

    bootstrap_rows = []
    valid_replicates = 0
    attempts = 0
    maximum_attempts = n_bootstrap * 5

    while (
        valid_replicates < n_bootstrap
        and attempts < maximum_attempts
    ):
        attempts += 1
        sampled_fold_results = []
        valid = True

        for fold, fold_data in fold_groups.items():
            sample = sample_date_blocks(
                fold_data,
                block_size_days,
                rng,
            )

            if sample["y_true"].nunique() < 2:
                valid = False
                break

            differences = (
                calculate_paired_metric_differences(
                    sample,
                    model_a,
                    model_b,
                )
            )

            differences["weight"] = len(sample)
            sampled_fold_results.append(
                differences
            )

        if not valid:
            continue

        sampled_fold_results = pd.DataFrame(
            sampled_fold_results
        )

        bootstrap_row = {
            "bootstrap_iteration":
                valid_replicates + 1
        }

        for metric in metrics:
            bootstrap_row[metric] = float(
                np.average(
                    sampled_fold_results[metric],
                    weights=sampled_fold_results[
                        "weight"
                    ],
                )
            )

        bootstrap_rows.append(
            bootstrap_row
        )

        valid_replicates += 1

    if valid_replicates < n_bootstrap:
        raise RuntimeError(
            f"Only obtained {valid_replicates} "
            f"valid replicates."
        )

    draws = pd.DataFrame(
        bootstrap_rows
    )

    alpha = 1 - confidence_level

    summary_rows = []

    for metric in metrics:
        values = draws[metric].to_numpy()

        summary_rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "block_size_days": block_size_days,
            "n_bootstrap": n_bootstrap,
            "point_estimate":
                point_estimates[metric],
            "bootstrap_mean":
                float(values.mean()),
            "bootstrap_standard_error":
                float(values.std(ddof=1)),
            "ci_lower":
                float(
                    np.quantile(
                        values,
                        alpha / 2,
                    )
                ),
            "ci_upper":
                float(
                    np.quantile(
                        values,
                        1 - alpha / 2,
                    )
                ),
            "probability_above_zero":
                float(np.mean(values > 0)),
            "probability_below_zero":
                float(np.mean(values < 0)),
        })

    return (
        pd.DataFrame(summary_rows),
        draws,
    )


def build_paired_prediction_table(oof_predictions: pd.DataFrame) -> pd.DataFrame:

    prediction_keys = [
        "fold",
        "row_id",
        "date",
        "US_L3NAME",
        "y_true",
    ]

    duplicates = oof_predictions.duplicated(
        prediction_keys + ["pipeline"],
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            "Duplicate pipeline predictions found."
        )

    paired = (
        oof_predictions
        .pivot(
            index=prediction_keys,
            columns="pipeline",
            values="probability",
        )
        .reset_index()
    )

    paired.columns.name = None

    required = {
        "extra_iterative__smote",
        "native__none",
        "rf_iterative__none",
        "median__smote",
    }

    missing = required - set(
        paired.columns
    )

    if missing:
        raise KeyError(
            f"Missing OOF pipelines: {sorted(missing)}"
        )

    assert paired[list(required)].notna().all().all()

    rows_per_fold_date = (
        paired
        .groupby(["fold", "date"])
        .size()
    )

    assert rows_per_fold_date.eq(13).all(), (
        "OOF table does not contain 13 ecoregions "
        "for every fold-date."
    )

    return paired


def main():

    oof_predictions = pd.read_csv(
        OUTPUT_DIR / "imputer_screen_oof_predictions.csv",
        parse_dates=["date"],
    )

    paired_predictions = build_paired_prediction_table(
        oof_predictions
    )

    paired_predictions.to_csv(
        OUTPUT_DIR / "finalist_paired_predictions.csv",
        index=False,
    )

    comparisons = [
        (
            "extra_smote_minus_native",
            "extra_iterative__smote",
            "native__none",
        ),
        (
            "extra_smote_minus_median_smote",
            "extra_iterative__smote",
            "median__smote",
        ),
        (
            "rf_none_minus_native",
            "rf_iterative__none",
            "native__none",
        ),
    ]

    summaries = []
    draws = []

    for comparison_name, model_a, model_b in comparisons:

        summary, bootstrap_draws = (
            paired_temporal_block_bootstrap(
                paired_predictions=paired_predictions,
                model_a=model_a,
                model_b=model_b,
                block_size_days=7,
                n_bootstrap=2000,
                confidence_level=0.95,
                random_state=RANDOM_STATE,
            )
        )

        summary["comparison"] = comparison_name
        bootstrap_draws["comparison"] = comparison_name

        summaries.append(summary)
        draws.append(bootstrap_draws)

    summary_df = pd.concat(
        summaries,
        ignore_index=True,
    )

    draws_df = pd.concat(
        draws,
        ignore_index=True,
    )
    print("Saving bootstrap summary")
    summary_df.to_csv(
        OUTPUT_DIR / "paired_block_bootstrap_7_summary.csv",
        index=False,
    )
    print("Done with summary")

    draws_df.to_csv(
        OUTPUT_DIR / "paired_block_bootstrap_7_draws.csv",
        index=False,
    )

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()