from __future__ import annotations

"""
Final one-time test for the frozen wildfire LightGBM model.

This script does only one thing:
    Evaluate the already-frozen model on the reserved 2023-2025 test period.

It does NOT:
    - retrain the model
    - tune hyperparameters
    - change features
    - recalibrate probabilities
    - change the decision threshold
    - perform model selection

Required files:
    FullDataSet3.csv
    artifacts/final_development/FINAL_FROZEN_SPECIFICATION.json
    artifacts/final_development/chosen_development_probability_model.joblib

Outputs:
    artifacts/final_test/final_test_metrics.json
    artifacts/final_test/final_test_predictions.csv
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wildfire_config import (
    ARTIFACTS_DIR,
    DATASET_PATH,
    DATE_COL,
    MISSING_RATE_COLUMNS,
    TARGET,
    YEAR_COL,
)
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
)


# =============================================================================
# 1. Paths
# =============================================================================

DATA_PATH = DATASET_PATH
DEVELOPMENT_DIR = ARTIFACTS_DIR / "final_development"
FINAL_SPEC_PATH = DEVELOPMENT_DIR / "FINAL_FROZEN_SPECIFICATION.json"
MODEL_PATH = DEVELOPMENT_DIR / "chosen_development_probability_model.joblib"
OUTPUT_DIR = ARTIFACTS_DIR / "final_test"


# =============================================================================
# 2. Small utilities
# =============================================================================

def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(json_safe(obj), indent=2))


# =============================================================================
# 3. Load the frozen specification
# =============================================================================

def load_frozen_specification() -> dict:
    if not FINAL_SPEC_PATH.exists():
        raise FileNotFoundError(
            f"Frozen specification not found: {FINAL_SPEC_PATH}"
        )

    spec = json.loads(FINAL_SPEC_PATH.read_text())

    if spec.get("status") != "FROZEN_PRE_TEST_SPECIFICATION":
        raise ValueError(
            "The specification is not marked as FROZEN_PRE_TEST_SPECIFICATION."
        )

    required_keys = [
        "features",
        "pipeline",
        "calibration_method",
        "threshold",
    ]

    missing = [key for key in required_keys if key not in spec]
    if missing:
        raise KeyError(f"Frozen specification is missing: {missing}")

    return spec


# =============================================================================
# 4. Load the reserved 2023-2025 test data
# =============================================================================

def load_test_data(features: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="raise")

    missing_rate_inputs = sorted(
        set(MISSING_RATE_COLUMNS) - set(df.columns)
    )
    if missing_rate_inputs:
        raise KeyError(
            "Cannot create row_weather_missing_rate. Missing columns: "
            f"{missing_rate_inputs}"
        )

    df["row_weather_missing_rate"] = (
        df[MISSING_RATE_COLUMNS].mean(axis=1)
    )

    missing_features = sorted(set(features) - set(df.columns))
    if missing_features:
        raise KeyError(
            f"Frozen test features are missing from the dataset: {missing_features}"
        )

    test = (
        df.loc[df[YEAR_COL].between(2023, 2025)]
        .sort_values([DATE_COL, "US_L3NAME"])
        .reset_index(drop=True)
    )

    if test.empty:
        raise ValueError("The reserved 2023-2025 test set is empty.")

    # Guardrail: this file must evaluate only the reserved test years.
    observed_years = sorted(test[YEAR_COL].unique().tolist())
    if observed_years != [2023, 2024, 2025]:
        raise ValueError(
            f"Expected test years [2023, 2024, 2025], found {observed_years}"
        )

    # Keep categorical type compatible with development.
    if "US_L3NAME" in features:
        test["US_L3NAME"] = test["US_L3NAME"].astype("object")

    X_test = test.loc[:, features].copy()
    y_test = test[TARGET].astype(int).copy()

    if y_test.nunique() != 2:
        raise ValueError("Test target must contain both classes.")

    return X_test, y_test, test


# =============================================================================
# 5. Evaluate the frozen model once
# =============================================================================

def evaluate_test(
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: float,
) -> dict:
    prediction = (probability >= threshold).astype(int)

    prevalence = float(y_true.mean())
    ap = float(average_precision_score(y_true, probability))

    return {
        "n_rows": int(len(y_true)),
        "n_fire": int(y_true.sum()),
        "prevalence": prevalence,
        "average_precision": ap,
        "ap_over_prevalence": ap / prevalence,
        "brier_score": float(
            brier_score_loss(y_true, probability)
        ),
        "log_loss": float(
            log_loss(y_true, probability, labels=[0, 1])
        ),
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "f2": float(
            fbeta_score(
                y_true,
                prediction,
                beta=2,
                zero_division=0,
            )
        ),
        "predicted_positive_count": int(prediction.sum()),
    }


# =============================================================================
# 6. Main
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = load_frozen_specification()

    features = list(spec["features"])
    threshold = float(spec["threshold"])

    print("============================================================")
    print("FINAL RESERVED TEST: 2023-2025")
    print("============================================================")
    print(f"Frozen pipeline:     {spec['pipeline']}")
    print(f"Calibration:         {spec['calibration_method']}")
    print(f"Frozen threshold:    {threshold:.8f}")
    print(f"Number of features:  {len(features)}")
    print()
    print("No tuning, retraining, recalibration, or threshold changes")
    print("are performed in this script.")
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Frozen probability model not found: {MODEL_PATH}"
        )

    frozen_probability_model = joblib.load(MODEL_PATH)

    X_test, y_test, test = load_test_data(features)

    probability = frozen_probability_model.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)

    metrics = evaluate_test(
        y_true=y_test,
        probability=probability,
        threshold=threshold,
    )

    # Save row-level predictions so later error analysis can be performed
    # without rerunning or modifying the final test model.
    prediction_columns = [
        column
        for column in [
            DATE_COL,
            YEAR_COL,
            "US_L3CODE",
            "US_L3NAME",
        ]
        if column in test.columns
    ]

    predictions = test.loc[:, prediction_columns].copy()
    predictions["y_true"] = y_test.to_numpy(dtype=int)
    predictions["fire_probability"] = probability
    predictions["predicted_fire"] = prediction

    predictions.to_csv(
        OUTPUT_DIR / "final_test_predictions.csv",
        index=False,
    )

    final_output = {
        "status": "FINAL_TEST_COMPLETE",
        "test_period": "2023-2025",
        "pipeline": spec["pipeline"],
        "calibration_method": spec["calibration_method"],
        "threshold_rule": spec.get("threshold_rule"),
        "threshold": threshold,
        "features": features,
        "metrics": metrics,
        "warning": (
            "These are the final reserved-test results. "
            "Do not modify the model specification based on them."
        ),
    }

    save_json(
        OUTPUT_DIR / "final_test_metrics.json",
        final_output,
    )

    print("FINAL TEST RESULTS")
    print("------------------------------------------------------------")
    print(f"Rows:                  {metrics['n_rows']:,}")
    print(f"Fire observations:     {metrics['n_fire']:,}")
    print(f"Prevalence:            {metrics['prevalence']:.6f}")
    print(f"Average precision:     {metrics['average_precision']:.6f}")
    print(f"AP / prevalence:       {metrics['ap_over_prevalence']:.6f}")
    print(f"Brier score:           {metrics['brier_score']:.6f}")
    print(f"Log loss:              {metrics['log_loss']:.6f}")
    print(f"Precision:             {metrics['precision']:.6f}")
    print(f"Recall:                {metrics['recall']:.6f}")
    print(f"F1:                    {metrics['f1']:.6f}")
    print(f"F2:                    {metrics['f2']:.6f}")
    print()
    print("Saved:")
    print(OUTPUT_DIR / "final_test_metrics.json")
    print(OUTPUT_DIR / "final_test_predictions.csv")
    print()
    print("FINAL TEST COMPLETE.")
    print("Do not retune or alter the frozen pipeline using these results.")


if __name__ == "__main__":
    main()
