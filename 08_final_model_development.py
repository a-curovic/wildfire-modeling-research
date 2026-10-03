from __future__ import annotations

"""
Final LightGBM development for the frozen wildfire methodology.

Purpose
-------
This file starts AFTER feature, sampler, and imputer screening are finished.
It performs only the remaining pre-test development stages:

1. Tune the three frozen finalist pipelines on 2013-2020 using date-based CV.
2. Compare the tuned finalists on the held-out 2021-2022 validation period.
3. For the selected pipeline, fit calibration on 2021 and compare
   uncalibrated / sigmoid / isotonic probabilities on 2022.
4. Choose the final F2 threshold on 2022.
5. Save a frozen model specification for the separate final-test script.

IMPORTANT
---------
* This script NEVER uses 2023-2025 for model selection or evaluation.
* Average precision is the primary model-selection metric.
* Brier score and log loss are secondary probability-quality metrics.
* The frozen 36-feature set is not changed here.
* SMOTE is implemented with SMOTENC because US_L3NAME is categorical.
* The final 2023-2025 test belongs in a separate _09_final_test.py file.
"""

import json
import shutil
from pathlib import Path

from wildfire_transformer import SkewAwareScaler
import joblib
import numpy as np
import pandas as pd

from wildfire_config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    DATE_COL,
    MISSING_RATE_COLUMNS,
    PRIMARY_FEATURES,
    RANDOM_STATE,
    TARGET,
    YEAR_COL,
)
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from joblib import Memory
from lightgbm import LGBMClassifier
from scipy.stats import randint, loguniform, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.frozen import FrozenEstimator
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OrdinalEncoder


# =============================================================================
# 1. Configuration
# =============================================================================

DATA_PATH = DATASET_PATH
OUTPUT_DIR = ARTIFACTS_DIR / "final_development"
CACHE_DIR = OUTPUT_DIR / "cache"

N_SPLITS = 4
N_SEARCH_ITERATIONS = 60
SEARCH_N_JOBS = 4
SAMPLING_STRATEGY = 0.25

# Existing tuning results can be reused when all expected artifacts are present.
REUSE_EXISTING_TUNING = True

FINALISTS: list[tuple[str, str]] = [
    ("native", "none"),
    ("median", "smote"),
    ("extra_iterative", "smote"),
]


# =============================================================================
# 2. Small utilities
# =============================================================================


def pipeline_name(imputer_name: str, imbalance_name: str) -> str:
    return f"{imputer_name}__{imbalance_name}"


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(json_safe(obj), indent=2))


def evaluate_probabilities(y_true: pd.Series, probability: np.ndarray) -> dict[str, float]:
    prevalence = float(y_true.mean())
    ap = float(average_precision_score(y_true, probability))
    return {
        "average_precision": ap,
        "prevalence": prevalence,
        "ap_over_prevalence": ap / prevalence,
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def choose_threshold_by_f2(
    y_true: pd.Series,
    probability: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)

    f2 = (
        5.0 * precision[:-1] * recall[:-1]
        / (4.0 * precision[:-1] + recall[:-1] + 1e-12)
    )

    best_idx = int(np.nanargmax(f2))
    best_threshold = float(thresholds[best_idx])

    curve = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
            "f2": f2,
        }
    )
    return best_threshold, curve


def evaluate_threshold(
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "f2": float(fbeta_score(y_true, prediction, beta=2, zero_division=0)),
    }


# =============================================================================
# 3. Data loading and chronological splits
# =============================================================================


def load_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Return train (2013-2020) and validation (2021-2022) only."""

    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="raise")

    missing_inputs = sorted(set(MISSING_RATE_COLUMNS) - set(df.columns))
    if missing_inputs:
        raise KeyError(
            "Cannot create row_weather_missing_rate. Missing columns: "
            f"{missing_inputs}"
        )

    df["row_weather_missing_rate"] = df[MISSING_RATE_COLUMNS].mean(axis=1)

    if len(PRIMARY_FEATURES) != 36 or len(set(PRIMARY_FEATURES)) != 36:
        raise ValueError("PRIMARY_FEATURES must contain exactly 36 unique features.")

    missing_features = sorted(set(PRIMARY_FEATURES) - set(df.columns))
    if missing_features:
        raise KeyError(f"Frozen features missing from dataset: {missing_features}")

    # Explicit chronological development partitions.
    train = (
        df.loc[df[YEAR_COL].between(2013, 2020)]
        .sort_values([DATE_COL, "US_L3NAME"])
        .reset_index(drop=True)
    )
    validation = (
        df.loc[df[YEAR_COL].between(2021, 2022)]
        .sort_values([DATE_COL, "US_L3NAME"])
        .reset_index(drop=True)
    )

    if train.empty or validation.empty:
        raise ValueError("Training or validation partition is empty.")

    # Guardrail: this development script must never contain test years.
    if train[YEAR_COL].max() > 2020 or validation[YEAR_COL].max() > 2022:
        raise RuntimeError("A post-2022 row entered final development.")

    for part in (train, validation):
        for column in CATEGORICAL_FEATURES:
            part[column] = part[column].astype("object")

    X_train = train.loc[:, PRIMARY_FEATURES].copy()
    y_train = train[TARGET].astype(int).copy()
    dates_train = train[DATE_COL].copy()

    X_val = validation.loc[:, PRIMARY_FEATURES].copy()
    y_val = validation[TARGET].astype(int).copy()
    dates_val = validation[DATE_COL].copy()

    if y_train.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError("Both train and validation must contain both target classes.")

    return X_train, y_train, dates_train, X_val, y_val, dates_val


def make_date_based_splits(
    dates: pd.Series,
    n_splits: int = N_SPLITS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create TimeSeriesSplit folds on UNIQUE dates, then map back to rows."""

    dates = pd.to_datetime(dates).reset_index(drop=True)
    unique_dates = np.array(sorted(dates.unique()))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    row_splits: list[tuple[np.ndarray, np.ndarray]] = []

    for train_date_idx, val_date_idx in splitter.split(unique_dates):
        train_dates = unique_dates[train_date_idx]
        val_dates = unique_dates[val_date_idx]

        train_rows = np.flatnonzero(dates.isin(train_dates).to_numpy())
        val_rows = np.flatnonzero(dates.isin(val_dates).to_numpy())

        if not set(dates.iloc[train_rows]).isdisjoint(set(dates.iloc[val_rows])):
            raise RuntimeError("The same calendar date appears on both sides of a CV fold.")

        row_splits.append((train_rows, val_rows))

    return row_splits


# =============================================================================
# 4. Frozen preprocessing pipelines
# =============================================================================

def make_numeric_pipeline(imputer_name: str, scale_for_sampling: bool):
    if imputer_name == "native":
        if scale_for_sampling:
            raise ValueError("SMOTENC requires complete numeric data.")
        return "passthrough"

    if imputer_name == "median":
        imputer = SimpleImputer(strategy="median")
    elif imputer_name == "extra_iterative":
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=15,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            max_iter=10,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown finalist imputer: {imputer_name}")

    steps = [("imputer", imputer)]
    if scale_for_sampling:
        steps.append(("scale_before_sampling", SkewAwareScaler()))

    return SklearnPipeline(steps)


def make_preprocessor(
    feature_names: list[str],
    imputer_name: str,
    scale_for_sampling: bool,
) -> tuple[ColumnTransformer, list[int]]:
    categorical = [x for x in CATEGORICAL_FEATURES if x in feature_names]
    numeric = [x for x in feature_names if x not in categorical]

    numeric_pipeline = make_numeric_pipeline(imputer_name, scale_for_sampling)

    categorical_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    dtype=np.int64,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

    categorical_indices = list(range(len(numeric), len(numeric) + len(categorical)))
    return preprocessor, categorical_indices


def build_pipeline(
    imputer_name: str,
    imbalance_name: str,
    feature_names: list[str],
    memory: Memory | None = None,
) -> tuple[Pipeline, list[int]]:
    if (imputer_name, imbalance_name) not in FINALISTS:
        raise ValueError(f"Pipeline is not a frozen finalist: {(imputer_name, imbalance_name)}")

    scale_for_sampling = imbalance_name == "smote"
    preprocessor, categorical_indices = make_preprocessor(
        feature_names,
        imputer_name,
        scale_for_sampling,
    )

    if imbalance_name == "none":
        sampler = "passthrough"
    elif imbalance_name == "smote":
        sampler = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=SAMPLING_STRATEGY,
            random_state=RANDOM_STATE,
            k_neighbors=5,
        )
    else:
        raise ValueError(f"Unknown finalist imbalance method: {imbalance_name}")

    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=1,  # outer RandomizedSearchCV handles parallelism
        verbosity=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("sampler", sampler),
            ("model", model),
        ],
        memory=memory,
    )

    return pipeline, categorical_indices


# =============================================================================
# 5. Stage 1: tune the three finalists on 2013-2020
# =============================================================================


def tuning_search_space() -> dict:
    return {
        "model__n_estimators": randint(150, 801),
        "model__learning_rate": loguniform(0.01, 0.15),
        "model__num_leaves": randint(15, 65),
        "model__max_depth": [-1, 4, 5, 6, 7, 8, 10, 12],
        "model__min_child_samples": randint(20, 151),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_alpha": loguniform(1e-4, 10),
        "model__reg_lambda": loguniform(1e-4, 10),
    }


def tune_finalist(
    imputer_name: str,
    imbalance_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> Pipeline:
    name = pipeline_name(imputer_name, imbalance_name)

    model_path = OUTPUT_DIR / f"best_estimator_{name}.joblib"
    params_path = OUTPUT_DIR / f"best_params_{name}.json"
    cv_path = OUTPUT_DIR / f"cv_results_{name}.csv"

    if (
        REUSE_EXISTING_TUNING
        and model_path.exists()
        and params_path.exists()
        and cv_path.exists()
    ):
        print(f"Loading existing tuning result: {name}", flush=True)
        return joblib.load(model_path)

    print(f"\nTUNING: {name}", flush=True)

    cache_path = CACHE_DIR / name
    memory = Memory(location=cache_path, verbose=0)

    pipeline, categorical_indices = build_pipeline(
        imputer_name,
        imbalance_name,
        X_train.columns.tolist(),
        memory=memory,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=tuning_search_space(),
        n_iter=N_SEARCH_ITERATIONS,
        scoring={
            "average_precision": "average_precision",
            "brier": "neg_brier_score",
            "log_loss": "neg_log_loss",
        },
        refit="average_precision",
        cv=cv_splits,
        n_jobs=SEARCH_N_JOBS,
        verbose=2,
        random_state=RANDOM_STATE,
        error_score="raise",
        return_train_score=True,
    )

    search.fit(
        X_train,
        y_train,
        model__categorical_feature=categorical_indices,
    )

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(cv_path, index=False)
    save_json(params_path, search.best_params_)

    # Remove the runtime cache from the estimator before saving it.
    best_estimator = search.best_estimator_
    best_estimator.memory = None
    joblib.dump(best_estimator, model_path)

    print(f"Best CV AP: {search.best_score_:.6f}", flush=True)
    print(f"Best params: {search.best_params_}", flush=True)

    # Cache is only a speed aid; it is not a research artifact.
    if cache_path.exists():
        shutil.rmtree(cache_path, ignore_errors=True)

    return best_estimator


# =============================================================================
# 6. Stage 2: compare tuned finalists on 2021-2022
# =============================================================================


def compare_on_validation(
    fitted_models: dict[str, Pipeline],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    dates_val: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    prediction_frames = []

    for name, model in fitted_models.items():
        probability = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_probabilities(y_val, probability)

        rows.append({"pipeline": name, **metrics})

        prediction_frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(dates_val).to_numpy(),
                    "y_true": y_val.to_numpy(dtype=int),
                    "probability": probability,
                    "pipeline": name,
                }
            )
        )

    comparison = (
        pd.DataFrame(rows)
        .sort_values(
            ["average_precision", "brier_score", "log_loss"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    return comparison, predictions


# =============================================================================
# 7. Stage 3: calibration on 2021 -> evaluation on 2022
# =============================================================================


def calibration_stage(
    chosen_name: str,
    chosen_model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    dates_val: pd.Series,
) -> tuple[str, object, pd.DataFrame, pd.DataFrame]:
    years = pd.to_datetime(dates_val).dt.year
    fit_mask = years.eq(2021).to_numpy()
    eval_mask = years.eq(2022).to_numpy()

    if not fit_mask.any() or not eval_mask.any():
        raise ValueError("Calibration stage requires both 2021 and 2022 observations.")

    X_cal = X_val.loc[fit_mask].copy()
    y_cal = y_val.loc[fit_mask].copy()

    X_eval = X_val.loc[eval_mask].copy()
    y_eval = y_val.loc[eval_mask].copy()
    dates_eval = dates_val.loc[eval_mask].copy()

    # The base pipeline is already trained on 2013-2020. Freeze it and fit only
    # the calibration mapping on 2021.
    frozen = FrozenEstimator(chosen_model)

    sigmoid = CalibratedClassifierCV(estimator=frozen, method="sigmoid")
    sigmoid.fit(X_cal, y_cal)

    isotonic = CalibratedClassifierCV(estimator=frozen, method="isotonic")
    isotonic.fit(X_cal, y_cal)

    methods: dict[str, object] = {
        "uncalibrated": chosen_model,
        "sigmoid": sigmoid,
        "isotonic": isotonic,
    }

    metric_rows = []
    prediction_rows = []

    for method_name, model in methods.items():
        probability = model.predict_proba(X_eval)[:, 1]
        metrics = evaluate_probabilities(y_eval, probability)

        metric_rows.append(
            {
                "pipeline": chosen_name,
                "calibration": method_name,
                **metrics,
            }
        )

        prediction_rows.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(dates_eval).to_numpy(),
                    "y_true": y_eval.to_numpy(dtype=int),
                    "probability": probability,
                    "calibration": method_name,
                    "pipeline": chosen_name,
                }
            )
        )

    metrics_df = pd.DataFrame(metric_rows)

    # Calibration is selected by Brier first, then log loss.
    chosen_calibration = (
        metrics_df.sort_values(
            ["brier_score", "log_loss"],
            ascending=[True, True],
        )
        .iloc[0]["calibration"]
    )

    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    chosen_calibrator = methods[str(chosen_calibration)]

    return str(chosen_calibration), chosen_calibrator, metrics_df, predictions_df


# =============================================================================
# 8. Main: stages 1-5 only
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, dates_train, X_val, y_val, dates_val = load_data()
    cv_splits = make_date_based_splits(dates_train)

    print("Final development only: train=2013-2020, validation=2021-2022")
    print("2023-2025 is not used in this script.\n")

    # -------------------------------------------------------------------------
    # Stage 1: tune all three frozen finalists.
    # -------------------------------------------------------------------------
    fitted_models: dict[str, Pipeline] = {}

    for imputer_name, imbalance_name in FINALISTS:
        name = pipeline_name(imputer_name, imbalance_name)
        fitted_models[name] = tune_finalist(
            imputer_name,
            imbalance_name,
            X_train,
            y_train,
            cv_splits,
        )

    # -------------------------------------------------------------------------
    # Stage 2: compare tuned models on 2021-2022.
    # Primary selection criterion = validation average precision.
    # -------------------------------------------------------------------------
    validation_comparison, validation_predictions = compare_on_validation(
        fitted_models,
        X_val,
        y_val,
        dates_val,
    )

    validation_comparison.to_csv(
        OUTPUT_DIR / "validation_2021_2022_comparison.csv",
        index=False,
    )
    validation_predictions.to_csv(
        OUTPUT_DIR / "validation_2021_2022_predictions.csv",
        index=False,
    )

    chosen_name = str(validation_comparison.iloc[0]["pipeline"])
    chosen_model = fitted_models[chosen_name]

    print("\n2021-2022 tuned-finalist comparison:")
    print(validation_comparison.to_string(index=False))
    print(f"\nSelected by validation AP: {chosen_name}")

    # -------------------------------------------------------------------------
    # Stage 3: fit calibration on 2021 and choose it by 2022 Brier/log loss.
    # -------------------------------------------------------------------------
    (
        chosen_calibration,
        chosen_probability_model,
        calibration_comparison,
        calibration_predictions,
    ) = calibration_stage(
        chosen_name,
        chosen_model,
        X_val,
        y_val,
        dates_val,
    )

    calibration_comparison.to_csv(
        OUTPUT_DIR / "calibration_2022_comparison.csv",
        index=False,
    )
    calibration_predictions.to_csv(
        OUTPUT_DIR / "calibration_2022_predictions.csv",
        index=False,
    )

    print("\n2022 calibration comparison (calibration fitted on 2021):")
    print(calibration_comparison.to_string(index=False))
    print(f"\nChosen calibration: {chosen_calibration}")

    # -------------------------------------------------------------------------
    # Stage 4: choose final F2 threshold on 2022 only.
    # -------------------------------------------------------------------------
    years_val = pd.to_datetime(dates_val).dt.year
    threshold_mask = years_val.eq(2022).to_numpy()

    X_threshold = X_val.loc[threshold_mask].copy()
    y_threshold = y_val.loc[threshold_mask].copy()

    threshold_probability = chosen_probability_model.predict_proba(X_threshold)[:, 1]

    threshold, threshold_curve = choose_threshold_by_f2(
        y_threshold,
        threshold_probability,
    )
    threshold_metrics = evaluate_threshold(
        y_threshold,
        threshold_probability,
        threshold,
    )

    threshold_curve.to_csv(
        OUTPUT_DIR / "threshold_curve_2022.csv",
        index=False,
    )
    save_json(
        OUTPUT_DIR / "threshold_metrics_2022.json",
        threshold_metrics,
    )

    print("\nChosen 2022 F2 threshold:")
    print(json.dumps(threshold_metrics, indent=2))

    # -------------------------------------------------------------------------
    # Stage 5: freeze the complete pre-test specification.
    # -------------------------------------------------------------------------
    chosen_imputer, chosen_imbalance = chosen_name.split("__", maxsplit=1)
    best_params_path = OUTPUT_DIR / f"best_params_{chosen_name}.json"
    chosen_best_params = json.loads(best_params_path.read_text())

    chosen_validation_row = (
        validation_comparison
        .loc[validation_comparison["pipeline"].eq(chosen_name)]
        .iloc[0]
        .to_dict()
    )

    chosen_calibration_row = (
        calibration_comparison
        .loc[calibration_comparison["calibration"].eq(chosen_calibration)]
        .iloc[0]
        .to_dict()
    )

    final_specification = {
        "status": "FROZEN_PRE_TEST_SPECIFICATION",
        "warning": "Do not modify this specification using 2023-2025 results.",
        "data_ranges": {
            "hyperparameter_training": "2013-2020",
            "model_selection_validation": "2021-2022",
            "calibration_fit": "2021",
            "calibration_evaluation_and_threshold": "2022",
            "reserved_final_test": "2023-2025",
        },
        "feature_set_name": "compact_36_no_transmission",
        "features": PRIMARY_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "pipeline": chosen_name,
        "imputer": chosen_imputer,
        "imbalance_method": chosen_imbalance,
        "sampling_strategy": (
            SAMPLING_STRATEGY if chosen_imbalance == "smote" else None
        ),
        "lightgbm_best_params": chosen_best_params,
        "selection_metric": "average_precision",
        "validation_result": chosen_validation_row,
        "calibration_method": chosen_calibration,
        "calibration_selection_rule": "lowest 2022 Brier score; log loss as tie-breaker",
        "calibration_result_2022": chosen_calibration_row,
        "threshold_rule": "maximize F2 on 2022",
        "threshold": threshold,
        "threshold_metrics_2022": threshold_metrics,
        "random_state": RANDOM_STATE,
        "cv_folds": N_SPLITS,
        "random_search_iterations": N_SEARCH_ITERATIONS,
    }

    save_json(
        OUTPUT_DIR / "FINAL_FROZEN_SPECIFICATION.json",
        final_specification,
    )

    # Save the fitted development objects for audit/reproducibility.
    joblib.dump(
        chosen_model,
        OUTPUT_DIR / "chosen_development_base_model.joblib",
    )
    joblib.dump(
        chosen_probability_model,
        OUTPUT_DIR / "chosen_development_probability_model.joblib",
    )

    print("\n============================================================")
    print("FINAL DEVELOPMENT COMPLETE")
    print("============================================================")
    print(f"Chosen pipeline:    {chosen_name}")
    print(f"Calibration:        {chosen_calibration}")
    print(f"F2 threshold:       {threshold:.8f}")
    print("Frozen specification saved to:")
    print(OUTPUT_DIR / "FINAL_FROZEN_SPECIFICATION.json")
    print("\nSTOP HERE. Do not inspect 2023-2025 in this script.")
    print("The next step is the separate _09_final_test.py workflow.")


if __name__ == "__main__":
    main()
