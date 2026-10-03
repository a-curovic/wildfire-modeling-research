"""
Leakage-safe screening of imbalance and imputation methods for the frozen
wildfire LightGBM feature set, including categorical ecoregion support.

Workflow
--------
1. sampler_screen: fixed LightGBM, compare valid mixed-data imbalance methods.
2. imputer_screen: fixed selected imbalance method(s), compare numeric imputers.

Important
---------
* US_L3NAME is retained as a categorical predictor.
* Numeric imputers are fitted only to numeric predictors.
* US_L3NAME is imputed separately and ordinal-encoded inside each training fold.
* SMOTE is implemented as SMOTENC because the data contain a categorical feature.
* ADASYN and BorderlineSMOTE are intentionally excluded: they are not
  categorical-aware and would interpolate arbitrary category codes.
* The script does not calibrate, choose thresholds, or touch 2023-2025 test data.
"""
from __future__ import annotations

import time
from typing import Iterable
import traceback
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
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

DATA_PATH = DATASET_PATH
OUTPUT_DIR = ARTIFACTS_DIR / "screening"

# Change between runs.
STAGE = "imputer_screen"  # sampler_screen | imputer_screen

# Fill after sampler screening.
BEST_IMBALANCE_METHODS = ["none","smote"]


N_SPLITS = 4
GAP_DAYS = 0
SAMPLING_STRATEGY = 0.25


# -----------------------------------------------------------------------------
# Data and chronological CV
# -----------------------------------------------------------------------------
def load_primary_features() -> list[str]:
    return list(PRIMARY_FEATURES)


def load_development_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="raise")

    missing_inputs = sorted(set(MISSING_RATE_COLUMNS) - set(df.columns))
    if missing_inputs:
        raise KeyError(
            "Cannot create row_weather_missing_rate. Missing source columns: "
            f"{missing_inputs}"
        )

    df["row_weather_missing_rate"] = df[MISSING_RATE_COLUMNS].mean(axis=1)

    dev = (
        df.loc[df[YEAR_COL].between(2013, 2020)]
        .sort_values([DATE_COL, "US_L3NAME"])
        .reset_index(drop=True)
    )

    features = load_primary_features()
    missing = sorted(set(features) - set(dev.columns))
    if missing:
        raise KeyError(f"Frozen features missing from dataset: {missing}")

    undeclared_categorical = sorted(
        set(dev.loc[:, features].select_dtypes(exclude=["number"]).columns)
        - set(CATEGORICAL_FEATURES)
    )
    if undeclared_categorical:
        raise TypeError(
            "Non-numeric predictors must be declared in CATEGORICAL_FEATURES. "
            f"Found: {undeclared_categorical}"
        )

    for column in CATEGORICAL_FEATURES:
        dev[column] = dev[column].astype("object")

    X = dev.loc[:, features].copy()
    y = dev[TARGET].astype(int).copy()
    dates = dev[DATE_COL].copy()

    if y.nunique() != 2:
        raise ValueError("Target must contain both classes.")

    return X, y, dates


def make_date_based_splits(
    dates: pd.Series,
    n_splits: int = N_SPLITS,
    gap_days: int = GAP_DAYS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dates = pd.to_datetime(dates).reset_index(drop=True)
    unique_dates = np.array(sorted(dates.unique()))
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap_days)

    row_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_date_idx, val_date_idx in splitter.split(unique_dates):
        train_dates = unique_dates[train_date_idx]
        val_dates = unique_dates[val_date_idx]

        train_rows = np.flatnonzero(dates.isin(train_dates).to_numpy())
        val_rows = np.flatnonzero(dates.isin(val_dates).to_numpy())

        assert set(dates.iloc[train_rows]).isdisjoint(set(dates.iloc[val_rows]))
        row_splits.append((train_rows, val_rows))

    return row_splits

class SkewAwareScaler(BaseEstimator, TransformerMixin):
    """Robust-scale highly skewed numeric columns and standard-scale the rest."""

    def __init__(self, skew_threshold: float = 1.0):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = X.columns.to_list()
        skew = X.skew(numeric_only=True)
        high = skew[skew.abs() > self.skew_threshold].index.tolist()
        low = skew[skew.abs() <= self.skew_threshold].index.tolist()

        transformers = []
        if high:
            transformers.append(("high_skew", RobustScaler(), high))
        if low:
            transformers.append(("low_skew", StandardScaler(), low))

        self.preprocessor_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.preprocessor_.fit(X)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        return self.preprocessor_.transform(X)




def make_numeric_imputer(name: str):
    if name == "native":
        return "passthrough"
    if name == "median":
        return SimpleImputer(strategy="median")
    if name == "bayes_iterative":
        return IterativeImputer(
            estimator=BayesianRidge(), max_iter=10, random_state=RANDOM_STATE
        )
    if name == "knn_imputer":
        return KNNImputer(n_neighbors=10)
    if name == "iterative_knn":
        return IterativeImputer(
            estimator=KNeighborsRegressor(n_neighbors=5),
            max_iter=10,
            random_state=RANDOM_STATE,
        )
    if name == "rf_iterative":
        return IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=15, random_state=RANDOM_STATE, n_jobs=1
            ),
            max_iter=10,
            random_state=RANDOM_STATE,
        )
    if name == "extra_iterative":
        return IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=15, random_state=RANDOM_STATE, n_jobs=1
            ),
            max_iter=10,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown imputer: {name}")


def make_numeric_pipeline(imputer_name: str, scale_for_sampling: bool):
    imputer = make_numeric_imputer(imputer_name)
    pre_scale_imputers = {"bayes_iterative", "knn_imputer", "iterative_knn"}

    if imputer_name == "native":
        if scale_for_sampling:
            raise ValueError("SMOTENC requires complete numeric data; choose an imputer.")
        return "passthrough"

    steps = []
    if imputer_name in pre_scale_imputers:
        steps.append(("scale_before_imputation", SkewAwareScaler()))
        steps.append(("imputer", imputer))
    else:
        steps.append(("imputer", imputer))
        if scale_for_sampling:
            steps.append(("scale_before_sampling", SkewAwareScaler()))

    return SklearnPipeline(steps)


def make_preprocessor(feature_names: list[str], imputer_name: str, scale_for_sampling: bool) -> tuple[ColumnTransformer, list[int]]:
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

    # ColumnTransformer outputs numeric columns first, then categorical columns.
    categorical_indices = list(range(len(numeric), len(numeric) + len(categorical)))
    return preprocessor, categorical_indices


def make_sampler(name: str, categorical_indices: list[int]):
    if name in {"none", "class_weight"}:
        return "passthrough"
    if name == "random_over":
        return RandomOverSampler(
            sampling_strategy=SAMPLING_STRATEGY,
            random_state=RANDOM_STATE,
        )
    if name == "smote":
        return SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=SAMPLING_STRATEGY,
            random_state=RANDOM_STATE,
            k_neighbors=5,
        )
    if name == "smote_tomek":
        smote = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=SAMPLING_STRATEGY,
            random_state=RANDOM_STATE,
            k_neighbors=5,
        )
        return SMOTETomek(smote=smote, random_state=RANDOM_STATE)
    if name in {"adasyn", "borderline_smote", "tomek"}:
        raise ValueError(
            f"{name} is excluded from the primary mixed-data screen because it "
            "has no categorical-aware treatment comparable to SMOTENC."
        )
    raise ValueError(f"Unknown imbalance method: {name}")


def make_fixed_model(class_weight=None, n_jobs: int = 1) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        min_child_samples=50,
        subsample=1.0,
        colsample_bytree=1.0,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        verbosity=-1,
    )

def assert_unique_fold_results(fold_results: pd.DataFrame) -> None:
    
    keys = ["imputer", "imbalance_method","fold"]
    
    duplicated = fold_results.duplicated(keys, keep=False)
    
    if duplicated.any():
        duplicate_rows = (fold_results.loc[duplicated,keys].sort_values(keys))
        
        raise ValueError(f"Duplicated configuration fold results found: \n {duplicate_rows.to_string(index=False)}")

def build_screening_pipeline(imputer_name: str, imbalance_name: str, feature_names: list[str]) -> tuple[Pipeline, list[int]]:
    if imputer_name == "native" and imbalance_name in {"smote", "smote_tomek"}:
        raise ValueError("SMOTENC cannot operate on NaNs; use a numeric imputer.")

    scale_for_sampling = imbalance_name in {"smote", "smote_tomek"}
    preprocessor, categorical_indices = make_preprocessor(
        feature_names=feature_names,
        imputer_name=imputer_name,
        scale_for_sampling=scale_for_sampling,
    )

    class_weight = "balanced" if imbalance_name == "class_weight" else None
    sampler = make_sampler(imbalance_name, categorical_indices)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("sampler", sampler),
            ("model", make_fixed_model(class_weight=class_weight, n_jobs=3)),
        ]
    )
    return pipeline, categorical_indices


def evaluate_configuration(X: pd.DataFrame, y: pd.Series, dates: pd.Series, splits: Iterable[tuple[np.ndarray, np.ndarray]], imputer_name: str, imbalance_name: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    pipeline, categorical_indices = build_screening_pipeline(
        imputer_name=imputer_name,
        imbalance_name=imbalance_name,
        feature_names=X.columns.tolist(),
    )
    rows = []

    prediction_rows =[]
    pipeline_name = f"{imputer_name}__{imbalance_name}"

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        start = time.perf_counter()
        fitted = clone(pipeline)
        fitted.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            model__categorical_feature=categorical_indices,
        )
        y_val = y.iloc[val_idx]
        probability = fitted.predict_proba(X.iloc[val_idx])[:, 1]
        
        fold_predictions = pd.DataFrame({
            "row_id": val_idx.astype(int),
            
            "fold":fold,
            
            "date": pd.to_datetime(dates.iloc[val_idx]).to_numpy(),
            
            "US_L3NAME": X.iloc[val_idx]["US_L3NAME"].astype(str).to_numpy(),
            
            "y_true": y_val.to_numpy(dtype=int),

            "probability": probability,

            "imputer": imputer_name,

            "imbalance_method": imbalance_name,

            "pipeline": pipeline_name
        })
        
        assert len(fold_predictions) == len(val_idx)
        prediction_rows.append(fold_predictions)
        elapsed = time.perf_counter() - start

        prevalence = float(y_val.mean())
        ap = float(average_precision_score(y_val, probability))
        rows.append(
            {
                "imputer": imputer_name,
                "imbalance_method": imbalance_name,
                "fold": fold,
                "validation_start": str(pd.to_datetime(dates.iloc[val_idx]).min().date()),
                "validation_end": str(pd.to_datetime(dates.iloc[val_idx]).max().date()),
                "validation_rows": int(len(val_idx)),
                "validation_fires": int(y_val.sum()),
                "validation_prevalence": prevalence,
                "average_precision": ap,
                "ap_over_prevalence": ap / prevalence,
                "brier_score": float(brier_score_loss(y_val, probability)),
                "log_loss": float(log_loss(y_val, probability, labels=[0, 1])),
                "runtime_seconds": elapsed,
            }
        )
    fold_results = pd.DataFrame(rows)
    
    oof_predictions = pd.concat(prediction_rows, ignore_index=True)
    
    duplicate_prediction_keys = oof_predictions.duplicated(["pipeline","fold","row_id"])
    
    assert not duplicate_prediction_keys.any(), "Duplicated OOF prediction rows found."
    
    return fold_results, oof_predictions


def summarize(fold_results: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_results.groupby(["imputer", "imbalance_method"], as_index=False)
        .agg(
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
            mean_ap_over_prevalence=("ap_over_prevalence", "mean"),
            mean_brier=("brier_score", "mean"),
            mean_log_loss=("log_loss", "mean"),
            total_runtime_seconds=("runtime_seconds", "sum"),
        )
        .sort_values(
            ["mean_average_precision", "mean_log_loss"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def stage_configurations() -> list[tuple[str, str]]:
    if STAGE == "sampler_screen":
        return [
            ("native", "none"),
            ("native", "class_weight"),
            ("median", "none"),
            ("median", "class_weight"),
            ("median", "random_over"),
            # Names retained for convenience; these use SMOTENC internally.
            ("median", "smote"),
            ("median", "smote_tomek"),
        ]

    if STAGE == "imputer_screen":
        imputers = [
            "native",
            "median",
            "bayes_iterative",
            "knn_imputer",
            "iterative_knn",
            "rf_iterative",
            "extra_iterative"
        ]
        
        configurations = []
        
        for imbalance in BEST_IMBALANCE_METHODS:
            for imputer in imputers:
                
                if imputer == "native" and imbalance != "none":
                    continue
            
                pair = (imputer,imbalance)
            
                if pair not in configurations:
                    configurations.append(pair)
        
        assert len(configurations) == len(set(configurations)), ("Duplicate screening configurations were generated")
        
        return configurations

    raise ValueError(f"Unknown STAGE: {STAGE}")







def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, dates = load_development_data()
    splits = make_date_based_splits(dates)

    all_results = []
    all_predictions = []
    for imputer_name, imbalance_name in stage_configurations():
        print(f"\nRunning: imputer={imputer_name}, imbalance={imbalance_name}", flush=True)
        try:
            result, predictions = evaluate_configuration(
                X=X,
                y=y,
                dates=dates,
                splits=splits,
                imputer_name=imputer_name,
                imbalance_name=imbalance_name,
            )
            all_results.append(result)
            all_predictions.append(predictions)
            
            prediction_checkpoint = pd.concat(all_predictions, ignore_index=True)
            prediction_checkpoint.to_csv(OUTPUT_DIR / f"{STAGE}_oof_predictions_checkpoint.csv", index=False)
            checkpoint = pd.concat(all_results, ignore_index=True)
            checkpoint.to_csv(OUTPUT_DIR / f"{STAGE}_checkpoint.csv", index=False)
            
            summarize(checkpoint).to_csv(OUTPUT_DIR / f"{STAGE}_checkpoint_summary.csv", index=False)
            
            print(f"Completed and checkpointed: {imputer_name} + {imbalance_name}", flush=True)
            
        except Exception as exc:
            print(f"FAILED: {imputer_name} + {imbalance_name}", flush=True)
            
            print(f"Exception type: {type(exc).__name__}",flush=True)
            
            print(f"Exception message: {exc}", flush=True)
            
            traceback.print_exc()

    if not all_results:
        raise RuntimeError("Every configuration failed.")

    fold_results = pd.concat(all_results, ignore_index=True)
    assert_unique_fold_results(fold_results)
    summary = summarize(fold_results)
    
    oof_predictions = pd.concat(all_predictions, ignore_index=True)

    oof_path = OUTPUT_DIR / f"{STAGE}_oof_predictions.csv"

    oof_predictions.to_csv(oof_path, index=False)

    print(f"Saved OOF predictions to {oof_path}")

    fold_path = OUTPUT_DIR / f"{STAGE}_fold_results.csv"
    summary_path = OUTPUT_DIR / f"{STAGE}_summary.csv"
    fold_results.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved {fold_path} and {summary_path}")
    
    fold_comparison = fold_results.pivot_table(
    index="fold",
    columns=["imputer", "imbalance_method"],
    values=[
        "average_precision",
        "brier_score",
        "log_loss",
    ])

    native = (
    fold_results
    .query("imputer == 'native' and imbalance_method == 'none'")
    .set_index("fold")
    )

    smote = (
        fold_results
        .query("imputer == 'median' and imbalance_method == 'smote'")
        .set_index("fold")
    )

    comparison = pd.DataFrame({
        "ap_difference": (
            smote["average_precision"] - native["average_precision"]
        ),
        "brier_difference": (
            smote["brier_score"] - native["brier_score"]
        ),
        "log_loss_difference": (
            smote["log_loss"] - native["log_loss"]
        ),
    })

    print(comparison)
    print(fold_comparison)

if __name__ == "__main__":
    main()
