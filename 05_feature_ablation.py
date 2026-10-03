"""Feature-family ablation and compact feature-set confirmation.

A fixed LightGBM model is used so the feature question is separated from later
hyperparameter tuning. Finalists are then checked across chronological folds and
compared with paired temporal block bootstrap.
"""

import json
from wildfire_config import ARTIFACTS_DIR, DATASET_PATH
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

TARGET = "fire_occurred"
FIXED_THRESHOLD = 0.10


OUTPUT_DIR = ARTIFACTS_DIR / "ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_PATH)
df["date"] = pd.to_datetime(df["date"])
df["US_L3NAME"] = df["US_L3NAME"].astype("category")

missing_rate_columns = [
    "ceiling_height_m_missingRate",
    "dew_point_c_missingRate",
    "temp_c_missingRate",
    "wind_type_missingRate",
    "sea_level_pressure_hPa_missingRate",
    "vis_m_missingRate",
    "wind_speed_ms_missingRate",
    "wind_dir_deg_missingRate",
]

df["row_weather_missing_rate"] = (
    df[missing_rate_columns]
    .mean(axis=1)
)



# Year is deliberately NOT excluded here because it should be tested as
# part of the temporal family.
permanent_drop = {
    "date",
    "US_L3CODE",
    "fire_count",
    TARGET,
    "season",

    "ceiling_height_m_missingFlag_count",
    "dew_point_c_missingFlag_count",
    "wind_type_missingFlag_count",
    "wind_dir_deg_missingFlag_count",
    "wind_speed_ms_missingFlag_count",
    "sea_level_pressure_hPa_missingFlag_count",
    "vis_m_missingFlag_count"
}


temperature_family = {
    "temp_c_max",
    "temp_c_mean",
    "temp_c_min",
    "temp_c_std",
    "DTR",
    "sat_vp"
}

dew_vpd_family = {
    "dew_point_c_max",
    "dew_point_c_mean",
    "dew_point_c_min",
    "A_sat_vp",
    "RH_pct",
    "VPD_kpa",
}

wind_family = {
    "ew_wind_mean",
    "ns_wind_mean",
    "wind_speed_ms_max",
    "wind_speed_ms_mean",
    "wind_speed_ms_min",
    "wind_speed_ms_std",
    "HDWI_proxy",
}

visibility_family = {
    "vis_m_max",
    "vis_m_mean",
    "vis_m_min",
}

pressure_family = {
    "sea_level_pressure_hPa_max",
    "sea_level_pressure_hPa_mean",
    "sea_level_pressure_hPa_min",
}

ceiling_family = {
    "ceiling_height_m_max",
    "ceiling_height_m_mean",
    "ceiling_height_m_min",
}

vegetation_family = {
    "ndvi_mean",
    "ndvi_std",
    "ndvi_min",
    "ndvi_max",
    "evi_mean",
    "evi_std",
    "evi_min",
    "evi_max",
}

observation_count_family = {
    "ceiling_height_m_count",
    "dew_point_c_count",
    "ew_wind_count",
    "ns_wind_count",
    "sea_level_pressure_hPa_count",
    "temp_c_count",
    "vis_m_count",
    "wind_speed_ms_count",
}

missingness_family = {
    "ceiling_height_m_missingRate",
    "dew_point_c_missingRate",
    "sea_level_pressure_hPa_missingRate",
    "temp_c_missingFlag_count",
    "temp_c_missingRate",
    "vis_m_missingRate",
    "wind_dir_deg_missingRate",
    "wind_speed_ms_missingRate",
    "wind_type_missingRate",
    "row_weather_missing_rate",
}

spatial_family = {
    "road_km_per_100km2",
    "trans_km_per_100km2",
    "population_density",
    "roads_per_pop_density",
    "transmission_risk",
}

temporal_family = {
    "year",
    "month",
}

families = {
    "temperature": temperature_family,
    "dew_vpd": dew_vpd_family,
    "wind": wind_family,
    "visibility": visibility_family,
    "pressure": pressure_family,
    "ceiling": ceiling_family,
    "vegetation": vegetation_family,
    "observation_counts": observation_count_family,
    "missingness": missingness_family,
    "spatial": spatial_family,
    "temporal": temporal_family,
    "ecoregion": {"US_L3NAME"},
}


def keep_only(family, retained):
    """Drop every family member except the explicitly retained columns."""
    family = set(family)
    retained = set(retained)

    unknown = retained - family
    if unknown:
        raise ValueError(
            f"Retained columns are not in the family: {sorted(unknown)}"
        )

    return family - retained


experiments = {
    "baseline_all_features": set(),
}

# Leave-one-family-out experiments
for family_name, family_columns in families.items():
    experiments[f"drop_{family_name}"] = set(family_columns)


# Selected combined family removals
experiments.update({
    "drop_spatial_and_counts":
        spatial_family | observation_count_family,

    "drop_spatial_counts_and_vegetation":
        spatial_family | observation_count_family | vegetation_family,

    "drop_counts_and_vegetation":
        observation_count_family | vegetation_family,

    "drop_counts_and_visibility":
        observation_count_family | visibility_family,
})


# Within-temperature compact variants
experiments.update({
    "temperature_keep_mean":
        keep_only(temperature_family, {"temp_c_mean"}),

    "temperature_keep_max":
        keep_only(temperature_family, {"temp_c_max"}),

    "temperature_keep_min":
        keep_only(temperature_family, {"temp_c_min"}),

    "temperature_keep_std":
        keep_only(temperature_family, {"temp_c_std"}),

    "temperature_keep_dtr":
        keep_only(temperature_family, {"DTR"}),
        
    "temperature_keep_dtr":
        keep_only(temperature_family, {"sat_vp"}),

    "temperature_keep_mean_and_dtr":
        keep_only(temperature_family,{"temp_c_mean", "DTR"})
})


# Within-visibility compact variants
experiments.update({
    "visibility_keep_min":
        keep_only(visibility_family, {"vis_m_min"}),

    "visibility_keep_mean":
        keep_only(visibility_family, {"vis_m_mean"}),

    "visibility_keep_max":
        keep_only(visibility_family, {"vis_m_max"}),

    "visibility_keep_mean_and_max":
        keep_only(visibility_family,{"vis_m_mean", "vis_m_max"})
})

# Within-Dew compact variants
experiments.update({
    "dew_keep_min":
        keep_only(dew_vpd_family, {"dew_point_c_min"}),

    "dew_keep_mean":
        keep_only(dew_vpd_family, {"dew_point_c_mean"}),

    "dew_keep_max":
        keep_only(dew_vpd_family, {"dew_point_c_max"}),

    "dew_keep_mean_and_max":
        keep_only(dew_vpd_family,{"RH_pct", "dew_point_c_mean"}),
    
    "dew_keep_RH_pct":
        keep_only(dew_vpd_family, {"RH_pct"}),
})

# Within-wind compact variants
experiments.update({
    "wind_keep_HDWI_proxy":
        keep_only(wind_family, {"HDWI_proxy"}),

    "wind_keep_ew_wind":
        keep_only(wind_family, {"ew_wind_mean"}),

    "wind_keep_max":
        keep_only(wind_family, {"wind_speed_ms_max"}),

    "wind_keep_HDWI_proxy_and_max":
        keep_only(wind_family,{"HDWI_proxy", "wind_speed_ms_max"}),
        
    "wind_keep_HDWI_proxy_and_ew_wind":
        keep_only(wind_family,{"HDWI_proxy", "ew_wind_mean"})
})

# Within-pressure compact variants
experiments.update({
    "pressure_keep_mean":
            keep_only(pressure_family, {"sea_level_pressure_hPa_mean"}),
    
    "pressure_keep_max":
        keep_only(pressure_family, {"sea_level_pressure_hPa_max"}),
    
    "pressure_keep_min":
        keep_only(pressure_family, {"sea_level_pressure_hPa_min"})
})


# Within-ceiling compact variants
experiments.update({
    "ceiling_keep_min":
        keep_only(ceiling_family, {"ceiling_height_m_min"}),

    "ceiling_keep_mean":
        keep_only(ceiling_family, {"ceiling_height_m_mean"}),

    "ceiling_keep_max":
        keep_only(ceiling_family, {"ceiling_height_m_max"})
})

# Within-vegetation compact variants
experiments.update({
    "vegetation_keep_evi_mean":
        keep_only(vegetation_family, {"evi_mean"}),

    "vegetation_keep_ndvi_mean":
        keep_only(vegetation_family, {"ndvi_mean"}),

    "vegetation_keep_ndvi_std":
        keep_only(vegetation_family, {"ndvi_std"}),
        
    "vegetation_keep_evi_std":
        keep_only(vegetation_family, {"evi_std"}),

    "vegetation_keep_evi_mean_and_std":
        keep_only(vegetation_family,{"evi_mean", "evi_std"}),
        
    "vegetation_keep_ndvi_mean_and_std":
        keep_only(vegetation_family,{"ndvi_mean", "ndvi_std"}),
})

# Within-missingness compact variants
experiments.update({
    "missingness_keep_row_Rate":
        keep_only(missingness_family, {"row_weather_missing_rate"}),

    "missingness_keep_temp_count":
        keep_only(missingness_family, {"temp_c_missingFlag_count"}),

    "missingness_keep_vis_Rate":
        keep_only(missingness_family, {"vis_m_missingRate"}),

    "missingness_keep_row_Rate_and_vis_Rate":
        keep_only(missingness_family,{"row_weather_missing_rate", "vis_m_missingRate"}),
        
    "missingness_keep_row_Rate_and_temp_count":
        keep_only(missingness_family,{"row_weather_missing_rate", "temp_c_missingFlag_count"})
})

# Within-spatial compact variants
experiments.update({
    "spatial_keep_population_density":
        keep_only(spatial_family, {"population_density"}),

    "spatial_keep_trans":
        keep_only(spatial_family, {"trans_km_per_100km2"}),

    "spatial_keep_road":
        keep_only(spatial_family, {"road_km_per_100km2"}),
    
    "spatial_keep_roads_per_pop_density":
        keep_only(spatial_family, {"roads_per_pop_density"}),
        
    "spatial_keep_trans_risk":
        keep_only(spatial_family, {"transmission_risk"}),

    "spatial_keep_population_density_and_trans_risk":
        keep_only(spatial_family,{"population_density", "transmission_risk"}),
        
    "spatial_keep_road_and_trans_risk":
        keep_only(spatial_family,{"road_km_per_100km2", "transmission_risk"}),
    
    "spatial_keep_population_density_and_trans_risk_and_roads_per_pop_density":
        keep_only(spatial_family,{"population_density", "transmission_risk","roads_per_pop_density"})
})

experiments.update({
    "temp_keep_year":
        keep_only(temporal_family, {"year"}),

    "temp_keep_month":
        keep_only(temporal_family, {"month"})
})

##To make additional checks

experiments.update({
    "temperature_keep_mean_and_std":
        keep_only(temperature_family, {"temp_c_mean","temp_c_std"}),

    "wind_keep_road":
        keep_only(wind_family, {"HDWI_proxy","wind_speed_ms_max","ew_wind_mean"}),
    
    "dew_keep_VPD_kpa":
        keep_only(dew_vpd_family, {"VPD_kpa"}),
        
    "primary_compact_candidate":
        (keep_only(vegetation_family, {"evi_mean","evi_std"}) | keep_only(observation_count_family, set()) | keep_only(pressure_family,{"sea_level_pressure_hPa_mean"}) | 
        keep_only(ceiling_family,{"ceiling_height_m_mean"}) | keep_only(visibility_family,{"vis_m_max"}) | keep_only(temporal_family,{"month"}) | keep_only(spatial_family, {"trans_km_per_100km2"})),

    "primary_compact_candidate_without_transmission":
        (keep_only(vegetation_family, {"evi_mean","evi_std"}) | keep_only(observation_count_family, set()) | keep_only(pressure_family,{"sea_level_pressure_hPa_mean"}) |
        keep_only(ceiling_family,{"ceiling_height_m_mean"}) | keep_only(visibility_family,{"vis_m_max"}) | keep_only(temporal_family,{"month"}) | keep_only(spatial_family, set())),

    "primary_compact_candidate_road_density":
        (keep_only(vegetation_family, {"evi_mean","evi_std"}) | keep_only(observation_count_family, set()) | keep_only(pressure_family,{"sea_level_pressure_hPa_mean"}) | 
        keep_only(ceiling_family,{"ceiling_height_m_mean"}) | keep_only(visibility_family,{"vis_m_max"}) | keep_only(temporal_family,{"month"}) | keep_only(spatial_family, {"roads_per_pop_density"})),

    "Without_ecoregion_with_trans":
        {"US_L3NAME"} | keep_only(spatial_family,{"trans_km_per_100km2"}),

"primary_compact_candidate_without_evi_mean":
        (keep_only(vegetation_family, {"evi_std"}) | keep_only(observation_count_family, set()) | keep_only(pressure_family,{"sea_level_pressure_hPa_mean"}) | 
        keep_only(ceiling_family,{"ceiling_height_m_mean"}) | keep_only(visibility_family,{"vis_m_max"}) | keep_only(temporal_family,{"month"}) | keep_only(spatial_family, {"trans_km_per_100km2"})),

"primary_compact_candidate_with_roads":
        (keep_only(vegetation_family, {"evi_mean","evi_std"}) | keep_only(observation_count_family, set()) | keep_only(pressure_family,{"sea_level_pressure_hPa_mean"}) | 
        keep_only(ceiling_family,{"ceiling_height_m_mean"}) | keep_only(visibility_family,{"vis_m_max"}) | keep_only(temporal_family,{"month"}) | keep_only(spatial_family, {"road_km_per_100km2"})),

})


train = df.loc[df["year"].between(2013, 2020)].copy()
val = df.loc[df["year"].between(2021, 2022)].copy()

y_train = train[TARGET].astype(int)
y_val = val[TARGET].astype(int)

validation_prevalence = y_val.mean()


results = []
prediction_table = val[
    ["date", "US_L3NAME", TARGET]
].reset_index(drop=True).copy()

feature_sets = {}

for experiment_name, additional_drop in experiments.items():

    columns_to_drop = permanent_drop | additional_drop

    unknown_columns = columns_to_drop - set(df.columns)
    if unknown_columns:
        raise KeyError(
            f"{experiment_name} contains unknown columns: "
            f"{sorted(unknown_columns)}"
        )

    x_train = train.drop(columns=sorted(columns_to_drop))
    x_val = val.drop(columns=sorted(columns_to_drop))

    if list(x_train.columns) != list(x_val.columns):
        raise ValueError(
            f"Train/validation columns differ for {experiment_name}"
        )

    unsupported = x_train.select_dtypes(
        exclude=["number", "category", "bool"]
    ).columns.tolist()

    if unsupported:
        raise TypeError(
            f"{experiment_name} contains unsupported dtypes: {unsupported}"
        )

    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        min_child_samples=50,
        subsample=1.0,
        colsample_bytree=1.0,
        class_weight=None,
        random_state=1,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(x_train, y_train)

    probability = model.predict_proba(x_val)[:, 1]
    prediction = (probability >= FIXED_THRESHOLD).astype(int)

    ap = average_precision_score(y_val, probability)
    brier = brier_score_loss(y_val, probability)
    ll = log_loss(y_val, probability)

    results.append({
        "experiment": experiment_name,
        "n_features": x_train.shape[1],
        "n_dropped_from_baseline": len(additional_drop),
        "average_precision": ap,
        "ap_lift_over_prevalence": ap / validation_prevalence,
        "brier_score": brier,
        "log_loss": ll,
        "threshold": FIXED_THRESHOLD,
        "precision_at_fixed_threshold": precision_score(
            y_val, prediction, zero_division=0
        ),
        "recall_at_fixed_threshold": recall_score(
            y_val, prediction, zero_division=0
        ),
        "f1_at_fixed_threshold": f1_score(
            y_val, prediction, zero_division=0
        ),
        "f2_at_fixed_threshold": fbeta_score(
            y_val,
            prediction,
            beta=2.0,
            zero_division=0,
        ),
        "balanced_accuracy_at_fixed_threshold":
            balanced_accuracy_score(y_val, prediction),
        "mcc_at_fixed_threshold":
            matthews_corrcoef(y_val, prediction),
        "columns": x_train.columns
    })

    prediction_table[f"prob__{experiment_name}"] = probability
    feature_sets[experiment_name] = x_train.columns.tolist()


results_df = pd.DataFrame(results)

baseline = (results_df.set_index("experiment").loc["baseline_all_features"])

results_df["delta_average_precision"] = (results_df["average_precision"] - baseline["average_precision"])

# Positive values mean improvement for these two columns.
results_df["brier_improvement"] = (baseline["brier_score"] - results_df["brier_score"])

results_df["log_loss_improvement"] = (baseline["log_loss"] - results_df["log_loss"])

results_df = results_df.sort_values("average_precision", ascending=False)

results_df.to_csv(OUTPUT_DIR / "ablation_validation_results.csv", index=False)

prediction_table.to_csv(OUTPUT_DIR / "ablation_validation_predictions.csv", index=False)

with open(OUTPUT_DIR / "ablation_feature_sets.json", "w") as file:
    json.dump(feature_sets, file, indent=2)

with pd.option_context("display.max_rows",None, "display.max_columns",None):
    print(results_df.to_string(index=False))


##Finalists

development_df = (
    df.loc[df["year"].between(2013, 2020)]
    .sort_values(["date", "US_L3NAME"])
    .reset_index(drop=True)
)

# Confirm exactly 13 ecoregions per date.
rows_per_date = development_df.groupby("date").size()

assert rows_per_date.eq(13).all(), ("At least one date does not contain exactly 13 rows.")

unique_dates = np.sort(development_df["date"].unique())

tscv = TimeSeriesSplit(n_splits=4)

finalists = {
    "baseline_all_features":
        experiments["baseline_all_features"],

    "primary_compact_candidate":
        experiments["primary_compact_candidate"],

    "primary_compact_candidate_without_transmission":
        experiments[
            "primary_compact_candidate_without_transmission"
        ],

    "spatial_keep_road":
        experiments["spatial_keep_road"],

    "primary_compact_candidate_with_roads":
        experiments["primary_compact_candidate_with_roads"],
}

confirmation_results = []
prediction_frames = []

for experiment_name, additional_drop in finalists.items():

    columns_to_drop = permanent_drop | additional_drop

    for fold, (train_date_idx, val_date_idx) in enumerate(tscv.split(unique_dates),start=1):
        train_dates = unique_dates[train_date_idx]
        val_dates = unique_dates[val_date_idx]

        fold_train = development_df.loc[development_df["date"].isin(train_dates)].copy()

        fold_val = development_df.loc[development_df["date"].isin(val_dates)].copy()

        validation_rows = len(fold_val)
        validation_fires = int(fold_val["fire_occurred"].sum())
        validation_fire_rate = (fold_val["fire_occurred"].mean())


        x_train = fold_train.drop(columns=sorted(columns_to_drop))
        y_train = (fold_train["fire_occurred"].astype(int))

        x_val = fold_val.drop(columns=sorted(columns_to_drop))
        y_val = (fold_val["fire_occurred"].astype(int))

        assert list(x_train.columns) == list(x_val.columns)

        
        model = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=8,
            min_child_samples=50,
            subsample=1.0,
            colsample_bytree=1.0,
            class_weight=None,
            random_state=1,
            n_jobs=-1,
            verbosity=-1,
        )

        model.fit(x_train, y_train)

        probability = model.predict_proba(x_val)[:, 1]
        
        
        fold_prediction = fold_val[[
            "date",
            "US_L3CODE",
            "US_L3NAME",
            "fire_occurred"
        ]].copy()
        
        fold_prediction["fold"] = fold
        fold_prediction["experiment"] = experiment_name
        fold_prediction["probability"] = probability
        
        prediction_frames.append(fold_prediction)

        ap = average_precision_score(y_val, probability)

        brier = brier_score_loss(y_val, probability)

        fold_log_loss = log_loss(y_val, probability)

        ap_over_prevalence = (ap / validation_fire_rate if validation_fire_rate > 0 else np.nan)


        confirmation_results.append({
            "experiment": experiment_name,
            "fold": fold,

            "train_start":
                fold_train["date"].min(),

            "train_end":
                fold_train["date"].max(),

            "validation_start":
                fold_val["date"].min(),

            "validation_end":
                fold_val["date"].max(),

            "training_rows":
                len(fold_train),

            "validation_rows":
                validation_rows,

            "validation_fires":
                validation_fires,

            "validation_fire_rate":
                validation_fire_rate,

            "n_features":
                x_train.shape[1],

            "average_precision":
                ap,

            "ap_over_prevalence":
                ap_over_prevalence,

            "brier_score":
                brier,

            "log_loss":
                fold_log_loss,
        })

        print(
            f"{experiment_name} | Fold {fold} | "
            f"Fires: {validation_fires}/{validation_rows} "
            f"({validation_fire_rate:.4f}) | "
            f"AP: {ap:.4f} | "
            f"AP/prevalence: {ap_over_prevalence:.2f} | "
            f"Brier: {brier:.4f}"
        )


from math import ceil


def calculate_metric_differences(
    sample_df: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> dict:
    """
    Calculate paired metric differences as model A minus model B.

    Interpretation
    --------------
    AP difference:
        Positive favors model A.

    Brier difference:
        Negative favors model A because lower Brier is better.

    Log-loss difference:
        Negative favors model A because lower log loss is better.
    """

    y_true = sample_df["fire_occurred"].to_numpy()
    prob_a = sample_df[model_a].to_numpy()
    prob_b = sample_df[model_b].to_numpy()

    # AP requires positive observations for meaningful interpretation.
    if np.unique(y_true).size < 2:
        raise ValueError(
            "Bootstrap sample contains only one target class."
        )

    return {
        "ap_difference": (
            average_precision_score(y_true, prob_a)
            - average_precision_score(y_true, prob_b)
        ),
        "brier_difference": (
            brier_score_loss(y_true, prob_a)
            - brier_score_loss(y_true, prob_b)
        ),
        "log_loss_difference": (
            log_loss(y_true, prob_a, labels=[0, 1])
            - log_loss(y_true, prob_b, labels=[0, 1])
        ),
    }

from math import ceil
def sample_temporal_blocks(
    fold_df: pd.DataFrame,
    block_size_days: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Moving-block bootstrap within one validation fold.

    Entire dates are sampled, so all 13 ecoregions remain together.
    Consecutive dates are sampled in blocks to preserve short-term
    temporal dependence.
    """

    fold_df = (
        fold_df
        .sort_values(["date", "US_L3CODE"])
        .reset_index(drop=True)
        .copy()
    )

    # --------------------------------------------------------
    # Prepare dates without iterating over pandas DatetimeArray
    # --------------------------------------------------------

    fold_df["date"] = pd.to_datetime(
        fold_df["date"],
        errors="raise",
    ).dt.normalize()

    assert not fold_df["date"].isna().any(), (
        "Validation fold contains missing dates."
    )

    date_values = fold_df["date"].to_numpy(
        dtype="datetime64[ns]"
    )

    # np.unique removes duplicates and returns sorted dates.
    dates = np.unique(date_values)

    n_dates = len(dates)

    if n_dates == 0:
        raise ValueError(
            "Fold contains no validation dates."
        )

    block_size = min(
        block_size_days,
        n_dates,
    )

    # --------------------------------------------------------
    # Check the panel structure
    # --------------------------------------------------------

    rows_per_date = fold_df.groupby("date").size()

    assert rows_per_date.eq(13).all(), (
        "A validation date does not contain 13 ecoregion rows."
    )

    # Each NumPy datetime key maps to all 13 corresponding rows.
    positions_by_date = {
        date: np.flatnonzero(date_values == date)
        for date in dates
    }

    assert all(
        len(positions) == 13
        for positions in positions_by_date.values()
    ), (
        "At least one date does not map to exactly 13 rows."
    )

    # --------------------------------------------------------
    # Sample contiguous blocks of dates
    # --------------------------------------------------------

    n_blocks = ceil(n_dates / block_size)
    largest_start = n_dates - block_size

    if largest_start == 0:
        starts = np.zeros(
            n_blocks,
            dtype=int,
        )
    else:
        starts = rng.integers(
            low=0,
            high=largest_start + 1,
            size=n_blocks,
        )

    sampled_dates = np.concatenate([
        dates[start:start + block_size]
        for start in starts
    ])[:n_dates]

    # Repeated dates intentionally produce repeated rows.
    sampled_positions = np.concatenate([
        positions_by_date[date]
        for date in sampled_dates
    ])

    sampled_df = (
        fold_df
        .iloc[sampled_positions]
        .reset_index(drop=True)
    )

    expected_rows = n_dates * 13

    assert len(sampled_df) == expected_rows, (
        f"Expected {expected_rows} sampled rows, "
        f"found {len(sampled_df)}."
    )

    return sampled_df


def paired_temporal_block_bootstrap(predictions_df: pd.DataFrame, model_a: str, model_b: str, block_size_days: int = 30, n_bootstrap: int = 2000, confidence_level: float = 0.95, random_state: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Paired temporal block bootstrap of model A minus model B.

    The same sampled observations are used for both models in every
    bootstrap replicate.

    Metrics are first calculated separately within each fold and then
    aggregated across folds using validation-row weights.
    """

    required_columns = {
        "fold",
        "date",
        "US_L3CODE",
        "US_L3NAME",
        "fire_occurred",
        model_a,
        model_b,
    }

    missing_columns = (
        required_columns
        - set(predictions_df.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    rng = np.random.default_rng(random_state)

    fold_groups = {
        fold: fold_df.copy()
        for fold, fold_df in predictions_df.groupby(
            "fold",
            sort=True,
        )
    }

    metric_names = [
        "ap_difference",
        "brier_difference",
        "log_loss_difference",
    ]


    original_fold_results = []

    for fold, fold_df in fold_groups.items():

        differences = calculate_metric_differences(
            fold_df,
            model_a,
            model_b,
        )

        differences["fold"] = fold
        differences["weight"] = len(fold_df)

        original_fold_results.append(differences)

    original_fold_df = pd.DataFrame(
        original_fold_results
    )

    original_point_estimates = {
        metric: np.average(
            original_fold_df[metric],
            weights=original_fold_df["weight"],
        )
        for metric in metric_names
    }


    bootstrap_records = []

    valid_replicates = 0
    attempts = 0
    max_attempts = n_bootstrap * 5

    while (
        valid_replicates < n_bootstrap
        and attempts < max_attempts
    ):
        attempts += 1
        sampled_fold_results = []
        replicate_valid = True

        for fold, fold_df in fold_groups.items():

            sampled_fold = sample_temporal_blocks(
                fold_df=fold_df,
                block_size_days=block_size_days,
                rng=rng,
            )

            # In the unlikely event that block sampling creates a
            # one-class fold, discard the complete replicate.
            if sampled_fold["fire_occurred"].nunique() < 2:
                replicate_valid = False
                break

            differences = calculate_metric_differences(
                sampled_fold,
                model_a,
                model_b,
            )

            differences["fold"] = fold
            differences["weight"] = len(sampled_fold)

            sampled_fold_results.append(differences)

        if not replicate_valid:
            continue

        sampled_fold_df = pd.DataFrame(
            sampled_fold_results
        )

        bootstrap_record = {
            "bootstrap_iteration": valid_replicates + 1,
        }

        for metric in metric_names:
            bootstrap_record[metric] = np.average(
                sampled_fold_df[metric],
                weights=sampled_fold_df["weight"],
            )

        bootstrap_records.append(
            bootstrap_record
        )

        valid_replicates += 1

    if valid_replicates < n_bootstrap:
        raise RuntimeError(
            f"Only obtained {valid_replicates} valid bootstrap "
            f"replicates after {attempts} attempts."
        )

    bootstrap_df = pd.DataFrame(
        bootstrap_records
    )


    alpha = 1.0 - confidence_level
    lower_quantile = alpha / 2
    upper_quantile = 1.0 - alpha / 2

    summary_records = []

    for metric in metric_names:

        values = bootstrap_df[metric].to_numpy()

        summary_records.append({
            "model_a": model_a,
            "model_b": model_b,
            "comparison": f"{model_a} minus {model_b}",
            "metric": metric,
            "block_size_days": block_size_days,
            "n_bootstrap": n_bootstrap,
            "point_estimate":
                original_point_estimates[metric],
            "bootstrap_mean":
                values.mean(),
            "bootstrap_standard_error":
                values.std(ddof=1),
            "confidence_level":
                confidence_level,
            "ci_lower":
                np.quantile(
                    values,
                    lower_quantile,
                ),
            "ci_upper":
                np.quantile(
                    values,
                    upper_quantile,
                ),
            "bootstrap_probability_above_zero":
                np.mean(values > 0),
            "bootstrap_probability_below_zero":
                np.mean(values < 0),
        })

    summary_df = pd.DataFrame(
        summary_records
    )

    return summary_df, bootstrap_df
c_f = pd.DataFrame(confirmation_results)

print("\nFinalist fold results\n")

print(
    c_f[
        [
            "experiment",
            "fold",
            "validation_start",
            "validation_end",
            "validation_rows",
            "validation_fires",
            "validation_fire_rate",
            "n_features",
            "average_precision",
            "ap_over_prevalence",
            "brier_score",
            "log_loss",
        ]
    ].to_string(index=False)
)

c_f.to_csv(
    OUTPUT_DIR / "finalist_chronological_fold_results.csv",
    index=False,
)

predictions_long = pd.concat(prediction_frames, ignore_index=True)

prediction_keys = [
    "fold",
    "date",
    "US_L3CODE",
    "US_L3NAME",
    "fire_occurred"
]

# Every experiment should provide exactly one probability for each
# fold-date-ecoregion observation.
duplicate_predictions = predictions_long.duplicated(prediction_keys + ["experiment"]).sum()

assert duplicate_predictions == 0, (f"Found {duplicate_predictions} duplicated prediction rows.")

predictions_wide = (
    predictions_long
    .pivot(
        index=prediction_keys,
        columns="experiment",
        values="probability",
    )
    .reset_index()
)

predictions_wide.columns.name = None

required_models = [
    "baseline_all_features",
    "primary_compact_candidate",
    "primary_compact_candidate_without_transmission"
]

missing_models = (set(required_models) - set(predictions_wide.columns))

assert not missing_models, (f"Missing prediction columns: {sorted(missing_models)}")

assert predictions_wide[required_models].notna().all().all(), ("At least one model is missing validation predictions.")

# Confirm that each date still contains all 13 ecoregions.
rows_per_fold_date = (
    predictions_wide
    .groupby(["fold", "date"])
    .size()
)

assert rows_per_fold_date.eq(13).all(), ("At least one fold-date does not contain all 13 ecoregions.")

predictions_wide = (
    predictions_wide
    .sort_values(
        ["fold", "date", "US_L3CODE"]
    )
    .reset_index(drop=True)
)

predictions_wide.to_csv(OUTPUT_DIR / "finalist_paired_validation_predictions.csv", index=False,)

bootstrap_comparisons = [
    {
        "comparison_name":
            "without_transmission_minus_baseline",

        "model_a":
            "primary_compact_candidate_without_transmission",

        "model_b":
            "baseline_all_features",
    },
    {
        "comparison_name":
            "without_transmission_minus_with_transmission",

        "model_a":
            "primary_compact_candidate_without_transmission",

        "model_b":
            "primary_compact_candidate",
    },
]

all_bootstrap_summaries = []
all_bootstrap_draws = []

for comparison in bootstrap_comparisons:

    summary_df, draws_df = (
        paired_temporal_block_bootstrap(
            predictions_df=predictions_wide,
            model_a=comparison["model_a"],
            model_b=comparison["model_b"],
            block_size_days=30,
            n_bootstrap=2000,
            confidence_level=0.95,
            random_state=1,
        )
    )

    summary_df["comparison_name"] = (
        comparison["comparison_name"]
    )

    draws_df["comparison_name"] = (
        comparison["comparison_name"]
    )

    all_bootstrap_summaries.append(
        summary_df
    )

    all_bootstrap_draws.append(
        draws_df
    )

bootstrap_summary = pd.concat(
    all_bootstrap_summaries,
    ignore_index=True,
)

bootstrap_draws = pd.concat(
    all_bootstrap_draws,
    ignore_index=True,
)

print("\nPaired temporal block-bootstrap results\n")

print(
    bootstrap_summary[
        [
            "comparison_name",
            "metric",
            "point_estimate",
            "bootstrap_standard_error",
            "ci_lower",
            "ci_upper",
            "bootstrap_probability_above_zero",
            "bootstrap_probability_below_zero",
        ]
    ].to_string(index=False)
)

bootstrap_summary.to_csv(OUTPUT_DIR / "paired_temporal_bootstrap_summary.csv", index=False)

bootstrap_draws.to_csv(OUTPUT_DIR / "paired_temporal_bootstrap_draws.csv", index=False)
