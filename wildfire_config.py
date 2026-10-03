from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATASET_PATH = PROCESSED_DIR / "fullDataSet3.csv"
TARGET = "fire_occurred"
DATE_COL = "date"
YEAR_COL = "year"
RANDOM_STATE = 42

PRIMARY_FEATURES = [
    "dew_point_c_max",
    "dew_point_c_mean",
    "dew_point_c_min",
    "dew_point_c_missingRate",
    "ew_wind_mean",
    "ns_wind_mean",
    "sea_level_pressure_hPa_mean",
    "sea_level_pressure_hPa_missingRate",
    "temp_c_max",
    "temp_c_mean",
    "temp_c_min",
    "temp_c_missingFlag_count",
    "temp_c_missingRate",
    "temp_c_std",
    "vis_m_max",
    "vis_m_missingRate",
    "wind_dir_deg_missingRate",
    "wind_speed_ms_max",
    "wind_speed_ms_mean",
    "wind_speed_ms_min",
    "wind_speed_ms_missingRate",
    "wind_speed_ms_std",
    "wind_type_missingRate",
    "month",
    "evi_mean",
    "evi_std",
    "sat_vp",
    "A_sat_vp",
    "RH_pct",
    "VPD_kpa",
    "HDWI_proxy",
    "DTR",
    "row_weather_missing_rate",
    "US_L3NAME",
    "ceiling_height_m_mean",
    "ceiling_height_m_missingRate",
]

CATEGORICAL_FEATURES = ["US_L3NAME"]

MISSING_RATE_COLUMNS = [
    "ceiling_height_m_missingRate",
    "dew_point_c_missingRate",
    "temp_c_missingRate",
    "wind_type_missingRate",
    "sea_level_pressure_hPa_missingRate",
    "vis_m_missingRate",
    "wind_speed_ms_missingRate",
    "wind_dir_deg_missingRate",
]
