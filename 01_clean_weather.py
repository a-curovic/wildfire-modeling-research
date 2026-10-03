"""Clean one NOAA Global Hourly weather file.

The raw fields store several values inside comma-separated strings. This file
keeps the variables used later in the project, converts them to usable units,
and adds missing-value flags before daily aggregation.
"""

from datetime import datetime

import numpy as np
import pandas as pd


def separate_wind(df: pd.DataFrame) -> pd.DataFrame:
    """Split NOAA WND values without letting malformed rows stop the run.

    WND is expected to contain exactly five comma-separated components. If a
    source row contains a different number of components, its wind values are
    treated as missing rather than shifted into the wrong fields. The caller
    can audit those rows before cleaning.
    """
    wnd = df["WND"].astype("string")
    valid = wnd.str.count(",").eq(4)

    # n=4 guarantees exactly five output columns for valid NOAA WND records.
    wind = wnd.where(valid).str.split(",", n=4, expand=True)
    wind = wind.reindex(columns=range(5))
    wind.columns = [
        "wind_dir_deg_str",
        "wind_dir_qc",
        "wind_type",
        "wind_speed_raw",
        "wind_speed_qc",
    ]
    wind = wind.drop(columns=["wind_dir_qc", "wind_speed_qc"])
    return pd.concat([df.drop(columns=["WND"]), wind], axis=1)


def transform_wind(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["wind_speed_ms"] = pd.to_numeric(df["wind_speed_raw"], errors="coerce")
    df["wind_speed_ms"] = df["wind_speed_ms"].replace(9999, np.nan) * 0.1
    df["wind_speed_ms_missingFlag"] = df["wind_speed_ms"].isna().astype("int8")
    df = df.drop(columns=["wind_speed_raw"])

    df["wind_dir_deg"] = pd.to_numeric(df["wind_dir_deg_str"], errors="coerce")
    df["wind_dir_deg"] = df["wind_dir_deg"].replace(999, np.nan)
    df["wind_dir_deg_missingFlag"] = df["wind_dir_deg"].isna().astype("int8")
    df = df.drop(columns=["wind_dir_deg_str"])

    df["wind_type"] = df["wind_type"].replace("9", np.nan).astype("string")
    df["wind_type_missingFlag"] = df["wind_type"].isna().astype("int8")
    df["wind_type"] = df["wind_type"].astype("category")
    return df


def clean_ceiling(df: pd.DataFrame) -> pd.DataFrame:
    ceiling = df["CIG"].astype(str).str.split(",", expand=True)
    ceiling.columns = ["height", "qc", "method", "cavok"]
    df["ceiling_height_m"] = pd.to_numeric(ceiling["height"], errors="coerce")
    df["ceiling_height_m"] = df["ceiling_height_m"].replace(99999, np.nan)

    # Values at or above 22 km indicate that no low ceiling was observed.
    df["ceiling_height_m"] = df["ceiling_height_m"].mask(df["ceiling_height_m"] >= 22000)
    df["ceiling_height_m_missingFlag"] = df["ceiling_height_m"].isna().astype("int8")
    return df.drop(columns=["CIG"])


def clean_visibility(df: pd.DataFrame) -> pd.DataFrame:
    vis = df["VIS"].astype(str).str.split(",", expand=True)
    df["vis_m"] = pd.to_numeric(vis[0], errors="coerce").replace(999999, np.nan)
    df["vis_m_missingFlag"] = df["vis_m"].isna().astype("int8")
    return df.drop(columns=["VIS"])


def clean_temperature(df: pd.DataFrame) -> pd.DataFrame:
    temp = df["TMP"].astype(str).str.split(",", expand=True)
    df["temp_c"] = pd.to_numeric(temp[0], errors="coerce").replace(9999, np.nan) / 10.0
    df["temp_c_missingFlag"] = df["temp_c"].isna().astype("int8")
    return df.drop(columns=["TMP"])


def clean_dew_point(df: pd.DataFrame) -> pd.DataFrame:
    dew = df["DEW"].astype(str).str.split(",", expand=True)
    df["dew_point_c"] = pd.to_numeric(dew[0], errors="coerce").replace(9999, np.nan) / 10.0
    df["dew_point_c_missingFlag"] = df["dew_point_c"].isna().astype("int8")
    return df.drop(columns=["DEW"])


def clean_pressure(df: pd.DataFrame) -> pd.DataFrame:
    pressure = df["SLP"].astype(str).str.split(",", expand=True)
    df["sea_level_pressure_hPa"] = (
        pd.to_numeric(pressure[0], errors="coerce").replace(99999, np.nan) / 10.0
    )
    df["sea_level_pressure_hPa_missingFlag"] = (
        df["sea_level_pressure_hPa"].isna().astype("int8")
    )
    return df.drop(columns=["SLP"])


def clean_file(df: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned hourly fields used by the rest of the project."""
    df = df[
        [
            "STATION",
            "DATE",
            "LATITUDE",
            "LONGITUDE",
            "ELEVATION",
            "WND",
            "CIG",
            "VIS",
            "TMP",
            "DEW",
            "SLP",
        ]
    ].drop_duplicates()

    df["DATE"] = df["DATE"].apply(datetime.fromisoformat)
    df = separate_wind(df)
    df = transform_wind(df)
    df = clean_ceiling(df)
    df = clean_visibility(df)
    df = clean_temperature(df)
    df = clean_dew_point(df)
    df = clean_pressure(df)
    return df
