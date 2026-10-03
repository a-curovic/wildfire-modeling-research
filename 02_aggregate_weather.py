"""Clean the NOAA station-year files and aggregate them to station-day rows.

The original workflow wrote a very large combined hourly CSV and read it back
before aggregation. This version does the same daily aggregation one file at a
time, which avoids that unnecessary intermediate file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from wildfire_config import INTERIM_DIR, RAW_DIR
import importlib.util

# Numbered filenames are convenient for the workflow but cannot be imported normally.
_cleaner_path = Path(__file__).resolve().parent / "01_clean_weather.py"
_spec = importlib.util.spec_from_file_location("clean_weather", _cleaner_path)
_clean_weather = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_clean_weather)
clean_file = _clean_weather.clean_file


WEATHER_ROOT = RAW_DIR / "ncei_global_hourly_CA_2013_2025"
OUTPUT_PATH = INTERIM_DIR / "weather_daily_agg.csv"

RAW_COLUMNS = [
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

AGGREGATION = {
    "temp_c": ["mean", "min", "max", "std", "count"],
    "wind_speed_ms": ["mean", "max", "std", "min", "count"],
    "ew_wind": ["mean", "count"],
    "ns_wind": ["mean", "count"],
    "dew_point_c": ["mean", "min", "max", "count"],
    "sea_level_pressure_hPa": ["mean", "min", "max", "count"],
    "ceiling_height_m": ["mean", "min", "max", "count"],
    "vis_m": ["mean", "min", "max", "count"],
    "wind_dir_deg_missingFlag": ["mean", "count"],
    "wind_type_missingFlag": ["mean", "count"],
    "wind_speed_ms_missingFlag": ["mean", "count"],
    "temp_c_missingFlag": ["mean", "count"],
    "dew_point_c_missingFlag": ["mean", "count"],
    "sea_level_pressure_hPa_missingFlag": ["mean", "count"],
    "ceiling_height_m_missingFlag": ["mean", "count"],
    "vis_m_missingFlag": ["mean", "count"],
}


def read_station_file(path: Path) -> tuple[pd.DataFrame, bool]:
    """Read only the NOAA fields used by the project.

    The fast C parser is used first. A small number of downloaded station files
    can end with a malformed/truncated final record; if that happens, the file
    is retried with pandas' Python parser and only malformed rows are skipped.
    The caller records every fallback so the data issue remains visible.
    """
    try:
        frame = pd.read_csv(
            path,
            usecols=RAW_COLUMNS,
            low_memory=False,
        )
        return frame, False
    except pd.errors.ParserError:
        frame = pd.read_csv(
            path,
            usecols=RAW_COLUMNS,
            engine="python",
            on_bad_lines="skip",
        )
        return frame, True


def aggregate_one_file(path: Path) -> tuple[pd.DataFrame, bool, int]:
    raw, used_fallback = read_station_file(path)

    # NOAA WND should contain exactly five comma-separated components.
    # A malformed WND row is kept for its other weather variables, while its
    # wind fields are treated as missing by clean_file().
    malformed_wnd_rows = int(
        raw["WND"].astype("string").str.count(",").ne(4).sum()
    )

    hourly = clean_file(raw)

    hourly["ew_wind"] = (
        np.sin(np.deg2rad(hourly["wind_dir_deg"])) * hourly["wind_speed_ms"] * -1
    )
    hourly["ns_wind"] = (
        np.cos(np.deg2rad(hourly["wind_dir_deg"])) * hourly["wind_speed_ms"] * -1
    )
    hourly = hourly.drop(columns=["wind_dir_deg"])
    hourly["DATE"] = pd.to_datetime(hourly["DATE"]).dt.floor("D")

    daily = (
        hourly.groupby(["STATION", "LATITUDE", "LONGITUDE", "DATE"], as_index=False)
        .agg(AGGREGATION)
    )
    daily.columns = ["_".join(c).rstrip("_") for c in daily.columns.to_flat_index()]

    rename_map = {
        c: c.replace("_missingFlag_mean", "_missingRate")
        for c in daily.columns
        if c.endswith("_missingFlag_mean")
    }
    daily = daily.rename(columns=rename_map).rename(columns={"DATE": "date"})
    return daily, used_fallback, malformed_wnd_rows


def main() -> None:
    files = [
        p for p in WEATHER_ROOT.rglob("*.csv")
        if p.name.lower() != "isd-history.csv"
    ]
    if not files:
        raise FileNotFoundError(f"No NOAA CSV files found under {WEATHER_ROOT}")

    print(f"Found {len(files)} station-year weather files.")
    daily_parts = []
    fallback_files = []
    malformed_wnd_files = []

    for i, path in enumerate(sorted(files), start=1):
        try:
            daily, used_fallback, malformed_wnd_rows = aggregate_one_file(path)
            daily_parts.append(daily)
            if used_fallback:
                fallback_files.append(path)
                print(f"Warning: skipped malformed CSV row(s) while reading {path}")
            if malformed_wnd_rows:
                malformed_wnd_files.append((path, malformed_wnd_rows))
                print(
                    f"Warning: treated {malformed_wnd_rows} malformed WND row(s) "
                    f"as missing wind data in {path}"
                )
        except Exception as exc:
            raise RuntimeError(f"Failed while processing {path}") from exc

        if i % 100 == 0 or i == len(files):
            print(f"Processed {i}/{len(files)} files")

    weather_daily = pd.concat(daily_parts, ignore_index=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    weather_daily.to_csv(OUTPUT_PATH, index=False)

    warning_path = INTERIM_DIR / "weather_parse_warnings.txt"
    warning_sections = []

    if fallback_files:
        warning_sections.append(
            "Files read with the tolerant Python parser because the original CSV "
            "contained malformed row(s):\n"
            + "\n".join(str(path) for path in fallback_files)
        )

    if malformed_wnd_files:
        warning_sections.append(
            "Files containing malformed NOAA WND values. These rows were kept, "
            "but wind fields were treated as missing:\n"
            + "\n".join(
                f"{path} | malformed WND rows: {count}"
                for path, count in malformed_wnd_files
            )
        )

    if warning_sections:
        warning_path.write_text("\n\n".join(warning_sections) + "\n")
        print(f"Weather warnings saved to {warning_path}")
    elif warning_path.exists():
        warning_path.unlink()

    print(f"Saved {len(weather_daily):,} station-day rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
