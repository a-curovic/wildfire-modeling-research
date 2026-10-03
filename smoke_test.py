"""Small local smoke test for the cleaned repository.

This does not run the research pipeline or reproduce model results. It only
checks a few basic pieces with synthetic data before the repository is pushed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from wildfire_config import PRIMARY_FEATURES
from wildfire_transformer import SkewAwareScaler


ROOT = Path(__file__).resolve().parent


def load_numbered_module(filename: str, module_name: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weather_cleaning_and_aggregation() -> None:
    aggregate_weather = load_numbered_module(
        "02_aggregate_weather.py",
        "aggregate_weather",
    )

    raw = pd.DataFrame(
        {
            "STATION": ["TEST001", "TEST001"],
            "DATE": ["2025-07-01T00:00:00", "2025-07-01T12:00:00"],
            "LATITUDE": [35.0, 35.0],
            "LONGITUDE": [-120.0, -120.0],
            "ELEVATION": [100.0, 100.0],
            "WND": ["090,1,N,0050,1", "180,1,N,0070,1"],
            "CIG": ["01000,1,N,N", "01200,1,N,N"],
            "VIS": ["016093,1,N,1", "016093,1,N,1"],
            "TMP": ["+0150,1", "+0170,1"],
            "DEW": ["+0100,1", "+0110,1"],
            "SLP": ["10132,1", "10120,1"],
        }
    )

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "station.csv"
        raw.to_csv(path, index=False)
        daily = aggregate_weather.aggregate_one_file(path)

    assert len(daily) == 1
    assert daily.loc[0, "temp_c_mean"] == 16.0
    assert daily.loc[0, "temp_c_min"] == 15.0
    assert daily.loc[0, "temp_c_max"] == 17.0
    assert daily.loc[0, "temp_c_count"] == 2
    assert np.isclose(daily.loc[0, "wind_speed_ms_mean"], 6.0)


def test_scaler() -> None:
    X = pd.DataFrame(
        {
            "roughly_symmetric": [1.0, 2.0, 3.0, 4.0, 5.0],
            "strongly_skewed": [1.0, 1.0, 1.0, 1.0, 100.0],
        }
    )
    scaler = SkewAwareScaler(skew_threshold=1.0)
    transformed = scaler.fit_transform(X)
    assert transformed.shape == X.shape
    assert np.isfinite(transformed).all()


def test_frozen_feature_list() -> None:
    assert len(PRIMARY_FEATURES) == 36
    assert len(set(PRIMARY_FEATURES)) == 36
    assert "US_L3NAME" in PRIMARY_FEATURES


def main() -> None:
    test_weather_cleaning_and_aggregation()
    test_scaler()
    test_frozen_feature_list()
    print("Smoke test passed.")
    print("Checked weather cleaning/aggregation, SkewAwareScaler, and the 36-feature list.")


if __name__ == "__main__":
    main()
