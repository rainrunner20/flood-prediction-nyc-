"""One-step sensor forecasting. Never train on a randomly shuffled time series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FREQUENCY = pd.Timedelta(minutes=5)
LAGS = (0, 1, 2, 3)


def read_sensor_data(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    columns = [column for column in frame.columns if column.startswith("fs-")]
    if "Time" not in frame or not columns:
        raise ValueError("Expected Time and at least one fs-* sensor column.")
    if len(frame) < 100:
        raise ValueError("At least 100 observations are required for this baseline.")
    try:
        frame["Time"] = pd.to_datetime(
            frame["Time"], format="%Y/%m/%d %H:%M:%S.%f", utc=True, errors="raise"
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("Time must use the source CSV timestamp format.") from exc
    if frame["Time"].isna().any() or frame["Time"].duplicated().any():
        raise ValueError("Timestamps may not be missing or duplicated.")
    if not frame["Time"].is_monotonic_increasing:
        raise ValueError("Timestamps must be chronologically sorted.")
    if not frame["Time"].diff().iloc[1:].eq(FREQUENCY).all():
        raise ValueError("This evaluation requires uninterrupted five-minute readings.")

    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    matrix = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Sensor values must be numeric and finite.")
    if (matrix < 0).any():
        raise ValueError("Sensor values must be non-negative.")
    return frame, columns


def run_forecast(frame: pd.DataFrame, columns: list[str]) -> dict:
    if not columns or any(column not in frame for column in columns):
        raise ValueError("Valid sensor column names are required.")
    features = pd.concat(
        [frame[columns].shift(lag).add_suffix(f"_lag{lag}") for lag in LAGS],
        axis=1,
    )
    labels = frame[columns].shift(-1)
    # Each row uses readings available at or before time t to forecast t+5m.
    valid = features.notna().all(axis=1) & labels.notna().all(axis=1)
    features = features.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].reset_index(drop=True)
    times = frame.loc[valid, "Time"].reset_index(drop=True)

    if len(features) < 75:
        raise ValueError("Not enough complete windows after adding lag features.")
    boundary = int(len(features) * 0.8)
    # Leave one row between train and test so even the final training label
    # cannot use an observation from the holdout feature window.
    x_train, y_train = features.iloc[:boundary-1], labels.iloc[:boundary-1]
    x_test, y_test = features.iloc[boundary:], labels.iloc[boundary:]
    if len(x_train) < 40 or len(x_test) < 10:
        raise ValueError("Not enough data for a chronological train/test holdout.")

    estimator = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])
    estimator.fit(x_train, y_train)
    forecast = np.maximum(estimator.predict(x_test), 0.0)
    actual = y_test.to_numpy(dtype=float)
    # The last observed sensor reading is the most important baseline to beat.
    persistence = x_test[[f"{column}_lag0" for column in columns]].to_numpy(dtype=float)
    metrics = []
    for j, name in enumerate(columns):
        ridge_mae = float(np.mean(np.abs(actual[:, j] - forecast[:, j])))
        naive_mae = float(np.mean(np.abs(actual[:, j] - persistence[:, j])))
        metrics.append({
            "sensor": name,
            "ridge_mae": round(ridge_mae, 8),
            "persistence_mae": round(naive_mae, 8),
            "ridge_beats_baseline": ridge_mae < naive_mae,
        })

    return {
        "dataset": "Week 2&3/flood_sensor_data_Zhang.csv",
        "target": "Next five-minute sensor reading (NOT a flood-event label)",
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "last_train_feature_time": times.iloc[boundary - 2].isoformat(),
        "first_test_feature_time": times.iloc[boundary].isoformat(),
        "holdout_fraction": 0.2,
        "lags": list(LAGS),
        "sensors": metrics,
        "plot": {
            "time": [t.isoformat() for t in times.iloc[boundary:]],
            "actual": {name: actual[:, j].tolist() for j, name in enumerate(columns)},
            "ridge": {name: forecast[:, j].tolist() for j, name in enumerate(columns)},
            "persistence": {name: persistence[:, j].tolist() for j, name in enumerate(columns)},
        },
        "methodology": "Train-only standardization, Ridge(alpha=10), chronological 80/20 split, one-row label gap, no shuffle.",
        "limitations": "291 historical rows, no geolocation mapping, uncertain sensor units, no weather features or labeled flood events.",
    }
