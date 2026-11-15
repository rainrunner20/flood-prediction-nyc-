# SensorLab: Five-minute sensor forecasting research

This repository retains the original experimental flood-ML notebooks and
datasets. Its new `sensor_forecast` module is a **reproducible, tested
one-step forecasting benchmark** using the included sensor file.

**Important:** These data do not establish a validated NYC flood-prediction
system. The benchmark predicts the next numeric sensor measurement, not
whether a flood will occur. The source lacks documented NYC station locations,
units, verified event labels and longitudinal coverage.

## What is implemented

- Checks sensor schema, timestamps, missing values and five-minute cadence.
- Uses the current and previous three readings to predict the next reading.
- Chronological 80/20 holdout with a one-row boundary gap.
- Train-only standardization and multi-output Ridge regression.
- Persistence baseline that repeats the most recent reading.
- Per-sensor out-of-sample MAE and a standalone dark HTML report.
- Offline regression tests and GitHub Actions dataset smoke test.

### Run on Windows PowerShell

```powershell
git clone https://github.com/rainrunner20/flood-prediction-nyc-.git
cd flood-prediction-nyc-
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-forecast.txt
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m sensor_forecast --input "Week 2&3/flood_sensor_data_Zhang.csv" --output reports/metrics.json --html reports/dashboard.html --sensor fs-00096
Start-Process reports/dashboard.html
```

Generated `reports/` files are ignored by Git. No cloud credentials or live
sensor access are needed. See [methodology](docs/FORECAST_METHODOLOGY.md).

## Roadmap, not claims

- [x] Reproducible sensor forecast and persistence benchmark
- [x] Chronological evaluation and standalone visualization
- [x] Input tests and automated CI smoke test
- [ ] Independent sensor metadata/units and NYC geocoding
- [ ] Rainfall, tide, flood labels and true citywide risk classification
- [ ] Spatial generalization and multi-season backtesting
- [ ] Real-time data ingestion, monitored deployment and validated alerting

Older notebooks remain as historical learning artifacts, not production ML
pipelines. This repository should not be cited as a validated flood warning tool.
