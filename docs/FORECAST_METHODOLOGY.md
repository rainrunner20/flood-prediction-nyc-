# Sensor forecasting benchmark: assumptions and reproducibility

## Actual source

This project includes a historical file:
`Week 2&3/flood_sensor_data_Zhang.csv`. Inspection of that file shows
291 five-minute rows with 13 `fs-*` sensor columns from September 29–30, 2023.
The source does not include clear sensor units, station locations, official
flood-event labels, or a demonstrated data provenance chain. Its title does not
prove that measurements represent NYC-wide flooding.

The Kaggle `flood_kaggle.csv` dataset in older notebooks is separate; results
from one dataset must not be described as validated predictions on the other.

## What the model predicts

The **next five-minute reading at each sensor**, not a flood occurrence.
For each timestamp t, features are readings at t, t-5, t-10 and t-15.
The target is the measurement at t+5.

We use a chronological ~80/20 train/test split with a one-row gap at the
boundary. All standardization and regression fitting happen on training data
only. The baseline simply repeats the last sensor reading. Ridge regression
with alpha=10 is an interpretable, deterministic first comparator; negative
predictions are clamped to zero because the input values are nonnegative.

The report records per-sensor mean absolute error (MAE). We do not claim a
superior model unless its measured MAE beats persistence on the untouched
holdout. We deliberately avoid extrapolating to official flood warnings.

## Real-world requirements before any flood-risk claims

- Verify exact sensor metadata, units, calibration and station geolocations.
- Join trustworthy rainfall, tide, terrain and verified historical flooding labels.
- Extend observations across multiple seasons and storms.
- Use time-based cross validation with temporal gaps and drift checks.
- Evaluate rare flood events with appropriate recall, precision, calibration,
  false-alert and lead-time metrics.
- Add monitoring, retraining criteria and human review before decision support.

## Run

```powershell
py -m pip install -r requirements-forecast.txt
py -m sensor_forecast --input "Week 2&3/flood_sensor_data_Zhang.csv" --output reports/metrics.json --html reports/dashboard.html --sensor fs-00096
Start-Process reports/dashboard.html
py -m unittest discover -s tests -v
```

This is a testable **research baseline**, not a deployed or operational
life-safety service.
