import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sensor_forecast.model import read_sensor_data, run_forecast
from sensor_forecast.report import render_html


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "sensor.csv"
        rng = np.random.default_rng(42)
        n = 130
        times = pd.date_range("2023-09-29", periods=n, freq="5min", tz="UTC")
        self.frame = pd.DataFrame({
            "Time": times.strftime("%Y/%m/%d %H:%M:%S.%f"),
            "fs-00001": 0.5 + 0.1 * np.sin(np.arange(n) / 9),
            "fs-00002": np.maximum(0, rng.normal(0.1, 0.01, n)),
        })
        self.frame.to_csv(self.path, index=False)

    def test_realistic_schema_and_holdout(self):
        data, columns = read_sensor_data(self.path)
        result = run_forecast(data, columns)
        self.assertEqual(columns, ["fs-00001", "fs-00002"])
        self.assertEqual(result["train_rows"] + result["test_rows"], len(data) - 5)
        self.assertLess(result["last_train_feature_time"], result["first_test_feature_time"])
        self.assertEqual(len(result["plot"]["actual"]["fs-00001"]), result["test_rows"])
        self.assertTrue(all(metric["ridge_mae"] >= 0 for metric in result["sensors"]))

    def test_baseline_uses_current_observed_value_not_future(self):
        data, columns = read_sensor_data(self.path)
        result = run_forecast(data, columns)
        observed = result["plot"]["actual"]["fs-00001"]
        persistent = result["plot"]["persistence"]["fs-00001"]
        self.assertNotEqual(observed, persistent)
        self.assertAlmostEqual(persistent[1], observed[0])

    def test_rejects_non_regular_timestamps(self):
        frame = self.frame.drop(index=3)
        frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "uninterrupted"):
            read_sensor_data(self.path)

    def test_rejects_duplicate_times_and_negative_values(self):
        frame = self.frame.copy()
        frame.loc[2, "Time"] = frame.loc[1, "Time"]
        frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            read_sensor_data(self.path)
        frame = self.frame.copy()
        frame.loc[4, "fs-00001"] = -0.1
        frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            read_sensor_data(self.path)

    def test_missing_columns_and_short_dataset(self):
        frame = self.frame.drop(columns=["fs-00001", "fs-00002"])
        frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "fs-"):
            read_sensor_data(self.path)
        self.frame.iloc[:20].to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "100"):
            read_sensor_data(self.path)

    def test_report_contains_reproducible_values_and_escapes_sensor(self):
        data, columns = read_sensor_data(self.path)
        result = run_forecast(data, columns)
        report = render_html(result, "fs-00001")
        self.assertIn("NOT a validated NYC flood warning", report)
        self.assertIn("fs-00001", report)
        with self.assertRaisesRegex(ValueError, "Unknown sensor"):
            render_html(result, '<script>')

if __name__ == "__main__":
    unittest.main()
