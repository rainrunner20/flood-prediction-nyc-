"""Command-line evaluation runner that saves exact metrics and an HTML brief."""

import argparse
import json
from pathlib import Path

from sensor_forecast.model import read_sensor_data, run_forecast
from sensor_forecast.report import render_html


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a five-minute sensor forecast")
    parser.add_argument("--input", type=Path, default=Path("Week 2&3/flood_sensor_data_Zhang.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/sensor_metrics.json"))
    parser.add_argument("--html", type=Path, default=Path("reports/sensor_dashboard.html"))
    parser.add_argument("--sensor", default="fs-00096", help="Sensor for the forecast plot")
    args = parser.parse_args(argv)

    try:
        frame, columns = read_sensor_data(args.input)
        result = run_forecast(frame, columns)
        rendered = render_html(result, args.sensor)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.html.write_text(rendered, encoding="utf-8")
    print(f"Metrics: {args.output}\nDashboard: {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
