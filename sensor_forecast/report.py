"""Standalone HTML evaluation dashboard, with no scripts or external assets."""

from html import escape


def render_html(result: dict, sensor: str) -> str:
    plot = result["plot"]
    if sensor not in plot["actual"]:
        raise ValueError("Unknown sensor: " + sensor)

    actual = plot["actual"][sensor]
    ridge = plot["ridge"][sensor]
    naive = plot["persistence"][sensor]
    values = actual + ridge + naive
    minimum, maximum = min(values), max(values)
    span = max(maximum - minimum, 1e-9)
    width, height = 840, 210

    def points(series: list[float]) -> str:
        return " ".join(
            f"{32 + index * width / max(len(series) - 1, 1):.2f},"
            f"{18 + (maximum - value) * height / span:.2f}"
            for index, value in enumerate(series)
        )

    rows = []
    wins = 0
    for metric in result["sensors"]:
        wins += int(metric["ridge_beats_baseline"])
        name = escape(metric["sensor"])
        result_label = "Ridge wins" if metric["ridge_beats_baseline"] else "Baseline wins / tie"
        rows.append(
            "<tr><th scope='row'>" + name + "</th><td>"
            + f"{metric['ridge_mae']:.8f}</td><td>{metric['persistence_mae']:.8f}</td>"
            + "<td>" + result_label + "</td></tr>"
        )
    label = escape(sensor)
    total = len(result["sensors"])
    return f'''<!doctype html><html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SensorLab | Forecast evaluation</title>
<style>
:root{{color-scheme:dark;font-family:Inter,system-ui,Arial,sans-serif;background:#0b1019;color:#e9f2ff}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 90% 0%,#183f43,transparent 35%),#0b1019;min-height:100vh}}
main{{max-width:1160px;margin:0 auto;padding:58px 28px 90px}}.eyebrow{{color:#9af1d3;font-size:12px;letter-spacing:.18em;font-weight:800}}
h1{{font-size:clamp(42px,7vw,78px);letter-spacing:-.06em;margin:14px 0 12px;line-height:1.08}}
.intro{{max-width:720px;color:#a4b6c7;line-height:1.7}}.notice{{display:inline-block;color:#f3d9a3;background:#2b2219;border:1px solid #7a5f2b;padding:9px 13px;border-radius:100px;font-size:12px;margin-top:14px}}
.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:32px 0}}.card,.panel{{border:1px solid #2d3c4b;border-radius:18px;background:#141e2a;padding:25px}}
.title{{font-size:12px;color:#9dabbc;text-transform:uppercase;letter-spacing:.13em}}.value{{font-weight:800;font-size:39px;letter-spacing:-.04em;margin-top:14px}}
.panel{{margin-top:18px}}h2{{font-size:24px;letter-spacing:-.03em;margin:0 0 6px}}
.caption{{font-size:13px;color:#a1b0bc;line-height:1.6}}.graph{{width:100%;height:auto;margin-top:22px}}
.legend{{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;color:#c5cfdb;margin-top:16px}}.swatch{{display:inline-block;width:20px;height:3px;vertical-align:middle;margin-right:7px}}
table{{border-collapse:collapse;width:100%;margin-top:20px}}th,td{{padding:13px 9px;text-align:left;border-bottom:1px solid #2b3a49;font-size:14px}}td{{color:#c2cfdb}}
small{{color:#8ca2b3;line-height:1.65}}.overflow{{overflow-x:auto}}
@media(max-width:650px){{main{{padding:35px 15px}}.stats{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">SENSORLAB / REPRODUCIBLE FORECASTING</div>
<h1>Measure, then predict<span style="color:#9cf4d7">.</span></h1>
<p class="intro">A five-minute-ahead sensor forecasting experiment using a chronological holdout.
Compare a simple Ridge regression model against repeating the most recent measurement.
The numbers below come from the dataset, not illustrative estimates.</p>
<div class="notice">NOT a validated NYC flood warning or flood probability system</div>
<section class="stats" aria-label="Evaluation summary">
<div class="card"><div class="title">Training examples</div><div class="value">{result["train_rows"]:,}</div></div>
<div class="card"><div class="title">Holdout examples</div><div class="value">{result["test_rows"]:,}</div></div>
<div class="card"><div class="title">Sensors beating baseline</div><div class="value">{wins} / {total}</div></div>
</section>
<section class="panel"><h2>Forecast versus actual — {label}</h2>
<p class="caption">Evaluation only, original sensor units unknown. All lines use the same holdout timestamps and vertical scale.</p>
<svg class="graph" role="img" aria-label="Actual, Ridge and persistence forecast sequences for {label}"
viewBox="0 0 904 256" xmlns="http://www.w3.org/2000/svg">
<path d="M32 18V228H872" stroke="#506578" fill="none" stroke-width="1"/>
<polyline points="{points(naive)}" fill="none" stroke="#f3b968" stroke-width="2.4" opacity=".85"/>
<polyline points="{points(ridge)}" fill="none" stroke="#81b4ff" stroke-width="2.4"/>
<polyline points="{points(actual)}" fill="none" stroke="#7ef0b8" stroke-width="2.6"/>
</svg>
<div class="legend"><span><i class="swatch" style="background:#7ef0b8"></i>Observed</span>
<span><i class="swatch" style="background:#81b4ff"></i>Ridge forecast</span>
<span><i class="swatch" style="background:#f3b968"></i>Persistence baseline</span></div></section>
<section class="panel"><h2>All sensors</h2><p class="caption">Mean absolute error (MAE); lower is better. Scales differ by sensor, so don't compare absolute MAE across different sensors.</p>
<div class="overflow"><table><thead><tr><th>Sensor</th><th>Ridge MAE</th><th>Baseline MAE</th><th>Winner</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>
<p style="margin-top:28px"><small>{escape(result["methodology"])}<br>
{escape(result["limitations"])}<br>Five-minute sensor readings are not flood-event labels. No map-level risk or life-safety decisions should rely on this benchmark.</small></p>
</main></body></html>'''
