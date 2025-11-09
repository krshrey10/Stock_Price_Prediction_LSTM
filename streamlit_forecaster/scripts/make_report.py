# scripts/make_report.py
import os, sys, pandas as pd, datetime as dt
from jinja2 import Template

TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{{ title }}</title>
<style>
 body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,Inter,Arial; margin:32px;}
 h1,h2{margin:0 0 8px}
 table{border-collapse:collapse; width:100%; margin:14px 0}
 th,td{border:1px solid #ddd; padding:6px 8px; font-size:14px}
 .muted{color:#6b7280}
</style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="muted">{{ now }}</div>
  <h2>Latest Forecast</h2>
  {{ table_html|safe }}
  <h2>Notes</h2>
  <ul>
    <li>Uncertainty shown via model-specific intervals.</li>
    <li>Backtesting: walk-forward expanding windows; RMSE/MAE/MAPE + DM tests.</li>
  </ul>
</body>
</html>
"""

def main(csv_path, out_html):
    df = pd.read_csv(csv_path)
    html = Template(TEMPLATE).render(
        title="Stock Forecaster — Report",
        now=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        table_html=df.to_html(index=False)
    )
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print("Saved", out_html)

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "outputs/backtests/AAPL_backtest.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/report.html"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    main(csv, out)
