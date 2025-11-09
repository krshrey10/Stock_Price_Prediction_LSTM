import argparse, numpy as np, pandas as pd
from scipy.stats import t as student_t
from pathlib import Path

def metrics(df):
    e = df["y"] - df["yhat"]
    mae = e.abs().mean()
    rmse = np.sqrt((e**2).mean())
    mape = (e.abs()/(df["y"].abs()+1e-12)).mean()*100
    return pd.Series({"MAE": mae, "RMSE": rmse, "MAPE": mape})

def dm_test(y, e1, e2, h=1):
    d = e1**2 - e2**2
    T = len(d)
    if T < 5 or np.allclose(d.var(ddof=1), 0):
        return np.nan, np.nan, T
    dm = d.mean() / np.sqrt(d.var(ddof=1) / T + 1e-12)
    p = 2*(1 - student_t.cdf(abs(dm), df=T-1))
    return float(dm), float(p), int(T)

def main(path):
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"\nLoaded {path} with {len(df)} rows")

    # 1) Metrics per model
    met = df.groupby("model").apply(metrics).sort_values("RMSE")
    print("\n=== Metrics (lower is better) ===")
    print(met.round(4))

    # 2) DM vs GBDT (for Naive & MA)
    y = df.pivot_table(index=["fold","date"], columns="model", values="y")
    yhat = df.pivot_table(index=["fold","date"], columns="model", values="yhat")
    rows = []
    for name in ["Naive", "MA(5)"]:
        if name not in yhat.columns or "GBDT" not in yhat.columns:
            continue
        idx = y.index.intersection(yhat.index)
        sub = pd.DataFrame({
            "y": y.loc[idx]["GBDT"],
            "m1": yhat.loc[idx][name],
            "m2": yhat.loc[idx]["GBDT"]
        }).dropna()
        dm, p, n = dm_test(sub["y"].values, (sub["y"]-sub["m1"]).values, (sub["y"]-sub["m2"]).values, h=1)
        rows.append({"model": name, "vs": "GBDT", "DM": dm, "p_value": p, "n": n})
    dm_df = pd.DataFrame(rows)
    print("\n=== Diebold–Mariano vs GBDT ===")
    print(dm_df)

    # 3) Save a README-ready markdown table
    md = ["| Model | MAE | RMSE | MAPE |",
          "|------:|----:|-----:|-----:|"]
    for m, r in met.round(4).iterrows():
        md.append(f"| {m} | {r.MAE} | {r.RMSE} | {r.MAPE} |")
    out_md = Path("outputs") / "results_summary.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved summary table → {out_md}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=r"outputs/backtests/AAPL_backtest.csv")
    args = ap.parse_args()
    main(args.csv)
