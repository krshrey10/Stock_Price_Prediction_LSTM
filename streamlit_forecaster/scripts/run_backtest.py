import argparse
from pathlib import Path
import pandas as pd
from src.forecast import run_backtest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--horizon", type=int, default=None)
    args = p.parse_args()

    out = run_backtest(args.ticker, horizon=args.horizon)
    Path("outputs/backtests").mkdir(parents=True, exist_ok=True)
    out_path = Path(f"outputs/backtests/{args.ticker}_backtest.csv")
    out["backtests"].to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print("\nDiebold–Mariano vs GBDT:\n", out["dm"])

