from __future__ import annotations
import yaml
import numpy as np
import pandas as pd

from .data import load_prices
from .features import make_feature_frame
from .split import expanding_walk_forward
from .eval import diebold_mariano
from .models import baseline, gbdt


def _to_1d_aligned(pred, index: pd.Index) -> np.ndarray:
    """Coerce any prediction object to a 1-D float array aligned to `index`."""
    if isinstance(pred, pd.Series):
        return pred.reindex(index).astype(float).values
    if isinstance(pred, pd.DataFrame):
        return pred.iloc[:, 0].reindex(index).astype(float).values
    arr = np.asarray(pred).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Prediction is not 1-D after squeeze: shape={arr.shape}")
    if arr.shape[0] != len(index):
        raise ValueError(f"Prediction length {arr.shape[0]} != target length {len(index)}")
    return arr.astype(float)


def run_backtest(ticker: str, cfg_path: str = 'config.yaml', horizon: int | None = None) -> dict:
    # --- load config and data ---
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
    if horizon is None:
        horizon = cfg['split']['horizon']

    df_price = load_prices(ticker, cfg['data']['start'], cfg['data']['end'], cfg['data']['interval'])
    df = make_feature_frame(df_price, cfg)

    # --- walk-forward backtest ---
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(
        expanding_walk_forward(df, cfg['split']['min_train_period'], cfg['split']['step'], horizon=horizon),
        start=1,
    ):
        train, test = df.loc[tr_idx], df.loc[te_idx]
        target_col = 'y_t+1' if horizon == 1 else 'y7'
        y_true = test[target_col]

        # Models (fast set): Naive, MA(5), GBDT (quantiles)
        yhat_naive = baseline.naive_last_value(train, test)
        yhat_ma = baseline.moving_average(train, test)
        g_out = gbdt.fit_predict_quantiles(
            train, test,
            quantiles=(0.1, 0.5, 0.9),
            params=cfg['models']['gbdt']['params'],
        )

        models = {
            'Naive': yhat_naive,
            'MA(5)': yhat_ma,
            'GBDT': g_out['yhat'],
        }

        for name, pred in models.items():
            pred_vals = _to_1d_aligned(pred, y_true.index)
            for t, p in zip(y_true.index, pred_vals):
                rows.append({'fold': fold, 'date': t, 'model': name, 'y': float(y_true.loc[t]), 'yhat': float(p)})

    # If no folds produced (e.g., min_train too large), return empty-but-valid result
    if not rows:
        empty = pd.DataFrame(columns=['fold', 'date', 'model', 'y', 'yhat'])
        return {"backtests": empty, "dm": pd.DataFrame(columns=['model', 'vs', 'DM', 'p_value'])}

    results = pd.DataFrame(rows)
    results.set_index('date', inplace=True)

    # --- Diebold–Mariano: compare Naive & MA against GBDT (robust) ---
    dm_rows = []
    baseline_name = 'GBDT'
    res = results.reset_index()  # bring 'date' back as a column

    # y_true per (fold, date) from baseline rows
    y_true = (
        res[res['model'] == baseline_name]
        .set_index(['fold', 'date'])['y']
    )

    # wide preds per model
    preds = (
        res[res['model'].isin([baseline_name, 'Naive', 'MA(5)'])]
        .pivot_table(index=['fold', 'date'], columns='model', values='yhat')
    )

    idx = y_true.index.intersection(preds.index)
    y_true = y_true.loc[idx]
    preds = preds.loc[idx]

    for name in ['Naive', 'MA(5)']:
        if name not in preds.columns or baseline_name not in preds.columns:
            continue
        e1 = (y_true - preds[name]).dropna()
        e2 = (y_true - preds[baseline_name]).dropna()
        common = e1.index.intersection(e2.index)
        if len(common) == 0:
            continue
        dm = diebold_mariano(y_true.loc[common], e1.loc[common], e2.loc[common], h=horizon, loss='mse')
        dm_rows.append({'model': name, 'vs': baseline_name, **dm})

    dm_df = pd.DataFrame(dm_rows)

    # --- return final dict ---
    return {"backtests": results.reset_index(), "dm": dm_df}
