from typing import Iterator, Tuple
import pandas as pd

def expanding_walk_forward(df: pd.DataFrame, min_train: int, step: int, horizon: int = 1) -> Iterator[tuple[pd.Index, pd.Index]]:
    n = len(df)
    start_test = min_train
    while start_test + horizon <= n:
        end_test = min(start_test + step, n - horizon + 1)
        train_idx = df.index[:start_test]
        test_idx = df.index[start_test:end_test]
        yield train_idx, test_idx
        start_test = end_test
