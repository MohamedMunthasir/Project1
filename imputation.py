"""
imputation.py
Flexible imputation for numeric and categorical columns.
"""
from typing import Dict, Optional
import pandas as pd


def impute_dataframe(df: pd.DataFrame, strategy_map: Optional[Dict[str, str]] = None, inplace: bool = False) -> pd.DataFrame:
    """
    Impute missing values column-wise.

    strategy_map example: {'age': 'median', 'gender': 'mode', 'salary': 'mean'}
    Supported strategies:
      - numeric: 'mean', 'median', 'constant'
      - categorical: 'mode', 'constant'
    If a column is not in strategy_map, defaults:
      - numeric -> 'mean'
      - categorical -> 'mode'

    Returns new DataFrame (unless inplace=True).
    """
    out = df if inplace else df.copy()
    s_map = strategy_map or {}

    for col in out.columns:
        miss = out[col].isna().sum()
        if miss == 0:
            continue

        dtype_is_numeric = pd.api.types.is_numeric_dtype(out[col])
        strat = s_map.get(col, "mean" if dtype_is_numeric else "mode")

        if strat == "mean":
            if not dtype_is_numeric:
                # try coercion, then mean
                out[col] = pd.to_numeric(out[col], errors="coerce")
            val = out[col].mean()
            out[col] = out[col].fillna(val)
        elif strat == "median":
            if not dtype_is_numeric:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            val = out[col].median()
            out[col] = out[col].fillna(val)
        elif strat == "mode":
            mode_vals = out[col].mode()
            fill = mode_vals.iloc[0] if not mode_vals.empty else ""
            out[col] = out[col].fillna(fill)
        elif strat == "constant":
            fill = 0 if dtype_is_numeric else "missing"
            out[col] = out[col].fillna(fill)
        else:
            raise ValueError(f"Unsupported strategy '{strat}' for column '{col}'")

        # debug/info
        imputed = miss - out[col].isna().sum()
        print(f"Imputed {imputed} values in column '{col}' using strategy '{strat}'")

    return out
