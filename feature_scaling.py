"""
feature_scaling.py
Safe feature scaling utilities. Attempts to use sklearn; if not installed, uses fallback implementations.
"""
from typing import List, Tuple
import pandas as pd
import numpy as np

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def _simple_minmax_scale(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float)
    min_v = np.nanmin(arr)
    max_v = np.nanmax(arr)
    if np.isclose(max_v, min_v):
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def scale_dataframe(df: pd.DataFrame, columns: List[str], method: str = "standard") -> Tuple[pd.DataFrame, object]:
    """
    Scale specified columns and return (scaled_df, scaler_object).
    If sklearn isn't available, returns basic fallback scaler object (None).
    method: 'standard' | 'minmax' | 'robust'
    """
    out = df.copy()
    cols = []
    for c in columns:
        # coerce to numeric where possible
        out[c] = pd.to_numeric(out[c], errors="coerce")
        cols.append(c)

    zero_var = [c for c in cols if out[c].nunique(dropna=True) <= 1 or out[c].std(ddof=0) == 0]
    if zero_var:
        print("Warning: zero-variance or single-unique columns (these will become zeros after scaling):", zero_var)

    scaler = None
    if SKLEARN_AVAILABLE:
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError("Unknown scaling method: " + method)

        # fill NaNs with 0 temporarily for fit_transform (user can choose other behavior)
        out[cols] = scaler.fit_transform(out[cols].fillna(0))
    else:
        # fallback minmax per column
        print("sklearn not available — using fallback min-max scaling per column.")
        for c in cols:
            out[c] = _simple_minmax_scale(out[c].to_numpy())
        scaler = None

    return out, scaler
