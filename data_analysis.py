"""
data_analysis.py
Provides summary statistics and quick exploratory information for CLI usage.
"""
from typing import Optional
import pandas as pd


def summary_table(df: pd.DataFrame, sample_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Returns a summary DataFrame with columns:
    column, dtype, null_count, unique_count, top_value, top_freq
    """
    if sample_rows and sample_rows > 0:
        df = df.sample(min(sample_rows, len(df)))

    rows = []
    for col in df.columns:
        dtype = df[col].dtype
        nulls = int(df[col].isna().sum())
        uniq = int(df[col].nunique(dropna=True))
        top_val = None
        top_freq = 0
        if uniq > 0:
            vc = df[col].value_counts(dropna=True)
            if not vc.empty:
                top_val = vc.index[0]
                top_freq = int(vc.iloc[0])
        rows.append({
            "column": col,
            "dtype": str(dtype),
            "null_count": nulls,
            "unique_count": uniq,
            "top_value": top_val,
            "top_freq": top_freq
        })

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, sample_rows: Optional[int] = None):
    s = summary_table(df, sample_rows)
    # Print a neat (not too wide) view for CLI
    print(s.to_string(index=False))
