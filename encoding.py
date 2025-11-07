"""
encoding.py
Provides automatic encoding options for categorical columns.
Supports:
- Label Encoding
- One-Hot Encoding
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def detect_categorical_columns(df: pd.DataFrame):
    """Return a list of categorical/object columns."""
    return [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name == "category"]


def label_encode(df: pd.DataFrame, columns=None):
    """Apply label encoding to specified columns (or all categorical if None)."""
    df_copy = df.copy()
    le = LabelEncoder()
    cols = columns or detect_categorical_columns(df_copy)

    for c in cols:
        try:
            df_copy[c] = le.fit_transform(df_copy[c].astype(str))
            print(f"Label encoded column: {c}")
        except Exception as e:
            print(f"Skipping column '{c}' due to error: {e}")
    return df_copy


def one_hot_encode(df: pd.DataFrame, columns=None, drop_first=True):
    """Apply one-hot encoding using pandas.get_dummies."""
    cols = columns or detect_categorical_columns(df)
    if not cols:
        print("No categorical columns found to one-hot encode.")
        return df
    print(f"One-hot encoding columns: {cols}")
    return pd.get_dummies(df, columns=cols, drop_first=drop_first)
