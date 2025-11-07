"""
data_input.py
Robust file reading utilities for ML-Preprocessor-CLI
"""
import os
from typing import Optional
import pandas as pd


def read_file(path: str, normalize_columns: bool = True, lowercase: bool = True, sample_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Read CSV/TSV/XLSX file and return DataFrame.

    Args:
        path: path to file
        normalize_columns: strip whitespace and collapse internal spaces
        lowercase: convert column names to lowercase
        sample_rows: if >0, return a sample of the DataFrame (useful for quick testing)

    Raises:
        FileNotFoundError, ValueError
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".csv", ".txt"]:
            # use python engine and sniff delimiter
            df = pd.read_csv(path, sep=None, engine="python")
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t")
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Supported: .csv, .tsv, .txt, .xls, .xlsx")
    except Exception as e:
        raise ValueError(f"Failed to read '{path}': {e}")

    if normalize_columns:
        new_cols = []
        for c in df.columns:
            if isinstance(c, str):
                c2 = " ".join(c.strip().split())  # collapse multiple spaces
                c2 = c2.lower() if lowercase else c2
                new_cols.append(c2)
            else:
                new_cols.append(c)
        df.columns = new_cols

    if sample_rows and isinstance(sample_rows, int) and sample_rows > 0:
        return df.sample(min(sample_rows, len(df))).reset_index(drop=True)

    return df
