# """
# encoding.py
# Provides automatic encoding options for categorical columns.
# Supports:
# - Label Encoding
# - One-Hot Encoding
# """
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder


# def detect_categorical_columns(df: pd.DataFrame):
#     """Return a list of categorical/object columns."""
#     return [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name == "category"]


# def label_encode(df: pd.DataFrame, columns=None):
#     """Apply label encoding to specified columns (or all categorical if None)."""
#     df_copy = df.copy()
#     le = LabelEncoder()
#     cols = columns or detect_categorical_columns(df_copy)

#     for c in cols:
#         try:
#             df_copy[c] = le.fit_transform(df_copy[c].astype(str))
#             print(f"Label encoded column: {c}")
#         except Exception as e:
#             print(f"Skipping column '{c}' due to error: {e}")
#     return df_copy


# def one_hot_encode(df: pd.DataFrame, columns=None, drop_first=True):
#     """Apply one-hot encoding using pandas.get_dummies."""
#     cols = columns or detect_categorical_columns(df)
#     if not cols:
#         print("No categorical columns found to one-hot encode.")
#         return df
#     print(f"One-hot encoding columns: {cols}")
#     return pd.get_dummies(df, columns=cols, drop_first=drop_first)


"""
encoding.py (IMPROVED with cardinality checking)
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def detect_categorical_columns(df: pd.DataFrame, max_cardinality: int = 50):
    """
    Return categorical columns with reasonable cardinality for encoding.
    Excludes high-cardinality columns like IDs, Names, etc.
    """
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            unique_count = df[col].nunique()
            cardinality_ratio = unique_count / len(df)
            
            # Only include if cardinality is reasonable
            if unique_count <= max_cardinality and cardinality_ratio < 0.5:
                cat_cols.append(col)
            else:
                print(f"⚠️  Skipping high-cardinality column '{col}' ({unique_count} unique values)")
    
    return cat_cols


def label_encode(df: pd.DataFrame, columns=None):
    """Apply label encoding to specified columns"""
    df_copy = df.copy()
    le = LabelEncoder()
    cols = columns or detect_categorical_columns(df_copy)
    
    for c in cols:
        try:
            df_copy[c] = le.fit_transform(df_copy[c].astype(str))
            print(f"✓ Label encoded column: {c}")
        except Exception as e:
            print(f"⚠️  Skipping column '{c}' due to error: {e}")
    return df_copy


def one_hot_encode(df: pd.DataFrame, columns=None, drop_first=True):
    """Apply one-hot encoding with cardinality check"""
    cols = columns or detect_categorical_columns(df)
    
    if not cols:
        print("No suitable categorical columns found for encoding.")
        return df
    
    # Additional safety check
    safe_cols = []
    for col in cols:
        unique = df[col].nunique()
        if unique > 50:
            print(f"⚠️  Skipping '{col}' - too many unique values ({unique})")
        else:
            safe_cols.append(col)
    
    if not safe_cols:
        print("No columns passed cardinality check.")
        return df
    
    print(f"✓ One-hot encoding columns: {safe_cols}")
    return pd.get_dummies(df, columns=safe_cols, drop_first=drop_first)
