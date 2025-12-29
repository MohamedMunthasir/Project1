"""
Improved visualization module for ML-Preprocessor-CLI
- Accepts column numbers (1-based) or names (case-insensitive)
- Selects numeric columns for correlation/pairplots, coerces when possible
- Provides safer fallbacks for categorical data (one-hot) when needed
- Uses seaborn/matplotlib in a robust way, with friendly prompts

Replace your existing visualization.py with this file (or import/merge changes).
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Sequence, Union


class DataVisualization:
    def __init__(self, data: pd.DataFrame):
        # Expect data columns already normalized to lowercase by DataInput
        self.data = data

    def _print_available_columns(self):
        print("\nAvailable columns:")
        for i, col in enumerate(self.data.columns, 1):
            print(f"{i}. {col}")

    def _parse_column_selection(self, raw: str) -> List[str]:
        """Accepts comma-separated input of either column numbers (1-based) or column names.
        Returns list of valid column names (in original case as in DataFrame).
        Raises ValueError on invalid selections.
        """
        if not raw:
            raise ValueError("No input provided")

        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        cols = list(self.data.columns)
        selected = []
        for tok in tokens:
            # try numeric index
            try:
                idx = int(tok)
                if 1 <= idx <= len(cols):
                    selected.append(cols[idx - 1])
                    continue
                else:
                    raise ValueError(f"Index {idx} is out of range")
            except ValueError:
                # treat as column name (case-insensitive)
                matches = [c for c in cols if c.lower() == tok.lower()]
                if matches:
                    selected.append(matches[0])
                else:
                    raise ValueError(f"Column '{tok}' not found")

        # dedupe while preserving order
        seen = set()
        final = []
        for c in selected:
            if c not in seen:
                final.append(c)
                seen.add(c)
        return final

    def select_columns_prompt(self) -> List[str]:
        self._print_available_columns()
        raw = input("\nEnter column numbers or names (comma-separated): ")
        try:
            cols = self._parse_column_selection(raw)
        except ValueError as e:
            print("Selection error:", e)
            return []
        return cols

    def analyze_data(self):
        print("\nData types:\n", self.data.dtypes)
        print("\nNull values:\n", self.data.isnull().sum())
        print("\nBasic statistics:\n", self.data.describe(include='all'))

    def scatterplot(self, x_col: str, y_col: str):
        # require numeric columns; try coercion if necessary
        df = self.data[[x_col, y_col]].copy()
        for col in [x_col, y_col]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if df[[x_col, y_col]].dropna().shape[0] == 0:
            print("No numeric data available for scatterplot after coercion.")
            return

        plt.figure(figsize=(9, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col)
        plt.title(f"Scatterplot: {y_col} vs {x_col}")
        plt.tight_layout()
        plt.show()

    def boxplot(self, x_col: str, y_col: str):
        plt.figure(figsize=(9, 6))
        # If y is numeric, good; else try coercion
        df = self.data[[x_col, y_col]].copy()
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')

        sns.boxplot(data=df, x=x_col, y=y_col)
        plt.title(f"Boxplot of {y_col} by {x_col}")
        plt.tight_layout()
        plt.show()

    def pairplot(self, columns: Sequence[str]):
        # pairplot works best with numeric columns; coerce where possible
        df = self.data.loc[:, columns].copy()
        coerced = df.apply(lambda s: pd.to_numeric(s, errors='coerce'))
        numeric_cols = coerced.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_cols:
            print("No numeric columns available for pairplot after coercion. Try selecting other columns.")
            return

        plt.figure()
        sns.pairplot(coerced[numeric_cols].dropna())
        plt.suptitle('Pairplot for Selected Numeric Columns', y=1.02)
        plt.show()

    def heatmap(self, columns: Sequence[str]):
        # Build a numeric dataframe for correlation
        df = self.data.loc[:, columns].copy()
        # 1) try selecting numeric dtypes
        numeric_df = df.select_dtypes(include=["number"]).copy()
        # 2) coerce any convertible columns
        if numeric_df.shape[1] == 0:
            coerced = df.apply(lambda s: pd.to_numeric(s, errors='coerce'))
            numeric_df = coerced.select_dtypes(include=["number"]).copy()

        # 3) if still no numeric columns, fallback to one-hot encoding for categorical
        if numeric_df.shape[1] == 0:
            print("No numeric columns detected — using one-hot encoding for categorical columns to compute correlations.")
            df_encoded = pd.get_dummies(df, drop_first=True)
            if df_encoded.shape[1] == 0:
                print("Unable to create numeric features from selected columns for correlation.")
                return
            corr = df_encoded.corr()
        else:
            corr = numeric_df.corr()

        plt.figure(figsize=(max(6, corr.shape[0]), max(6, corr.shape[1])))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation heatmap')
        plt.tight_layout()
        plt.show()

    def multivariate_analysis(self):
        cols = self.select_columns_prompt()
        if not cols:
            print("No valid columns selected.")
            return
        print("\nMultivariate Analysis Options:\n1. Pairplot\n2. Heatmap")
        choice = input("Choose the type of multivariate analysis (1 or 2): ")
        if choice == '1':
            self.pairplot(cols)
        elif choice == '2':
            self.heatmap(cols)
        else:
            print("Invalid choice")

    def run_visualization(self):
        print("Would you like to analyze the dataset before visualization? (yes/no)")
        if input().strip().lower() in ('yes', 'y'):
            self.analyze_data()

        print("\nVisualization Options:\n1. Scatterplot\n2. Boxplot\n3. Multivariate Analysis")
        choice = input("Choose the type of visualization (1-3): ")

        if choice not in {'1', '2', '3'}:
            print("Invalid choice")
            return

        if choice in {'1', '2'}:
            cols = self.select_columns_prompt()
            if len(cols) < 2:
                print("You need to select at least two columns for this plot.")
                return
            if choice == '1':
                self.scatterplot(cols[0], cols[1])
            else:
                self.boxplot(cols[0], cols[1])
        else:
            self.multivariate_analysis()

    # Add to DataVisualization class in visualization.py

def missing_data_heatmap(self):
    """Visualize missing data patterns"""
    import numpy as np
    
    plt.figure(figsize=(12, 8))
    
    # Create binary matrix: 1 for missing, 0 for present
    missing_matrix = self.data.isna().astype(int)
    
    if missing_matrix.sum().sum() == 0:
        print("No missing data to visualize")
        return
    
    sns.heatmap(missing_matrix, cbar=True, yticklabels=False, 
                cmap='YlOrRd', cbar_kws={'label': 'Missing Data'})
    plt.title('Missing Data Heatmap')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.tight_layout()
    plt.show()

