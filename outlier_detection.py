"""
outlier_detection.py
Robust outlier detection and handling for numerical columns
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from scipy import stats


class OutlierDetector:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.outlier_info = {}
    
    def detect_iqr(self, columns: List[str], multiplier: float = 1.5) -> Dict:
        """Detect outliers using IQR method"""
        outliers = {}
        for col in columns:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            
            outlier_mask = (self.df[col] < lower) | (self.df[col] > upper)
            outlier_count = outlier_mask.sum()
            
            outliers[col] = {
                'count': int(outlier_count),
                'percentage': round(100 * outlier_count / len(self.df), 2),
                'lower_bound': lower,
                'upper_bound': upper,
                'indices': self.df[outlier_mask].index.tolist()
            }
        
        self.outlier_info['iqr'] = outliers
        return outliers
    
    def detect_zscore(self, columns: List[str], threshold: float = 3.0) -> Dict:
        """Detect outliers using Z-score method"""
        outliers = {}
        for col in columns:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outlier_mask = pd.Series(False, index=self.df.index)
            outlier_mask[self.df[col].notna()] = z_scores > threshold
            
            outlier_count = outlier_mask.sum()
            
            outliers[col] = {
                'count': int(outlier_count),
                'percentage': round(100 * outlier_count / len(self.df), 2),
                'threshold': threshold,
                'indices': self.df[outlier_mask].index.tolist()
            }
        
        self.outlier_info['zscore'] = outliers
        return outliers
    
    def remove_outliers(self, method: str = 'iqr', columns: List[str] = None) -> pd.DataFrame:
        """Remove detected outliers"""
        if method not in self.outlier_info:
            raise ValueError(f"Run detect_{method} first")
        
        outliers = self.outlier_info[method]
        all_outlier_indices = set()
        
        for col, info in outliers.items():
            if columns is None or col in columns:
                all_outlier_indices.update(info['indices'])
        
        cleaned_df = self.df.drop(index=list(all_outlier_indices))
        print(f"Removed {len(all_outlier_indices)} outlier rows ({len(all_outlier_indices)/len(self.df)*100:.2f}%)")
        
        return cleaned_df.reset_index(drop=True)
    
    def cap_outliers(self, method: str = 'iqr', columns: List[str] = None) -> pd.DataFrame:
        """Cap outliers at boundaries instead of removing"""
        if method != 'iqr':
            raise ValueError("Capping only supported for IQR method")
        
        if 'iqr' not in self.outlier_info:
            raise ValueError("Run detect_iqr first")
        
        df_capped = self.df.copy()
        outliers = self.outlier_info['iqr']
        
        for col, info in outliers.items():
            if columns is None or col in columns:
                df_capped[col] = df_capped[col].clip(
                    lower=info['lower_bound'],
                    upper=info['upper_bound']
                )
                print(f"Capped outliers in column '{col}'")
        
        return df_capped
    
    def print_summary(self, method: str = 'iqr'):
        """Print formatted outlier summary"""
        if method not in self.outlier_info:
            print(f"No outlier detection results for method '{method}'")
            return
        
        print(f"\n{'='*60}")
        print(f"OUTLIER DETECTION SUMMARY ({method.upper()} Method)")
        print(f"{'='*60}\n")
        
        outliers = self.outlier_info[method]
        for col, info in outliers.items():
            print(f"Column: {col}")
            print(f"  Outliers: {info['count']} ({info['percentage']}%)")
            if 'lower_bound' in info:
                print(f"  Valid Range: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
            print()
