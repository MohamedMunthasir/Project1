"""
data_quality.py
Comprehensive data quality assessment and reporting
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime


class DataQualityAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report = {}
    
    def analyze_completeness(self) -> Dict:
        """Analyze missing values across dataset"""
        completeness = {}
        total_cells = len(self.df) * len(self.df.columns)
        total_missing = self.df.isna().sum().sum()
        
        completeness['overall'] = {
            'completeness_pct': round(100 * (1 - total_missing/total_cells), 2),
            'missing_cells': int(total_missing),
            'total_cells': int(total_cells)
        }
        
        column_missing = {}
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            column_missing[col] = {
                'missing_count': int(missing_count),
                'missing_pct': round(100 * missing_count / len(self.df), 2),
                'completeness_pct': round(100 * (1 - missing_count/len(self.df)), 2)
            }
        
        completeness['by_column'] = column_missing
        self.report['completeness'] = completeness
        return completeness
    
    def analyze_duplicates(self) -> Dict:
        """Detect and analyze duplicate rows"""
        duplicates = {}
        
        # Full duplicates
        duplicate_mask = self.df.duplicated()
        dup_count = duplicate_mask.sum()
        
        duplicates['full_duplicates'] = {
            'count': int(dup_count),
            'percentage': round(100 * dup_count / len(self.df), 2),
            'indices': self.df[duplicate_mask].index.tolist()
        }
        
        # Column-wise duplicates
        column_dups = {}
        for col in self.df.columns:
            dup_in_col = self.df[col].duplicated().sum()
            unique_vals = self.df[col].nunique()
            column_dups[col] = {
                'duplicate_count': int(dup_in_col),
                'unique_count': int(unique_vals),
                'uniqueness_pct': round(100 * unique_vals / len(self.df), 2)
            }
        
        duplicates['by_column'] = column_dups
        self.report['duplicates'] = duplicates
        return duplicates
    
    def analyze_consistency(self) -> Dict:
        """Check data type consistency and format"""
        consistency = {}
        
        for col in self.df.columns:
            col_data = self.df[col].dropna()
            
            # Type consistency
            actual_type = self.df[col].dtype
            inferred_type = pd.api.types.infer_dtype(col_data, skipna=True)
            
            # Cardinality check
            cardinality = self.df[col].nunique()
            cardinality_ratio = cardinality / len(self.df)
            
            # Suggest appropriate type
            if cardinality_ratio < 0.05 and cardinality < 50:
                suggested_type = 'categorical'
            elif actual_type == 'object' and inferred_type in ['string', 'mixed']:
                suggested_type = 'text'
            else:
                suggested_type = str(actual_type)
            
            consistency[col] = {
                'current_type': str(actual_type),
                'inferred_type': inferred_type,
                'suggested_type': suggested_type,
                'cardinality': int(cardinality),
                'cardinality_ratio': round(cardinality_ratio, 4),
                'is_high_cardinality': cardinality_ratio > 0.5
            }
        
        self.report['consistency'] = consistency
        return consistency
    
    def analyze_distributions(self) -> Dict:
        """Analyze statistical distributions of numeric columns"""
        distributions = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            distributions[col] = {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'q25': float(col_data.quantile(0.25)),
                'q75': float(col_data.quantile(0.75)),
                'zeros_count': int((col_data == 0).sum()),
                'zeros_pct': round(100 * (col_data == 0).sum() / len(col_data), 2)
            }
        
        self.report['distributions'] = distributions
        return distributions
    
    def generate_full_report(self) -> Dict:
        """Generate comprehensive quality report"""
        print("Generating data quality report...")
        
        self.analyze_completeness()
        self.analyze_duplicates()
        self.analyze_consistency()
        self.analyze_distributions()
        
        # Overall score
        completeness_score = self.report['completeness']['overall']['completeness_pct']
        duplicate_penalty = self.report['duplicates']['full_duplicates']['percentage']
        
        quality_score = completeness_score * 0.6 + (100 - duplicate_penalty) * 0.4
        
        self.report['overall_quality_score'] = round(quality_score, 2)
        self.report['generated_at'] = datetime.now().isoformat()
        
        return self.report
    
    def print_summary(self):
        """Print formatted quality report summary"""
        if not self.report:
            self.generate_full_report()
        
        print(f"\n{'='*70}")
        print(f"DATA QUALITY REPORT")
        print(f"{'='*70}\n")
        
        print(f"Overall Quality Score: {self.report['overall_quality_score']}/100")
        print(f"Dataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"Generated: {self.report['generated_at']}")
        
        print(f"\n{'-'*70}")
        print("COMPLETENESS")
        print(f"{'-'*70}")
        comp = self.report['completeness']['overall']
        print(f"Completeness: {comp['completeness_pct']}%")
        print(f"Missing Cells: {comp['missing_cells']:,} / {comp['total_cells']:,}")
        
        print(f"\n{'-'*70}")
        print("DUPLICATES")
        print(f"{'-'*70}")
        dup = self.report['duplicates']['full_duplicates']
        print(f"Duplicate Rows: {dup['count']} ({dup['percentage']}%)")
        
        print(f"\n{'-'*70}")
        print("DATA TYPES & CONSISTENCY")
        print(f"{'-'*70}")
        for col, info in list(self.report['consistency'].items())[:10]:
            print(f"{col:30} {info['current_type']:15} → {info['suggested_type']:15}")
        
        if len(self.report['consistency']) > 10:
            print(f"... and {len(self.report['consistency']) - 10} more columns")
        
        print(f"\n{'='*70}\n")
