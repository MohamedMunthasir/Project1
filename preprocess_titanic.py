"""
preprocess_titanic.py
Proper Titanic preprocessing workflow
"""
import pandas as pd
from data_input import read_file
from imputation import impute_dataframe
from outlier_detection import OutlierDetector
from encoding import one_hot_encode, detect_categorical_columns
from feature_scaling import scale_dataframe
from data_quality import DataQualityAnalyzer

# 1. Load data
print("Loading Titanic dataset...")
df = read_file("titanic_original.csv")
print(f"Original shape: {df.shape}")

# 2. Drop useless columns (high cardinality, not predictive)
print("\n🗑️  Dropping useless columns...")
drop_cols = ['passengerid', 'name', 'ticket', 'cabin']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
print(f"Shape after dropping: {df.shape}")

# 3. Quality check BEFORE
print("\n📊 BEFORE PREPROCESSING:")
qa_before = DataQualityAnalyzer(df)
qa_before.generate_full_report()
print(f"Quality Score: {qa_before.report['overall_quality_score']}/100")

# 4. Handle missing values
print("\n🧹 Handling missing values...")
df = impute_dataframe(df)

# 5. Handle outliers
print("\n🎯 Handling outliers...")
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
detector = OutlierDetector(df)
outliers = detector.detect_iqr(numeric_cols)
df = detector.cap_outliers()

# 6. Encode categorical (only Sex and Embarked)
print("\n🏷️  Encoding categorical variables...")
cat_cols = detect_categorical_columns(df, max_cardinality=10)
print(f"Encoding: {cat_cols}")
df = one_hot_encode(df, cat_cols)

# 7. Scale numeric features
print("\n⚖️  Scaling numeric features...")
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
df, _ = scale_dataframe(df, numeric_cols, method="standard")

# 8. Quality check AFTER
print("\n📊 AFTER PREPROCESSING:")
qa_after = DataQualityAnalyzer(df)
qa_after.generate_full_report()
print(f"Quality Score: {qa_after.report['overall_quality_score']}/100")

# 9. Save
print("\n💾 Saving processed dataset...")
df.to_csv("titanic_preprocessed.csv", index=False)
print(f"✅ DONE! Final shape: {df.shape}")
print(f"Saved to: titanic_preprocessed.csv")

# 10. Show comparison
print(f"\n{'='*60}")
print("PREPROCESSING SUMMARY")
print(f"{'='*60}")
print(f"Before: Quality Score = {qa_before.report['overall_quality_score']}/100")
print(f"After:  Quality Score = {qa_after.report['overall_quality_score']}/100")
print(f"Columns: {qa_before.df.shape[1]} → {df.shape[1]}")
print(f"Rows: {qa_before.df.shape[0]} → {df.shape[0]}")
