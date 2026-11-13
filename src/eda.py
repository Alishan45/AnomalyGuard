"""
Exploratory Data Analysis (EDA) module for AnomalyGuard.
Inspects schema, distributions, missingness, and plausible anomaly patterns.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

def compute_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes comprehensive high-level summary of the dataset."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    
    summary = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "total_missing_values": total_missing,
        "missing_per_column": missing_counts,
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024))
    }
    return summary

def get_numeric_distribution_stats(df: pd.DataFrame, numeric_cols: List[str] = None) -> pd.DataFrame:
    """Computes key distribution metrics: Mean, Std, Median, IQR, Min, Max, Skewness."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude synthetic ground-truth flags if present from EDA distribution table
        numeric_cols = [c for c in numeric_cols if c not in ['is_synthetic_anomaly']]
        
    stats = []
    for col in numeric_cols:
        series = df[col].dropna()
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        stats.append({
            "Feature": col,
            "Count": int(series.count()),
            "Mean": float(series.mean()),
            "Std": float(series.std()),
            "Median": float(series.median()),
            "IQR": float(iqr),
            "Min": float(series.min()),
            "Max": float(series.max()),
            "Skewness": float(series.skew())
        })
    return pd.DataFrame(stats)

def compute_correlation_matrix(df: pd.DataFrame, numeric_cols: List[str] = None) -> pd.DataFrame:
    """Computes Pearson correlation matrix for numerical features."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['is_synthetic_anomaly']]
    return df[numeric_cols].corr()
