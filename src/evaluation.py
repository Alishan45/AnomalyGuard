"""
Evaluation, Threshold Sensitivity, and Interpretability Module for AnomalyGuard.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def evaluate_threshold_sensitivity(
    anomaly_scores: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    percentile_thresholds: List[float] = [90.0, 95.0, 97.5, 99.0, 99.5]
) -> pd.DataFrame:
    """
    Evaluates flag counts and performance metrics across multiple percentile thresholds.
    """
    results = []
    n_total = len(anomaly_scores)
    
    for pct in percentile_thresholds:
        cutoff = np.percentile(anomaly_scores, pct)
        flags = (anomaly_scores >= cutoff).astype(int)
        flag_count = int(np.sum(flags))
        flag_rate_pct = (flag_count / n_total) * 100.0
        
        row = {
            "Percentile_Cutoff": pct,
            "Score_Threshold": round(float(cutoff), 4),
            "Flagged_Count": flag_count,
            "Flag_Rate_%": round(flag_rate_pct, 2)
        }
        
        if ground_truth is not None:
            tp = int(np.sum((flags == 1) & (ground_truth == 1)))
            fp = int(np.sum((flags == 1) & (ground_truth == 0)))
            fn = int(np.sum((flags == 0) & (ground_truth == 1)))
            tn = int(np.sum((flags == 0) & (ground_truth == 0)))
            prec = precision_score(ground_truth, flags, zero_division=0)
            rec = recall_score(ground_truth, flags, zero_division=0)
            f1 = f1_score(ground_truth, flags, zero_division=0)
            
            row.update({
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "F1_Score": round(float(f1), 4),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn
            })
            
        results.append(row)
        
    return pd.DataFrame(results)


def compute_comprehensive_metrics(
    scores: np.ndarray,
    binary_preds: np.ndarray,
    ground_truth: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """Computes full set of classification and ranking metrics given ground truth."""
    prec = precision_score(ground_truth, binary_preds, zero_division=0)
    rec = recall_score(ground_truth, binary_preds, zero_division=0)
    f1 = f1_score(ground_truth, binary_preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(ground_truth, scores)
    except Exception:
        roc_auc = float('nan')
        
    try:
        pr_auc = average_precision_score(ground_truth, scores)
    except Exception:
        pr_auc = float('nan')
        
    cm = confusion_matrix(ground_truth, binary_preds)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "Model": model_name,
        "ROC_AUC": round(float(roc_auc), 4),
        "PR_AUC": round(float(pr_auc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1_Score": round(float(f1), 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Flagged_Total": int(tp + fp)
    }


def explain_top_anomalies(
    df: pd.DataFrame,
    anomaly_scores: np.ndarray,
    binary_flags: np.ndarray,
    top_k: int = 15,
    explain_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Ranks the most suspicious records and provides human-interpretable feature deviation notes.
    """
    if explain_cols is None:
        explain_cols = ['amount', 'distance_from_home_km', 'distance_from_last_txn_km',
                        'time_since_last_txn_hours', 'device_risk_score', 'failed_attempts_last_24h']
        explain_cols = [c for c in explain_cols if c in df.columns]
        
    ranked_indices = np.argsort(-anomaly_scores)[:top_k]
    
    # Compute population medians and IQRs for reference
    medians = df[explain_cols].median()
    iqrs = df[explain_cols].quantile(0.75) - df[explain_cols].quantile(0.25)
    
    explanation_rows = []
    for rank, idx in enumerate(ranked_indices, start=1):
        row = df.iloc[idx].to_dict()
        score = anomaly_scores[idx]
        flag = binary_flags[idx]
        
        # Determine which features deviate most (> 3 * IQR above median)
        drivers = []
        for col in explain_cols:
            val = float(row[col])
            med = float(medians[col])
            iqr = float(iqrs[col]) + 1e-4
            dev = (val - med) / iqr
            if dev >= 3.0:
                drivers.append(f"{col} elevated ({val:.1f} vs med {med:.1f})")
            elif col == 'time_since_last_txn_hours' and val < 0.1:
                drivers.append(f"rapid velocity ({val:.2f} hrs)")
                
        explanation_text = "; ".join(drivers) if drivers else "Multivariate combination anomaly"
        
        res = {
            "Rank": rank,
            "Anomaly_Score": round(float(score), 4),
            "Flagged": bool(flag == 1),
            "Transaction_ID": row.get("transaction_id", f"IDX_{idx}"),
            "Amount": row.get("amount", None),
            "Category": row.get("merchant_category", "N/A"),
            "Channel": row.get("channel", "N/A"),
            "Device_Risk": row.get("device_risk_score", None),
            "Key_Review_Drivers": explanation_text
        }
        if "is_synthetic_anomaly" in row:
            res["Ground_Truth"] = int(row["is_synthetic_anomaly"])
        explanation_rows.append(res)
        
    return pd.DataFrame(explanation_rows)
