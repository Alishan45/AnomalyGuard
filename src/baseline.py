"""
Baseline Anomaly Detection Module for AnomalyGuard.
Implements statistical (Z-Score / IQR) and business rule heuristics.
Serves as Tuesday's benchmark against unsupervised machine learning models.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

class RuleBasedStatisticalBaseline:
    """
    Statistical and Rule-based heuristic baseline.
    Computes univariate Z-scores, extreme IQR fences, and multi-factor domain rules.
    """
    def __init__(self, z_score_threshold: float = 3.0, velocity_cutoff_kmh: float = 800.0):
        self.z_score_threshold = z_score_threshold
        self.velocity_cutoff_kmh = velocity_cutoff_kmh
        self.feature_stats_: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False
        
    def fit(self, df: pd.DataFrame):
        """Calculates reference statistics (mean, std, quartiles) on training data."""
        numeric_cols = ['amount', 'distance_from_home_km', 'device_risk_score']
        for col in numeric_cols:
            if col in df.columns:
                series = df[col].astype(float)
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                self.feature_stats_[col] = {
                    'mean': float(series.mean()),
                    'std': float(series.std() + 1e-6),
                    'median': float(series.median()),
                    'q25': float(q25),
                    'q75': float(q75),
                    'iqr': float(q75 - q25)
                }
        self.is_fitted = True
        return self

    def predict_scores_and_flags(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Generates anomaly scores and binary flags using heuristic rules & statistical fences.
        Returns:
            - scores: continuous composite anomaly score in [0, 1]
            - flags: binary anomaly flag (0 or 1)
            - rule_triggers: DataFrame showing which specific rules fired
        """
        if not self.is_fitted:
            self.fit(df)
            
        n = len(df)
        rule_triggers = pd.DataFrame(index=df.index)
        
        # Rule 1: High Physical Travel Velocity (Impossible Travel > 800 km/h)
        if 'distance_from_last_txn_km' in df.columns and 'time_since_last_txn_hours' in df.columns:
            velocity = df['distance_from_last_txn_km'] / (df['time_since_last_txn_hours'] + 0.01)
            rule_triggers['rule_impossible_travel'] = (velocity > self.velocity_cutoff_kmh).astype(int)
        else:
            rule_triggers['rule_impossible_travel'] = 0
            
        # Rule 2: Extreme Amount (Above Upper IQR Fence Q75 + 3.0 * IQR or Z-score > 3)
        if 'amount' in df.columns and 'amount' in self.feature_stats_:
            stats = self.feature_stats_['amount']
            upper_fence = stats['q75'] + 3.0 * stats['iqr']
            z_amt = (df['amount'] - stats['mean']) / stats['std']
            rule_triggers['rule_extreme_amount'] = ((df['amount'] > upper_fence) | (z_amt > self.z_score_threshold)).astype(int)
        else:
            rule_triggers['rule_extreme_amount'] = 0
            
        # Rule 3: High Device Risk & Authentication Failures
        if 'device_risk_score' in df.columns and 'failed_attempts_last_24h' in df.columns:
            rule_triggers['rule_credential_risk'] = ((df['device_risk_score'] > 0.70) & (df['failed_attempts_last_24h'] >= 2)).astype(int)
        else:
            rule_triggers['rule_credential_risk'] = 0
            
        # Rule 4: International High-Risk
        if 'is_international' in df.columns and 'device_risk_score' in df.columns:
            rule_triggers['rule_suspicious_intl'] = ((df['is_international'] == 1) & (df['device_risk_score'] > 0.75)).astype(int)
        else:
            rule_triggers['rule_suspicious_intl'] = 0
            
        # Composite score calculation: normalized sum of triggers + continuous component
        trigger_count = rule_triggers.sum(axis=1)
        
        # Add continuous normalized score
        cont_score = np.zeros(n)
        if 'amount' in df.columns and 'amount' in self.feature_stats_:
            amt_z = np.clip((df['amount'] - self.feature_stats_['amount']['mean']) / self.feature_stats_['amount']['std'], 0, 10)
            cont_score += amt_z / 10.0
        if 'device_risk_score' in df.columns:
            cont_score += df['device_risk_score'].values
            
        composite_score = np.clip((trigger_count * 0.5) + (cont_score * 0.25), 0.0, 1.0)
        binary_flags = (trigger_count >= 1).astype(int)
        
        return composite_score.values, binary_flags.values, rule_triggers
