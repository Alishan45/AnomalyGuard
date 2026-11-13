"""
Anomaly Detection Models Module for AnomalyGuard.
Implements:
  1. Isolation Forest (with calibrated anomaly scoring)
  2. Local Outlier Factor (LOF) with careful prediction semantics
  3. Random Seed Stability Evaluator
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import spearmanr

class IsolationForestDetector:
    """
    Unsupervised Isolation Forest anomaly detector.
    Isolates observations by randomly selecting a feature and split value.
    Anomalies require fewer splits (shorter path lengths) to isolate.
    """
    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.025,
        max_samples: Any = 'auto',
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.min_raw_score_ = 0.0
        self.max_raw_score_ = 1.0

    def fit(self, X: np.ndarray):
        """Fit model on scaled feature matrix without labels."""
        self.model.fit(X)
        self.is_fitted = True
        
        # Raw decision function: higher means normal, lower/negative means anomalous
        raw_scores = self.model.decision_function(X)
        # We invert so higher means MORE anomalous: raw_anomaly = -decision_function
        raw_anomaly = -raw_scores
        self.min_raw_score_ = float(np.min(raw_anomaly))
        self.max_raw_score_ = float(np.max(raw_anomaly))
        return self

    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Computes calibrated anomaly score in [0, 1] range.
        Higher score = more anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        raw_anomaly = -self.model.decision_function(X)
        range_span = self.max_raw_score_ - self.min_raw_score_
        if range_span <= 1e-9:
            range_span = 1.0
        normalized = (raw_anomaly - self.min_raw_score_) / range_span
        return np.clip(normalized, 0.0, 1.0)

    def predict_flags(self, X: np.ndarray, custom_threshold: Optional[float] = None) -> np.ndarray:
        """
        Returns binary flags (1 = anomaly, 0 = normal).
        If custom_threshold is provided, thresholds calibrated score >= custom_threshold.
        Otherwise uses sklearn's contamination cut.
        """
        if custom_threshold is not None:
            scores = self.predict_anomaly_scores(X)
            return (scores >= custom_threshold).astype(int)
        
        # Scikit-learn outputs -1 for anomaly, 1 for inlier
        raw_preds = self.model.predict(X)
        return (raw_preds == -1).astype(int)


class LocalOutlierFactorDetector:
    """
    Local Outlier Factor (LOF) Detector.
    Prediction Semantics:
      - LOF measures the local density of a sample compared to its k-nearest neighbors.
      - Samples that have a substantially lower density than their neighbors are considered outliers.
      - We configure novelty=True so it can score new points out-of-sample and provide decision_function.
    """
    def __init__(
        self,
        n_neighbors: int = 25,
        contamination: float = 0.025,
        metric: str = 'minkowski'
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            metric=self.metric,
            n_jobs=-1
        )
        self.is_fitted = False
        self.min_raw_score_ = 0.0
        self.max_raw_score_ = 1.0

    def fit(self, X: np.ndarray):
        """Fit LOF model."""
        self.model.fit(X)
        self.is_fitted = True
        raw_scores = self.model.decision_function(X)
        raw_anomaly = -raw_scores
        self.min_raw_score_ = float(np.min(raw_anomaly))
        self.max_raw_score_ = float(np.max(raw_anomaly))
        return self

    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes calibrated anomaly score in [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        raw_anomaly = -self.model.decision_function(X)
        range_span = self.max_raw_score_ - self.min_raw_score_
        if range_span <= 1e-9:
            range_span = 1.0
        normalized = (raw_anomaly - self.min_raw_score_) / range_span
        return np.clip(normalized, 0.0, 1.0)

    def predict_flags(self, X: np.ndarray, custom_threshold: Optional[float] = None) -> np.ndarray:
        """Returns binary anomaly flags (1 = anomaly, 0 = normal)."""
        if custom_threshold is not None:
            scores = self.predict_anomaly_scores(X)
            return (scores >= custom_threshold).astype(int)
        raw_preds = self.model.predict(X)
        return (raw_preds == -1).astype(int)


def evaluate_seed_stability(
    X: np.ndarray,
    seeds: List[int] = [42, 101, 2024, 7, 99],
    n_estimators: int = 100,
    contamination: float = 0.025
) -> Dict[str, Any]:
    """
    Evaluates Isolation Forest ranking stability across different random seeds
    using pairwise Spearman rank correlation.
    """
    scores_by_seed = {}
    for seed in seeds:
        iso = IsolationForestDetector(n_estimators=n_estimators, contamination=contamination, random_state=seed)
        iso.fit(X)
        scores_by_seed[seed] = iso.predict_anomaly_scores(X)
        
    correlations = []
    seed_list = list(scores_by_seed.keys())
    for i in range(len(seed_list)):
        for j in range(i + 1, len(seed_list)):
            s1, s2 = seed_list[i], seed_list[j]
            rho, _ = spearmanr(scores_by_seed[s1], scores_by_seed[s2])
            correlations.append({
                "Seed_A": s1,
                "Seed_B": s2,
                "Spearman_Rho": float(rho)
            })
            
    corr_df = pd.DataFrame(correlations)
    mean_rho = float(corr_df["Spearman_Rho"].mean())
    min_rho = float(corr_df["Spearman_Rho"].min())
    
    return {
        "pairwise_correlations": corr_df,
        "mean_spearman_rho": mean_rho,
        "min_spearman_rho": min_rho,
        "is_highly_stable": bool(mean_rho >= 0.90)
    }
