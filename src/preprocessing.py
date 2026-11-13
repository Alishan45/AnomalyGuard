"""
Preprocessing and Feature Engineering module for AnomalyGuard.
Implements robust scaling and domain-specific feature transformations
without data leakage.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

class TransactionPreprocessor:
    """
    Transforms raw transaction tables into scaled numerical feature matrices.
    Strictly avoids data leakage: ground-truth labels and identifiers are isolated.
    """
    def __init__(self, use_log_transforms: bool = True, scale_features: bool = True):
        self.use_log_transforms = use_log_transforms
        self.scale_features = scale_features
        self.scaler = RobustScaler()
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False
        
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives domain-specific behavioral signals."""
        feat_df = pd.DataFrame(index=df.index)
        
        # 1. Base numeric columns if present
        if 'amount' in df.columns:
            feat_df['amount'] = df['amount'].astype(float)
            if self.use_log_transforms:
                feat_df['log_amount'] = np.log1p(np.maximum(0, df['amount'].astype(float)))
                
        if 'distance_from_home_km' in df.columns:
            feat_df['distance_from_home_km'] = df['distance_from_home_km'].astype(float)
            if self.use_log_transforms:
                feat_df['log_dist_home'] = np.log1p(np.maximum(0, df['distance_from_home_km'].astype(float)))
                
        if 'distance_from_last_txn_km' in df.columns:
            feat_df['distance_from_last_txn_km'] = df['distance_from_last_txn_km'].astype(float)
            
        if 'time_since_last_txn_hours' in df.columns:
            feat_df['time_since_last_txn_hours'] = df['time_since_last_txn_hours'].astype(float)
            
        # 2. Velocity (Physical Travel Feasibility in km/h)
        if 'distance_from_last_txn_km' in df.columns and 'time_since_last_txn_hours' in df.columns:
            # Add small epsilon to avoid divide-by-zero
            velocity = df['distance_from_last_txn_km'].astype(float) / (df['time_since_last_txn_hours'].astype(float) + 0.01)
            feat_df['travel_velocity_kmh'] = velocity
            if self.use_log_transforms:
                feat_df['log_velocity'] = np.log1p(np.maximum(0, velocity))
                
        # 3. Security Risk Features
        if 'device_risk_score' in df.columns:
            feat_df['device_risk_score'] = df['device_risk_score'].astype(float)
            
        if 'failed_attempts_last_24h' in df.columns:
            feat_df['failed_attempts_last_24h'] = df['failed_attempts_last_24h'].astype(float)
            
        if 'device_risk_score' in df.columns and 'failed_attempts_last_24h' in df.columns:
            feat_df['risk_interaction'] = df['device_risk_score'].astype(float) * (df['failed_attempts_last_24h'].astype(float) + 1.0)
            
        if 'is_international' in df.columns:
            feat_df['is_international'] = df['is_international'].astype(float)
            
        # 4. Temporal Features from timestamp if present
        if 'timestamp' in df.columns:
            try:
                dt_series = pd.to_datetime(df['timestamp'])
                feat_df['hour_of_day'] = dt_series.dt.hour.astype(float)
                # Cyclical hour encoding
                feat_df['hour_sin'] = np.sin(2 * np.pi * feat_df['hour_of_day'] / 24.0)
                feat_df['hour_cos'] = np.cos(2 * np.pi * feat_df['hour_of_day'] / 24.0)
                feat_df['is_night_txn'] = ((dt_series.dt.hour >= 0) & (dt_series.dt.hour <= 5)).astype(float)
            except Exception:
                pass
                
        # 5. One-hot or frequency encoding for categorical columns if present
        cat_cols = ['merchant_category', 'channel']
        for col in cat_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
                feat_df = pd.concat([feat_df, dummies], axis=1)
                
        # Fill any remaining NaNs with column median
        feat_df = feat_df.fillna(feat_df.median())
        return feat_df

    def fit(self, df: pd.DataFrame):
        """Fit preprocessor and scaler on training data."""
        feat_df = self._engineer_features(df)
        self.feature_names_ = feat_df.columns.tolist()
        
        if self.scale_features:
            self.scaler.fit(feat_df.values)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
            
        feat_df = self._engineer_features(df)
        # Ensure exact column alignment
        for col in self.feature_names_:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        feat_df = feat_df[self.feature_names_]
        
        if self.scale_features:
            scaled_values = self.scaler.transform(feat_df.values)
        else:
            scaled_values = feat_df.values
            
        return scaled_values, feat_df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
