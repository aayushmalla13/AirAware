"""
Data Preprocessor for Deep Learning Models

This module provides data preprocessing utilities for deep learning models,
including feature engineering, normalization, and sequence creation.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    # Sequence parameters
    context_length: int = 96  # Input sequence length
    prediction_length: int = 24  # Forecast horizon
    stride: int = 1  # Step size for sequence creation
    
    # Feature engineering
    use_lags: bool = True
    lag_periods: List[int] = None  # [1, 24, 168] for 1h, 24h, 168h lags
    use_rolling_features: bool = True
    rolling_windows: List[int] = None  # [3, 24, 168] for 3h, 24h, 168h windows
    use_difference_features: bool = True
    use_cyclical_features: bool = True
    
    # Normalization
    scaler_type: str = "standard"  # "standard", "minmax", "robust", "none"
    fit_scaler_on_train_only: bool = True
    
    # Missing data handling
    missing_data_strategy: str = "forward_fill"  # "forward_fill", "backward_fill", "interpolate", "drop"
    max_missing_ratio: float = 0.5  # Maximum ratio of missing data allowed
    
    # Feature selection
    use_feature_selection: bool = False
    feature_selection_method: str = "correlation"  # "correlation", "mutual_info", "variance"
    max_features: int = 50
    
    # Data validation
    validate_data: bool = True
    outlier_detection: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    outlier_threshold: float = 3.0
    
    def __post_init__(self):
        if self.lag_periods is None:
            self.lag_periods = [1, 24, 168]  # 1h, 24h, 168h
        if self.rolling_windows is None:
            self.rolling_windows = [3, 24, 168]  # 3h, 24h, 168h

class DeepDataPreprocessor:
    """Data preprocessor for deep learning models"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.scalers = {}
        self.feature_names = []
        self.target_scaler = None
        self.is_fitted = False
        self.feature_importance = {}
        
    def fit_transform(self, data: pd.DataFrame, target_col: str = 'pm25') -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Fit preprocessor and transform data"""
        logger.info("🔧 Fitting and transforming data...")
        
        # Validate data
        if self.config.validate_data:
            self._validate_data(data, target_col)
        
        # Handle missing data
        data_clean = self._handle_missing_data(data)
        
        # Feature engineering
        data_engineered = self._engineer_features(data_clean, target_col)
        
        # Feature selection
        if self.config.use_feature_selection:
            data_selected = self._select_features(data_engineered, target_col)
        else:
            data_selected = data_engineered
        
        # Normalization
        data_normalized = self._normalize_features(data_selected, target_col)
        
        # Create sequences
        X, y = self._create_sequences(data_normalized, target_col)
        
        # Store feature names
        self.feature_names = [col for col in data_normalized.columns if col != target_col]
        
        self.is_fitted = True
        
        logger.info(f"✅ Data preprocessing completed: {X.shape[0]} sequences, {X.shape[2]} features")
        
        return X, y, {
            'feature_names': self.feature_names,
            'n_features': X.shape[2],
            'sequence_length': X.shape[1],
            'prediction_length': y.shape[1]
        }
    
    def transform(self, data: pd.DataFrame, target_col: str = 'pm25') -> Tuple[np.ndarray, np.ndarray]:
        """Transform data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        logger.info("🔧 Transforming data...")
        
        # Handle missing data
        data_clean = self._handle_missing_data(data)
        
        # Feature engineering
        data_engineered = self._engineer_features(data_clean, target_col)
        
        # Feature selection
        if self.config.use_feature_selection:
            data_selected = self._select_features(data_engineered, target_col)
        else:
            data_selected = data_engineered
        
        # Normalization
        data_normalized = self._normalize_features(data_selected, target_col)
        
        # Create sequences
        X, y = self._create_sequences(data_normalized, target_col)
        
        logger.info(f"✅ Data transformation completed: {X.shape[0]} sequences")
        
        return X, y
    
    def _validate_data(self, data: pd.DataFrame, target_col: str):
        """Validate input data"""
        logger.info("Validating data...")
        
        # Check required columns
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Check for sufficient data
        if len(data) < self.config.context_length + self.config.prediction_length:
            raise ValueError(f"Insufficient data: need at least {self.config.context_length + self.config.prediction_length} samples")
        
        # Check for missing data ratio
        missing_ratio = data[target_col].isnull().sum() / len(data)
        if missing_ratio > self.config.max_missing_ratio:
            raise ValueError(f"Too much missing data in target: {missing_ratio:.2%} > {self.config.max_missing_ratio:.2%}")
        
        # Outlier detection
        if self.config.outlier_detection:
            self._detect_outliers(data, target_col)
        
        logger.info("✅ Data validation passed")
    
    def _detect_outliers(self, data: pd.DataFrame, target_col: str):
        """Detect and handle outliers"""
        logger.info("Detecting outliers...")
        
        target_data = data[target_col].dropna()
        
        if self.config.outlier_method == "iqr":
            Q1 = target_data.quantile(0.25)
            Q3 = target_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (target_data < lower_bound) | (target_data > upper_bound)
            
        elif self.config.outlier_method == "zscore":
            z_scores = np.abs((target_data - target_data.mean()) / target_data.std())
            outliers = z_scores > self.config.outlier_threshold
            
        else:
            logger.warning(f"Unknown outlier method: {self.config.outlier_method}")
            return
        
        outlier_ratio = outliers.sum() / len(target_data)
        logger.info(f"Outlier ratio: {outlier_ratio:.2%}")
        
        if outlier_ratio > 0.1:  # More than 10% outliers
            logger.warning(f"High outlier ratio detected: {outlier_ratio:.2%}")
    
    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data"""
        logger.info("Handling missing data...")
        
        data_clean = data.copy()
        
        if self.config.missing_data_strategy == "forward_fill":
            data_clean = data_clean.fillna(method='ffill')
        elif self.config.missing_data_strategy == "backward_fill":
            data_clean = data_clean.fillna(method='bfill')
        elif self.config.missing_data_strategy == "interpolate":
            data_clean = data_clean.interpolate()
        elif self.config.missing_data_strategy == "drop":
            data_clean = data_clean.dropna()
        else:
            logger.warning(f"Unknown missing data strategy: {self.config.missing_data_strategy}")
        
        # Fill any remaining NaNs with mean
        data_clean = data_clean.fillna(data_clean.mean())
        
        logger.info("✅ Missing data handled")
        return data_clean
    
    def _engineer_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Engineer features for deep learning"""
        logger.info("Engineering features...")
        
        data_engineered = data.copy()
        
        # Lag features
        if self.config.use_lags:
            for lag in self.config.lag_periods:
                data_engineered[f'{target_col}_lag_{lag}'] = data_engineered[target_col].shift(lag)
        
        # Rolling features
        if self.config.use_rolling_features:
            for window in self.config.rolling_windows:
                data_engineered[f'{target_col}_rolling_{window}_mean'] = data_engineered[target_col].rolling(window).mean()
                data_engineered[f'{target_col}_rolling_{window}_std'] = data_engineered[target_col].rolling(window).std()
                data_engineered[f'{target_col}_rolling_{window}_min'] = data_engineered[target_col].rolling(window).min()
                data_engineered[f'{target_col}_rolling_{window}_max'] = data_engineered[target_col].rolling(window).max()
        
        # Difference features
        if self.config.use_difference_features:
            data_engineered[f'{target_col}_diff_1'] = data_engineered[target_col].diff(1)
            data_engineered[f'{target_col}_diff_24'] = data_engineered[target_col].diff(24)
        
        # Cyclical features
        if self.config.use_cyclical_features and 'datetime_utc' in data_engineered.columns:
            data_engineered['hour_sin'] = np.sin(2 * np.pi * data_engineered['datetime_utc'].dt.hour / 24)
            data_engineered['hour_cos'] = np.cos(2 * np.pi * data_engineered['datetime_utc'].dt.hour / 24)
            data_engineered['day_sin'] = np.sin(2 * np.pi * data_engineered['datetime_utc'].dt.dayofweek / 7)
            data_engineered['day_cos'] = np.cos(2 * np.pi * data_engineered['datetime_utc'].dt.dayofweek / 7)
            data_engineered['month_sin'] = np.sin(2 * np.pi * data_engineered['datetime_utc'].dt.month / 12)
            data_engineered['month_cos'] = np.cos(2 * np.pi * data_engineered['datetime_utc'].dt.month / 12)
        
        # Fill NaN values created by feature engineering
        data_engineered = data_engineered.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"✅ Feature engineering completed: {len(data_engineered.columns)} features")
        return data_engineered
    
    def _select_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Select most important features"""
        logger.info("Selecting features...")
        
        feature_cols = [col for col in data.columns if col != target_col]
        
        if self.config.feature_selection_method == "correlation":
            # Select features based on correlation with target
            correlations = data[feature_cols].corrwith(data[target_col]).abs()
            selected_features = correlations.nlargest(self.config.max_features).index.tolist()
            
        elif self.config.feature_selection_method == "variance":
            # Select features based on variance
            variances = data[feature_cols].var()
            selected_features = variances.nlargest(self.config.max_features).index.tolist()
            
        else:
            logger.warning(f"Unknown feature selection method: {self.config.feature_selection_method}")
            selected_features = feature_cols[:self.config.max_features]
        
        # Store feature importance
        self.feature_importance = dict(zip(selected_features, range(len(selected_features))))
        
        # Select features
        selected_data = data[selected_features + [target_col]]
        
        logger.info(f"✅ Feature selection completed: {len(selected_features)} features selected")
        return selected_data
    
    def _normalize_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Normalize features"""
        logger.info("Normalizing features...")
        
        data_normalized = data.copy()
        feature_cols = [col for col in data.columns if col != target_col]
        
        # Create scaler
        if self.config.scaler_type == "standard":
            scaler = StandardScaler()
        elif self.config.scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif self.config.scaler_type == "robust":
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaler type: {self.config.scaler_type}")
            return data_normalized
        
        # Fit and transform features
        if not self.is_fitted:
            data_normalized[feature_cols] = scaler.fit_transform(data[feature_cols])
            self.scalers['features'] = scaler
        else:
            data_normalized[feature_cols] = self.scalers['features'].transform(data[feature_cols])
        
        # Normalize target
        if not self.is_fitted:
            target_scaler = StandardScaler()
            data_normalized[target_col] = target_scaler.fit_transform(data[[target_col]]).flatten()
            self.target_scaler = target_scaler
        else:
            if self.target_scaler is not None:
                data_normalized[target_col] = self.target_scaler.transform(data[[target_col]]).flatten()
        
        logger.info("✅ Feature normalization completed")
        return data_normalized
    
    def _create_sequences(self, data: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for deep learning"""
        logger.info("Creating sequences...")
        
        feature_cols = [col for col in data.columns if col != target_col]
        features = data[feature_cols].values
        target = data[target_col].values
        
        sequences = []
        targets = []
        
        for i in range(0, len(data) - self.config.context_length - self.config.prediction_length + 1, self.config.stride):
            seq_features = features[i:i+self.config.context_length]
            seq_target = target[i+self.config.context_length:i+self.config.context_length+self.config.prediction_length]
            
            sequences.append(seq_features)
            targets.append(seq_target)
        
        if not sequences:
            # Handle case where data is too short
            seq_features = features[-self.config.context_length:]
            seq_target = target[-self.config.prediction_length:]
            sequences = [seq_features]
            targets = [seq_target]
        
        X = np.array(sequences)
        y = np.array(targets)
        
        logger.info(f"✅ Sequence creation completed: {X.shape[0]} sequences")
        return X, y
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform target values"""
        if self.target_scaler is None:
            return y
        
        return self.target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance
    
    def save_preprocessor(self, path: str):
        """Save preprocessor state"""
        preprocessor_state = {
            'config': self.config,
            'scalers': self.scalers,
            'target_scaler': self.target_scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'w') as f:
            json.dump(preprocessor_state, f, indent=2, default=str)
        
        logger.info(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path: str):
        """Load preprocessor state"""
        with open(path, 'r') as f:
            preprocessor_state = json.load(f)
        
        self.config = preprocessor_state['config']
        self.scalers = preprocessor_state['scalers']
        self.target_scaler = preprocessor_state['target_scaler']
        self.feature_names = preprocessor_state['feature_names']
        self.feature_importance = preprocessor_state['feature_importance']
        self.is_fitted = preprocessor_state['is_fitted']
        
        logger.info(f"Preprocessor loaded from {path}")
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get preprocessing summary"""
        return {
            'config': self.config,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted,
            'scaler_type': self.config.scaler_type
        }
