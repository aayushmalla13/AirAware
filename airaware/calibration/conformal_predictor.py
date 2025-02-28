"""
Conformal Predictor for Uncertainty Quantification

This module implements conformal prediction methods for providing
reliable prediction intervals with finite-sample guarantees.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
import warnings

logger = logging.getLogger(__name__)

@dataclass
class ConformalConfig:
    """Configuration for conformal prediction"""
    # Calibration parameters
    alpha: float = 0.1  # Significance level (1 - coverage)
    method: str = "quantile"  # "quantile", "normalized", "locally_adaptive"
    
    # Data splitting
    calibration_ratio: float = 0.2  # Ratio of data for calibration
    random_state: Optional[int] = None
    
    # Adaptive parameters
    use_adaptive: bool = False
    adaptation_window: int = 100  # Window for adaptive conformal
    
    # Quantile parameters
    quantile_method: str = "linear"  # "linear", "higher", "lower", "midpoint", "nearest"
    
    # Normalization parameters
    normalize_scores: bool = True
    normalization_method: str = "standard"  # "standard", "robust", "minmax"
    
    # Time series specific
    respect_temporal_order: bool = True
    use_rolling_calibration: bool = False
    rolling_window: int = 1000

class ConformalPredictor:
    """Conformal predictor for uncertainty quantification"""
    
    def __init__(self, config: ConformalConfig):
        self.config = config
        self.is_fitted = False
        self.calibration_scores = None
        self.quantile_threshold = None
        self.calibration_data = None
        
    def fit(self, model: BaseEstimator, X_cal: np.ndarray, y_cal: np.ndarray, 
            X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None) -> 'ConformalPredictor':
        """Fit the conformal predictor using calibration data"""
        logger.info("🔧 Fitting conformal predictor...")
        
        # Train the base model if training data is provided
        if X_train is not None and y_train is not None:
            logger.info("Training base model...")
            model.fit(X_train, y_train)
        
        # Get predictions on calibration data
        y_pred_cal = model.predict(X_cal)
        
        # Calculate conformity scores
        if self.config.method == "quantile":
            scores = self._calculate_quantile_scores(y_cal, y_pred_cal)
        elif self.config.method == "normalized":
            scores = self._calculate_normalized_scores(y_cal, y_pred_cal, X_cal, model)
        elif self.config.method == "locally_adaptive":
            scores = self._calculate_locally_adaptive_scores(y_cal, y_pred_cal, X_cal)
        else:
            raise ValueError(f"Unknown conformal method: {self.config.method}")
        
        # Store calibration scores
        self.calibration_scores = scores
        
        # Calculate quantile threshold
        self.quantile_threshold = np.quantile(
            scores, 1 - self.config.alpha, method=self.config.quantile_method
        )
        
        # Store calibration data for adaptive methods
        if self.config.use_adaptive:
            self.calibration_data = {
                'X': X_cal,
                'y': y_cal,
                'y_pred': y_pred_cal,
                'scores': scores
            }
        
        self.is_fitted = True
        logger.info(f"✅ Conformal predictor fitted with {len(scores)} calibration samples")
        logger.info(f"Quantile threshold: {self.quantile_threshold:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray, model: BaseEstimator) -> Dict[str, np.ndarray]:
        """Make conformal predictions with uncertainty intervals"""
        if not self.is_fitted:
            raise ValueError("Conformal predictor must be fitted before making predictions")
        
        # Get point predictions
        y_pred = model.predict(X)
        
        # Calculate prediction intervals
        if self.config.method == "quantile":
            intervals = self._predict_quantile_intervals(y_pred)
        elif self.config.method == "normalized":
            intervals = self._predict_normalized_intervals(y_pred, X, model)
        elif self.config.method == "locally_adaptive":
            intervals = self._predict_locally_adaptive_intervals(y_pred, X)
        else:
            raise ValueError(f"Unknown conformal method: {self.config.method}")
        
        return {
            'predictions': y_pred,
            'lower_bound': intervals['lower'],
            'upper_bound': intervals['upper'],
            'interval_width': intervals['upper'] - intervals['lower'],
            'coverage_probability': 1 - self.config.alpha
        }
    
    def _calculate_quantile_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate conformity scores using absolute residuals"""
        return np.abs(y_true - y_pred)
    
    def _calculate_normalized_scores(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   X: np.ndarray, model: BaseEstimator) -> np.ndarray:
        """Calculate normalized conformity scores"""
        residuals = np.abs(y_true - y_pred)
        
        if self.config.normalize_scores:
            # Estimate residual variance using a simple model
            # This is a simplified approach - in practice, you might use a more sophisticated method
            residual_std = np.std(residuals)
            if residual_std > 0:
                normalized_scores = residuals / residual_std
            else:
                normalized_scores = residuals
        else:
            normalized_scores = residuals
        
        return normalized_scores
    
    def _calculate_locally_adaptive_scores(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                         X: np.ndarray) -> np.ndarray:
        """Calculate locally adaptive conformity scores"""
        # Simple local adaptation based on k-nearest neighbors
        # This is a simplified implementation
        residuals = np.abs(y_true - y_pred)
        
        # For now, return regular residuals
        # In practice, you would implement a more sophisticated local adaptation
        return residuals
    
    def _predict_quantile_intervals(self, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict intervals using quantile method"""
        lower = y_pred - self.quantile_threshold
        upper = y_pred + self.quantile_threshold
        
        return {'lower': lower, 'upper': upper}
    
    def _predict_normalized_intervals(self, y_pred: np.ndarray, X: np.ndarray, 
                                    model: BaseEstimator) -> Dict[str, np.ndarray]:
        """Predict intervals using normalized method"""
        # Estimate residual variance for each prediction
        # This is a simplified approach
        if hasattr(self, 'calibration_scores') and self.config.normalize_scores:
            residual_std = np.std(self.calibration_scores)
        else:
            residual_std = 1.0
        
        interval_width = self.quantile_threshold * residual_std
        lower = y_pred - interval_width
        upper = y_pred + interval_width
        
        return {'lower': lower, 'upper': upper}
    
    def _predict_locally_adaptive_intervals(self, y_pred: np.ndarray, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict intervals using locally adaptive method"""
        # For now, use the same as quantile method
        # In practice, you would implement local adaptation
        return self._predict_quantile_intervals(y_pred)
    
    def update_adaptive(self, X_new: np.ndarray, y_new: np.ndarray, model: BaseEstimator):
        """Update the conformal predictor with new data (adaptive conformal)"""
        if not self.config.use_adaptive:
            logger.warning("Adaptive updating is not enabled")
            return
        
        if self.calibration_data is None:
            logger.warning("No calibration data available for adaptive updating")
            return
        
        # Get predictions for new data
        y_pred_new = model.predict(X_new)
        
        # Calculate new scores
        if self.config.method == "quantile":
            new_scores = self._calculate_quantile_scores(y_new, y_pred_new)
        elif self.config.method == "normalized":
            new_scores = self._calculate_normalized_scores(y_new, y_pred_new, X_new, model)
        elif self.config.method == "locally_adaptive":
            new_scores = self._calculate_locally_adaptive_scores(y_new, y_pred_new, X_new)
        
        # Update calibration scores using rolling window
        if self.config.use_rolling_calibration:
            # Keep only the most recent scores
            window_size = min(self.config.rolling_window, len(self.calibration_scores))
            self.calibration_scores = np.concatenate([
                self.calibration_scores[-window_size:],
                new_scores
            ])
        else:
            # Add new scores
            self.calibration_scores = np.concatenate([self.calibration_scores, new_scores])
        
        # Update quantile threshold
        self.quantile_threshold = np.quantile(
            self.calibration_scores, 1 - self.config.alpha, method=self.config.quantile_method
        )
        
        logger.info(f"Updated conformal predictor with {len(new_scores)} new samples")
        logger.info(f"New quantile threshold: {self.quantile_threshold:.4f}")
    
    def get_coverage(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    intervals: Dict[str, np.ndarray]) -> float:
        """Calculate empirical coverage of prediction intervals"""
        lower = intervals['lower']
        upper = intervals['upper']
        
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        return coverage
    
    def get_interval_width(self, intervals: Dict[str, np.ndarray]) -> float:
        """Calculate average width of prediction intervals"""
        return np.mean(intervals['upper'] - intervals['lower'])
    
    def save_predictor(self, path: str):
        """Save the conformal predictor"""
        predictor_state = {
            'config': self.config,
            'calibration_scores': self.calibration_scores,
            'quantile_threshold': self.quantile_threshold,
            'calibration_data': self.calibration_data,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'w') as f:
            json.dump(predictor_state, f, indent=2, default=str)
        
        logger.info(f"Conformal predictor saved to {path}")
    
    def load_predictor(self, path: str):
        """Load the conformal predictor"""
        with open(path, 'r') as f:
            predictor_state = json.load(f)
        
        self.config = predictor_state['config']
        self.calibration_scores = np.array(predictor_state['calibration_scores'])
        self.quantile_threshold = predictor_state['quantile_threshold']
        self.calibration_data = predictor_state['calibration_data']
        self.is_fitted = predictor_state['is_fitted']
        
        logger.info(f"Conformal predictor loaded from {path}")
    
    def get_predictor_info(self) -> Dict[str, Any]:
        """Get predictor information"""
        return {
            'method': self.config.method,
            'alpha': self.config.alpha,
            'coverage_probability': 1 - self.config.alpha,
            'is_fitted': self.is_fitted,
            'n_calibration_samples': len(self.calibration_scores) if self.calibration_scores is not None else 0,
            'quantile_threshold': self.quantile_threshold,
            'use_adaptive': self.config.use_adaptive
        }
