"""
Adaptive Conformal Prediction

This module implements adaptive conformal prediction methods that can
update prediction intervals in real-time as new data arrives.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from sklearn.base import BaseEstimator, RegressorMixin
from collections import deque
import warnings

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveConformalConfig:
    """Configuration for adaptive conformal prediction"""
    # Calibration parameters
    alpha: float = 0.1  # Significance level
    initial_calibration_size: int = 100  # Initial calibration set size
    
    # Adaptation parameters
    adaptation_rate: float = 0.01  # Learning rate for adaptation
    adaptation_method: str = "online"  # "online", "batch", "sliding_window"
    
    # Online adaptation
    use_online_adaptation: bool = True
    online_update_frequency: int = 1  # Update every N samples
    
    # Sliding window parameters
    use_sliding_window: bool = False
    window_size: int = 1000
    min_window_size: int = 100
    
    # Batch adaptation
    use_batch_adaptation: bool = False
    batch_size: int = 50
    batch_update_frequency: int = 10  # Update every N batches
    
    # Robustness parameters
    use_robust_estimation: bool = True
    robust_quantile: float = 0.95  # Use 95th percentile for robust estimation
    
    # Time series specific
    respect_temporal_order: bool = True
    use_temporal_weighting: bool = False
    temporal_decay: float = 0.99  # Exponential decay factor
    
    # Monitoring
    monitor_coverage: bool = True
    coverage_target: float = 0.9  # Target coverage probability
    coverage_tolerance: float = 0.05  # Tolerance for coverage deviation

class AdaptiveConformalPredictor:
    """Adaptive conformal predictor for real-time uncertainty quantification"""
    
    def __init__(self, config: AdaptiveConformalConfig):
        self.config = config
        self.is_fitted = False
        self.calibration_scores = deque(maxlen=config.window_size if config.use_sliding_window else None)
        self.quantile_threshold = None
        self.adaptation_history = []
        self.coverage_history = []
        self.n_samples_seen = 0
        self.n_updates = 0
        
        # Online adaptation state
        self.online_quantile_estimator = None
        self.online_mean = 0.0
        self.online_variance = 0.0
        
        # Batch adaptation state
        self.batch_buffer = []
        self.batch_count = 0
        
    def fit(self, model: BaseEstimator, X_cal: np.ndarray, y_cal: np.ndarray,
            X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None) -> 'AdaptiveConformalPredictor':
        """Fit the adaptive conformal predictor using initial calibration data"""
        logger.info("🔧 Fitting adaptive conformal predictor...")
        
        # Train the base model if training data is provided
        if X_train is not None and y_train is not None:
            logger.info("Training base model...")
            model.fit(X_train, y_train)
        
        # Get predictions on calibration data
        y_pred_cal = model.predict(X_cal)
        
        # Calculate initial conformity scores
        scores = np.abs(y_cal - y_pred_cal)
        
        # Store initial calibration scores
        for score in scores:
            self.calibration_scores.append(score)
        
        # Calculate initial quantile threshold
        self.quantile_threshold = np.quantile(scores, 1 - self.config.alpha)
        
        # Initialize online adaptation
        if self.config.use_online_adaptation:
            self._initialize_online_adaptation(scores)
        
        self.is_fitted = True
        self.n_samples_seen = len(scores)
        
        logger.info(f"✅ Adaptive conformal predictor fitted with {len(scores)} calibration samples")
        logger.info(f"Initial quantile threshold: {self.quantile_threshold:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray, model: BaseEstimator) -> Dict[str, np.ndarray]:
        """Make adaptive conformal predictions"""
        if not self.is_fitted:
            raise ValueError("Adaptive conformal predictor must be fitted before making predictions")
        
        # Get point predictions
        y_pred = model.predict(X)
        
        # Calculate prediction intervals using current threshold
        lower_bound = y_pred - self.quantile_threshold
        upper_bound = y_pred + self.quantile_threshold
        
        return {
            'predictions': y_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound,
            'coverage_probability': 1 - self.config.alpha,
            'quantile_threshold': self.quantile_threshold
        }
    
    def update(self, X_new: np.ndarray, y_new: np.ndarray, model: BaseEstimator):
        """Update the adaptive conformal predictor with new data"""
        if not self.is_fitted:
            raise ValueError("Adaptive conformal predictor must be fitted before updating")
        
        # Get predictions for new data
        y_pred_new = model.predict(X_new)
        
        # Calculate new conformity scores
        new_scores = np.abs(y_new - y_pred_new)
        
        # Update based on adaptation method
        if self.config.adaptation_method == "online":
            self._update_online(new_scores)
        elif self.config.adaptation_method == "batch":
            self._update_batch(new_scores)
        elif self.config.adaptation_method == "sliding_window":
            self._update_sliding_window(new_scores)
        else:
            raise ValueError(f"Unknown adaptation method: {self.config.adaptation_method}")
        
        # Update sample count
        self.n_samples_seen += len(new_scores)
        
        # Monitor coverage if enabled
        if self.config.monitor_coverage:
            self._monitor_coverage(y_new, y_pred_new)
        
        logger.info(f"Updated adaptive conformal predictor with {len(new_scores)} new samples")
        logger.info(f"New quantile threshold: {self.quantile_threshold:.4f}")
    
    def _initialize_online_adaptation(self, initial_scores: np.ndarray):
        """Initialize online adaptation parameters"""
        self.online_mean = np.mean(initial_scores)
        self.online_variance = np.var(initial_scores)
        self.online_quantile_estimator = OnlineQuantileEstimator(
            quantile=1 - self.config.alpha,
            learning_rate=self.config.adaptation_rate
        )
        
        # Initialize with initial scores
        for score in initial_scores:
            self.online_quantile_estimator.update(score)
    
    def _update_online(self, new_scores: np.ndarray):
        """Update using online adaptation"""
        for score in new_scores:
            # Update online quantile estimator
            self.online_quantile_estimator.update(score)
            
            # Update online mean and variance
            self._update_online_statistics(score)
            
            # Add to calibration scores
            self.calibration_scores.append(score)
        
        # Update quantile threshold
        self.quantile_threshold = self.online_quantile_estimator.get_quantile()
        self.n_updates += 1
        
        # Store adaptation history
        self.adaptation_history.append({
            'n_samples': self.n_samples_seen,
            'quantile_threshold': self.quantile_threshold,
            'mean_score': self.online_mean,
            'variance_score': self.online_variance
        })
    
    def _update_batch(self, new_scores: np.ndarray):
        """Update using batch adaptation"""
        # Add new scores to batch buffer
        self.batch_buffer.extend(new_scores)
        self.batch_count += 1
        
        # Update if batch is full or update frequency reached
        if (len(self.batch_buffer) >= self.config.batch_size or 
            self.batch_count % self.config.batch_update_frequency == 0):
            
            # Calculate new quantile threshold from batch
            batch_threshold = np.quantile(self.batch_buffer, 1 - self.config.alpha)
            
            # Update quantile threshold with adaptation rate
            self.quantile_threshold = (1 - self.config.adaptation_rate) * self.quantile_threshold + \
                                    self.config.adaptation_rate * batch_threshold
            
            # Add batch scores to calibration scores
            for score in self.batch_buffer:
                self.calibration_scores.append(score)
            
            # Clear batch buffer
            self.batch_buffer = []
            self.n_updates += 1
            
            # Store adaptation history
            self.adaptation_history.append({
                'n_samples': self.n_samples_seen,
                'quantile_threshold': self.quantile_threshold,
                'batch_size': len(self.batch_buffer)
            })
    
    def _update_sliding_window(self, new_scores: np.ndarray):
        """Update using sliding window adaptation"""
        # Add new scores to calibration scores (deque automatically maintains window size)
        for score in new_scores:
            self.calibration_scores.append(score)
        
        # Recalculate quantile threshold from current window
        if len(self.calibration_scores) >= self.config.min_window_size:
            self.quantile_threshold = np.quantile(list(self.calibration_scores), 1 - self.config.alpha)
            self.n_updates += 1
            
            # Store adaptation history
            self.adaptation_history.append({
                'n_samples': self.n_samples_seen,
                'quantile_threshold': self.quantile_threshold,
                'window_size': len(self.calibration_scores)
            })
    
    def _update_online_statistics(self, score: float):
        """Update online mean and variance"""
        # Online mean update
        self.online_mean = (1 - self.config.adaptation_rate) * self.online_mean + \
                          self.config.adaptation_rate * score
        
        # Online variance update (simplified)
        if self.n_samples_seen > 1:
            self.online_variance = (1 - self.config.adaptation_rate) * self.online_variance + \
                                 self.config.adaptation_rate * (score - self.online_mean) ** 2
    
    def _monitor_coverage(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Monitor empirical coverage"""
        # Calculate prediction intervals
        lower_bound = y_pred - self.quantile_threshold
        upper_bound = y_pred + self.quantile_threshold
        
        # Calculate empirical coverage
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        
        # Store coverage history
        self.coverage_history.append({
            'n_samples': self.n_samples_seen,
            'coverage': coverage,
            'target_coverage': 1 - self.config.alpha,
            'coverage_error': abs(coverage - (1 - self.config.alpha))
        })
        
        # Check if coverage is within tolerance
        if abs(coverage - self.config.coverage_target) > self.config.coverage_tolerance:
            logger.warning(f"Coverage deviation detected: {coverage:.3f} vs target {self.config.coverage_target:.3f}")
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation history"""
        if not self.adaptation_history:
            return {}
        
        recent_history = self.adaptation_history[-10:]  # Last 10 updates
        
        return {
            'n_samples_seen': self.n_samples_seen,
            'n_updates': self.n_updates,
            'current_threshold': self.quantile_threshold,
            'threshold_range': {
                'min': min(h['quantile_threshold'] for h in self.adaptation_history),
                'max': max(h['quantile_threshold'] for h in self.adaptation_history),
                'mean': np.mean([h['quantile_threshold'] for h in self.adaptation_history])
            },
            'recent_updates': recent_history,
            'coverage_history': self.coverage_history[-10:] if self.coverage_history else []
        }
    
    def save_predictor(self, path: str):
        """Save the adaptive conformal predictor"""
        predictor_state = {
            'config': self.config,
            'calibration_scores': list(self.calibration_scores),
            'quantile_threshold': self.quantile_threshold,
            'adaptation_history': self.adaptation_history,
            'coverage_history': self.coverage_history,
            'n_samples_seen': self.n_samples_seen,
            'n_updates': self.n_updates,
            'online_mean': self.online_mean,
            'online_variance': self.online_variance,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'w') as f:
            json.dump(predictor_state, f, indent=2, default=str)
        
        logger.info(f"Adaptive conformal predictor saved to {path}")
    
    def load_predictor(self, path: str):
        """Load the adaptive conformal predictor"""
        with open(path, 'r') as f:
            predictor_state = json.load(f)
        
        self.config = predictor_state['config']
        self.calibration_scores = deque(predictor_state['calibration_scores'], 
                                      maxlen=self.config.window_size if self.config.use_sliding_window else None)
        self.quantile_threshold = predictor_state['quantile_threshold']
        self.adaptation_history = predictor_state['adaptation_history']
        self.coverage_history = predictor_state['coverage_history']
        self.n_samples_seen = predictor_state['n_samples_seen']
        self.n_updates = predictor_state['n_updates']
        self.online_mean = predictor_state['online_mean']
        self.online_variance = predictor_state['online_variance']
        self.is_fitted = predictor_state['is_fitted']
        
        logger.info(f"Adaptive conformal predictor loaded from {path}")
    
    def get_predictor_info(self) -> Dict[str, Any]:
        """Get predictor information"""
        return {
            'method': 'adaptive_conformal',
            'alpha': self.config.alpha,
            'coverage_probability': 1 - self.config.alpha,
            'is_fitted': self.is_fitted,
            'n_calibration_samples': len(self.calibration_scores),
            'quantile_threshold': self.quantile_threshold,
            'n_samples_seen': self.n_samples_seen,
            'n_updates': self.n_updates,
            'adaptation_method': self.config.adaptation_method
        }

class OnlineQuantileEstimator:
    """Online quantile estimator for adaptive conformal prediction"""
    
    def __init__(self, quantile: float = 0.9, learning_rate: float = 0.01):
        self.quantile = quantile
        self.learning_rate = learning_rate
        self.quantile_estimate = 0.0
        self.n_updates = 0
    
    def update(self, value: float):
        """Update quantile estimate with new value"""
        if self.n_updates == 0:
            self.quantile_estimate = value
        else:
            # Simple online quantile update
            if value > self.quantile_estimate:
                self.quantile_estimate += self.learning_rate * (1 - self.quantile)
            else:
                self.quantile_estimate -= self.learning_rate * self.quantile
        
        self.n_updates += 1
    
    def get_quantile(self) -> float:
        """Get current quantile estimate"""
        return self.quantile_estimate
