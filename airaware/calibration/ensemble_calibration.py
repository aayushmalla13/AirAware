"""
Ensemble Calibration

This module implements calibration methods for ensemble models,
including cross-validation-based calibration and ensemble-specific
uncertainty quantification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

logger = logging.getLogger(__name__)

@dataclass
class EnsembleCalibrationConfig:
    """Configuration for ensemble calibration"""
    # Calibration parameters
    alpha: float = 0.1  # Significance level
    n_folds: int = 5  # Number of CV folds
    
    # Ensemble parameters
    ensemble_method: str = "mean"  # "mean", "median", "weighted_mean"
    use_ensemble_uncertainty: bool = True
    uncertainty_method: str = "std"  # "std", "iqr", "quantile"
    
    # Cross-validation parameters
    cv_method: str = "kfold"  # "kfold", "timeseries"
    use_stratified_cv: bool = False
    random_state: Optional[int] = None
    
    # Calibration methods
    use_quantile_calibration: bool = True
    use_conformal_calibration: bool = True
    use_bayesian_calibration: bool = False
    
    # Quantile calibration
    quantile_levels: List[float] = None  # Will be set to [alpha/2, 1-alpha/2] if None
    
    # Conformal calibration
    conformal_method: str = "split"  # "split", "cv", "jackknife"
    conformal_alpha: float = 0.1
    
    # Bayesian calibration
    bayesian_prior: str = "uniform"  # "uniform", "normal", "beta"
    bayesian_samples: int = 1000
    
    # Validation
    use_validation_set: bool = True
    validation_size: float = 0.2
    
    # Monitoring
    monitor_calibration: bool = True
    calibration_tolerance: float = 0.05

class EnsembleCalibrator:
    """Calibrator for ensemble models"""
    
    def __init__(self, config: EnsembleCalibrationConfig):
        self.config = config
        self.is_fitted = False
        self.ensemble_models = []
        self.calibration_scores = []
        self.quantile_thresholds = {}
        self.conformal_thresholds = {}
        self.bayesian_params = {}
        self.calibration_history = []
        
        # Set default quantile levels
        if self.config.quantile_levels is None:
            self.config.quantile_levels = [self.config.alpha/2, 1 - self.config.alpha/2]
    
    def fit(self, ensemble_models: List[BaseEstimator], X_cal: np.ndarray, y_cal: np.ndarray,
            X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None) -> 'EnsembleCalibrator':
        """Fit the ensemble calibrator"""
        logger.info("🔧 Fitting ensemble calibrator...")
        
        self.ensemble_models = ensemble_models
        
        # Train ensemble models if training data is provided
        if X_train is not None and y_train is not None:
            logger.info("Training ensemble models...")
            for i, model in enumerate(self.ensemble_models):
                logger.info(f"Training model {i+1}/{len(self.ensemble_models)}")
                model.fit(X_train, y_train)
        
        # Get ensemble predictions on calibration data
        ensemble_preds = self._get_ensemble_predictions(X_cal)
        
        # Calculate calibration scores
        self.calibration_scores = self._calculate_calibration_scores(y_cal, ensemble_preds)
        
        # Fit different calibration methods
        if self.config.use_quantile_calibration:
            self._fit_quantile_calibration(y_cal, ensemble_preds)
        
        if self.config.use_conformal_calibration:
            self._fit_conformal_calibration(y_cal, ensemble_preds)
        
        if self.config.use_bayesian_calibration:
            self._fit_bayesian_calibration(y_cal, ensemble_preds)
        
        self.is_fitted = True
        
        logger.info(f"✅ Ensemble calibrator fitted with {len(self.ensemble_models)} models")
        logger.info(f"Calibration scores calculated for {len(self.calibration_scores)} samples")
        
        return self
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make calibrated ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble calibrator must be fitted before making predictions")
        
        # Get ensemble predictions
        ensemble_preds = self._get_ensemble_predictions(X)
        
        # Calculate prediction intervals using different methods
        results = {
            'predictions': ensemble_preds['mean'],
            'ensemble_std': ensemble_preds['std'],
            'ensemble_iqr': ensemble_preds['iqr']
        }
        
        # Add quantile-based intervals
        if self.config.use_quantile_calibration:
            quantile_intervals = self._get_quantile_intervals(ensemble_preds)
            results.update(quantile_intervals)
        
        # Add conformal intervals
        if self.config.use_conformal_calibration:
            conformal_intervals = self._get_conformal_intervals(ensemble_preds)
            results.update(conformal_intervals)
        
        # Add Bayesian intervals
        if self.config.use_bayesian_calibration:
            bayesian_intervals = self._get_bayesian_intervals(ensemble_preds)
            results.update(bayesian_intervals)
        
        return results
    
    def _get_ensemble_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all ensemble models"""
        predictions = []
        
        for model in self.ensemble_models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)  # (n_models, n_samples)
        
        # Calculate ensemble statistics
        if self.config.ensemble_method == "mean":
            mean_pred = np.mean(predictions, axis=0)
        elif self.config.ensemble_method == "median":
            mean_pred = np.median(predictions, axis=0)
        else:
            mean_pred = np.mean(predictions, axis=0)
        
        # Calculate uncertainty measures
        std_pred = np.std(predictions, axis=0)
        iqr_pred = np.percentile(predictions, 75, axis=0) - np.percentile(predictions, 25, axis=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'iqr': iqr_pred,
            'individual': predictions
        }
    
    def _calculate_calibration_scores(self, y_true: np.ndarray, ensemble_preds: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate calibration scores for ensemble predictions"""
        if self.config.uncertainty_method == "std":
            uncertainty = ensemble_preds['std']
        elif self.config.uncertainty_method == "iqr":
            uncertainty = ensemble_preds['iqr']
        else:
            uncertainty = ensemble_preds['std']
        
        # Calculate normalized residuals
        residuals = np.abs(y_true - ensemble_preds['mean'])
        scores = residuals / (uncertainty + 1e-8)  # Add small epsilon to avoid division by zero
        
        return scores
    
    def _fit_quantile_calibration(self, y_true: np.ndarray, ensemble_preds: Dict[str, np.ndarray]):
        """Fit quantile-based calibration"""
        logger.info("Fitting quantile calibration...")
        
        # Calculate quantiles of calibration scores
        for level in self.config.quantile_levels:
            threshold = np.quantile(self.calibration_scores, level)
            self.quantile_thresholds[level] = threshold
        
        logger.info(f"Quantile thresholds: {self.quantile_thresholds}")
    
    def _fit_conformal_calibration(self, y_true: np.ndarray, ensemble_preds: Dict[str, np.ndarray]):
        """Fit conformal calibration"""
        logger.info("Fitting conformal calibration...")
        
        # Calculate conformal threshold
        threshold = np.quantile(self.calibration_scores, 1 - self.config.conformal_alpha)
        self.conformal_thresholds['conformal'] = threshold
        
        logger.info(f"Conformal threshold: {threshold:.4f}")
    
    def _fit_bayesian_calibration(self, y_true: np.ndarray, ensemble_preds: Dict[str, np.ndarray]):
        """Fit Bayesian calibration"""
        logger.info("Fitting Bayesian calibration...")
        
        # Simple Bayesian calibration using beta distribution
        # This is a simplified implementation
        n_samples = len(self.calibration_scores)
        n_within_interval = np.sum(self.calibration_scores <= 1.0)  # Assuming 1.0 is the target threshold
        
        # Beta distribution parameters
        alpha_param = n_within_interval + 1
        beta_param = n_samples - n_within_interval + 1
        
        self.bayesian_params = {
            'alpha': alpha_param,
            'beta': beta_param,
            'mean': alpha_param / (alpha_param + beta_param),
            'variance': (alpha_param * beta_param) / ((alpha_param + beta_param) ** 2 * (alpha_param + beta_param + 1))
        }
        
        logger.info(f"Bayesian parameters: {self.bayesian_params}")
    
    def _get_quantile_intervals(self, ensemble_preds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get quantile-based prediction intervals"""
        intervals = {}
        
        for level, threshold in self.quantile_thresholds.items():
            if level < 0.5:  # Lower bound
                lower_bound = ensemble_preds['mean'] - threshold * ensemble_preds['std']
                intervals[f'quantile_lower_{level:.3f}'] = lower_bound
            else:  # Upper bound
                upper_bound = ensemble_preds['mean'] + threshold * ensemble_preds['std']
                intervals[f'quantile_upper_{level:.3f}'] = upper_bound
        
        return intervals
    
    def _get_conformal_intervals(self, ensemble_preds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get conformal prediction intervals"""
        threshold = self.conformal_thresholds['conformal']
        
        lower_bound = ensemble_preds['mean'] - threshold * ensemble_preds['std']
        upper_bound = ensemble_preds['mean'] + threshold * ensemble_preds['std']
        
        return {
            'conformal_lower': lower_bound,
            'conformal_upper': upper_bound,
            'conformal_width': upper_bound - lower_bound
        }
    
    def _get_bayesian_intervals(self, ensemble_preds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get Bayesian prediction intervals"""
        # Simplified Bayesian intervals
        mean_coverage = self.bayesian_params['mean']
        std_coverage = np.sqrt(self.bayesian_params['variance'])
        
        # Adjust uncertainty based on Bayesian parameters
        adjusted_std = ensemble_preds['std'] * (1 + std_coverage)
        
        lower_bound = ensemble_preds['mean'] - 1.96 * adjusted_std
        upper_bound = ensemble_preds['mean'] + 1.96 * adjusted_std
        
        return {
            'bayesian_lower': lower_bound,
            'bayesian_upper': upper_bound,
            'bayesian_width': upper_bound - lower_bound,
            'bayesian_coverage': mean_coverage
        }
    
    def evaluate_calibration(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate calibration quality"""
        if not self.is_fitted:
            raise ValueError("Ensemble calibrator must be fitted before evaluation")
        
        # Get predictions
        predictions = self.predict(X_test)
        
        # Calculate empirical coverage for different methods
        coverage_results = {}
        
        # Quantile coverage
        if self.config.use_quantile_calibration:
            for level in self.config.quantile_levels:
                if level < 0.5:
                    lower_key = f'quantile_lower_{level:.3f}'
                    if lower_key in predictions:
                        coverage = np.mean(y_test >= predictions[lower_key])
                        coverage_results[f'quantile_lower_{level:.3f}_coverage'] = coverage
                else:
                    upper_key = f'quantile_upper_{level:.3f}'
                    if upper_key in predictions:
                        coverage = np.mean(y_test <= predictions[upper_key])
                        coverage_results[f'quantile_upper_{level:.3f}_coverage'] = coverage
        
        # Conformal coverage
        if self.config.use_conformal_calibration:
            if 'conformal_lower' in predictions and 'conformal_upper' in predictions:
                coverage = np.mean((y_test >= predictions['conformal_lower']) & 
                                 (y_test <= predictions['conformal_upper']))
                coverage_results['conformal_coverage'] = coverage
        
        # Bayesian coverage
        if self.config.use_bayesian_calibration:
            if 'bayesian_lower' in predictions and 'bayesian_upper' in predictions:
                coverage = np.mean((y_test >= predictions['bayesian_lower']) & 
                                 (y_test <= predictions['bayesian_upper']))
                coverage_results['bayesian_coverage'] = coverage
        
        # Calculate calibration error
        calibration_errors = {}
        for method, coverage in coverage_results.items():
            if 'conformal' in method:
                target_coverage = 1 - self.config.conformal_alpha
            elif 'bayesian' in method:
                target_coverage = 0.95  # 95% confidence interval
            else:
                target_coverage = 1 - self.config.alpha
            
            calibration_errors[f'{method}_error'] = abs(coverage - target_coverage)
        
        # Calculate prediction accuracy
        mae = mean_absolute_error(y_test, predictions['predictions'])
        rmse = np.sqrt(mean_squared_error(y_test, predictions['predictions']))
        
        results = {
            'mae': mae,
            'rmse': rmse,
            'coverage_results': coverage_results,
            'calibration_errors': calibration_errors
        }
        
        return results
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get calibration summary"""
        if not self.is_fitted:
            return {}
        
        summary = {
            'n_models': len(self.ensemble_models),
            'n_calibration_samples': len(self.calibration_scores),
            'calibration_methods': {
                'quantile': self.config.use_quantile_calibration,
                'conformal': self.config.use_conformal_calibration,
                'bayesian': self.config.use_bayesian_calibration
            },
            'quantile_thresholds': self.quantile_thresholds,
            'conformal_thresholds': self.conformal_thresholds,
            'bayesian_params': self.bayesian_params
        }
        
        return summary
    
    def save_calibrator(self, path: str):
        """Save the ensemble calibrator"""
        calibrator_state = {
            'config': self.config,
            'calibration_scores': self.calibration_scores,
            'quantile_thresholds': self.quantile_thresholds,
            'conformal_thresholds': self.conformal_thresholds,
            'bayesian_params': self.bayesian_params,
            'calibration_history': self.calibration_history,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'w') as f:
            json.dump(calibrator_state, f, indent=2, default=str)
        
        logger.info(f"Ensemble calibrator saved to {path}")
    
    def load_calibrator(self, path: str):
        """Load the ensemble calibrator"""
        with open(path, 'r') as f:
            calibrator_state = json.load(f)
        
        self.config = calibrator_state['config']
        self.calibration_scores = calibrator_state['calibration_scores']
        self.quantile_thresholds = calibrator_state['quantile_thresholds']
        self.conformal_thresholds = calibrator_state['conformal_thresholds']
        self.bayesian_params = calibrator_state['bayesian_params']
        self.calibration_history = calibrator_state['calibration_history']
        self.is_fitted = calibrator_state['is_fitted']
        
        logger.info(f"Ensemble calibrator loaded from {path}")
    
    def get_calibrator_info(self) -> Dict[str, Any]:
        """Get calibrator information"""
        return {
            'method': 'ensemble_calibration',
            'alpha': self.config.alpha,
            'coverage_probability': 1 - self.config.alpha,
            'is_fitted': self.is_fitted,
            'n_models': len(self.ensemble_models),
            'n_calibration_samples': len(self.calibration_scores),
            'calibration_methods': {
                'quantile': self.config.use_quantile_calibration,
                'conformal': self.config.use_conformal_calibration,
                'bayesian': self.config.use_bayesian_calibration
            }
        }
