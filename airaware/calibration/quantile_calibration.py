"""
Quantile Calibration for Uncertainty Quantification

This module implements quantile-based calibration methods for improving
the reliability of prediction intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
import warnings

logger = logging.getLogger(__name__)

@dataclass
class QuantileCalibrationConfig:
    """Configuration for quantile calibration"""
    # Calibration parameters
    target_quantiles: List[float] = None  # Target quantiles to calibrate
    calibration_method: str = "isotonic"  # "isotonic", "linear", "spline"
    
    # Data splitting
    calibration_ratio: float = 0.2
    random_state: Optional[int] = None
    
    # Calibration validation
    validate_calibration: bool = True
    validation_ratio: float = 0.1
    
    # Smoothing parameters
    use_smoothing: bool = False
    smoothing_factor: float = 0.1
    
    # Time series specific
    respect_temporal_order: bool = True
    use_rolling_calibration: bool = False
    rolling_window: int = 1000
    
    def __post_init__(self):
        if self.target_quantiles is None:
            self.target_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

class QuantileCalibrator:
    """Quantile calibrator for improving prediction interval reliability"""
    
    def __init__(self, config: QuantileCalibrationConfig):
        self.config = config
        self.is_fitted = False
        self.calibration_models = {}
        self.calibration_data = None
        self.quantile_mappings = {}
        
    def fit(self, model: BaseEstimator, X_cal: np.ndarray, y_cal: np.ndarray,
            X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None) -> 'QuantileCalibrator':
        """Fit the quantile calibrator using calibration data"""
        logger.info("🔧 Fitting quantile calibrator...")
        
        # Train the base model if training data is provided
        if X_train is not None and y_train is not None:
            logger.info("Training base model...")
            model.fit(X_train, y_train)
        
        # Get predictions on calibration data
        y_pred_cal = model.predict(X_cal)
        
        # Calculate residuals
        residuals = y_cal - y_pred_cal
        
        # Fit calibration models for each quantile
        for quantile in self.config.target_quantiles:
            logger.info(f"Calibrating quantile {quantile}...")
            
            # Calculate empirical quantiles of residuals
            empirical_quantile = np.quantile(residuals, quantile)
            
            # Create calibration data
            cal_X = y_pred_cal.reshape(-1, 1)  # Use predictions as features
            cal_y = np.full_like(y_pred_cal, empirical_quantile)  # Target is the empirical quantile
            
            # Fit calibration model
            if self.config.calibration_method == "isotonic":
                cal_model = IsotonicRegression(out_of_bounds='clip')
            elif self.config.calibration_method == "linear":
                cal_model = LinearRegression()
            else:
                raise ValueError(f"Unknown calibration method: {self.config.calibration_method}")
            
            cal_model.fit(cal_X, cal_y)
            self.calibration_models[quantile] = cal_model
            
            # Store quantile mapping
            self.quantile_mappings[quantile] = empirical_quantile
        
        # Store calibration data
        self.calibration_data = {
            'X': X_cal,
            'y': y_cal,
            'y_pred': y_pred_cal,
            'residuals': residuals
        }
        
        self.is_fitted = True
        logger.info(f"✅ Quantile calibrator fitted for {len(self.config.target_quantiles)} quantiles")
        
        return self
    
    def predict(self, X: np.ndarray, model: BaseEstimator) -> Dict[str, np.ndarray]:
        """Make calibrated quantile predictions"""
        if not self.is_fitted:
            raise ValueError("Quantile calibrator must be fitted before making predictions")
        
        # Get point predictions
        y_pred = model.predict(X)
        
        # Calculate calibrated quantiles
        calibrated_quantiles = {}
        for quantile in self.config.target_quantiles:
            cal_model = self.calibration_models[quantile]
            cal_X = y_pred.reshape(-1, 1)
            calibrated_quantile = cal_model.predict(cal_X)
            calibrated_quantiles[quantile] = calibrated_quantile
        
        # Calculate prediction intervals
        lower_quantile = min(self.config.target_quantiles)
        upper_quantile = max(self.config.target_quantiles)
        
        lower_bound = y_pred + calibrated_quantiles[lower_quantile]
        upper_bound = y_pred + calibrated_quantiles[upper_quantile]
        
        # Calculate median prediction
        median_quantile = 0.5
        if median_quantile in calibrated_quantiles:
            median_pred = y_pred + calibrated_quantiles[median_quantile]
        else:
            median_pred = y_pred
        
        return {
            'predictions': median_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound,
            'calibrated_quantiles': calibrated_quantiles,
            'coverage_probability': upper_quantile - lower_quantile
        }
    
    def get_calibration_quality(self, X_val: np.ndarray, y_val: np.ndarray, 
                              model: BaseEstimator) -> Dict[str, float]:
        """Evaluate calibration quality on validation data"""
        if not self.is_fitted:
            raise ValueError("Quantile calibrator must be fitted before evaluation")
        
        # Get calibrated predictions
        predictions = self.predict(X_val, model)
        
        # Calculate empirical coverage for each quantile
        coverage_metrics = {}
        for quantile in self.config.target_quantiles:
            if quantile in predictions['calibrated_quantiles']:
                calibrated_quantile = predictions['calibrated_quantiles'][quantile]
                y_pred = model.predict(X_val)
                
                # Calculate empirical coverage
                if quantile <= 0.5:
                    # Lower quantile
                    coverage = np.mean(y_val <= y_pred + calibrated_quantile)
                else:
                    # Upper quantile
                    coverage = np.mean(y_val >= y_pred + calibrated_quantile)
                
                coverage_metrics[f'coverage_{quantile}'] = coverage
                coverage_metrics[f'coverage_error_{quantile}'] = abs(coverage - quantile)
        
        # Calculate overall calibration metrics
        lower_bound = predictions['lower_bound']
        upper_bound = predictions['upper_bound']
        
        # Empirical coverage of the prediction interval
        interval_coverage = np.mean((y_val >= lower_bound) & (y_val <= upper_bound))
        expected_coverage = predictions['coverage_probability']
        
        coverage_metrics['interval_coverage'] = interval_coverage
        coverage_metrics['expected_coverage'] = expected_coverage
        coverage_metrics['coverage_error'] = abs(interval_coverage - expected_coverage)
        
        # Average interval width
        coverage_metrics['avg_interval_width'] = np.mean(upper_bound - lower_bound)
        
        return coverage_metrics
    
    def plot_calibration_curve(self, X_val: np.ndarray, y_val: np.ndarray, 
                             model: BaseEstimator, save_path: Optional[str] = None):
        """Plot calibration curves for each quantile"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return
        
        if not self.is_fitted:
            raise ValueError("Quantile calibrator must be fitted before plotting")
        
        # Get calibrated predictions
        predictions = self.predict(X_val, model)
        y_pred = model.predict(X_val)
        
        # Create subplots for each quantile
        n_quantiles = len(self.config.target_quantiles)
        fig, axes = plt.subplots(2, (n_quantiles + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_quantiles > 1 else [axes]
        
        for i, quantile in enumerate(self.config.target_quantiles):
            if quantile in predictions['calibrated_quantiles']:
                calibrated_quantile = predictions['calibrated_quantiles'][quantile]
                
                # Calculate empirical quantiles
                empirical_quantiles = []
                theoretical_quantiles = []
                
                # Sort predictions and calculate empirical quantiles
                sorted_indices = np.argsort(y_pred)
                sorted_y_pred = y_pred[sorted_indices]
                sorted_y_val = y_val[sorted_indices]
                sorted_calibrated = calibrated_quantile[sorted_indices]
                
                # Calculate empirical coverage at different prediction levels
                n_bins = 20
                for j in range(n_bins):
                    start_idx = j * len(sorted_y_pred) // n_bins
                    end_idx = (j + 1) * len(sorted_y_pred) // n_bins
                    
                    if start_idx < end_idx:
                        bin_y_pred = sorted_y_pred[start_idx:end_idx]
                        bin_y_val = sorted_y_val[start_idx:end_idx]
                        bin_calibrated = sorted_calibrated[start_idx:end_idx]
                        
                        # Calculate empirical coverage
                        if quantile <= 0.5:
                            empirical_coverage = np.mean(bin_y_val <= bin_y_pred + bin_calibrated)
                        else:
                            empirical_coverage = np.mean(bin_y_val >= bin_y_pred + bin_calibrated)
                        
                        empirical_quantiles.append(empirical_coverage)
                        theoretical_quantiles.append(quantile)
                
                # Plot calibration curve
                axes[i].plot(theoretical_quantiles, empirical_quantiles, 'o-', label=f'Quantile {quantile}')
                axes[i].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                axes[i].set_xlabel('Theoretical Quantile')
                axes[i].set_ylabel('Empirical Quantile')
                axes[i].set_title(f'Calibration Curve - Quantile {quantile}')
                axes[i].legend()
                axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration curves saved to {save_path}")
        
        plt.show()
    
    def save_calibrator(self, path: str):
        """Save the quantile calibrator"""
        calibrator_state = {
            'config': self.config,
            'calibration_models': self.calibration_models,
            'calibration_data': self.calibration_data,
            'quantile_mappings': self.quantile_mappings,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'w') as f:
            json.dump(calibrator_state, f, indent=2, default=str)
        
        logger.info(f"Quantile calibrator saved to {path}")
    
    def load_calibrator(self, path: str):
        """Load the quantile calibrator"""
        with open(path, 'r') as f:
            calibrator_state = json.load(f)
        
        self.config = calibrator_state['config']
        self.calibration_models = calibrator_state['calibration_models']
        self.calibration_data = calibrator_state['calibration_data']
        self.quantile_mappings = calibrator_state['quantile_mappings']
        self.is_fitted = calibrator_state['is_fitted']
        
        logger.info(f"Quantile calibrator loaded from {path}")
    
    def get_calibrator_info(self) -> Dict[str, Any]:
        """Get calibrator information"""
        return {
            'method': self.config.calibration_method,
            'target_quantiles': self.config.target_quantiles,
            'is_fitted': self.is_fitted,
            'n_calibration_samples': len(self.calibration_data['y']) if self.calibration_data else 0,
            'calibrated_quantiles': list(self.calibration_models.keys())
        }
