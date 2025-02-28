"""
Calibration Enhancements

This module provides advanced calibration enhancements including
temporal calibration, multi-target calibration, and calibration
monitoring for production systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class TemporalCalibrationConfig:
    """Configuration for temporal calibration"""
    # Temporal parameters
    use_temporal_weighting: bool = True
    temporal_decay: float = 0.99  # Exponential decay factor
    temporal_window: int = 1000  # Window for temporal calibration
    
    # Seasonality parameters
    use_seasonal_calibration: bool = True
    seasonal_periods: List[int] = None  # Will be set to [24, 168] if None (hourly, weekly)
    
    # Trend parameters
    use_trend_calibration: bool = True
    trend_window: int = 100  # Window for trend detection
    
    # Holiday effects
    use_holiday_calibration: bool = False
    holiday_dates: List[str] = None  # List of holiday dates
    
    # Time-of-day effects
    use_hourly_calibration: bool = True
    hourly_bins: int = 6  # Number of hourly bins (4-hour periods)
    
    # Day-of-week effects
    use_weekly_calibration: bool = True
    
    def __post_init__(self):
        if self.seasonal_periods is None:
            self.seasonal_periods = [24, 168]  # Hourly and weekly

class TemporalCalibrator:
    """Temporal calibrator for time series data"""
    
    def __init__(self, config: TemporalCalibrationConfig):
        self.config = config
        self.is_fitted = False
        self.temporal_weights = {}
        self.seasonal_calibrators = {}
        self.trend_calibrator = None
        self.hourly_calibrators = {}
        self.weekly_calibrators = {}
        self.calibration_history = []
        
    def fit(self, timestamps: pd.Series, calibration_scores: np.ndarray) -> 'TemporalCalibrator':
        """Fit temporal calibrator"""
        logger.info("🔧 Fitting temporal calibrator...")
        
        # Convert timestamps to datetime if needed
        if not isinstance(timestamps.iloc[0], pd.Timestamp):
            timestamps = pd.to_datetime(timestamps)
        
        # Calculate temporal weights
        if self.config.use_temporal_weighting:
            self._calculate_temporal_weights(timestamps, calibration_scores)
        
        # Fit seasonal calibrators
        if self.config.use_seasonal_calibration:
            self._fit_seasonal_calibrators(timestamps, calibration_scores)
        
        # Fit trend calibrator
        if self.config.use_trend_calibration:
            self._fit_trend_calibrator(timestamps, calibration_scores)
        
        # Fit hourly calibrators
        if self.config.use_hourly_calibration:
            self._fit_hourly_calibrators(timestamps, calibration_scores)
        
        # Fit weekly calibrators
        if self.config.use_weekly_calibration:
            self._fit_weekly_calibrators(timestamps, calibration_scores)
        
        self.is_fitted = True
        logger.info("✅ Temporal calibrator fitted")
        
        return self
    
    def _calculate_temporal_weights(self, timestamps: pd.Series, calibration_scores: np.ndarray):
        """Calculate temporal weights for calibration scores"""
        # Exponential decay weights
        n_samples = len(timestamps)
        weights = np.array([self.config.temporal_decay ** (n_samples - i - 1) for i in range(n_samples)])
        weights = weights / np.sum(weights)  # Normalize
        
        self.temporal_weights = {
            'weights': weights,
            'decay_factor': self.config.temporal_decay
        }
    
    def _fit_seasonal_calibrators(self, timestamps: pd.Series, calibration_scores: np.ndarray):
        """Fit seasonal calibrators"""
        for period in self.config.seasonal_periods:
            # Calculate seasonal component
            seasonal_component = timestamps.dt.hour if period == 24 else timestamps.dt.dayofweek
            
            # Group by seasonal component
            seasonal_groups = {}
            for i, component in enumerate(seasonal_component):
                if component not in seasonal_groups:
                    seasonal_groups[component] = []
                seasonal_groups[component].append(calibration_scores[i])
            
            # Calculate seasonal statistics
            seasonal_stats = {}
            for component, scores in seasonal_groups.items():
                seasonal_stats[component] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'count': len(scores)
                }
            
            self.seasonal_calibrators[period] = seasonal_stats
    
    def _fit_trend_calibrator(self, timestamps: pd.Series, calibration_scores: np.ndarray):
        """Fit trend calibrator"""
        # Calculate rolling statistics
        window = self.config.trend_window
        rolling_means = []
        rolling_stds = []
        
        for i in range(len(calibration_scores)):
            start_idx = max(0, i - window + 1)
            window_scores = calibration_scores[start_idx:i+1]
            rolling_means.append(np.mean(window_scores))
            rolling_stds.append(np.std(window_scores))
        
        self.trend_calibrator = {
            'rolling_means': rolling_means,
            'rolling_stds': rolling_stds,
            'window_size': window
        }
    
    def _fit_hourly_calibrators(self, timestamps: pd.Series, calibration_scores: np.ndarray):
        """Fit hourly calibrators"""
        # Create hourly bins
        hours = timestamps.dt.hour
        bin_size = 24 // self.config.hourly_bins
        
        for bin_idx in range(self.config.hourly_bins):
            start_hour = bin_idx * bin_size
            end_hour = min((bin_idx + 1) * bin_size, 24)
            
            # Find samples in this hour bin
            mask = (hours >= start_hour) & (hours < end_hour)
            bin_scores = calibration_scores[mask]
            
            if len(bin_scores) > 0:
                self.hourly_calibrators[bin_idx] = {
                    'start_hour': start_hour,
                    'end_hour': end_hour,
                    'mean': np.mean(bin_scores),
                    'std': np.std(bin_scores),
                    'count': len(bin_scores)
                }
    
    def _fit_weekly_calibrators(self, timestamps: pd.Series, calibration_scores: np.ndarray):
        """Fit weekly calibrators"""
        days_of_week = timestamps.dt.dayofweek
        
        for day in range(7):
            mask = days_of_week == day
            day_scores = calibration_scores[mask]
            
            if len(day_scores) > 0:
                self.weekly_calibrators[day] = {
                    'mean': np.mean(day_scores),
                    'std': np.std(day_scores),
                    'count': len(day_scores)
                }
    
    def predict(self, timestamps: pd.Series, base_scores: np.ndarray) -> np.ndarray:
        """Apply temporal calibration to scores"""
        if not self.is_fitted:
            raise ValueError("Temporal calibrator must be fitted before prediction")
        
        # Convert timestamps to datetime if needed
        if not isinstance(timestamps.iloc[0], pd.Timestamp):
            timestamps = pd.to_datetime(timestamps)
        
        calibrated_scores = base_scores.copy()
        
        # Apply temporal weighting
        if self.config.use_temporal_weighting and 'weights' in self.temporal_weights:
            # Use the most recent weights
            weights = self.temporal_weights['weights']
            if len(weights) >= len(calibrated_scores):
                calibrated_scores = calibrated_scores * weights[:len(calibrated_scores)]
        
        # Apply seasonal calibration
        if self.config.use_seasonal_calibration:
            for period, calibrator in self.seasonal_calibrators.items():
                if period == 24:  # Hourly
                    hours = timestamps.dt.hour
                    for i, hour in enumerate(hours):
                        if hour in calibrator:
                            seasonal_mean = calibrator[hour]['mean']
                            calibrated_scores[i] = calibrated_scores[i] * (1 + 0.1 * (seasonal_mean - 1))
                else:  # Weekly
                    days = timestamps.dt.dayofweek
                    for i, day in enumerate(days):
                        if day in calibrator:
                            seasonal_mean = calibrator[day]['mean']
                            calibrated_scores[i] = calibrated_scores[i] * (1 + 0.1 * (seasonal_mean - 1))
        
        # Apply hourly calibration
        if self.config.use_hourly_calibration:
            hours = timestamps.dt.hour
            bin_size = 24 // self.config.hourly_bins
            
            for i, hour in enumerate(hours):
                bin_idx = min(hour // bin_size, self.config.hourly_bins - 1)
                if bin_idx in self.hourly_calibrators:
                    hourly_mean = self.hourly_calibrators[bin_idx]['mean']
                    calibrated_scores[i] = calibrated_scores[i] * (1 + 0.05 * (hourly_mean - 1))
        
        # Apply weekly calibration
        if self.config.use_weekly_calibration:
            days = timestamps.dt.dayofweek
            for i, day in enumerate(days):
                if day in self.weekly_calibrators:
                    weekly_mean = self.weekly_calibrators[day]['mean']
                    calibrated_scores[i] = calibrated_scores[i] * (1 + 0.05 * (weekly_mean - 1))
        
        return calibrated_scores
    
    def get_temporal_summary(self) -> Dict[str, Any]:
        """Get temporal calibration summary"""
        if not self.is_fitted:
            return {}
        
        summary = {
            'temporal_weights': self.temporal_weights,
            'seasonal_calibrators': {k: len(v) for k, v in self.seasonal_calibrators.items()},
            'trend_calibrator': self.trend_calibrator is not None,
            'hourly_calibrators': len(self.hourly_calibrators),
            'weekly_calibrators': len(self.weekly_calibrators)
        }
        
        return summary

@dataclass
class MultiTargetCalibrationConfig:
    """Configuration for multi-target calibration"""
    # Target parameters
    target_columns: List[str] = None  # Will be set to ['pm25', 'pm10', 'no2'] if None
    target_weights: Dict[str, float] = None  # Will be set to equal weights if None
    
    # Calibration parameters
    use_joint_calibration: bool = True
    use_individual_calibration: bool = True
    joint_alpha: float = 0.1
    individual_alpha: float = 0.1
    
    # Correlation parameters
    use_correlation_calibration: bool = True
    correlation_threshold: float = 0.7
    
    # Ensemble parameters
    use_ensemble_calibration: bool = True
    ensemble_method: str = "weighted_average"  # "weighted_average", "voting", "stacking"
    
    def __post_init__(self):
        if self.target_columns is None:
            self.target_columns = ['pm25', 'pm10', 'no2']
        if self.target_weights is None:
            self.target_weights = {col: 1.0/len(self.target_columns) for col in self.target_columns}

class MultiTargetCalibrator:
    """Multi-target calibrator for multiple air quality parameters"""
    
    def __init__(self, config: MultiTargetCalibrationConfig):
        self.config = config
        self.is_fitted = False
        self.individual_calibrators = {}
        self.joint_calibrator = None
        self.correlation_matrix = None
        self.ensemble_weights = {}
        
    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray, 
            base_models: Dict[str, BaseEstimator]) -> 'MultiTargetCalibrator':
        """Fit multi-target calibrator"""
        logger.info("🔧 Fitting multi-target calibrator...")
        
        # Fit individual calibrators
        if self.config.use_individual_calibration:
            self._fit_individual_calibrators(X_cal, y_cal, base_models)
        
        # Fit joint calibrator
        if self.config.use_joint_calibration:
            self._fit_joint_calibrator(X_cal, y_cal, base_models)
        
        # Calculate correlations
        if self.config.use_correlation_calibration:
            self._calculate_correlations(y_cal)
        
        # Fit ensemble calibrator
        if self.config.use_ensemble_calibration:
            self._fit_ensemble_calibrator(X_cal, y_cal, base_models)
        
        self.is_fitted = True
        logger.info("✅ Multi-target calibrator fitted")
        
        return self
    
    def _fit_individual_calibrators(self, X_cal: np.ndarray, y_cal: np.ndarray, 
                                  base_models: Dict[str, BaseEstimator]):
        """Fit individual calibrators for each target"""
        for target in self.config.target_columns:
            if target in base_models:
                # Get predictions for this target
                y_pred = base_models[target].predict(X_cal)
                y_true = y_cal[:, self.config.target_columns.index(target)]
                
                # Calculate calibration scores
                scores = np.abs(y_true - y_pred)
                
                # Store calibrator
                self.individual_calibrators[target] = {
                    'scores': scores,
                    'threshold': np.quantile(scores, 1 - self.config.individual_alpha)
                }
    
    def _fit_joint_calibrator(self, X_cal: np.ndarray, y_cal: np.ndarray, 
                            base_models: Dict[str, BaseEstimator]):
        """Fit joint calibrator across all targets"""
        joint_scores = []
        
        for target in self.config.target_columns:
            if target in base_models:
                y_pred = base_models[target].predict(X_cal)
                y_true = y_cal[:, self.config.target_columns.index(target)]
                scores = np.abs(y_true - y_pred)
                joint_scores.append(scores)
        
        if joint_scores:
            # Combine scores across targets
            combined_scores = np.mean(joint_scores, axis=0)
            self.joint_calibrator = {
                'scores': combined_scores,
                'threshold': np.quantile(combined_scores, 1 - self.config.joint_alpha)
            }
    
    def _calculate_correlations(self, y_cal: np.ndarray):
        """Calculate correlations between targets"""
        if y_cal.shape[1] >= 2:
            self.correlation_matrix = np.corrcoef(y_cal.T)
        else:
            self.correlation_matrix = np.array([[1.0]])
    
    def _fit_ensemble_calibrator(self, X_cal: np.ndarray, y_cal: np.ndarray, 
                               base_models: Dict[str, BaseEstimator]):
        """Fit ensemble calibrator"""
        ensemble_scores = []
        
        for target in self.config.target_columns:
            if target in base_models:
                y_pred = base_models[target].predict(X_cal)
                y_true = y_cal[:, self.config.target_columns.index(target)]
                scores = np.abs(y_true - y_pred)
                ensemble_scores.append(scores)
        
        if ensemble_scores:
            # Calculate ensemble weights
            if self.config.ensemble_method == "weighted_average":
                # Weight by inverse of individual thresholds
                weights = []
                for target in self.config.target_columns:
                    if target in self.individual_calibrators:
                        weight = 1.0 / self.individual_calibrators[target]['threshold']
                        weights.append(weight)
                    else:
                        weights.append(1.0)
                
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                self.ensemble_weights = dict(zip(self.config.target_columns, weights))
    
    def predict(self, X: np.ndarray, base_models: Dict[str, BaseEstimator]) -> Dict[str, Dict[str, np.ndarray]]:
        """Make multi-target calibrated predictions"""
        if not self.is_fitted:
            raise ValueError("Multi-target calibrator must be fitted before prediction")
        
        results = {}
        
        for target in self.config.target_columns:
            if target in base_models:
                # Get base predictions
                y_pred = base_models[target].predict(X)
                
                # Get calibration threshold
                if target in self.individual_calibrators:
                    threshold = self.individual_calibrators[target]['threshold']
                else:
                    threshold = 1.0
                
                # Calculate prediction intervals
                lower_bound = y_pred - threshold
                upper_bound = y_pred + threshold
                
                results[target] = {
                    'predictions': y_pred,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'interval_width': upper_bound - lower_bound
                }
        
        return results
    
    def get_multi_target_summary(self) -> Dict[str, Any]:
        """Get multi-target calibration summary"""
        if not self.is_fitted:
            return {}
        
        summary = {
            'targets': self.config.target_columns,
            'individual_calibrators': len(self.individual_calibrators),
            'joint_calibrator': self.joint_calibrator is not None,
            'correlation_matrix': self.correlation_matrix.tolist() if self.correlation_matrix is not None else None,
            'ensemble_weights': self.ensemble_weights
        }
        
        return summary

@dataclass
class CalibrationMonitorConfig:
    """Configuration for calibration monitoring"""
    # Monitoring parameters
    monitor_frequency: int = 100  # Monitor every N predictions
    alert_threshold: float = 0.05  # Alert if coverage deviates by this amount
    min_samples_for_alert: int = 50  # Minimum samples before alerting
    
    # Drift detection
    use_drift_detection: bool = True
    drift_threshold: float = 0.1  # Drift detection threshold
    drift_window: int = 500  # Window for drift detection
    
    # Performance tracking
    track_coverage: bool = True
    track_interval_width: bool = True
    track_calibration_error: bool = True
    
    # Alerting
    use_alerts: bool = True
    alert_methods: List[str] = None  # Will be set to ['log', 'email'] if None
    
    def __post_init__(self):
        if self.alert_methods is None:
            self.alert_methods = ['log']

class CalibrationMonitor:
    """Monitor calibration performance in production"""
    
    def __init__(self, config: CalibrationMonitorConfig):
        self.config = config
        self.is_monitoring = False
        self.prediction_history = deque(maxlen=10000)
        self.coverage_history = deque(maxlen=1000)
        self.interval_width_history = deque(maxlen=1000)
        self.calibration_error_history = deque(maxlen=1000)
        self.alerts = []
        self.drift_detector = None
        
    def start_monitoring(self):
        """Start calibration monitoring"""
        self.is_monitoring = True
        logger.info("🔍 Started calibration monitoring")
    
    def stop_monitoring(self):
        """Stop calibration monitoring"""
        self.is_monitoring = False
        logger.info("⏹️ Stopped calibration monitoring")
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray, 
               y_lower: np.ndarray, y_upper: np.ndarray):
        """Update monitor with new predictions"""
        if not self.is_monitoring:
            return
        
        # Store prediction history
        for i in range(len(y_true)):
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'y_true': y_true[i],
                'y_pred': y_pred[i],
                'y_lower': y_lower[i],
                'y_upper': y_upper[i]
            })
        
        # Calculate metrics
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        interval_width = np.mean(y_upper - y_lower)
        calibration_error = abs(coverage - 0.9)  # Assuming 90% target coverage
        
        # Store metric history
        self.coverage_history.append(coverage)
        self.interval_width_history.append(interval_width)
        self.calibration_error_history.append(calibration_error)
        
        # Check for alerts
        if len(self.coverage_history) >= self.config.min_samples_for_alert:
            self._check_alerts()
        
        # Check for drift
        if self.config.use_drift_detection and len(self.coverage_history) >= self.config.drift_window:
            self._check_drift()
    
    def _check_alerts(self):
        """Check for calibration alerts"""
        recent_coverage = np.mean(list(self.coverage_history)[-self.config.min_samples_for_alert:])
        target_coverage = 0.9  # Assuming 90% target coverage
        
        if abs(recent_coverage - target_coverage) > self.config.alert_threshold:
            alert = {
                'timestamp': datetime.now(),
                'type': 'coverage_deviation',
                'message': f"Coverage deviation detected: {recent_coverage:.3f} vs target {target_coverage:.3f}",
                'severity': 'warning' if abs(recent_coverage - target_coverage) < 0.1 else 'critical'
            }
            
            self.alerts.append(alert)
            
            if 'log' in self.config.alert_methods:
                logger.warning(alert['message'])
    
    def _check_drift(self):
        """Check for calibration drift"""
        if len(self.coverage_history) < self.config.drift_window * 2:
            return
        
        # Compare recent vs historical coverage
        recent_coverage = np.mean(list(self.coverage_history)[-self.config.drift_window:])
        historical_coverage = np.mean(list(self.coverage_history)[-self.config.drift_window*2:-self.config.drift_window])
        
        drift = abs(recent_coverage - historical_coverage)
        
        if drift > self.config.drift_threshold:
            alert = {
                'timestamp': datetime.now(),
                'type': 'calibration_drift',
                'message': f"Calibration drift detected: {drift:.3f}",
                'severity': 'warning' if drift < 0.2 else 'critical'
            }
            
            self.alerts.append(alert)
            
            if 'log' in self.config.alert_methods:
                logger.warning(alert['message'])
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary"""
        if not self.is_monitoring:
            return {'status': 'not_monitoring'}
        
        summary = {
            'status': 'monitoring',
            'n_predictions': len(self.prediction_history),
            'recent_coverage': np.mean(list(self.coverage_history)[-100:]) if self.coverage_history else None,
            'recent_interval_width': np.mean(list(self.interval_width_history)[-100:]) if self.interval_width_history else None,
            'recent_calibration_error': np.mean(list(self.calibration_error_history)[-100:]) if self.calibration_error_history else None,
            'n_alerts': len(self.alerts),
            'recent_alerts': self.alerts[-5:] if self.alerts else []
        }
        
        return summary
    
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by severity"""
        if severity is None:
            return self.alerts
        else:
            return [alert for alert in self.alerts if alert['severity'] == severity]
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []
        logger.info("Cleared all calibration alerts")
