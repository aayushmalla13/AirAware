"""
Dynamic Ensemble Weighting for AirAware Baseline Models

This module implements dynamic ensemble weighting that adapts model weights
based on recent performance, providing more robust and adaptive forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import warnings
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class EnsembleWeightingConfig:
    """Configuration for dynamic ensemble weighting."""
    
    # Weighting methods
    weighting_method: str = "performance_based"  # performance_based, equal, adaptive
    
    # Performance window
    performance_window_hours: int = 168  # 1 week
    min_performance_samples: int = 24  # Minimum samples for reliable performance
    
    # Weight adaptation
    adaptation_rate: float = 0.1  # How quickly weights adapt (0-1)
    min_weight: float = 0.05  # Minimum weight for any model
    max_weight: float = 0.8  # Maximum weight for any model
    
    # Performance metrics
    primary_metric: str = "mae"  # mae, rmse, smape
    secondary_metrics: List[str] = None
    
    # Weight smoothing
    weight_smoothing: bool = True
    smoothing_factor: float = 0.3  # Exponential smoothing factor
    
    # Fallback behavior
    fallback_to_equal: bool = True  # Fall back to equal weights if insufficient data
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = ["rmse", "smape"]

class ModelPerformanceTracker:
    """Tracks model performance over time for dynamic weighting."""
    
    def __init__(self, config: EnsembleWeightingConfig):
        self.config = config
        self.performance_history = {}  # model_name -> [(timestamp, metrics), ...]
        self.current_weights = {}
        
    def update_performance(self, 
                          model_name: str,
                          timestamp: datetime,
                          metrics: Dict[str, float]):
        """Update performance history for a model."""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append((timestamp, metrics))
        
        # Keep only recent performance data
        cutoff_time = timestamp - pd.Timedelta(hours=self.config.performance_window_hours)
        self.performance_history[model_name] = [
            (ts, m) for ts, m in self.performance_history[model_name] 
            if ts >= cutoff_time
        ]
        
        logger.debug(f"Updated performance for {model_name}: {metrics}")
    
    def get_recent_performance(self, 
                             model_name: str,
                             timestamp: datetime) -> Optional[Dict[str, float]]:
        """Get recent performance metrics for a model."""
        if model_name not in self.performance_history:
            return None
        
        # Filter to recent performance
        cutoff_time = timestamp - pd.Timedelta(hours=self.config.performance_window_hours)
        recent_performance = [
            (ts, m) for ts, m in self.performance_history[model_name] 
            if ts >= cutoff_time
        ]
        
        if len(recent_performance) < self.config.min_performance_samples:
            return None
        
        # Calculate average performance
        metrics = {}
        for metric_name in [self.config.primary_metric] + self.config.secondary_metrics:
            values = [m[1].get(metric_name, float('inf')) for m in recent_performance]
            metrics[metric_name] = np.mean(values)
        
        return metrics
    
    def calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate a single performance score from multiple metrics."""
        primary_score = metrics.get(self.config.primary_metric, float('inf'))
        
        # Lower is better for error metrics
        if primary_score == float('inf'):
            return 0.0
        
        # Convert to performance score (higher is better)
        # Use inverse of error as performance score
        performance_score = 1.0 / (1.0 + primary_score)
        
        return performance_score

class DynamicEnsembleWeighter:
    """Dynamic ensemble weighting based on recent performance."""
    
    def __init__(self, config: EnsembleWeightingConfig):
        self.config = config
        self.performance_tracker = ModelPerformanceTracker(config)
        self.model_weights = {}
        self.weight_history = []
        
    def update_weights(self, 
                     model_names: List[str],
                     timestamp: datetime,
                     performance_data: Optional[Dict[str, Dict[str, float]]] = None):
        """Update ensemble weights based on recent performance."""
        
        if self.config.weighting_method == "equal":
            # Equal weights
            equal_weight = 1.0 / len(model_names)
            self.model_weights = {name: equal_weight for name in model_names}
            logger.info("Using equal weights for all models")
            return
        
        elif self.config.weighting_method == "performance_based":
            self._update_performance_based_weights(model_names, timestamp, performance_data)
        
        elif self.config.weighting_method == "adaptive":
            self._update_adaptive_weights(model_names, timestamp, performance_data)
        
        # Apply weight smoothing if enabled
        if self.config.weight_smoothing and self.weight_history:
            self._apply_weight_smoothing()
        
        # Store weight history
        self.weight_history.append({
            'timestamp': timestamp,
            'weights': self.model_weights.copy()
        })
        
        logger.info(f"Updated ensemble weights: {self.model_weights}")
    
    def _update_performance_based_weights(self, 
                                        model_names: List[str],
                                        timestamp: datetime,
                                        performance_data: Optional[Dict[str, Dict[str, float]]]):
        """Update weights based on recent performance."""
        
        performance_scores = {}
        valid_models = []
        
        for model_name in model_names:
            # Get recent performance
            if performance_data and model_name in performance_data:
                metrics = performance_data[model_name]
            else:
                metrics = self.performance_tracker.get_recent_performance(model_name, timestamp)
            
            if metrics is not None:
                score = self.performance_tracker.calculate_performance_score(metrics)
                performance_scores[model_name] = score
                valid_models.append(model_name)
                logger.debug(f"{model_name} performance score: {score:.3f}")
            else:
                logger.warning(f"Insufficient performance data for {model_name}")
        
        if not valid_models:
            logger.warning("No models with sufficient performance data, using equal weights")
            equal_weight = 1.0 / len(model_names)
            self.model_weights = {name: equal_weight for name in model_names}
            return
        
        # Calculate weights based on performance scores
        total_score = sum(performance_scores.values())
        
        if total_score == 0:
            # Fallback to equal weights
            equal_weight = 1.0 / len(model_names)
            self.model_weights = {name: equal_weight for name in model_names}
            return
        
        # Calculate raw weights
        raw_weights = {}
        for model_name in model_names:
            if model_name in performance_scores:
                raw_weight = performance_scores[model_name] / total_score
            else:
                raw_weight = self.config.min_weight  # Minimum weight for models without data
            
            raw_weights[model_name] = raw_weight
        
        # Apply weight constraints
        self.model_weights = self._apply_weight_constraints(raw_weights)
    
    def _update_adaptive_weights(self, 
                               model_names: List[str],
                               timestamp: datetime,
                               performance_data: Optional[Dict[str, Dict[str, float]]]):
        """Update weights using adaptive learning."""
        
        if not self.weight_history:
            # Initialize with equal weights
            equal_weight = 1.0 / len(model_names)
            self.model_weights = {name: equal_weight for name in model_names}
            return
        
        # Get previous weights
        previous_weights = self.weight_history[-1]['weights']
        
        # Calculate performance-based adjustments
        performance_adjustments = {}
        
        for model_name in model_names:
            if performance_data and model_name in performance_data:
                metrics = performance_data[model_name]
                score = self.performance_tracker.calculate_performance_score(metrics)
                
                # Calculate adjustment based on performance
                if model_name in previous_weights:
                    current_weight = previous_weights[model_name]
                    # Increase weight for better performance
                    adjustment = self.config.adaptation_rate * (score - 0.5)
                    performance_adjustments[model_name] = adjustment
                else:
                    performance_adjustments[model_name] = 0.0
            else:
                performance_adjustments[model_name] = 0.0
        
        # Apply adaptive updates
        new_weights = {}
        for model_name in model_names:
            if model_name in previous_weights:
                old_weight = previous_weights[model_name]
                adjustment = performance_adjustments.get(model_name, 0.0)
                new_weight = old_weight + adjustment
            else:
                new_weight = 1.0 / len(model_names)  # Equal weight for new models
            
            new_weights[model_name] = new_weight
        
        # Apply weight constraints
        self.model_weights = self._apply_weight_constraints(new_weights)
    
    def _apply_weight_constraints(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply minimum and maximum weight constraints."""
        
        # Normalize weights
        total_weight = sum(raw_weights.values())
        if total_weight == 0:
            equal_weight = 1.0 / len(raw_weights)
            return {name: equal_weight for name in raw_weights.keys()}
        
        # Apply constraints
        constrained_weights = {}
        for model_name, weight in raw_weights.items():
            constrained_weight = max(self.config.min_weight, 
                                   min(self.config.max_weight, weight))
            constrained_weights[model_name] = constrained_weight
        
        # Renormalize to ensure weights sum to 1
        total_constrained = sum(constrained_weights.values())
        if total_constrained > 0:
            for model_name in constrained_weights:
                constrained_weights[model_name] /= total_constrained
        
        return constrained_weights
    
    def _apply_weight_smoothing(self):
        """Apply exponential smoothing to weights."""
        if len(self.weight_history) < 2:
            return
        
        current_weights = self.model_weights.copy()
        previous_weights = self.weight_history[-1]['weights']
        
        smoothed_weights = {}
        for model_name in current_weights:
            if model_name in previous_weights:
                smoothed_weight = (self.config.smoothing_factor * current_weights[model_name] + 
                                 (1 - self.config.smoothing_factor) * previous_weights[model_name])
            else:
                smoothed_weight = current_weights[model_name]
            
            smoothed_weights[model_name] = smoothed_weight
        
        # Renormalize
        total_smoothed = sum(smoothed_weights.values())
        if total_smoothed > 0:
            for model_name in smoothed_weights:
                smoothed_weights[model_name] /= total_smoothed
        
        self.model_weights = smoothed_weights
    
    def get_weights(self) -> Dict[str, float]:
        """Get current ensemble weights."""
        return self.model_weights.copy()
    
    def get_weight_history(self) -> List[Dict[str, Any]]:
        """Get weight history for analysis."""
        return self.weight_history.copy()

class DynamicEnsemble:
    """Dynamic ensemble with adaptive weighting."""
    
    def __init__(self, 
                 models: Dict[str, Any],
                 config: Optional[EnsembleWeightingConfig] = None):
        self.models = models
        self.config = config or EnsembleWeightingConfig()
        self.weighter = DynamicEnsembleWeighter(self.config)
        self.is_fitted = False
        
    def fit(self, 
           data: pd.DataFrame,
           target_col: str = 'pm25',
           group_col: str = 'station_id') -> 'DynamicEnsemble':
        """Fit all ensemble models."""
        
        logger.info(f"Fitting {len(self.models)} models for dynamic ensemble")
        
        # Create a copy of model names to avoid dictionary modification during iteration
        model_names = list(self.models.keys())
        failed_models = []
        
        for model_name in model_names:
            try:
                model_class = self.models[model_name]
                model = model_class()
                model.fit(data, target_col=target_col, group_col=group_col)
                self.models[model_name] = model
                logger.info(f"✅ Fitted {model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to fit {model_name}: {e}")
                failed_models.append(model_name)
        
        # Remove failed models
        for model_name in failed_models:
            del self.models[model_name]
        
        if not self.models:
            raise ValueError("No models could be fitted successfully")
        
        # Initialize with equal weights
        model_names = list(self.models.keys())
        self.weighter.update_weights(model_names, datetime.now())
        
        self.is_fitted = True
        logger.info(f"Dynamic ensemble fitted with {len(self.models)} models")
        return self
    
    def predict(self, 
               timestamps: List[datetime],
               station_id: Optional[int] = None,
               update_weights: bool = True,
               performance_data: Optional[Dict[str, Dict[str, float]]] = None) -> Any:
        """Generate ensemble prediction with dynamic weighting."""
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get individual model predictions
        model_predictions = {}
        model_forecasts = {}
        
        for model_name, model in self.models.items():
            try:
                forecast = model.predict(timestamps, station_id=station_id)
                model_forecasts[model_name] = forecast
                model_predictions[model_name] = forecast.predictions
                logger.debug(f"Generated predictions for {model_name}")
            except Exception as e:
                logger.warning(f"Failed to generate predictions for {model_name}: {e}")
        
        if not model_predictions:
            raise ValueError("No models could generate predictions")
        
        # Update weights if requested
        if update_weights:
            model_names = list(model_predictions.keys())
            self.weighter.update_weights(
                model_names, 
                datetime.now(), 
                performance_data
            )
        
        # Get current weights
        weights = self.weighter.get_weights()
        
        # Calculate weighted ensemble prediction
        ensemble_predictions = []
        for i in range(len(timestamps)):
            weighted_prediction = 0.0
            total_weight = 0.0
            
            for model_name, predictions in model_predictions.items():
                if i < len(predictions):
                    weight = weights.get(model_name, 0.0)
                    weighted_prediction += weight * predictions[i]
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_predictions.append(weighted_prediction / total_weight)
            else:
                # Fallback to simple average
                valid_predictions = [pred[i] for pred in model_predictions.values() if i < len(pred)]
                ensemble_predictions.append(np.mean(valid_predictions) if valid_predictions else 0.0)
        
        # Create ensemble forecast object
        from .baseline_ensemble import EnsembleForecast
        
        # Convert timestamps to strings for Pydantic validation
        timestamp_strings = [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in timestamps]
        
        # Calculate horizon hours
        horizon_hours = len(timestamps)
        
        # Create confidence intervals (simple approximation)
        pred_std = np.std(ensemble_predictions) if len(ensemble_predictions) > 1 else 1.0
        confidence_intervals = {
            'lower': [max(0, pred - 1.96 * pred_std) for pred in ensemble_predictions],
            'upper': [pred + 1.96 * pred_std for pred in ensemble_predictions]
        }
        
        # Extract individual predictions for EnsembleForecast
        individual_predictions = {}
        for model_name, forecast in model_forecasts.items():
            individual_predictions[model_name] = forecast.predictions
        
        ensemble_forecast = EnsembleForecast(
            predictions=ensemble_predictions,
            timestamps=timestamp_strings,
            horizon_hours=horizon_hours,
            confidence_intervals=confidence_intervals,
            model_weights=weights,
            individual_forecasts=individual_predictions,
            forecast_metadata={"ensemble_method": "dynamic_weighted"}
        )
        
        logger.info(f"Generated dynamic ensemble forecast with weights: {weights}")
        return ensemble_forecast
    
    def update_performance(self, 
                          model_name: str,
                          timestamp: datetime,
                          metrics: Dict[str, float]):
        """Update performance for a specific model."""
        self.weighter.performance_tracker.update_performance(model_name, timestamp, metrics)
    
    def get_weight_history(self) -> List[Dict[str, Any]]:
        """Get weight evolution history."""
        return self.weighter.get_weight_history()
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current ensemble weights."""
        return self.weighter.get_weights()
