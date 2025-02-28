"""
Probabilistic Forecasting with Uncertainty Quantification

This module enhances baseline models with proper uncertainty quantification,
including conformal prediction, quantile regression, and ensemble uncertainty.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import warnings
from abc import ABC, abstractmethod

# Statistical imports
try:
    from scipy import stats
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import QuantileRegressor
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("Scipy/sklearn not available for probabilistic forecasting")

logger = logging.getLogger(__name__)

@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification."""
    
    # Conformal prediction
    conformal_alpha: float = 0.1  # 90% prediction intervals
    conformal_method: str = "naive"  # naive, jackknife, cv+
    
    # Quantile regression
    quantiles: List[float] = None
    quantile_method: str = "linear"  # linear, forest
    
    # Ensemble uncertainty
    ensemble_method: str = "bootstrap"  # bootstrap, dropout, bayesian
    n_bootstrap_samples: int = 100
    
    # Calibration
    calibration_method: str = "isotonic"  # isotonic, platt, none
    calibration_data_size: int = 1000
    
    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

class ProbabilisticForecast:
    """Probabilistic forecast with uncertainty quantification."""
    
    def __init__(self, 
                 predictions: List[float],
                 timestamps: List[datetime],
                 quantiles: Dict[float, List[float]],
                 confidence_intervals: Dict[str, List[float]],
                 uncertainty_method: str = "conformal",
                 calibration_score: Optional[float] = None):
        self.predictions = predictions
        self.timestamps = timestamps
        self.quantiles = quantiles
        self.confidence_intervals = confidence_intervals
        self.uncertainty_method = uncertainty_method
        self.calibration_score = calibration_score
        
    def get_prediction_interval(self, level: float = 0.9) -> Tuple[List[float], List[float]]:
        """Get prediction interval for given confidence level."""
        alpha = 1 - level
        lower_key = f"q{alpha/2:.3f}"
        upper_key = f"q{1-alpha/2:.3f}"
        
        if lower_key in self.quantiles and upper_key in self.quantiles:
            return self.quantiles[lower_key], self.quantiles[upper_key]
        elif f"q{alpha/2}" in self.quantiles and f"q{1-alpha/2}" in self.quantiles:
            return self.quantiles[f"q{alpha/2}"], self.quantiles[f"q{1-alpha/2}"]
        else:
            # Fallback to confidence intervals
            if f"q{alpha/2:.1f}" in self.confidence_intervals:
                return (self.confidence_intervals[f"q{alpha/2:.1f}"], 
                       self.confidence_intervals[f"q{1-alpha/2:.1f}"])
            else:
                # Default to 80% interval
                return (self.confidence_intervals.get("q0.1", self.predictions),
                       self.confidence_intervals.get("q0.9", self.predictions))

class ConformalPredictor:
    """Conformal prediction for uncertainty quantification."""
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.calibration_scores = None
        self.quantile_threshold = None
        
    def fit(self, residuals: List[float]):
        """Fit conformal predictor using calibration residuals."""
        residuals = np.array(residuals)
        
        if self.config.conformal_method == "naive":
            # Naive conformal prediction
            self.quantile_threshold = np.quantile(
                np.abs(residuals), 
                1 - self.config.conformal_alpha
            )
        elif self.config.conformal_method == "jackknife":
            # Jackknife conformal prediction
            n = len(residuals)
            scores = []
            for i in range(n):
                leave_one_out_residuals = np.concatenate([residuals[:i], residuals[i+1:]])
                scores.append(np.abs(residuals[i]) - np.mean(np.abs(leave_one_out_residuals)))
            self.quantile_threshold = np.quantile(scores, 1 - self.config.conformal_alpha)
        
        logger.info(f"Conformal predictor fitted with threshold: {self.quantile_threshold:.3f}")
    
    def predict_intervals(self, 
                        predictions: List[float],
                        historical_std: Optional[float] = None) -> Tuple[List[float], List[float]]:
        """Generate prediction intervals using conformal prediction."""
        if self.quantile_threshold is None:
            logger.warning("Conformal predictor not fitted, using default intervals")
            default_width = historical_std or np.std(predictions) * 2
            lower = [p - default_width for p in predictions]
            upper = [p + default_width for p in predictions]
            return lower, upper
        
        # Use conformal threshold
        width = self.quantile_threshold
        lower = [p - width for p in predictions]
        upper = [p + width for p in predictions]
        
        return lower, upper

class QuantileRegressor:
    """Quantile regression for probabilistic forecasting."""
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.models = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit quantile regression models for different quantiles."""
        if not SCIPY_AVAILABLE:
            logger.warning("Scipy not available, using simple quantile estimation")
            return
        
        for quantile in self.config.quantiles:
            try:
                if self.config.quantile_method == "linear":
                    # Use sklearn's QuantileRegressor with correct parameter name
                    model = QuantileRegressor(quantile=quantile, alpha=0.1)
                elif self.config.quantile_method == "forest":
                    model = RandomForestRegressor(
                        n_estimators=50,
                        random_state=42,
                        max_depth=5
                    )
                else:
                    continue
                
                model.fit(X, y)
                self.models[quantile] = model
                logger.debug(f"Fitted quantile model for {quantile}")
                
            except Exception as e:
                logger.warning(f"Failed to fit quantile model for {quantile}: {e}")
    
    def predict_quantiles(self, X: np.ndarray) -> Dict[float, List[float]]:
        """Predict quantiles for given features."""
        quantile_predictions = {}
        
        for quantile, model in self.models.items():
            try:
                predictions = model.predict(X)
                quantile_predictions[quantile] = predictions.tolist()
            except Exception as e:
                logger.warning(f"Failed to predict quantile {quantile}: {e}")
                # Fallback to simple quantile estimation
                quantile_predictions[quantile] = [np.quantile(X.mean(axis=1), quantile)] * len(X)
        
        return quantile_predictions

class EnsembleUncertainty:
    """Ensemble-based uncertainty quantification."""
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.base_models = []
        
    def create_bootstrap_samples(self, 
                               data: pd.DataFrame,
                               n_samples: int) -> List[pd.DataFrame]:
        """Create bootstrap samples for ensemble uncertainty."""
        bootstrap_samples = []
        
        for _ in range(n_samples):
            # Bootstrap sampling with replacement
            sample_indices = np.random.choice(
                len(data), 
                size=len(data), 
                replace=True
            )
            bootstrap_samples.append(data.iloc[sample_indices])
        
        return bootstrap_samples
    
    def fit_ensemble(self, 
                    data: pd.DataFrame,
                    model_class: Any,
                    target_col: str = 'pm25',
                    group_col: str = 'station_id') -> List[Any]:
        """Fit ensemble of models using bootstrap sampling."""
        if self.config.ensemble_method == "bootstrap":
            bootstrap_samples = self.create_bootstrap_samples(
                data, 
                self.config.n_bootstrap_samples
            )
            
            models = []
            for i, sample in enumerate(bootstrap_samples):
                try:
                    model = model_class()
                    model.fit(sample, target_col=target_col, group_col=group_col)
                    models.append(model)
                    if i % 10 == 0:
                        logger.info(f"Fitted bootstrap model {i+1}/{len(bootstrap_samples)}")
                except Exception as e:
                    logger.warning(f"Failed to fit bootstrap model {i}: {e}")
            
            self.base_models = models
            logger.info(f"Fitted {len(models)} bootstrap models")
        
        return self.base_models
    
    def predict_ensemble(self, 
                       timestamps: List[datetime],
                       station_id: Optional[int] = None) -> Dict[str, List[float]]:
        """Generate ensemble predictions with uncertainty."""
        if not self.base_models:
            logger.warning("No ensemble models available")
            return {}
        
        all_predictions = []
        
        for model in self.base_models:
            try:
                forecast = model.predict(timestamps, station_id=station_id)
                all_predictions.append(forecast.predictions)
            except Exception as e:
                logger.warning(f"Ensemble model prediction failed: {e}")
        
        if not all_predictions:
            logger.warning("No successful ensemble predictions")
            return {}
        
        # Convert to numpy array for easier computation
        predictions_array = np.array(all_predictions)
        
        # Calculate ensemble statistics
        ensemble_mean = np.mean(predictions_array, axis=0)
        ensemble_std = np.std(predictions_array, axis=0)
        
        # Calculate quantiles
        quantiles = {}
        for quantile in self.config.quantiles:
            quantiles[quantile] = np.quantile(predictions_array, quantile, axis=0).tolist()
        
        # Calculate confidence intervals
        confidence_intervals = {
            'q0.1': (ensemble_mean - 1.28 * ensemble_std).tolist(),
            'q0.5': ensemble_mean.tolist(),
            'q0.9': (ensemble_mean + 1.28 * ensemble_std).tolist()
        }
        
        return {
            'predictions': ensemble_mean.tolist(),
            'quantiles': quantiles,
            'confidence_intervals': confidence_intervals,
            'uncertainty': ensemble_std.tolist()
        }

class ProbabilisticBaselineWrapper:
    """Wrapper to add probabilistic forecasting to baseline models."""
    
    def __init__(self, 
                 base_model: Any,
                 config: Optional[UncertaintyConfig] = None):
        self.base_model = base_model
        self.config = config or UncertaintyConfig()
        self.conformal_predictor = ConformalPredictor(self.config)
        self.quantile_regressor = QuantileRegressor(self.config)
        self.ensemble_uncertainty = EnsembleUncertainty(self.config)
        self.is_fitted = False
        
    def fit(self, 
           data: pd.DataFrame,
           target_col: str = 'pm25',
           group_col: str = 'station_id') -> 'ProbabilisticBaselineWrapper':
        """Fit the base model and uncertainty quantification components."""
        
        # Fit base model
        self.base_model.fit(data, target_col=target_col, group_col=group_col)
        
        # Generate predictions for calibration
        train_predictions = []
        train_actuals = []
        
        # Sample data for calibration (to avoid overfitting)
        calibration_data = data.sample(
            min(len(data), self.config.calibration_data_size),
            random_state=42
        )
        
        for _, row in calibration_data.iterrows():
            try:
                # Generate prediction for this timestamp
                pred = self.base_model.predict([row['datetime_utc']], station_id=row[group_col])
                if pred and len(pred.predictions) > 0:
                    train_predictions.append(pred.predictions[0])
                    train_actuals.append(row[target_col])
            except Exception as e:
                logger.debug(f"Calibration prediction failed: {e}")
        
        if len(train_predictions) > 10:  # Need sufficient data for calibration
            # Calculate residuals for conformal prediction
            residuals = [actual - pred for actual, pred in zip(train_actuals, train_predictions)]
            self.conformal_predictor.fit(residuals)
            
            # Fit quantile regressor if we have features
            feature_cols = [col for col in data.columns 
                          if col not in [target_col, group_col, 'datetime_utc']]
            if feature_cols:
                X = calibration_data[feature_cols].values
                y = np.array(train_actuals)
                self.quantile_regressor.fit(X, y)
        
        self.is_fitted = True
        logger.info("Probabilistic baseline wrapper fitted successfully")
        return self
    
    def predict_probabilistic(self, 
                            timestamps: List[datetime],
                            station_id: Optional[int] = None,
                            features: Optional[pd.DataFrame] = None) -> ProbabilisticForecast:
        """Generate probabilistic forecast with uncertainty quantification."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get base predictions
        base_forecast = self.base_model.predict(timestamps, station_id=station_id)
        base_predictions = base_forecast.predictions
        
        # Generate uncertainty quantification
        quantiles = {}
        confidence_intervals = {}
        
        # Method 1: Conformal prediction
        if hasattr(base_forecast, 'confidence_intervals') and base_forecast.confidence_intervals:
            # Use existing confidence intervals
            confidence_intervals = base_forecast.confidence_intervals
        else:
            # Generate conformal prediction intervals
            historical_std = getattr(base_forecast, 'forecast_metadata', {}).get('residual_std', None)
            lower, upper = self.conformal_predictor.predict_intervals(
                base_predictions, 
                historical_std
            )
            confidence_intervals = {
                'q0.1': lower,
                'q0.5': base_predictions,
                'q0.9': upper
            }
        
        # Method 2: Quantile regression (if features available)
        if features is not None and self.quantile_regressor.models:
            feature_cols = [col for col in features.columns 
                          if col not in ['datetime_utc', 'station_id']]
            if feature_cols:
                X = features[feature_cols].values
                quantile_predictions = self.quantile_regressor.predict_quantiles(X)
                quantiles.update(quantile_predictions)
        
        # Method 3: Simple quantile estimation from base predictions
        if not quantiles:
            for quantile in self.config.quantiles:
                quantile_value = np.quantile(base_predictions, quantile)
                quantiles[quantile] = [quantile_value] * len(base_predictions)
        
        # Create probabilistic forecast
        return ProbabilisticForecast(
            predictions=base_predictions,
            timestamps=timestamps,
            quantiles=quantiles,
            confidence_intervals=confidence_intervals,
            uncertainty_method="conformal",
            calibration_score=None
        )
    
    def predict(self, 
               timestamps: List[datetime],
               station_id: Optional[int] = None) -> Any:
        """Standard prediction interface (returns base forecast)."""
        return self.base_model.predict(timestamps, station_id=station_id)
