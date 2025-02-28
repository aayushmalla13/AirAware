"""Ensemble of baseline forecasters for robust PM₂.₅ nowcasting."""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .seasonal_naive import SeasonalNaiveForecaster, SeasonalNaiveConfig
from .prophet_baseline import ProphetBaseline, ProphetConfig  
from .arima_baseline import ARIMABaseline, ARIMAConfig

logger = logging.getLogger(__name__)


class EnsembleConfig(BaseModel):
    """Configuration for baseline ensemble."""
    include_seasonal_naive: bool = Field(True, description="Include seasonal naive model")
    include_prophet: bool = Field(True, description="Include Prophet model")
    include_arima: bool = Field(True, description="Include ARIMA model")
    
    # Individual model configs
    seasonal_naive_config: SeasonalNaiveConfig = Field(default_factory=SeasonalNaiveConfig)
    prophet_config: ProphetConfig = Field(default_factory=ProphetConfig)
    arima_config: ARIMAConfig = Field(default_factory=ARIMAConfig)
    
    # Ensemble combination
    combination_method: str = Field("equal_weight", description="Combination method: equal_weight, performance_weight")
    validation_period_hours: int = Field(168, description="Hours to use for performance weighting")


class EnsembleForecast(BaseModel):
    """Ensemble forecast result."""
    horizon_hours: int
    predictions: List[float]
    timestamps: List[str]
    confidence_intervals: Dict[str, List[float]]
    individual_forecasts: Dict[str, List[float]]
    model_weights: Dict[str, float]
    forecast_metadata: Dict = Field(default_factory=dict)


class BaselineEnsemble:
    """
    Ensemble of baseline forecasters for robust PM₂.₅ predictions.
    
    Combines multiple baseline methods:
    - Seasonal-naive (strong for periodic patterns)
    - Prophet (trend + seasonality + optional regressors)
    - ARIMA (autoregressive dependencies)
    
    Uses performance-based weighting on validation data.
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.is_fitted = False
        self.models = {}
        self.model_weights = {}
        
        # Initialize individual models
        if self.config.include_seasonal_naive:
            self.models['seasonal_naive'] = SeasonalNaiveForecaster(self.config.seasonal_naive_config)
        
        if self.config.include_prophet:
            self.models['prophet'] = ProphetBaseline(self.config.prophet_config)
        
        if self.config.include_arima:
            self.models['arima'] = ARIMABaseline(self.config.arima_config)
        
        logger.info(f"BaselineEnsemble initialized with {len(self.models)} models")
    
    def fit(self, df: pd.DataFrame, target_col: str = 'pm25',
            group_col: Optional[str] = 'station_id') -> 'BaselineEnsemble':
        """
        Fit all baseline models in the ensemble.
        
        Args:
            df: Training data with datetime_utc, target, and optional features
            target_col: Name of target variable column
            group_col: Name of grouping column (e.g., station_id)
        """
        logger.info(f"Fitting baseline ensemble on {len(df):,} records")
        
        self.target_col = target_col
        self.group_col = group_col
        
        # Fit each model
        fitted_models = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Fitting {model_name} model...")
                model.fit(df, target_col, group_col)
                fitted_models[model_name] = model
                logger.info(f"✅ {model_name} fitted successfully")
                
            except Exception as e:
                logger.warning(f"❌ Failed to fit {model_name}: {e}")
        
        self.models = fitted_models
        
        # Calculate model weights based on validation performance
        if self.config.combination_method == "performance_weight":
            self.model_weights = self._calculate_performance_weights(df)
        else:
            # Equal weights
            n_models = len(self.models)
            self.model_weights = {name: 1.0/n_models for name in self.models.keys()}
        
        self.is_fitted = True
        logger.info(f"Baseline ensemble fitted with {len(self.models)} models")
        logger.info(f"Model weights: {self.model_weights}")
        
        return self
    
    def predict(self, timestamps: Union[List, pd.DatetimeIndex],
                station_id: Optional[int] = None,
                horizon_hours: Optional[int] = None,
                meteo_data: Optional[pd.DataFrame] = None) -> EnsembleForecast:
        """
        Generate ensemble predictions for given timestamps.
        
        Args:
            timestamps: Target prediction timestamps
            station_id: Station ID for grouped predictions
            horizon_hours: Forecast horizon for metadata
            meteo_data: Meteorological data for Prophet model
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        if isinstance(timestamps, list):
            timestamps = pd.to_datetime(timestamps)
        elif not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex([timestamps])
        
        # Get predictions from each model
        individual_forecasts = {}
        individual_confidence = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'prophet':
                    # Prophet may use meteorological data
                    forecast = model.predict(timestamps, station_id, horizon_hours, meteo_data)
                else:
                    forecast = model.predict(timestamps, station_id, horizon_hours)
                
                individual_forecasts[model_name] = forecast.predictions
                individual_confidence[model_name] = forecast.confidence_intervals
                
            except Exception as e:
                logger.warning(f"Failed to get prediction from {model_name}: {e}")
        
        if not individual_forecasts:
            raise ValueError("No models were able to generate predictions")
        
        # Combine predictions using weights
        ensemble_predictions = self._combine_predictions(individual_forecasts)
        
        # Combine confidence intervals
        ensemble_confidence = self._combine_confidence_intervals(individual_confidence)
        
        forecast_metadata = {
            'ensemble_method': self.config.combination_method,
            'models_used': list(individual_forecasts.keys()),
            'model_weights': self.model_weights,
            'station_id': station_id
        }
        
        return EnsembleForecast(
            horizon_hours=horizon_hours or len(timestamps),
            predictions=ensemble_predictions,
            timestamps=[ts.isoformat() for ts in timestamps],
            confidence_intervals=ensemble_confidence,
            individual_forecasts=individual_forecasts,
            model_weights=self.model_weights,
            forecast_metadata=forecast_metadata
        )
    
    def forecast(self, start_time: pd.Timestamp, horizon_hours: int,
                 station_id: Optional[int] = None,
                 meteo_data: Optional[pd.DataFrame] = None) -> EnsembleForecast:
        """
        Generate multi-step ahead ensemble forecasts.
        
        Args:
            start_time: Start time for forecast
            horizon_hours: Number of hours to forecast ahead
            station_id: Station ID for grouped forecasts
            meteo_data: Meteorological data for forecast period
        """
        timestamps = pd.date_range(
            start=start_time,
            periods=horizon_hours,
            freq='h'
        )
        
        return self.predict(timestamps, station_id, horizon_hours, meteo_data)
    
    def _calculate_performance_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate model weights based on validation performance."""
        
        logger.info("Calculating performance-based weights...")
        
        # Use the last portion of training data for validation
        val_hours = min(self.config.validation_period_hours, len(df) // 4)
        
        if val_hours < 24:
            logger.warning("Insufficient data for performance weighting, using equal weights")
            n_models = len(self.models)
            return {name: 1.0/n_models for name in self.models.keys()}
        
        # Split data
        split_idx = len(df) - val_hours
        train_data = df.iloc[:split_idx].copy()
        val_data = df.iloc[split_idx:].copy()
        
        # Temporarily fit models on training portion
        temp_models = {}
        for model_name, model in self.models.items():
            try:
                temp_model = model.__class__(model.config)
                temp_model.fit(train_data, self.target_col, self.group_col)
                temp_models[model_name] = temp_model
            except Exception as e:
                logger.warning(f"Failed to fit {model_name} for weight calculation: {e}")
        
        # Calculate validation errors
        model_errors = {}
        
        for model_name, model in temp_models.items():
            try:
                # Get validation predictions
                val_timestamps = val_data['datetime_utc'].tolist()
                
                if self.group_col:
                    # Group-wise predictions
                    all_predictions = []
                    all_actuals = []
                    
                    for group in val_data[self.group_col].unique():
                        group_data = val_data[val_data[self.group_col] == group]
                        group_timestamps = group_data['datetime_utc'].tolist()
                        
                        if len(group_timestamps) > 0:
                            forecast = model.predict(group_timestamps, group)
                            all_predictions.extend(forecast.predictions)
                            all_actuals.extend(group_data[self.target_col].tolist())
                else:
                    forecast = model.predict(val_timestamps)
                    all_predictions = forecast.predictions
                    all_actuals = val_data[self.target_col].tolist()
                
                # Calculate MAE
                predictions = np.array(all_predictions)
                actuals = np.array(all_actuals)
                
                mask = ~(np.isnan(predictions) | np.isnan(actuals))
                if np.sum(mask) > 0:
                    mae = np.mean(np.abs(predictions[mask] - actuals[mask]))
                    model_errors[model_name] = mae
                
            except Exception as e:
                logger.warning(f"Failed to calculate validation error for {model_name}: {e}")
        
        if not model_errors:
            # Fallback to equal weights
            n_models = len(self.models)
            return {name: 1.0/n_models for name in self.models.keys()}
        
        # Convert errors to weights (inverse of error)
        # Add small epsilon to avoid division by zero
        epsilon = 0.001
        inverse_errors = {name: 1.0 / (error + epsilon) for name, error in model_errors.items()}
        
        # Normalize to sum to 1
        total_inverse = sum(inverse_errors.values())
        weights = {name: inv_error / total_inverse for name, inv_error in inverse_errors.items()}
        
        logger.info(f"Performance-based weights calculated: {weights}")
        return weights
    
    def _combine_predictions(self, individual_forecasts: Dict[str, List[float]]) -> List[float]:
        """Combine individual model predictions using weights."""
        
        if not individual_forecasts:
            return []
        
        # Get prediction length
        n_predictions = len(list(individual_forecasts.values())[0])
        
        ensemble_predictions = []
        
        for i in range(n_predictions):
            weighted_sum = 0.0
            total_weight = 0.0
            
            for model_name, predictions in individual_forecasts.items():
                if i < len(predictions) and not np.isnan(predictions[i]):
                    weight = self.model_weights.get(model_name, 0.0)
                    weighted_sum += weight * predictions[i]
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_predictions.append(weighted_sum / total_weight)
            else:
                # Fallback to simple average
                valid_preds = [pred[i] for pred in individual_forecasts.values() 
                             if i < len(pred) and not np.isnan(pred[i])]
                if valid_preds:
                    ensemble_predictions.append(np.mean(valid_preds))
                else:
                    ensemble_predictions.append(25.0)  # Default PM2.5 value
        
        return ensemble_predictions
    
    def _combine_confidence_intervals(self, individual_confidence: Dict[str, Dict]) -> Dict[str, List[float]]:
        """Combine confidence intervals from individual models."""
        
        if not individual_confidence:
            return {}
        
        # Get all quantile levels
        all_quantiles = set()
        for conf_dict in individual_confidence.values():
            all_quantiles.update(conf_dict.keys())
        
        ensemble_confidence = {}
        
        for quantile in all_quantiles:
            quantile_predictions = {}
            
            # Collect predictions for this quantile from all models
            for model_name, conf_dict in individual_confidence.items():
                if quantile in conf_dict:
                    quantile_predictions[model_name] = conf_dict[quantile]
            
            # Combine using same weights as point predictions
            if quantile_predictions:
                ensemble_confidence[quantile] = self._combine_predictions(quantile_predictions)
        
        return ensemble_confidence
    
    def get_model_info(self) -> Dict:
        """Get information about the fitted ensemble."""
        
        if not self.is_fitted:
            return {"fitted": False}
        
        info = {
            "fitted": True,
            "ensemble_type": "baseline_ensemble",
            "combination_method": self.config.combination_method,
            "models": list(self.models.keys()),
            "model_weights": self.model_weights,
            "individual_model_info": {}
        }
        
        # Get info from individual models
        for model_name, model in self.models.items():
            try:
                info["individual_model_info"][model_name] = model.get_model_info()
            except Exception as e:
                info["individual_model_info"][model_name] = {"error": str(e)}
        
        return info


