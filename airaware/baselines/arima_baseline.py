"""ARIMA baseline forecaster for PM₂.₅ nowcasting."""

import logging
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ARIMAConfig(BaseModel):
    """Configuration for ARIMA forecaster."""
    order: Tuple[int, int, int] = Field((2, 1, 2), description="ARIMA order (p, d, q)")
    seasonal_order: Optional[Tuple[int, int, int, int]] = Field(
        None, description="Seasonal ARIMA order (P, D, Q, s)"
    )
    auto_arima: bool = Field(True, description="Use auto ARIMA for order selection")
    max_p: int = Field(5, description="Maximum AR order for auto ARIMA")
    max_q: int = Field(5, description="Maximum MA order for auto ARIMA")
    max_d: int = Field(2, description="Maximum differencing for auto ARIMA")
    seasonal: bool = Field(True, description="Include seasonal components")
    seasonal_period: int = Field(24, description="Seasonal period (hours)")
    
    # Fitting parameters
    trend: Optional[str] = Field("n", description="Trend component: 'n', 'c', 't', 'ct'")
    method: str = Field("lbfgs", description="Optimization method")
    maxiter: int = Field(50, description="Maximum iterations")
    
    # Confidence intervals
    alpha: float = Field(0.05, description="Significance level for confidence intervals")


class ARIMAForecast(BaseModel):
    """ARIMA forecast result."""
    horizon_hours: int
    predictions: List[float]
    timestamps: List[str]
    confidence_intervals: Dict[str, List[float]]
    model_order: Tuple[int, int, int]
    seasonal_order: Optional[Tuple[int, int, int, int]] = None
    forecast_metadata: Dict = Field(default_factory=dict)


class ARIMABaseline:
    """
    ARIMA baseline forecaster for PM₂.₅ time series.
    
    Uses AutoRegressive Integrated Moving Average models to capture
    temporal dependencies and trends in air quality data.
    Supports both non-seasonal and seasonal ARIMA variants.
    """
    
    def __init__(self, config: Optional[ARIMAConfig] = None):
        self.config = config or ARIMAConfig()
        self.is_fitted = False
        self.models = {}  # One model per station
        self.scalers = {}  # For numerical stability
        self.training_stats = {}
        
        # Try to import statsmodels (with fallback)
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.seasonal import seasonal_decompose
            self.ARIMA = ARIMA
            self.seasonal_decompose = seasonal_decompose
            self.statsmodels_available = True
            
            if self.config.auto_arima:
                try:
                    from pmdarima import auto_arima
                    self.auto_arima = auto_arima
                    self.auto_arima_available = True
                except ImportError:
                    logger.warning("pmdarima not available - will use manual ARIMA order")
                    self.auto_arima_available = False
            else:
                self.auto_arima_available = False
                
        except ImportError:
            logger.warning("statsmodels not available - will use simplified AR model")
            self.ARIMA = None
            self.statsmodels_available = False
            self.auto_arima_available = False
        
        logger.info(f"ARIMABaseline initialized (statsmodels: {self.statsmodels_available}, auto_arima: {self.auto_arima_available})")
    
    def fit(self, df: pd.DataFrame, target_col: str = 'pm25', 
            group_col: Optional[str] = 'station_id') -> 'ARIMABaseline':
        """
        Fit ARIMA models.
        
        Args:
            df: Training data with datetime_utc and target column
            target_col: Name of target variable column
            group_col: Name of grouping column (e.g., station_id)
        """
        logger.info(f"Fitting ARIMA model on {len(df):,} records")
        
        # Validate input data
        required_cols = ['datetime_utc', target_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.target_col = target_col
        self.group_col = group_col
        
        # Fit models for each group
        if group_col and group_col in df.columns:
            groups = df[group_col].unique()
            for group in groups:
                group_data = df[df[group_col] == group].copy()
                self.models[group], self.scalers[group], self.training_stats[group] = self._fit_single_model(group_data, group)
                logger.info(f"Fitted ARIMA model for group {group}")
        else:
            self.models['default'], self.scalers['default'], self.training_stats['default'] = self._fit_single_model(df, 'default')
            logger.info("Fitted single ARIMA model")
        
        self.is_fitted = True
        logger.info(f"ARIMA models fitted for {len(self.models)} groups")
        
        return self
    
    def predict(self, timestamps: Union[List, pd.DatetimeIndex], 
                station_id: Optional[int] = None,
                horizon_hours: Optional[int] = None) -> ARIMAForecast:
        """
        Generate ARIMA predictions for given timestamps.
        
        Args:
            timestamps: Target prediction timestamps
            station_id: Station ID for grouped predictions
            horizon_hours: Forecast horizon for metadata
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(timestamps, list):
            timestamps = pd.to_datetime(timestamps)
        elif not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex([timestamps])
        
        # Determine which model to use
        if self.group_col and station_id is not None:
            if station_id in self.models:
                model = self.models[station_id]
                scaler = self.scalers[station_id]
                stats = self.training_stats[station_id]
            else:
                logger.warning(f"Station {station_id} not found, using default model")
                model = list(self.models.values())[0]
                scaler = list(self.scalers.values())[0]
                stats = list(self.training_stats.values())[0]
        else:
            model = self.models['default']
            scaler = self.scalers['default']
            stats = self.training_stats['default']
        
        # Generate predictions
        n_periods = len(timestamps)
        
        if self.statsmodels_available and hasattr(model, 'forecast'):
            # Use statsmodels ARIMA
            forecast_result = model.forecast(steps=n_periods, alpha=self.config.alpha)
            
            if isinstance(forecast_result, tuple):
                predictions_scaled, conf_int = forecast_result
            else:
                predictions_scaled = forecast_result
                conf_int = None
            
            # Scale back to original range
            predictions = self._inverse_scale(predictions_scaled, scaler)
            
            # Extract confidence intervals
            if conf_int is not None:
                lower_scaled = conf_int[:, 0]
                upper_scaled = conf_int[:, 1]
                
                lower = self._inverse_scale(lower_scaled, scaler)
                upper = self._inverse_scale(upper_scaled, scaler)
                
                confidence_intervals = {
                    'q0.1': lower.tolist(),
                    'q0.5': predictions.tolist(),
                    'q0.9': upper.tolist()
                }
            else:
                # Simple confidence intervals based on residual std
                std_residual = stats.get('residual_std', stats.get('std', 1.0))
                margin = 1.64 * std_residual  # ~90% interval
                
                confidence_intervals = {
                    'q0.1': [max(0, pred - margin) for pred in predictions],
                    'q0.5': predictions.tolist(),
                    'q0.9': [pred + margin for pred in predictions]
                }
            
            model_order = getattr(model, 'order', self.config.order)
            seasonal_order = getattr(model, 'seasonal_order', self.config.seasonal_order)
            
        else:
            # Use fallback AR model
            predictions, confidence_intervals, model_order, seasonal_order = self._predict_fallback(
                model, n_periods, scaler, stats
            )
        
        forecast_metadata = {
            'model_type': 'arima' if self.statsmodels_available else 'simplified_ar',
            'model_order': model_order,
            'seasonal_order': seasonal_order,
            'station_id': station_id,
            'residual_std': stats.get('residual_std', 0.0)
        }
        
        return ARIMAForecast(
            horizon_hours=horizon_hours or len(timestamps),
            predictions=predictions if isinstance(predictions, list) else predictions.tolist(),
            timestamps=[ts.isoformat() for ts in timestamps],
            confidence_intervals=confidence_intervals,
            model_order=model_order,
            seasonal_order=seasonal_order,
            forecast_metadata=forecast_metadata
        )
    
    def forecast(self, start_time: pd.Timestamp, horizon_hours: int,
                 station_id: Optional[int] = None) -> ARIMAForecast:
        """
        Generate multi-step ahead forecasts.
        
        Args:
            start_time: Start time for forecast
            horizon_hours: Number of hours to forecast ahead
            station_id: Station ID for grouped forecasts
        """
        timestamps = pd.date_range(
            start=start_time,
            periods=horizon_hours,
            freq='h'
        )
        
        return self.predict(timestamps, station_id, horizon_hours)
    
    def _fit_single_model(self, data: pd.DataFrame, group_name: str) -> Tuple[object, Dict, Dict]:
        """Fit a single ARIMA model."""
        
        # Prepare time series data
        ts_data = data[['datetime_utc', self.target_col]].copy()
        ts_data = ts_data.sort_values('datetime_utc').dropna()
        
        if len(ts_data) < 10:
            logger.warning(f"Insufficient data for group {group_name}: {len(ts_data)} points")
            return self._fit_fallback_model(ts_data)
        
        # Scale data for numerical stability
        scaler = self._fit_scaler(ts_data[self.target_col])
        scaled_values = self._scale(ts_data[self.target_col], scaler)
        
        # Calculate training statistics
        stats = {
            'mean': ts_data[self.target_col].mean(),
            'std': ts_data[self.target_col].std(),
            'min': ts_data[self.target_col].min(),
            'max': ts_data[self.target_col].max(),
            'n_obs': len(ts_data)
        }
        
        if self.statsmodels_available:
            try:
                model = self._fit_statsmodels_arima(scaled_values, stats)
                
                # Calculate residual statistics
                if hasattr(model, 'resid'):
                    residuals = model.resid
                    stats['residual_std'] = np.std(residuals)
                    stats['residual_mean'] = np.mean(residuals)
                
                return model, scaler, stats
                
            except Exception as e:
                logger.warning(f"ARIMA fitting failed for group {group_name}: {e}, using fallback")
                return self._fit_fallback_model(ts_data)
        else:
            return self._fit_fallback_model(ts_data)
    
    def _fit_statsmodels_arima(self, scaled_values: np.ndarray, stats: Dict) -> object:
        """Fit ARIMA model using statsmodels."""
        
        if self.auto_arima_available and self.config.auto_arima:
            # Use auto ARIMA for order selection
            try:
                model = self.auto_arima(
                    scaled_values,
                    seasonal=self.config.seasonal,
                    m=self.config.seasonal_period,
                    max_p=self.config.max_p,
                    max_q=self.config.max_q,
                    max_d=self.config.max_d,
                    start_p=1,
                    start_q=1,
                    suppress_warnings=True,
                    stepwise=True,
                    error_action='ignore'
                )
                return model
                
            except Exception as e:
                logger.warning(f"Auto ARIMA failed: {e}, using manual order")
        
        # Use manual order
        order = self.config.order
        seasonal_order = self.config.seasonal_order
        
        if self.config.seasonal and seasonal_order is None:
            # Default seasonal order
            seasonal_order = (1, 1, 1, self.config.seasonal_period)
        
        try:
            model = self.ARIMA(
                scaled_values,
                order=order,
                seasonal_order=seasonal_order,
                trend=self.config.trend
            )
            
            fitted_model = model.fit(
                method=self.config.method,
                maxiter=self.config.maxiter
            )
            
            return fitted_model
            
        except Exception as e:
            logger.warning(f"ARIMA fitting with order {order} failed: {e}")
            # Try simpler order without trend
            simple_order = (1, 0, 1)
            
            try:
                model = self.ARIMA(
                    scaled_values,
                    order=simple_order,
                    trend='n'  # No trend
                )
                
                return model.fit()
            except Exception as e2:
                logger.warning(f"Simple ARIMA also failed: {e2}")
                # Final fallback - AR(1) only
                model = self.ARIMA(
                    scaled_values,
                    order=(1, 0, 0),
                    trend='n'
                )
                
                return model.fit()
    
    def _fit_fallback_model(self, data: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
        """Fit simplified AR model when statsmodels is not available."""
        
        target_values = data[self.target_col].values
        
        # Simple scaling
        scaler = self._fit_scaler(target_values)
        scaled_values = self._scale(target_values, scaler)
        
        # Calculate statistics
        stats = {
            'mean': np.mean(target_values),
            'std': np.std(target_values),
            'min': np.min(target_values),
            'max': np.max(target_values),
            'n_obs': len(target_values)
        }
        
        # Fit simple AR(1) model
        if len(scaled_values) > 1:
            # Calculate AR(1) coefficient
            x = scaled_values[:-1]
            y = scaled_values[1:]
            
            if len(x) > 0 and np.var(x) > 1e-10:
                ar_coef = np.corrcoef(x, y)[0, 1]
                intercept = np.mean(y) - ar_coef * np.mean(x)
            else:
                ar_coef = 0.0
                intercept = np.mean(scaled_values)
            
            # Calculate residual std
            if len(x) > 0:
                predictions = intercept + ar_coef * x
                residuals = y - predictions
                residual_std = np.std(residuals)
            else:
                residual_std = np.std(scaled_values)
        else:
            ar_coef = 0.0
            intercept = scaled_values[0] if len(scaled_values) > 0 else 0.0
            residual_std = 1.0
        
        model = {
            'type': 'fallback_ar1',
            'ar_coef': ar_coef,
            'intercept': intercept,
            'last_value': scaled_values[-1] if len(scaled_values) > 0 else 0.0,
            'residual_std': residual_std
        }
        
        stats['residual_std'] = residual_std
        
        return model, scaler, stats
    
    def _predict_fallback(self, model: Dict, n_periods: int, scaler: Dict, stats: Dict) -> Tuple[List[float], Dict, Tuple, Optional[Tuple]]:
        """Generate predictions using fallback AR model."""
        
        predictions_scaled = []
        current_value = model['last_value']
        
        for _ in range(n_periods):
            # AR(1) prediction
            next_value = model['intercept'] + model['ar_coef'] * current_value
            predictions_scaled.append(next_value)
            current_value = next_value
        
        # Scale back to original range
        predictions = self._inverse_scale(np.array(predictions_scaled), scaler)
        
        # Simple confidence intervals
        residual_std_scaled = model['residual_std']
        if callable(residual_std_scaled):
            residual_std_scaled = 1.0  # Default fallback value
        else:
            residual_std_scaled = float(residual_std_scaled)
        
        residual_std_original = residual_std_scaled * scaler['scale']
        
        confidence_intervals = {
            'q0.1': [max(0, pred - 1.64 * residual_std_original) for pred in predictions],
            'q0.5': predictions.tolist(),
            'q0.9': [pred + 1.64 * residual_std_original for pred in predictions]
        }
        
        model_order = (1, 0, 0)  # AR(1)
        seasonal_order = None
        
        return predictions.tolist(), confidence_intervals, model_order, seasonal_order
    
    def _fit_scaler(self, values: Union[pd.Series, np.ndarray]) -> Dict:
        """Fit a simple scaler for numerical stability."""
        
        values_array = np.array(values)
        values_clean = values_array[~np.isnan(values_array)]
        
        if len(values_clean) == 0:
            return {'mean': 0.0, 'scale': 1.0}
        
        mean = np.mean(values_clean)
        std = np.std(values_clean)
        scale = max(std, 1e-6)  # Avoid division by zero
        
        return {'mean': mean, 'scale': scale}
    
    def _scale(self, values: Union[pd.Series, np.ndarray], scaler: Dict) -> np.ndarray:
        """Scale values using fitted scaler."""
        
        values_array = np.array(values)
        return (values_array - scaler['mean']) / scaler['scale']
    
    def _inverse_scale(self, scaled_values: np.ndarray, scaler: Dict) -> np.ndarray:
        """Inverse scale values back to original range."""
        
        return scaled_values * scaler['scale'] + scaler['mean']
    
    def get_model_info(self) -> Dict:
        """Get information about the fitted models."""
        
        if not self.is_fitted:
            return {"fitted": False}
        
        info = {
            "fitted": True,
            "model_type": "arima" if self.statsmodels_available else "simplified_ar",
            "statsmodels_available": self.statsmodels_available,
            "auto_arima_available": self.auto_arima_available,
            "groups": list(self.models.keys()),
            "config": self.config.model_dump()
        }
        
        # Add model-specific info for each group
        info["group_models"] = {}
        for group, model in self.models.items():
            if hasattr(model, 'order'):
                # statsmodels ARIMA
                info["group_models"][group] = {
                    "order": getattr(model, 'order', None),
                    "seasonal_order": getattr(model, 'seasonal_order', None),
                    "aic": getattr(model, 'aic', None),
                    "bic": getattr(model, 'bic', None)
                }
            else:
                # Fallback model
                info["group_models"][group] = {
                    "type": model.get('type', 'unknown'),
                    "ar_coef": model.get('ar_coef', None),
                    "residual_std": model.get('residual_std', None)
                }
        
        return info
