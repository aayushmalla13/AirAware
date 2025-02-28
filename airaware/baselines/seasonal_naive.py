"""Seasonal-naive baseline forecaster for PM₂.₅ nowcasting."""

import logging
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SeasonalNaiveConfig(BaseModel):
    """Configuration for seasonal-naive forecaster."""
    seasonal_period: int = Field(168, description="Seasonal period in hours (168 = weekly)")
    fallback_periods: List[int] = Field(default=[24, 72], description="Fallback periods if seasonal period unavailable")
    min_history_periods: int = Field(2, description="Minimum number of seasonal periods needed")
    handle_missing: str = Field("interpolate", description="How to handle missing values: interpolate, forward_fill, drop")


class SeasonalNaiveForecast(BaseModel):
    """Seasonal-naive forecast result."""
    horizon_hours: int
    predictions: List[float]
    timestamps: List[str]
    seasonal_period_used: int
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    forecast_metadata: Dict = Field(default_factory=dict)


class SeasonalNaiveForecaster:
    """
    Seasonal-naive forecaster that predicts using the same hour from the previous seasonal period.
    
    This is a strong baseline for air quality data with weekly patterns (rush hours, weekend effects).
    The model looks back exactly one seasonal period (default: 1 week = 168 hours) and uses
    that value as the prediction.
    """
    
    def __init__(self, config: Optional[SeasonalNaiveConfig] = None):
        self.config = config or SeasonalNaiveConfig()
        self.is_fitted = False
        self.training_data = None
        self.seasonal_stats = {}
        
        logger.info(f"SeasonalNaiveForecaster initialized with period={self.config.seasonal_period}h")
    
    def fit(self, df: pd.DataFrame, target_col: str = 'pm25', 
            group_col: Optional[str] = 'station_id') -> 'SeasonalNaiveForecaster':
        """
        Fit the seasonal-naive model.
        
        Args:
            df: Training data with datetime_utc, target, and optional group columns
            target_col: Name of target variable column
            group_col: Name of grouping column (e.g., station_id)
        """
        logger.info(f"Fitting seasonal-naive model on {len(df):,} records")
        
        # Validate input data
        required_cols = ['datetime_utc', target_col]
        if group_col:
            required_cols.append(group_col)
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Prepare training data
        self.training_data = df[required_cols].copy()
        self.training_data['datetime_utc'] = pd.to_datetime(self.training_data['datetime_utc'])
        self.training_data = self.training_data.sort_values(['datetime_utc'])
        
        self.target_col = target_col
        self.group_col = group_col
        
        # Calculate seasonal statistics for each group
        if group_col:
            groups = self.training_data[group_col].unique()
            for group in groups:
                group_data = self.training_data[self.training_data[group_col] == group]
                self.seasonal_stats[group] = self._calculate_seasonal_stats(group_data)
        else:
            self.seasonal_stats['default'] = self._calculate_seasonal_stats(self.training_data)
        
        self.is_fitted = True
        logger.info(f"Seasonal-naive model fitted for {len(self.seasonal_stats)} groups")
        
        return self
    
    def predict(self, timestamps: Union[List, pd.DatetimeIndex], 
                station_id: Optional[int] = None,
                horizon_hours: Optional[int] = None) -> SeasonalNaiveForecast:
        """
        Generate seasonal-naive predictions for given timestamps.
        
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
        
        # Determine which seasonal stats to use
        if self.group_col and station_id is not None:
            if station_id in self.seasonal_stats:
                stats = self.seasonal_stats[station_id]
                group_data = self.training_data[self.training_data[self.group_col] == station_id]
            else:
                logger.warning(f"Station {station_id} not found in training data, using default")
                stats = list(self.seasonal_stats.values())[0]
                group_data = self.training_data
        else:
            # Use first available station's stats as default
            if self.seasonal_stats:
                stats = list(self.seasonal_stats.values())[0]
                group_data = self.training_data
            else:
                logger.warning("No seasonal stats available, using fallback")
                stats = {'seasonal_period_used': 24, 'mad': 1.0, 'seasonal_strength': 0.0}
                group_data = self.training_data
        
        # Generate predictions
        predictions = []
        seasonal_period_used = stats['seasonal_period_used']
        
        for ts in timestamps:
            # Look back exactly one seasonal period
            lookback_time = ts - pd.Timedelta(hours=seasonal_period_used)
            
            # Find the closest historical value
            pred_value = self._get_seasonal_value(group_data, lookback_time, stats)
            predictions.append(pred_value)
        
        # Calculate simple confidence intervals based on historical variability
        confidence_intervals = self._calculate_confidence_intervals(
            predictions, stats, confidence_levels=[0.1, 0.5, 0.9]
        )
        
        forecast_metadata = {
            'seasonal_period_used': seasonal_period_used,
            'training_data_points': len(group_data),
            'seasonal_strength': stats.get('seasonal_strength', 0.0),
            'mean_absolute_deviation': stats.get('mad', 0.0)
        }
        
        return SeasonalNaiveForecast(
            horizon_hours=horizon_hours or len(timestamps),
            predictions=predictions,
            timestamps=[ts.isoformat() for ts in timestamps],
            seasonal_period_used=seasonal_period_used,
            confidence_intervals=confidence_intervals,
            forecast_metadata=forecast_metadata
        )
    
    def forecast(self, start_time: pd.Timestamp, horizon_hours: int, 
                 station_id: Optional[int] = None) -> SeasonalNaiveForecast:
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
    
    def _calculate_seasonal_stats(self, data: pd.DataFrame) -> Dict:
        """Calculate seasonal statistics for the given data."""
        
        stats = {}
        
        # Determine the best seasonal period to use
        seasonal_period = self._find_best_seasonal_period(data)
        stats['seasonal_period_used'] = seasonal_period
        
        # Calculate seasonal strength (how much the seasonal pattern explains variance)
        stats['seasonal_strength'] = self._calculate_seasonal_strength(data, seasonal_period)
        
        # Calculate mean absolute deviation for confidence intervals
        if len(data) > 1:
            target_values = data[self.target_col].dropna()
            if len(target_values) > 1:
                stats['mad'] = np.mean(np.abs(target_values - target_values.mean()))
                stats['std'] = target_values.std()
                stats['mean'] = target_values.mean()
            else:
                stats['mad'] = 0.0
                stats['std'] = 0.0
                stats['mean'] = target_values.iloc[0] if len(target_values) > 0 else 25.0
        else:
            stats['mad'] = 0.0
            stats['std'] = 0.0
            stats['mean'] = 25.0  # Default PM2.5 value
        
        return stats
    
    def _find_best_seasonal_period(self, data: pd.DataFrame) -> int:
        """Find the best seasonal period based on available data."""
        
        # Check if we have enough data for the primary seasonal period
        min_required_hours = self.config.seasonal_period * self.config.min_history_periods
        
        if len(data) >= min_required_hours:
            return self.config.seasonal_period
        
        # Try fallback periods
        for period in self.config.fallback_periods:
            min_required = period * self.config.min_history_periods
            if len(data) >= min_required:
                logger.info(f"Using fallback seasonal period: {period}h (insufficient data for {self.config.seasonal_period}h)")
                return period
        
        # If no seasonal period works, use the maximum possible
        max_possible = len(data) // self.config.min_history_periods
        if max_possible > 0:
            logger.warning(f"Very limited data, using period: {max_possible}h")
            return max_possible
        
        # Absolute fallback
        logger.warning("Insufficient data for any seasonal period, using 24h")
        return 24
    
    def _calculate_seasonal_strength(self, data: pd.DataFrame, seasonal_period: int) -> float:
        """Calculate how strong the seasonal pattern is."""
        
        try:
            target_values = data[self.target_col].dropna()
            
            if len(target_values) < seasonal_period * 2:
                return 0.0
            
            # Group by hour within the seasonal period
            data_copy = data.copy()
            data_copy['hour_in_period'] = range(len(data_copy))
            data_copy['hour_in_period'] = data_copy['hour_in_period'] % seasonal_period
            
            # Calculate variance within each hour vs total variance
            hour_means = data_copy.groupby('hour_in_period')[self.target_col].mean()
            overall_mean = data_copy[self.target_col].mean()
            
            seasonal_variance = ((hour_means - overall_mean) ** 2).mean()
            total_variance = data_copy[self.target_col].var()
            
            if total_variance > 0:
                seasonal_strength = seasonal_variance / total_variance
                return min(1.0, seasonal_strength)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Could not calculate seasonal strength: {e}")
            return 0.0
    
    def _get_seasonal_value(self, data: pd.DataFrame, lookback_time: pd.Timestamp, stats: Dict) -> float:
        """Get the seasonal value for a specific lookback time."""
        
        # Find the closest historical value to the lookback time
        time_diffs = np.abs((data['datetime_utc'] - lookback_time).dt.total_seconds())
        
        if len(time_diffs) == 0:
            return stats['mean']
        
        closest_idx = time_diffs.idxmin()
        closest_value = data.loc[closest_idx, self.target_col]
        
        # Handle missing values
        if pd.isna(closest_value):
            # Try to find nearby non-missing values
            window = pd.Timedelta(hours=3)  # 3-hour window
            mask = np.abs(data['datetime_utc'] - lookback_time) <= window
            nearby_values = data[mask][self.target_col].dropna()
            
            if len(nearby_values) > 0:
                return nearby_values.mean()
            else:
                return stats['mean']
        
        return float(closest_value)
    
    def _calculate_confidence_intervals(self, predictions: List[float], stats: Dict, 
                                     confidence_levels: List[float]) -> Dict[str, List[float]]:
        """Calculate simple confidence intervals based on historical variability."""
        
        intervals = {}
        mad = stats.get('mad', 0.0)
        
        for confidence_level in confidence_levels:
            # Simple approach: use MAD to estimate confidence bounds
            # For normal distribution, ~68% of data is within 1 MAD of median
            z_score = self._confidence_to_z_score(confidence_level)
            margin = z_score * mad
            
            if confidence_level < 0.5:
                # Lower bound
                intervals[f'q{confidence_level:.1f}'] = [max(0, pred - margin) for pred in predictions]
            elif confidence_level > 0.5:
                # Upper bound
                intervals[f'q{confidence_level:.1f}'] = [pred + margin for pred in predictions]
            else:
                # Median (just the prediction)
                intervals[f'q{confidence_level:.1f}'] = predictions.copy()
        
        return intervals
    
    def _confidence_to_z_score(self, confidence_level: float) -> float:
        """Convert confidence level to approximate z-score."""
        
        # Simple mapping for common confidence levels
        if confidence_level <= 0.1:
            return 1.28  # ~90% interval
        elif confidence_level <= 0.25:
            return 0.67  # ~50% interval
        elif confidence_level >= 0.9:
            return 1.28  # ~90% interval
        elif confidence_level >= 0.75:
            return 0.67  # ~50% interval
        else:
            return 0.0  # Median
    
    def get_model_info(self) -> Dict:
        """Get information about the fitted model."""
        
        if not self.is_fitted:
            return {"fitted": False}
        
        info = {
            "fitted": True,
            "model_type": "seasonal_naive",
            "seasonal_periods": {},
            "training_data_size": len(self.training_data) if self.training_data is not None else 0,
            "groups": list(self.seasonal_stats.keys())
        }
        
        for group, stats in self.seasonal_stats.items():
            info["seasonal_periods"][group] = {
                "period_hours": stats['seasonal_period_used'],
                "seasonal_strength": stats.get('seasonal_strength', 0.0),
                "mean_value": stats.get('mean', 0.0),
                "std_value": stats.get('std', 0.0)
            }
        
        return info
