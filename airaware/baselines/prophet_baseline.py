"""Prophet baseline forecaster for PM₂.₅ nowcasting with meteorological features."""

import logging
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProphetConfig(BaseModel):
    """Configuration for Prophet forecaster."""
    growth: str = Field("linear", description="Growth trend: linear or logistic")
    seasonality_mode: str = Field("additive", description="Seasonality mode: additive or multiplicative")
    daily_seasonality: bool = Field(True, description="Include daily seasonality")
    weekly_seasonality: bool = Field(True, description="Include weekly seasonality")
    yearly_seasonality: bool = Field("auto", description="Include yearly seasonality")
    
    # Seasonality parameters
    seasonality_prior_scale: float = Field(10.0, description="Strength of seasonality")
    holidays_prior_scale: float = Field(10.0, description="Strength of holiday effects")
    changepoint_prior_scale: float = Field(0.05, description="Strength of changepoints")
    
    # Additional regressors
    include_meteorological: bool = Field(True, description="Include meteorological features")
    include_calendar_events: bool = Field(True, description="Include calendar events (holidays, weekends)")
    include_pollution_features: bool = Field(True, description="Include pollution-related features")
    
    meteo_features: List[str] = Field(
        default=["wind_speed", "t2m_celsius", "blh", "precip"], 
        description="Meteorological features to include"
    )
    
    pollution_features: List[str] = Field(
        default=["pm25_lag_1h", "pm25_lag_24h", "pm25_rolling_24h_mean"],
        description="Pollution-related features to include"
    )
    
    # External regressor settings
    regressor_prior_scale: float = Field(10.0, description="Prior scale for external regressors")
    regressor_mode: str = Field("additive", description="Mode for external regressors: additive or multiplicative")
    
    # Confidence intervals
    interval_width: float = Field(0.8, description="Width of uncertainty intervals")
    mcmc_samples: int = Field(0, description="Number of MCMC samples (0 for MAP)")


class ProphetForecast(BaseModel):
    """Prophet forecast result."""
    horizon_hours: int
    predictions: List[float]
    timestamps: List[str]
    confidence_intervals: Dict[str, List[float]]
    trend: List[float]
    seasonal_components: Dict[str, List[float]]
    forecast_metadata: Dict = Field(default_factory=dict)


class ProphetBaseline:
    """
    Prophet baseline forecaster for PM₂.₅ with optional meteorological regressors.
    
    Uses Facebook Prophet to capture trend, seasonality, and optional meteorological effects.
    This provides a strong time series baseline that can capture complex seasonal patterns
    and incorporate external regressors.
    """
    
    def __init__(self, config: Optional[ProphetConfig] = None):
        self.config = config or ProphetConfig()
        self.is_fitted = False
        self.models = {}  # One model per station
        self.feature_columns = []
        
        # Import Prophet (with fallback handling)
        try:
            from prophet import Prophet
            self.Prophet = Prophet
            self.prophet_available = True
        except ImportError:
            logger.warning("Prophet not available - will use simplified trend + seasonal model")
            self.Prophet = None
            self.prophet_available = False
        
        logger.info(f"ProphetBaseline initialized (Prophet available: {self.prophet_available})")
    
    def fit(self, df: pd.DataFrame, target_col: str = 'pm25', 
            group_col: Optional[str] = 'station_id') -> 'ProphetBaseline':
        """
        Fit Prophet models.
        
        Args:
            df: Training data with datetime_utc, target, and optional features
            target_col: Name of target variable column
            group_col: Name of grouping column (e.g., station_id)
        """
        logger.info(f"Fitting Prophet model on {len(df):,} records")
        
        # Validate input data
        required_cols = ['datetime_utc', target_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.target_col = target_col
        self.group_col = group_col
        
        # Determine available meteorological features
        if self.config.include_meteorological:
            available_meteo = [col for col in self.config.meteo_features if col in df.columns]
            self.feature_columns = available_meteo
            logger.info(f"Using meteorological features: {self.feature_columns}")
        else:
            self.feature_columns = []
        
        # Fit models for each group
        if group_col and group_col in df.columns:
            groups = df[group_col].unique()
            for group in groups:
                group_data = df[df[group_col] == group].copy()
                self.models[group] = self._fit_single_model(group_data)
                logger.info(f"Fitted Prophet model for group {group}")
        else:
            # Fit single model without grouping
            self.models[0] = self._fit_single_model(df)
            logger.info("Fitted single Prophet model")
        
        self.is_fitted = True
        logger.info(f"Prophet models fitted for {len(self.models)} groups")
        
        return self
    
    def predict(self, timestamps: Union[List, pd.DatetimeIndex], 
                station_id: Optional[int] = None,
                horizon_hours: Optional[int] = None,
                meteo_data: Optional[pd.DataFrame] = None) -> ProphetForecast:
        """
        Generate Prophet predictions for given timestamps.
        
        Args:
            timestamps: Target prediction timestamps
            station_id: Station ID for grouped predictions
            horizon_hours: Forecast horizon for metadata
            meteo_data: Meteorological data for predictions (optional)
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
            else:
                logger.warning(f"Station {station_id} not found, using default model")
                model = list(self.models.values())[0]
        else:
            # Use first available model as default
            if self.models:
                model = list(self.models.values())[0]
            else:
                logger.warning("No Prophet models available")
                return self._create_empty_forecast(timestamps)
        
        # Create future dataframe
        future_df = self._create_future_dataframe(timestamps, meteo_data)
        
        # Generate predictions
        if self.prophet_available:
            forecast = model.predict(future_df)
            predictions = forecast['yhat'].tolist()
            
            # Extract confidence intervals
            confidence_intervals = {
                'q0.1': forecast['yhat_lower'].tolist(),
                'q0.5': predictions.copy(),
                'q0.9': forecast['yhat_upper'].tolist()
            }
            
            # Extract trend and seasonal components
            trend = forecast['trend'].tolist()
            seasonal_components = {}
            
            if 'daily' in forecast.columns:
                seasonal_components['daily'] = forecast['daily'].tolist()
            if 'weekly' in forecast.columns:
                seasonal_components['weekly'] = forecast['weekly'].tolist()
            if 'yearly' in forecast.columns:
                seasonal_components['yearly'] = forecast['yearly'].tolist()
            
        else:
            # Fallback to simplified model
            predictions, confidence_intervals, trend, seasonal_components = self._predict_fallback(
                model, timestamps, future_df
            )
        
        forecast_metadata = {
            'model_type': 'prophet' if self.prophet_available else 'simplified_prophet',
            'features_used': self.feature_columns,
            'station_id': station_id
        }
        
        return ProphetForecast(
            horizon_hours=horizon_hours or len(timestamps),
            predictions=predictions,
            timestamps=[ts.isoformat() for ts in timestamps],
            confidence_intervals=confidence_intervals,
            trend=trend,
            seasonal_components=seasonal_components,
            forecast_metadata=forecast_metadata
        )
    
    def forecast(self, start_time: pd.Timestamp, horizon_hours: int,
                 station_id: Optional[int] = None,
                 meteo_data: Optional[pd.DataFrame] = None) -> ProphetForecast:
        """
        Generate multi-step ahead forecasts.
        
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
    
    def _fit_single_model(self, data: pd.DataFrame) -> Union[object, Dict]:
        """Fit a single Prophet model or fallback model."""
        
        if self.prophet_available:
            return self._fit_prophet_model(data)
        else:
            return self._fit_fallback_model(data)
    
    def _fit_prophet_model(self, data: pd.DataFrame) -> object:
        """Fit actual Prophet model with enhanced external regressors."""
        
        # Prepare data in Prophet format
        prophet_data = data[['datetime_utc', self.target_col]].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Convert timezone-aware datetime to naive (Prophet requirement)
        if prophet_data['ds'].dt.tz is not None:
            prophet_data['ds'] = prophet_data['ds'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Remove missing values
        prophet_data = prophet_data.dropna()
        
        # Add enhanced external regressors
        prophet_data = self._add_enhanced_regressors(prophet_data, data)
        
        # Initialize Prophet model
        model = self.Prophet(
            growth=self.config.growth,
            seasonality_mode=self.config.seasonality_mode,
            daily_seasonality=self.config.daily_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            yearly_seasonality=self.config.yearly_seasonality,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            holidays_prior_scale=self.config.holidays_prior_scale,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            interval_width=self.config.interval_width,
            mcmc_samples=self.config.mcmc_samples
        )
        
        # Add all available regressors
        regressor_columns = [col for col in prophet_data.columns if col not in ['ds', 'y']]
        for feature in regressor_columns:
            model.add_regressor(feature, prior_scale=self.config.regressor_prior_scale, mode=self.config.regressor_mode)
        
        # Fit model
        model.fit(prophet_data)
        
        return model
    
    def _add_enhanced_regressors(self, prophet_data: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced external regressors to Prophet data."""
        
        enhanced_data = prophet_data.copy()
        
        # Add meteorological regressors
        if self.config.include_meteorological:
            enhanced_data = self._add_meteorological_regressors(enhanced_data, original_data)
        
        # Add calendar event regressors
        if self.config.include_calendar_events:
            enhanced_data = self._add_calendar_regressors(enhanced_data)
        
        # Add pollution-related regressors
        if self.config.include_pollution_features:
            enhanced_data = self._add_pollution_regressors(enhanced_data)
        
        return enhanced_data
    
    def _add_meteorological_regressors(self, prophet_data: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Add meteorological external regressors."""
        
        enhanced_data = prophet_data.copy()
        
        # Wind speed (if available)
        if 'u10' in original_data.columns and 'v10' in original_data.columns:
            wind_speed = np.sqrt(original_data['u10']**2 + original_data['v10']**2)
            enhanced_data['wind_speed'] = wind_speed.values[:len(enhanced_data)]
        
        # Temperature in Celsius (if available)
        if 't2m' in original_data.columns:
            temp_celsius = original_data['t2m'] - 273.15
            enhanced_data['t2m_celsius'] = temp_celsius.values[:len(enhanced_data)]
        
        # Boundary layer height (if available)
        if 'blh' in original_data.columns:
            enhanced_data['blh'] = original_data['blh'].values[:len(enhanced_data)]
        
        # Precipitation (if available)
        if 'precip' in original_data.columns:
            enhanced_data['precip'] = original_data['precip'].values[:len(enhanced_data)]
        
        # Add derived meteorological features
        if 'wind_speed' in enhanced_data.columns:
            # Wind speed categories
            enhanced_data['wind_calm'] = (enhanced_data['wind_speed'] < 2).astype(int)
            enhanced_data['wind_light'] = ((enhanced_data['wind_speed'] >= 2) & 
                                         (enhanced_data['wind_speed'] < 5)).astype(int)
            enhanced_data['wind_moderate'] = ((enhanced_data['wind_speed'] >= 5) & 
                                            (enhanced_data['wind_speed'] < 10)).astype(int)
            enhanced_data['wind_strong'] = (enhanced_data['wind_speed'] >= 10).astype(int)
        
        if 't2m_celsius' in enhanced_data.columns:
            # Temperature categories
            enhanced_data['temp_cold'] = (enhanced_data['t2m_celsius'] < 10).astype(int)
            enhanced_data['temp_mild'] = ((enhanced_data['t2m_celsius'] >= 10) & 
                                        (enhanced_data['t2m_celsius'] < 25)).astype(int)
            enhanced_data['temp_warm'] = (enhanced_data['t2m_celsius'] >= 25).astype(int)
        
        return enhanced_data
    
    def _add_calendar_regressors(self, prophet_data: pd.DataFrame) -> pd.DataFrame:
        """Add calendar event external regressors."""
        
        enhanced_data = prophet_data.copy()
        
        # Weekend indicator
        enhanced_data['is_weekend'] = (enhanced_data['ds'].dt.dayofweek >= 5).astype(int)
        
        # Day of week (one-hot encoded)
        for i in range(7):
            enhanced_data[f'dow_{i}'] = (enhanced_data['ds'].dt.dayofweek == i).astype(int)
        
        # Hour of day (one-hot encoded for key hours)
        key_hours = [6, 7, 8, 9, 17, 18, 19, 20]  # Rush hours
        for hour in key_hours:
            enhanced_data[f'hour_{hour}'] = (enhanced_data['ds'].dt.hour == hour).astype(int)
        
        # Month (one-hot encoded)
        for month in range(1, 13):
            enhanced_data[f'month_{month}'] = (enhanced_data['ds'].dt.month == month).astype(int)
        
        # Season indicators
        enhanced_data['is_winter'] = enhanced_data['ds'].dt.month.isin([12, 1, 2]).astype(int)
        enhanced_data['is_spring'] = enhanced_data['ds'].dt.month.isin([3, 4, 5]).astype(int)
        enhanced_data['is_summer'] = enhanced_data['ds'].dt.month.isin([6, 7, 8]).astype(int)
        enhanced_data['is_autumn'] = enhanced_data['ds'].dt.month.isin([9, 10, 11]).astype(int)
        
        # Holiday indicators (Nepal-specific holidays)
        enhanced_data = self._add_nepal_holidays(enhanced_data)
        
        return enhanced_data
    
    def _add_nepal_holidays(self, prophet_data: pd.DataFrame) -> pd.DataFrame:
        """Add Nepal-specific holiday indicators."""
        
        enhanced_data = prophet_data.copy()
        
        # Major Nepal holidays (simplified)
        holidays = {
            'Dashain': [(9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (9, 22), (9, 23), (9, 24)],
            'Tihar': [(10, 8), (10, 9), (10, 10), (10, 11), (10, 12)],
            'New_Year': [(1, 1)],
            'Republic_Day': [(5, 29)],
            'Constitution_Day': [(9, 20)],
            'Independence_Day': [(8, 15)]
        }
        
        for holiday_name, dates in holidays.items():
            enhanced_data[f'holiday_{holiday_name}'] = 0
            for month, day in dates:
                enhanced_data[f'holiday_{holiday_name}'] += (
                    (enhanced_data['ds'].dt.month == month) & 
                    (enhanced_data['ds'].dt.day == day)
                ).astype(int)
        
        # General holiday indicator
        holiday_cols = [col for col in enhanced_data.columns if col.startswith('holiday_')]
        enhanced_data['is_holiday'] = enhanced_data[holiday_cols].sum(axis=1).astype(int)
        
        return enhanced_data
    
    def _add_pollution_regressors(self, prophet_data: pd.DataFrame) -> pd.DataFrame:
        """Add pollution-related external regressors."""
        
        enhanced_data = prophet_data.copy()
        
        # Lag features
        if 'y' in enhanced_data.columns:
            enhanced_data['pm25_lag_1h'] = enhanced_data['y'].shift(1)
            enhanced_data['pm25_lag_24h'] = enhanced_data['y'].shift(24)
            enhanced_data['pm25_lag_168h'] = enhanced_data['y'].shift(168)  # Weekly lag
            
            # Rolling statistics
            enhanced_data['pm25_rolling_24h_mean'] = enhanced_data['y'].rolling(24, min_periods=1).mean()
            enhanced_data['pm25_rolling_24h_std'] = enhanced_data['y'].rolling(24, min_periods=1).std()
            enhanced_data['pm25_rolling_168h_mean'] = enhanced_data['y'].rolling(168, min_periods=1).mean()
            
            # Fill NaN values with forward fill or mean
            enhanced_data['pm25_lag_1h'] = enhanced_data['pm25_lag_1h'].ffill().fillna(enhanced_data['y'].mean())
            enhanced_data['pm25_lag_24h'] = enhanced_data['pm25_lag_24h'].ffill().fillna(enhanced_data['y'].mean())
            enhanced_data['pm25_lag_168h'] = enhanced_data['pm25_lag_168h'].ffill().fillna(enhanced_data['y'].mean())
            
            # Fill NaN values in rolling statistics
            enhanced_data['pm25_rolling_24h_mean'] = enhanced_data['pm25_rolling_24h_mean'].fillna(enhanced_data['y'].mean())
            enhanced_data['pm25_rolling_24h_std'] = enhanced_data['pm25_rolling_24h_std'].fillna(enhanced_data['y'].std())
            enhanced_data['pm25_rolling_168h_mean'] = enhanced_data['pm25_rolling_168h_mean'].fillna(enhanced_data['y'].mean())
            
            # Pollution level categories
            enhanced_data['pm25_low'] = (enhanced_data['y'] < 25).astype(int)
            enhanced_data['pm25_moderate'] = ((enhanced_data['y'] >= 25) & 
                                            (enhanced_data['y'] < 50)).astype(int)
            enhanced_data['pm25_unhealthy'] = ((enhanced_data['y'] >= 50) & 
                                             (enhanced_data['y'] < 100)).astype(int)
            enhanced_data['pm25_very_unhealthy'] = (enhanced_data['y'] >= 100).astype(int)
        
        return enhanced_data
    
    def _fit_fallback_model(self, data: pd.DataFrame) -> Dict:
        """Fit simplified trend + seasonal model when Prophet is not available."""
        
        # Calculate basic statistics
        target_values = data[self.target_col].dropna()
        
        model = {
            'type': 'fallback',
            'mean': target_values.mean(),
            'std': target_values.std(),
            'trend_slope': 0.0,
            'seasonal_patterns': {}
        }
        
        if len(target_values) > 1:
            # Calculate simple linear trend
            data_clean = data.dropna(subset=[self.target_col]).copy()
            data_clean['hour_index'] = range(len(data_clean))
            
            if len(data_clean) > 10:
                # Simple linear regression for trend
                x = data_clean['hour_index'].values
                y = data_clean[self.target_col].values
                
                # Calculate slope
                x_mean = x.mean()
                y_mean = y.mean()
                
                numerator = ((x - x_mean) * (y - y_mean)).sum()
                denominator = ((x - x_mean) ** 2).sum()
                
                if denominator != 0:
                    model['trend_slope'] = numerator / denominator
            
            # Calculate hourly seasonal patterns
            data_clean['hour'] = data_clean['datetime_utc'].dt.hour
            hourly_means = data_clean.groupby('hour')[self.target_col].mean()
            overall_mean = data_clean[self.target_col].mean()
            
            model['seasonal_patterns']['hourly'] = (hourly_means - overall_mean).to_dict()
            
            # Calculate day-of-week patterns
            data_clean['dow'] = data_clean['datetime_utc'].dt.dayofweek
            dow_means = data_clean.groupby('dow')[self.target_col].mean()
            
            model['seasonal_patterns']['weekly'] = (dow_means - overall_mean).to_dict()
        
        return model
    
    def _create_future_dataframe(self, timestamps: pd.DatetimeIndex, 
                                meteo_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create future dataframe for predictions with enhanced regressors."""
        
        future_df = pd.DataFrame({'ds': timestamps})
        
        # Convert timezone-aware datetime to naive (Prophet requirement)
        if future_df['ds'].dt.tz is not None:
            future_df['ds'] = future_df['ds'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Add enhanced external regressors
        future_df = self._add_future_regressors(future_df, meteo_data)
        
        return future_df
    
    def _add_future_regressors(self, future_df: pd.DataFrame, meteo_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Add external regressors to future dataframe."""
        
        enhanced_future = future_df.copy()
        
        # Add meteorological regressors
        if self.config.include_meteorological:
            enhanced_future = self._add_future_meteorological_regressors(enhanced_future, meteo_data)
        
        # Add calendar event regressors
        if self.config.include_calendar_events:
            enhanced_future = self._add_future_calendar_regressors(enhanced_future)
        
        # Add pollution-related regressors (using defaults for future)
        if self.config.include_pollution_features:
            enhanced_future = self._add_future_pollution_regressors(enhanced_future)
        
        return enhanced_future
    
    def _add_future_meteorological_regressors(self, future_df: pd.DataFrame, meteo_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Add meteorological regressors to future dataframe."""
        
        enhanced_future = future_df.copy()
        
        if meteo_data is not None:
            # Align meteorological data with timestamps
            meteo_aligned = meteo_data.set_index('datetime_utc').reindex(future_df['ds'])
            
            # Wind speed (if available)
            if 'u10' in meteo_aligned.columns and 'v10' in meteo_aligned.columns:
                wind_speed = np.sqrt(meteo_aligned['u10']**2 + meteo_aligned['v10']**2)
                enhanced_future['wind_speed'] = wind_speed.fillna(2.0).values
            else:
                enhanced_future['wind_speed'] = 2.0
            
            # Temperature in Celsius (if available)
            if 't2m' in meteo_aligned.columns:
                temp_celsius = meteo_aligned['t2m'] - 273.15
                enhanced_future['t2m_celsius'] = temp_celsius.fillna(20.0).values
            else:
                enhanced_future['t2m_celsius'] = 20.0
            
            # Boundary layer height (if available)
            if 'blh' in meteo_aligned.columns:
                enhanced_future['blh'] = meteo_aligned['blh'].fillna(500.0).values
            else:
                enhanced_future['blh'] = 500.0
            
            # Precipitation (if available)
            if 'precip' in meteo_aligned.columns:
                enhanced_future['precip'] = meteo_aligned['precip'].fillna(0.0).values
            else:
                enhanced_future['precip'] = 0.0
        else:
            # Use default values
            enhanced_future['wind_speed'] = 2.0
            enhanced_future['t2m_celsius'] = 20.0
            enhanced_future['blh'] = 500.0
            enhanced_future['precip'] = 0.0
        
        # Add derived meteorological features
        # Wind speed categories
        enhanced_future['wind_calm'] = (enhanced_future['wind_speed'] < 2).astype(int)
        enhanced_future['wind_light'] = ((enhanced_future['wind_speed'] >= 2) & 
                                       (enhanced_future['wind_speed'] < 5)).astype(int)
        enhanced_future['wind_moderate'] = ((enhanced_future['wind_speed'] >= 5) & 
                                          (enhanced_future['wind_speed'] < 10)).astype(int)
        enhanced_future['wind_strong'] = (enhanced_future['wind_speed'] >= 10).astype(int)
        
        # Temperature categories
        enhanced_future['temp_cold'] = (enhanced_future['t2m_celsius'] < 10).astype(int)
        enhanced_future['temp_mild'] = ((enhanced_future['t2m_celsius'] >= 10) & 
                                      (enhanced_future['t2m_celsius'] < 25)).astype(int)
        enhanced_future['temp_warm'] = (enhanced_future['t2m_celsius'] >= 25).astype(int)
        
        return enhanced_future
    
    def _add_future_calendar_regressors(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar event regressors to future dataframe."""
        
        enhanced_future = future_df.copy()
        
        # Weekend indicator
        enhanced_future['is_weekend'] = (enhanced_future['ds'].dt.dayofweek >= 5).astype(int)
        
        # Day of week (one-hot encoded)
        for i in range(7):
            enhanced_future[f'dow_{i}'] = (enhanced_future['ds'].dt.dayofweek == i).astype(int)
        
        # Hour of day (one-hot encoded for key hours)
        key_hours = [6, 7, 8, 9, 17, 18, 19, 20]  # Rush hours
        for hour in key_hours:
            enhanced_future[f'hour_{hour}'] = (enhanced_future['ds'].dt.hour == hour).astype(int)
        
        # Month (one-hot encoded)
        for month in range(1, 13):
            enhanced_future[f'month_{month}'] = (enhanced_future['ds'].dt.month == month).astype(int)
        
        # Season indicators
        enhanced_future['is_winter'] = enhanced_future['ds'].dt.month.isin([12, 1, 2]).astype(int)
        enhanced_future['is_spring'] = enhanced_future['ds'].dt.month.isin([3, 4, 5]).astype(int)
        enhanced_future['is_summer'] = enhanced_future['ds'].dt.month.isin([6, 7, 8]).astype(int)
        enhanced_future['is_autumn'] = enhanced_future['ds'].dt.month.isin([9, 10, 11]).astype(int)
        
        # Holiday indicators (Nepal-specific holidays)
        enhanced_future = self._add_future_nepal_holidays(enhanced_future)
        
        return enhanced_future
    
    def _add_future_nepal_holidays(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """Add Nepal-specific holiday indicators to future dataframe."""
        
        enhanced_future = future_df.copy()
        
        # Major Nepal holidays (simplified)
        holidays = {
            'Dashain': [(9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20), (9, 21), (9, 22), (9, 23), (9, 24)],
            'Tihar': [(10, 8), (10, 9), (10, 10), (10, 11), (10, 12)],
            'New_Year': [(1, 1)],
            'Republic_Day': [(5, 29)],
            'Constitution_Day': [(9, 20)],
            'Independence_Day': [(8, 15)]
        }
        
        for holiday_name, dates in holidays.items():
            enhanced_future[f'holiday_{holiday_name}'] = 0
            for month, day in dates:
                enhanced_future[f'holiday_{holiday_name}'] += (
                    (enhanced_future['ds'].dt.month == month) & 
                    (enhanced_future['ds'].dt.day == day)
                ).astype(int)
        
        # General holiday indicator
        holiday_cols = [col for col in enhanced_future.columns if col.startswith('holiday_')]
        enhanced_future['is_holiday'] = enhanced_future[holiday_cols].sum(axis=1).astype(int)
        
        return enhanced_future
    
    def _add_future_pollution_regressors(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """Add pollution-related regressors to future dataframe (using defaults)."""
        
        enhanced_future = future_df.copy()
        
        # Use default values for future pollution features
        enhanced_future['pm25_lag_1h'] = 30.0  # Default PM2.5 level
        enhanced_future['pm25_lag_24h'] = 30.0
        enhanced_future['pm25_lag_168h'] = 30.0
        
        # Rolling statistics (using defaults)
        enhanced_future['pm25_rolling_24h_mean'] = 30.0
        enhanced_future['pm25_rolling_24h_std'] = 10.0
        enhanced_future['pm25_rolling_168h_mean'] = 30.0
        
        # Pollution level categories (default to moderate)
        enhanced_future['pm25_low'] = 0
        enhanced_future['pm25_moderate'] = 1
        enhanced_future['pm25_unhealthy'] = 0
        enhanced_future['pm25_very_unhealthy'] = 0
        
        return enhanced_future
    
    def _predict_fallback(self, model: Dict, timestamps: pd.DatetimeIndex, 
                         future_df: pd.DataFrame) -> Tuple[List[float], Dict, List[float], Dict]:
        """Generate predictions using fallback model."""
        
        predictions = []
        trend_values = []
        seasonal_daily = []
        seasonal_weekly = []
        
        base_time = timestamps[0] if len(timestamps) > 0 else pd.Timestamp.now()
        
        for i, ts in enumerate(timestamps):
            # Trend component
            hours_from_start = i
            trend = model['mean'] + model['trend_slope'] * hours_from_start
            
            # Seasonal components
            hour = ts.hour
            dow = ts.dayofweek
            
            hourly_seasonal = model['seasonal_patterns']['hourly'].get(hour, 0.0)
            weekly_seasonal = model['seasonal_patterns']['weekly'].get(dow, 0.0)
            
            # Combine components
            prediction = trend + hourly_seasonal + weekly_seasonal
            
            predictions.append(max(0.0, prediction))  # Ensure non-negative
            trend_values.append(trend)
            seasonal_daily.append(hourly_seasonal)
            seasonal_weekly.append(weekly_seasonal)
        
        # Simple confidence intervals
        std = model['std']
        confidence_intervals = {
            'q0.1': [max(0, pred - 1.28 * std) for pred in predictions],
            'q0.5': predictions.copy(),
            'q0.9': [pred + 1.28 * std for pred in predictions]
        }
        
        seasonal_components = {
            'daily': seasonal_daily,
            'weekly': seasonal_weekly
        }
        
        return predictions, confidence_intervals, trend_values, seasonal_components
    
    def get_model_info(self) -> Dict:
        """Get information about the fitted models."""
        
        if not self.is_fitted:
            return {"fitted": False}
        
        info = {
            "fitted": True,
            "model_type": "prophet" if self.prophet_available else "simplified_prophet",
            "prophet_available": self.prophet_available,
            "groups": list(self.models.keys()),
            "features_used": self.feature_columns,
            "config": self.config.model_dump()
        }
        
        return info
