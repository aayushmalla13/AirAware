"""Meteorological feature engineering for air quality forecasting."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MeteorologicalConfig(BaseModel):
    """Configuration for meteorological feature generation."""
    wind_features: bool = Field(True, description="Generate wind-based features")
    stability_features: bool = Field(True, description="Generate atmospheric stability features")
    comfort_features: bool = Field(True, description="Generate human comfort indices")
    pollution_potential: bool = Field(True, description="Generate pollution potential indicators")
    lag_hours: List[int] = Field(default=[1, 3, 6, 12], description="Meteorological lags")
    rolling_windows: List[int] = Field(default=[6, 12, 24], description="Rolling windows for met variables")


class MeteorologicalFeatureGenerator:
    """Generate meteorological features for air quality prediction."""
    
    def __init__(self, config: Optional[MeteorologicalConfig] = None):
        self.config = config or MeteorologicalConfig()
        logger.info("MeteorologicalFeatureGenerator initialized")
    
    def generate_features(self, df: pd.DataFrame, 
                         group_col: Optional[str] = 'station_id') -> pd.DataFrame:
        """Generate comprehensive meteorological features."""
        
        if df.empty:
            logger.warning("Empty dataframe provided")
            return df
        
        logger.info(f"Generating meteorological features for {len(df):,} records")
        
        df = df.copy()
        
        # Sort data properly
        sort_cols = [group_col, 'datetime_utc'] if group_col and group_col in df.columns else ['datetime_utc']
        df = df.sort_values(sort_cols).reset_index(drop=True)
        
        # Generate different types of meteorological features
        feature_generators = [
            self._add_wind_features,
            self._add_temperature_features,
            self._add_stability_features,
            self._add_comfort_indices,
            self._add_pollution_potential,
            self._add_meteorological_lags,
            self._add_meteorological_rolling,
            self._add_weather_patterns,
            self._add_boundary_layer_features
        ]
        
        for generator in feature_generators:
            try:
                df = generator(df, group_col)
            except Exception as e:
                logger.warning(f"Failed to generate features with {generator.__name__}: {e}")
        
        # Log feature summary
        met_features = [col for col in df.columns if col.startswith(('wind_', 'temp_', 'stability_', 'comfort_', 'pollution_', 'met_lag_', 'met_rolling_', 'weather_', 'bl_'))]
        logger.info(f"Generated {len(met_features)} meteorological features")
        
        return df
    
    def _add_wind_features(self, df: pd.DataFrame, group_col: Optional[str]) -> pd.DataFrame:
        """Add comprehensive wind-based features."""
        
        if not self.config.wind_features:
            return df
        
        logger.debug("Adding wind features")
        
        # Basic wind features (should already exist from ERA5)
        required_cols = ['u10', 'v10', 'wind_speed', 'wind_direction']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing wind columns for advanced features: {missing_cols}")
            return df
        
        # Wind categorization
        df['wind_speed_category'] = pd.cut(
            df['wind_speed'], 
            bins=[0, 1, 3, 5, 8, 12, float('inf')],
            labels=['calm', 'light', 'gentle', 'moderate', 'fresh', 'strong'],
            include_lowest=True
        ).astype(str)
        
        # Wind direction sectors (16 sectors)
        df['wind_direction_sector'] = ((df['wind_direction'] + 11.25) // 22.5) % 16
        
        # Wind consistency (variability in recent hours)
        rolling_windows = [6, 12, 24]
        
        for window in rolling_windows:
            if group_col and group_col in df.columns:
                # Wind speed variability
                df[f'wind_speed_std_{window}h'] = df.groupby(group_col)['wind_speed'].rolling(
                    window=window, min_periods=max(1, window//2)
                ).std().reset_index(level=0, drop=True)
                
                # Wind direction variability (circular statistics)
                df[f'wind_direction_consistency_{window}h'] = df.groupby(group_col).apply(
                    lambda x: x['wind_direction'].rolling(window=window, min_periods=max(1, window//2)).apply(
                        self._calculate_wind_direction_consistency
                    )
                ).reset_index(level=0, drop=True)
            else:
                df[f'wind_speed_std_{window}h'] = df['wind_speed'].rolling(
                    window=window, min_periods=max(1, window//2)
                ).std()
                
                df[f'wind_direction_consistency_{window}h'] = df['wind_direction'].rolling(
                    window=window, min_periods=max(1, window//2)
                ).apply(self._calculate_wind_direction_consistency)
        
        # Wind shear proxy (change in wind speed)
        for lag in [1, 3, 6]:
            if group_col and group_col in df.columns:
                df[f'wind_shear_{lag}h'] = df.groupby(group_col)['wind_speed'].diff(lag)
            else:
                df[f'wind_shear_{lag}h'] = df['wind_speed'].diff(lag)
        
        # Predominant wind direction (most common in last 24h)
        if group_col and group_col in df.columns:
            df['wind_direction_mode_24h'] = df.groupby(group_col)['wind_direction_sector'].rolling(
                window=24, min_periods=12
            ).apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[-1]).reset_index(level=0, drop=True)
        else:
            df['wind_direction_mode_24h'] = df['wind_direction_sector'].rolling(
                window=24, min_periods=12
            ).apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[-1])
        
        return df
    
    def _add_temperature_features(self, df: pd.DataFrame, group_col: Optional[str]) -> pd.DataFrame:
        """Add temperature-based features."""
        
        logger.debug("Adding temperature features")
        
        if 't2m_celsius' not in df.columns:
            logger.warning("No temperature data available for temperature features")
            return df
        
        # Temperature categories
        df['temp_category'] = pd.cut(
            df['t2m_celsius'],
            bins=[-float('inf'), 5, 15, 25, 35, float('inf')],
            labels=['cold', 'cool', 'mild', 'warm', 'hot']
        ).astype(str)
        
        # Daily temperature range proxy
        rolling_windows = [24, 48]
        
        for window in rolling_windows:
            if group_col and group_col in df.columns:
                temp_rolling = df.groupby(group_col)['t2m_celsius'].rolling(
                    window=window, min_periods=max(6, window//4)
                )
                df[f'temp_range_{window}h'] = (temp_rolling.max() - temp_rolling.min()).reset_index(level=0, drop=True)
                df[f'temp_amplitude_{window}h'] = (df['t2m_celsius'] - temp_rolling.mean()).reset_index(level=0, drop=True)
            else:
                temp_rolling = df['t2m_celsius'].rolling(window=window, min_periods=max(6, window//4))
                df[f'temp_range_{window}h'] = temp_rolling.max() - temp_rolling.min()
                df[f'temp_amplitude_{window}h'] = df['t2m_celsius'] - temp_rolling.mean()
        
        # Temperature gradient (rate of change)
        for lag in [1, 3, 6]:
            if group_col and group_col in df.columns:
                df[f'temp_gradient_{lag}h'] = df.groupby(group_col)['t2m_celsius'].diff(lag) / lag
            else:
                df[f'temp_gradient_{lag}h'] = df['t2m_celsius'].diff(lag) / lag
        
        return df
    
    def _add_stability_features(self, df: pd.DataFrame, group_col: Optional[str]) -> pd.DataFrame:
        """Add atmospheric stability features."""
        
        if not self.config.stability_features:
            return df
        
        logger.debug("Adding atmospheric stability features")
        
        required_cols = ['wind_speed', 't2m_celsius', 'blh']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for stability features: {missing_cols}")
            return df
        
        # Richardson number proxy (stability indicator)
        # Simplified: based on temperature gradient and wind speed
        df['stability_richardson_proxy'] = np.where(
            df['wind_speed'] > 0,
            df.get('temp_gradient_1h', 0) / (df['wind_speed'] ** 2 + 0.1),
            0
        )
        
        # Atmospheric stability classes (simplified Pasquill)
        # Based on wind speed, time of day, and cloud cover proxy
        hour = df['datetime_utc'].dt.hour if 'datetime_utc' in df.columns else 12
        
        stability_conditions = [
            (df['wind_speed'] < 2) & (hour.between(6, 18)),  # Unstable (day, low wind)
            (df['wind_speed'] < 2) & (~hour.between(6, 18)), # Stable (night, low wind)
            (df['wind_speed'].between(2, 5)) & (hour.between(6, 18)), # Neutral-unstable
            (df['wind_speed'].between(2, 5)) & (~hour.between(6, 18)), # Neutral-stable
            df['wind_speed'] > 5  # Neutral (high wind)
        ]
        
        stability_classes = ['unstable', 'stable', 'neutral_unstable', 'neutral_stable', 'neutral']
        df['stability_class'] = pd.Series(dtype=str)
        
        for condition, stability_class in zip(stability_conditions, stability_classes):
            df.loc[condition, 'stability_class'] = stability_class
        
        df['stability_class'] = df['stability_class'].fillna('neutral')
        
        # Mixing potential (combination of wind and boundary layer height)
        df['mixing_potential'] = df['wind_speed'] * np.log(df['blh'] + 1)
        
        # Ventilation coefficient
        df['ventilation_coefficient'] = df['wind_speed'] * df['blh']
        
        # Stability indicator (0-1 scale, 0=stable, 1=unstable)
        df['stability_indicator'] = np.clip(
            (df['wind_speed'] / 10.0 + 
             np.where(hour.between(6, 18), 0.3, -0.3) +
             np.log(df['blh'] + 1) / 10.0), 
            0, 1
        )
        
        return df
    
    def _add_comfort_indices(self, df: pd.DataFrame, group_col: Optional[str]) -> pd.DataFrame:
        """Add human comfort indices that correlate with air quality."""
        
        if not self.config.comfort_features:
            return df
        
        logger.debug("Adding comfort indices")
        
        if 't2m_celsius' not in df.columns or 'wind_speed' not in df.columns:
            logger.warning("Missing data for comfort indices")
            return df
        
        # Wind chill index (when temperature < 10°C)
        temp_c = df['t2m_celsius']
        wind_kmh = df['wind_speed'] * 3.6  # Convert m/s to km/h
        
        wind_chill_condition = temp_c < 10
        df['comfort_wind_chill'] = np.where(
            wind_chill_condition,
            13.12 + 0.6215 * temp_c - 11.37 * (wind_kmh ** 0.16) + 0.3965 * temp_c * (wind_kmh ** 0.16),
            temp_c
        )
        
        # Heat index (when temperature > 20°C and assuming some humidity)
        # Simplified version using temperature only
        heat_index_condition = temp_c > 20
        df['comfort_heat_index'] = np.where(
            heat_index_condition,
            temp_c + 0.1 * (temp_c - 20) ** 2,  # Simplified heat index
            temp_c
        )
        
        # Effective temperature (combines temperature and wind)
        df['comfort_effective_temp'] = temp_c - 0.4 * (temp_c - 10) * (wind_kmh / 10) ** 0.16
        
        return df
    
    def _add_pollution_potential(self, df: pd.DataFrame, group_col: Optional[str]) -> pd.DataFrame:
        """Add features specifically related to pollution potential."""
        
        if not self.config.pollution_potential:
            return df
        
        logger.debug("Adding pollution potential features")
        
        required_cols = ['wind_speed', 'blh']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for pollution potential: {missing_cols}")
            return df
        
        # Dispersion potential (higher = better dispersion)
        df['pollution_dispersion_potential'] = (
            np.log(df['wind_speed'] + 0.1) * 
            np.log(df['blh'] + 1) / 10.0
        )
        
        # Accumulation potential (inverse of dispersion)
        df['pollution_accumulation_potential'] = 1.0 / (df['pollution_dispersion_potential'] + 0.1)
        
        # Stagnation index (low wind + low mixing)
        df['pollution_stagnation_index'] = np.where(
            (df['wind_speed'] < 2) & (df['blh'] < 500),
            1.0,
            0.0
        )
        
        # Peak traffic alignment (higher pollution potential during traffic hours)
        hour = df['datetime_utc'].dt.hour if 'datetime_utc' in df.columns else 12
        traffic_hours = hour.isin([7, 8, 9, 17, 18, 19, 20])  # Morning and evening rush
        
        df['pollution_traffic_alignment'] = np.where(
            traffic_hours & (df['wind_speed'] < 3),
            1.0,
            0.0
        )
        
        return df
    
    def _add_meteorological_lags(self, df: pd.DataFrame, group_col: Optional[str]) -> pd.DataFrame:
        """Add lagged meteorological variables."""
        
        logger.debug(f"Adding meteorological lags: {self.config.lag_hours}")
        
        met_variables = ['wind_speed', 'wind_direction', 't2m_celsius', 'blh']
        available_vars = [var for var in met_variables if var in df.columns]
        
        for var in available_vars:
            for lag in self.config.lag_hours:
                feature_name = f"met_lag_{lag}h_{var}"
                
                if group_col and group_col in df.columns:
                    df[feature_name] = df.groupby(group_col)[var].shift(lag)
                else:
                    df[feature_name] = df[var].shift(lag)
        
        return df
    
    def _add_meteorological_rolling(self, df: pd.DataFrame, group_col: Optional[str]) -> pd.DataFrame:
        """Add rolling statistics for meteorological variables."""
        
        logger.debug(f"Adding meteorological rolling features: {self.config.rolling_windows}")
        
        met_variables = ['wind_speed', 't2m_celsius', 'blh']
        available_vars = [var for var in met_variables if var in df.columns]
        
        for var in available_vars:
            for window in self.config.rolling_windows:
                for stat in ['mean', 'std', 'min', 'max']:
                    feature_name = f"met_rolling_{window}h_{stat}_{var}"
                    
                    if group_col and group_col in df.columns:
                        rolling_series = df.groupby(group_col)[var].rolling(
                            window=window, min_periods=max(1, window//2)
                        )
                    else:
                        rolling_series = df[var].rolling(
                            window=window, min_periods=max(1, window//2)
                        )
                    
                    if stat == 'mean':
                        df[feature_name] = rolling_series.mean().reset_index(level=0, drop=True) if group_col else rolling_series.mean()
                    elif stat == 'std':
                        df[feature_name] = rolling_series.std().reset_index(level=0, drop=True) if group_col else rolling_series.std()
                    elif stat == 'min':
                        df[feature_name] = rolling_series.min().reset_index(level=0, drop=True) if group_col else rolling_series.min()
                    elif stat == 'max':
                        df[feature_name] = rolling_series.max().reset_index(level=0, drop=True) if group_col else rolling_series.max()
        
        return df
    
    def _add_weather_patterns(self, df: pd.DataFrame, group_col: Optional[str]) -> pd.DataFrame:
        """Add weather pattern recognition features."""
        
        logger.debug("Adding weather pattern features")
        
        # Temperature and wind patterns
        if 't2m_celsius' in df.columns and 'wind_speed' in df.columns:
            
            # Hot and calm conditions (poor air quality potential)
            df['weather_hot_calm'] = (
                (df['t2m_celsius'] > 25) & (df['wind_speed'] < 2)
            ).astype(int)
            
            # Cold and windy conditions (better air quality potential)
            df['weather_cold_windy'] = (
                (df['t2m_celsius'] < 15) & (df['wind_speed'] > 5)
            ).astype(int)
            
            # Temperature inversion conditions (very poor air quality potential)
            if group_col and group_col in df.columns:
                temp_trend = df.groupby(group_col)['t2m_celsius'].diff(6)  # 6-hour temperature change
            else:
                temp_trend = df['t2m_celsius'].diff(6)
            
            df['weather_inversion_potential'] = (
                (temp_trend > 0) & (df['wind_speed'] < 1.5) & 
                (df['datetime_utc'].dt.hour < 10 if 'datetime_utc' in df.columns else True)
            ).astype(int)
        
        return df
    
    def _add_boundary_layer_features(self, df: pd.DataFrame, group_col: Optional[str]) -> pd.DataFrame:
        """Add boundary layer specific features."""
        
        logger.debug("Adding boundary layer features")
        
        if 'blh' not in df.columns:
            logger.warning("No boundary layer height data available")
            return df
        
        # Boundary layer height categories
        df['bl_height_category'] = pd.cut(
            df['blh'],
            bins=[0, 200, 500, 1000, 2000, float('inf')],
            labels=['very_low', 'low', 'moderate', 'high', 'very_high']
        ).astype(str)
        
        # Boundary layer development (growth/shrinkage)
        for lag in [1, 3, 6]:
            if group_col and group_col in df.columns:
                df[f'bl_development_{lag}h'] = df.groupby(group_col)['blh'].diff(lag)
            else:
                df[f'bl_development_{lag}h'] = df['blh'].diff(lag)
        
        # Normalized boundary layer height (relative to daily max/min)
        rolling_window = 24
        
        if group_col and group_col in df.columns:
            bl_rolling = df.groupby(group_col)['blh'].rolling(
                window=rolling_window, min_periods=12, center=True
            )
            bl_min = bl_rolling.min().reset_index(level=0, drop=True)
            bl_max = bl_rolling.max().reset_index(level=0, drop=True)
        else:
            bl_rolling = df['blh'].rolling(window=rolling_window, min_periods=12, center=True)
            bl_min = bl_rolling.min()
            bl_max = bl_rolling.max()
        
        df['bl_height_normalized'] = np.where(
            (bl_max - bl_min) > 0,
            (df['blh'] - bl_min) / (bl_max - bl_min),
            0.5
        )
        
        return df
    
    def _calculate_wind_direction_consistency(self, directions: pd.Series) -> float:
        """Calculate wind direction consistency using circular statistics."""
        
        if directions.empty or directions.isnull().all():
            return 0.0
        
        # Convert degrees to radians
        directions_clean = directions.dropna()
        if len(directions_clean) < 2:
            return 1.0  # Single direction is perfectly consistent
        
        radians = np.radians(directions_clean)
        
        # Calculate mean direction vector components
        mean_sin = np.mean(np.sin(radians))
        mean_cos = np.mean(np.cos(radians))
        
        # Vector strength (0 = no consistency, 1 = perfect consistency)
        vector_strength = np.sqrt(mean_sin**2 + mean_cos**2)
        
        return float(vector_strength)
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """Get meteorological feature categories for analysis."""
        
        categories = {
            "wind_features": [
                "wind_speed_category", "wind_direction_sector", "wind_direction_mode_24h",
                "wind_speed_std_6h", "wind_speed_std_12h", "wind_speed_std_24h",
                "wind_direction_consistency_6h", "wind_direction_consistency_12h", "wind_direction_consistency_24h",
                "wind_shear_1h", "wind_shear_3h", "wind_shear_6h"
            ],
            "temperature_features": [
                "temp_category", "temp_range_24h", "temp_range_48h",
                "temp_amplitude_24h", "temp_amplitude_48h",
                "temp_gradient_1h", "temp_gradient_3h", "temp_gradient_6h"
            ],
            "stability_features": [
                "stability_richardson_proxy", "stability_class", "stability_indicator",
                "mixing_potential", "ventilation_coefficient"
            ],
            "comfort_features": [
                "comfort_wind_chill", "comfort_heat_index", "comfort_effective_temp"
            ],
            "pollution_potential": [
                "pollution_dispersion_potential", "pollution_accumulation_potential",
                "pollution_stagnation_index", "pollution_traffic_alignment"
            ],
            "boundary_layer_features": [
                "bl_height_category", "bl_height_normalized",
                "bl_development_1h", "bl_development_3h", "bl_development_6h"
            ],
            "weather_patterns": [
                "weather_hot_calm", "weather_cold_windy", "weather_inversion_potential"
            ]
        }
        
        # Add lag and rolling features dynamically
        met_variables = ['wind_speed', 'wind_direction', 't2m_celsius', 'blh']
        
        categories["meteorological_lags"] = [
            f"met_lag_{lag}h_{var}" 
            for var in met_variables 
            for lag in self.config.lag_hours
        ]
        
        categories["meteorological_rolling"] = [
            f"met_rolling_{window}h_{stat}_{var}"
            for var in ['wind_speed', 't2m_celsius', 'blh']
            for window in self.config.rolling_windows
            for stat in ['mean', 'std', 'min', 'max']
        ]
        
        return categories


