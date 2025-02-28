"""Temporal feature generation for time series forecasting."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TemporalConfig(BaseModel):
    """Configuration for temporal feature generation."""
    lag_hours: List[int] = Field(default=[1, 2, 3, 6, 12, 24], description="Lag features to create")
    rolling_windows: List[int] = Field(default=[6, 12, 24], description="Rolling window sizes (hours)")
    rolling_functions: List[str] = Field(default=["mean", "std", "min", "max"], description="Rolling statistics")
    seasonal_periods: List[int] = Field(default=[24, 168], description="Seasonal periods (24=daily, 168=weekly)")
    include_calendar: bool = Field(True, description="Include calendar features")
    include_cyclical: bool = Field(True, description="Include cyclical encoding")


class TemporalFeatureGenerator:
    """Generate comprehensive temporal features for time series forecasting."""
    
    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()
        logger.info("TemporalFeatureGenerator initialized")
    
    def generate_features(self, df: pd.DataFrame, 
                         target_col: str = 'pm25',
                         time_col: str = 'datetime_utc',
                         group_col: Optional[str] = 'station_id') -> pd.DataFrame:
        """Generate comprehensive temporal features."""
        
        if df.empty:
            logger.warning("Empty dataframe provided")
            return df
        
        logger.info(f"Generating temporal features for {len(df):,} records")
        
        # Ensure proper datetime column
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Sort by time (and group if specified)
        sort_cols = [group_col, time_col] if group_col and group_col in df.columns else [time_col]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        
        # Generate different types of temporal features
        feature_generators = [
            self._add_lag_features,
            self._add_rolling_features,
            self._add_calendar_features,
            self._add_cyclical_features,
            self._add_seasonal_features,
            self._add_time_since_features,
            self._add_trend_features
        ]
        
        for generator in feature_generators:
            try:
                df = generator(df, target_col, time_col, group_col)
            except Exception as e:
                logger.warning(f"Failed to generate features with {generator.__name__}: {e}")
        
        # Log feature summary
        new_features = [col for col in df.columns if col.startswith(('lag_', 'rolling_', 'calendar_', 'cyclical_', 'seasonal_', 'time_since_', 'trend_'))]
        logger.info(f"Generated {len(new_features)} temporal features")
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, target_col: str, 
                         time_col: str, group_col: Optional[str]) -> pd.DataFrame:
        """Add lag features for the target variable."""
        
        logger.debug(f"Adding lag features: {self.config.lag_hours}")
        
        for lag_hours in self.config.lag_hours:
            feature_name = f"lag_{lag_hours}h_{target_col}"
            
            if group_col and group_col in df.columns:
                # Group-wise lagging for multiple stations
                df[feature_name] = df.groupby(group_col)[target_col].shift(lag_hours)
            else:
                # Simple lagging for single time series
                df[feature_name] = df[target_col].shift(lag_hours)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, target_col: str,
                             time_col: str, group_col: Optional[str]) -> pd.DataFrame:
        """Add rolling window statistics."""
        
        logger.debug(f"Adding rolling features: windows={self.config.rolling_windows}, functions={self.config.rolling_functions}")
        
        for window in self.config.rolling_windows:
            for func in self.config.rolling_functions:
                feature_name = f"rolling_{window}h_{func}_{target_col}"
                
                if group_col and group_col in df.columns:
                    # Group-wise rolling for multiple stations
                    rolling_series = df.groupby(group_col)[target_col].rolling(
                        window=window, min_periods=max(1, window//2)
                    )
                else:
                    # Simple rolling for single time series
                    rolling_series = df[target_col].rolling(
                        window=window, min_periods=max(1, window//2)
                    )
                
                if func == "mean":
                    df[feature_name] = rolling_series.mean().reset_index(level=0, drop=True) if group_col else rolling_series.mean()
                elif func == "std":
                    df[feature_name] = rolling_series.std().reset_index(level=0, drop=True) if group_col else rolling_series.std()
                elif func == "min":
                    df[feature_name] = rolling_series.min().reset_index(level=0, drop=True) if group_col else rolling_series.min()
                elif func == "max":
                    df[feature_name] = rolling_series.max().reset_index(level=0, drop=True) if group_col else rolling_series.max()
        
        return df
    
    def _add_calendar_features(self, df: pd.DataFrame, target_col: str,
                              time_col: str, group_col: Optional[str]) -> pd.DataFrame:
        """Add calendar-based features."""
        
        if not self.config.include_calendar:
            return df
        
        logger.debug("Adding calendar features")
        
        dt = df[time_col]
        
        # Basic calendar features
        df['calendar_hour'] = dt.dt.hour
        df['calendar_day_of_week'] = dt.dt.dayofweek  # 0=Monday, 6=Sunday
        df['calendar_day_of_month'] = dt.dt.day
        df['calendar_day_of_year'] = dt.dt.dayofyear
        df['calendar_week_of_year'] = dt.dt.isocalendar().week
        df['calendar_month'] = dt.dt.month
        df['calendar_quarter'] = dt.dt.quarter
        df['calendar_year'] = dt.dt.year
        
        # Boolean calendar features
        df['calendar_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        df['calendar_is_month_start'] = dt.dt.is_month_start.astype(int)
        df['calendar_is_month_end'] = dt.dt.is_month_end.astype(int)
        df['calendar_is_quarter_start'] = dt.dt.is_quarter_start.astype(int)
        df['calendar_is_quarter_end'] = dt.dt.is_quarter_end.astype(int)
        
        # Season features (for Nepal/Kathmandu)
        month = dt.dt.month
        df['calendar_season'] = month.map({
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,   # Spring  
            6: 2, 7: 2, 8: 2,   # Summer (Monsoon)
            9: 3, 10: 3, 11: 3  # Autumn (Post-monsoon)
        })
        
        # Nepal-specific features
        df['calendar_is_monsoon'] = ((month >= 6) & (month <= 8)).astype(int)
        df['calendar_is_dry_season'] = ((month >= 10) | (month <= 5)).astype(int)
        
        return df
    
    def _add_cyclical_features(self, df: pd.DataFrame, target_col: str,
                              time_col: str, group_col: Optional[str]) -> pd.DataFrame:
        """Add cyclical encoding of temporal features."""
        
        if not self.config.include_cyclical:
            return df
        
        logger.debug("Adding cyclical features")
        
        dt = df[time_col]
        
        # Hour of day (24-hour cycle)
        hour = dt.dt.hour
        df['cyclical_hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['cyclical_hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (7-day cycle)
        day_of_week = dt.dt.dayofweek
        df['cyclical_dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['cyclical_dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Day of year (365-day cycle)
        day_of_year = dt.dt.dayofyear
        df['cyclical_doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        df['cyclical_doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Month (12-month cycle)
        month = dt.dt.month
        df['cyclical_month_sin'] = np.sin(2 * np.pi * month / 12)
        df['cyclical_month_cos'] = np.cos(2 * np.pi * month / 12)
        
        return df
    
    def _add_seasonal_features(self, df: pd.DataFrame, target_col: str,
                              time_col: str, group_col: Optional[str]) -> pd.DataFrame:
        """Add seasonal decomposition features."""
        
        logger.debug(f"Adding seasonal features for periods: {self.config.seasonal_periods}")
        
        dt = df[time_col]
        
        for period in self.config.seasonal_periods:
            # Calculate hour of period
            total_hours = (dt - dt.min()).dt.total_seconds() / 3600
            hour_in_period = total_hours % period
            
            # Cyclical encoding of seasonal position
            df[f'seasonal_{period}h_sin'] = np.sin(2 * np.pi * hour_in_period / period)
            df[f'seasonal_{period}h_cos'] = np.cos(2 * np.pi * hour_in_period / period)
        
        return df
    
    def _add_time_since_features(self, df: pd.DataFrame, target_col: str,
                                time_col: str, group_col: Optional[str]) -> pd.DataFrame:
        """Add time since specific events features."""
        
        logger.debug("Adding time since features")
        
        dt = df[time_col]
        
        # Time since start of data (normalized)
        time_since_start = (dt - dt.min()).dt.total_seconds() / 3600
        df['time_since_start_hours'] = time_since_start
        df['time_since_start_norm'] = time_since_start / time_since_start.max() if time_since_start.max() > 0 else 0
        
        # Time since start of day
        start_of_day = dt.dt.normalize()
        df['time_since_day_start'] = (dt - start_of_day).dt.total_seconds() / 3600
        
        # Time since start of week
        start_of_week = dt.dt.to_period('W').dt.start_time
        df['time_since_week_start'] = (dt - start_of_week).dt.total_seconds() / 3600
        
        # Time since start of month
        start_of_month = dt.dt.to_period('M').dt.start_time
        df['time_since_month_start'] = (dt - start_of_month).dt.total_seconds() / 3600
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame, target_col: str,
                           time_col: str, group_col: Optional[str]) -> pd.DataFrame:
        """Add trend-based features."""
        
        logger.debug("Adding trend features")
        
        # Linear trend over entire series
        df['trend_linear'] = range(len(df))
        
        # Trend within groups if group column exists
        if group_col and group_col in df.columns:
            df['trend_within_group'] = df.groupby(group_col).cumcount()
        
        # Local trend (rate of change)
        window_sizes = [6, 24]  # 6-hour and 24-hour trends
        
        for window in window_sizes:
            feature_name = f'trend_{window}h_slope'
            
            # Calculate rolling linear regression slope
            if group_col and group_col in df.columns:
                df[feature_name] = df.groupby(group_col)[target_col].rolling(
                    window=window, min_periods=max(2, window//2)
                ).apply(self._calculate_slope, raw=False).reset_index(level=0, drop=True)
            else:
                df[feature_name] = df[target_col].rolling(
                    window=window, min_periods=max(2, window//2)
                ).apply(self._calculate_slope, raw=False)
        
        return df
    
    def _calculate_slope(self, y: pd.Series) -> float:
        """Calculate linear regression slope for a window."""
        if len(y) < 2 or y.isnull().all():
            return 0.0
        
        # Remove NaN values
        valid_data = y.dropna()
        if len(valid_data) < 2:
            return 0.0
        
        x = np.arange(len(valid_data))
        
        # Calculate slope using least squares
        try:
            slope = np.polyfit(x, valid_data.values, 1)[0]
            return float(slope)
        except (np.linalg.LinAlgError, np.RankWarning):
            return 0.0
    
    def get_feature_importance_categories(self) -> Dict[str, List[str]]:
        """Get feature categories for importance analysis."""
        
        categories = {
            "lag_features": [f"lag_{h}h_pm25" for h in self.config.lag_hours],
            "rolling_features": [
                f"rolling_{w}h_{f}_pm25" 
                for w in self.config.rolling_windows 
                for f in self.config.rolling_functions
            ],
            "calendar_features": [
                "calendar_hour", "calendar_day_of_week", "calendar_day_of_month",
                "calendar_month", "calendar_season", "calendar_is_weekend",
                "calendar_is_monsoon", "calendar_is_dry_season"
            ],
            "cyclical_features": [
                "cyclical_hour_sin", "cyclical_hour_cos",
                "cyclical_dow_sin", "cyclical_dow_cos",
                "cyclical_doy_sin", "cyclical_doy_cos",
                "cyclical_month_sin", "cyclical_month_cos"
            ],
            "seasonal_features": [
                f"seasonal_{p}h_sin" for p in self.config.seasonal_periods
            ] + [
                f"seasonal_{p}h_cos" for p in self.config.seasonal_periods
            ],
            "trend_features": [
                "trend_linear", "trend_within_group",
                "trend_6h_slope", "trend_24h_slope"
            ],
            "time_since_features": [
                "time_since_start_hours", "time_since_start_norm",
                "time_since_day_start", "time_since_week_start", "time_since_month_start"
            ]
        }
        
        return categories
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate generated temporal features."""
        
        validation_results = {
            "total_features": 0,
            "missing_values": {},
            "infinite_values": {},
            "constant_features": [],
            "correlated_features": [],
            "feature_ranges": {}
        }
        
        # Get temporal feature columns
        temporal_cols = [col for col in df.columns 
                        if any(col.startswith(prefix) for prefix in 
                              ['lag_', 'rolling_', 'calendar_', 'cyclical_', 'seasonal_', 'time_since_', 'trend_'])]
        
        validation_results["total_features"] = len(temporal_cols)
        
        for col in temporal_cols:
            # Check missing values
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                validation_results["missing_values"][col] = missing_count
            
            # Check infinite values
            if df[col].dtype in ['float64', 'float32']:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    validation_results["infinite_values"][col] = inf_count
            
            # Check constant features
            if df[col].nunique() <= 1:
                validation_results["constant_features"].append(col)
            
            # Feature ranges
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                validation_results["feature_ranges"][col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std())
                }
        
        # Check for highly correlated features
        if len(temporal_cols) > 1:
            corr_matrix = df[temporal_cols].corr().abs()
            # Find pairs with correlation > 0.95
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            validation_results["correlated_features"] = high_corr_pairs
        
        logger.info(f"Temporal feature validation complete: {len(temporal_cols)} features analyzed")
        
        return validation_results


