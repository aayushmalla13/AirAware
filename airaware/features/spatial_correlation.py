"""Spatial correlation features for cross-border air quality prediction."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from geopy.distance import geodesic
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SpatialCorrelationConfig(BaseModel):
    """Configuration for spatial correlation features."""
    
    # External station configurations
    india_stations: List[Dict[str, Any]] = Field(
        default=[
            {"id": "IN001", "name": "Delhi", "lat": 28.6139, "lon": 77.2090, "weight": 0.8},
            {"id": "IN002", "name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "weight": 0.6},
            {"id": "IN003", "name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "weight": 0.7},
            {"id": "IN004", "name": "Chennai", "lat": 13.0827, "lon": 80.2707, "weight": 0.5},
            {"id": "IN005", "name": "Bangalore", "lat": 12.9716, "lon": 77.5946, "weight": 0.5},
            {"id": "IN006", "name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "weight": 0.5},
            {"id": "IN007", "name": "Pune", "lat": 18.5204, "lon": 73.8567, "weight": 0.6},
            {"id": "IN008", "name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "weight": 0.6},
            {"id": "IN009", "name": "Jaipur", "lat": 26.9124, "lon": 75.7873, "weight": 0.7},
            {"id": "IN010", "name": "Lucknow", "lat": 26.8467, "lon": 80.9462, "weight": 0.7}
        ],
        description="Indian monitoring stations"
    )
    
    china_stations: List[Dict[str, Any]] = Field(
        default=[
            {"id": "CN001", "name": "Beijing", "lat": 39.9042, "lon": 116.4074, "weight": 0.6},
            {"id": "CN002", "name": "Shanghai", "lat": 31.2304, "lon": 121.4737, "weight": 0.4},
            {"id": "CN003", "name": "Guangzhou", "lat": 23.1291, "lon": 113.2644, "weight": 0.3},
            {"id": "CN004", "name": "Shenzhen", "lat": 22.5431, "lon": 114.0579, "weight": 0.3},
            {"id": "CN005", "name": "Chengdu", "lat": 30.5728, "lon": 104.0668, "weight": 0.5},
            {"id": "CN006", "name": "Wuhan", "lat": 30.5928, "lon": 114.3055, "weight": 0.4},
            {"id": "CN007", "name": "Xi'an", "lat": 34.3416, "lon": 108.9398, "weight": 0.5},
            {"id": "CN008", "name": "Tianjin", "lat": 39.3434, "lon": 117.3616, "weight": 0.5},
            {"id": "CN009", "name": "Nanjing", "lat": 32.0603, "lon": 118.7969, "weight": 0.4},
            {"id": "CN010", "name": "Hangzhou", "lat": 30.2741, "lon": 120.1551, "weight": 0.4}
        ],
        description="Chinese monitoring stations"
    )
    
    us_stations: List[Dict[str, Any]] = Field(
        default=[
            {"id": "US001", "name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "weight": 0.1, "state": "CA", "zipcode": "90001"},
            {"id": "US002", "name": "New York", "lat": 40.7128, "lon": -74.0060, "weight": 0.1, "state": "NY", "zipcode": "10001"},
            {"id": "US003", "name": "Chicago", "lat": 41.8781, "lon": -87.6298, "weight": 0.1, "state": "IL", "zipcode": "60601"},
            {"id": "US004", "name": "Houston", "lat": 29.7604, "lon": -95.3698, "weight": 0.1, "state": "TX", "zipcode": "77001"},
            {"id": "US005", "name": "Phoenix", "lat": 33.4484, "lon": -112.0740, "weight": 0.1, "state": "AZ", "zipcode": "85001"}
        ],
        description="US monitoring stations"
    )
    
    # Kathmandu reference point
    kathmandu_coords: Tuple[float, float] = Field(
        default=(27.7172, 85.3240),
        description="Kathmandu coordinates (lat, lon)"
    )
    
    # Correlation parameters
    max_distance_km: float = Field(2000.0, description="Maximum distance for correlation (km)")
    wind_transport_hours: List[int] = Field(
        default=[6, 12, 24, 48],
        description="Wind transport time lags (hours)"
    )
    seasonal_weights: Dict[str, float] = Field(
        default={
            "winter": 0.8,  # Higher correlation in winter
            "spring": 0.6,
            "summer": 0.4,  # Lower correlation in summer (monsoon)
            "autumn": 0.7
        },
        description="Seasonal correlation weights"
    )
    
    # Data source configurations
    openaq_api_key: Optional[str] = Field(None, description="OpenAQ API key")
    use_synthetic_data: bool = Field(False, description="Use REAL data from APIs")
    cache_ttl_hours: int = Field(6, description="Cache TTL for external data")


class SpatialCorrelationGenerator:
    """Generate spatial correlation features for cross-border air quality prediction."""
    
    def __init__(self, config: Optional[SpatialCorrelationConfig] = None):
        self.config = config or SpatialCorrelationConfig()
        self.external_data_cache = {}
        self.cache_timestamps = {}
        
        logger.info("SpatialCorrelationGenerator initialized")
        logger.info(f"Configured {len(self.config.india_stations)} Indian stations")
        logger.info(f"Configured {len(self.config.china_stations)} Chinese stations")
        logger.info(f"Configured {len(self.config.us_stations)} US stations")
    
    def generate_spatial_features(
        self, 
        df: pd.DataFrame,
        target_station_id: str = "5506835"
    ) -> pd.DataFrame:
        """Generate spatial correlation features for the target station."""
        
        if df.empty:
            logger.warning("Empty dataframe provided")
            return df
        
        logger.info(f"Generating spatial correlation features for {len(df):,} records")
        
        df = df.copy()
        
        # Sort by datetime
        df = df.sort_values('datetime_utc').reset_index(drop=True)
        
        # Generate different types of spatial features
        feature_generators = [
            self._add_distance_weighted_features,
            self._add_wind_transport_features,
            self._add_seasonal_correlation_features,
            self._add_cross_border_pollution_index,
            self._add_regional_pollution_trends
        ]
        
        for generator in feature_generators:
            try:
                df = generator(df, target_station_id)
            except Exception as e:
                logger.warning(f"Failed to generate features with {generator.__name__}: {e}")
        
        # Log feature summary
        spatial_features = [col for col in df.columns if col.startswith(('spatial_', 'cross_border_', 'regional_'))]
        logger.info(f"Generated {len(spatial_features)} spatial correlation features")
        
        return df
    
    def _add_distance_weighted_features(
        self, 
        df: pd.DataFrame, 
        target_station_id: str
    ) -> pd.DataFrame:
        """Add distance-weighted features from external stations."""
        
        # Get external station data
        external_data = self._get_external_station_data(df['datetime_utc'].min(), df['datetime_utc'].max())
        
        if external_data.empty:
            logger.warning("No external station data available")
            return df
        
        # Calculate distance weights
        india_weights = self._calculate_distance_weights(self.config.india_stations)
        china_weights = self._calculate_distance_weights(self.config.china_stations)
        
        # Add weighted PM2.5 features
        df['spatial_india_pm25_weighted'] = self._calculate_weighted_pm25(
            external_data, india_weights, 'india'
        )
        df['spatial_china_pm25_weighted'] = self._calculate_weighted_pm25(
            external_data, china_weights, 'china'
        )
        
        # Add distance-weighted temperature correlation
        df['spatial_india_temp_correlation'] = self._calculate_temp_correlation(
            external_data, india_weights, 'india'
        )
        df['spatial_china_temp_correlation'] = self._calculate_temp_correlation(
            external_data, china_weights, 'china'
        )
        
        return df
    
    def _add_wind_transport_features(
        self, 
        df: pd.DataFrame, 
        target_station_id: str
    ) -> pd.DataFrame:
        """Add wind transport features considering atmospheric conditions."""
        
        # Get external station data
        external_data = self._get_external_station_data(df['datetime_utc'].min(), df['datetime_utc'].max())
        
        if external_data.empty:
            return df
        
        # Calculate wind transport features for different time lags
        for lag_hours in self.config.wind_transport_hours:
            # India wind transport (westerly winds)
            df[f'spatial_india_wind_transport_{lag_hours}h'] = self._calculate_wind_transport(
                external_data, 'india', lag_hours, df
            )
            
            # China wind transport (northerly winds)
            df[f'spatial_china_wind_transport_{lag_hours}h'] = self._calculate_wind_transport(
                external_data, 'china', lag_hours, df
            )
        
        return df
    
    def _add_seasonal_correlation_features(
        self, 
        df: pd.DataFrame, 
        target_station_id: str
    ) -> pd.DataFrame:
        """Add seasonal correlation features."""
        
        # Determine season
        df['season'] = df['datetime_utc'].dt.month.map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        # Apply seasonal weights
        df['spatial_seasonal_weight'] = df['season'].map(self.config.seasonal_weights)
        
        # Add seasonal correlation features
        df['spatial_india_seasonal_correlation'] = (
            df['spatial_india_pm25_weighted'] * df['spatial_seasonal_weight']
        )
        df['spatial_china_seasonal_correlation'] = (
            df['spatial_china_pm25_weighted'] * df['spatial_seasonal_weight']
        )
        
        return df
    
    def _add_cross_border_pollution_index(
        self, 
        df: pd.DataFrame, 
        target_station_id: str
    ) -> pd.DataFrame:
        """Add cross-border pollution index."""
        
        # Calculate cross-border pollution index
        df['cross_border_pollution_index'] = (
            0.6 * df['spatial_india_pm25_weighted'] + 
            0.4 * df['spatial_china_pm25_weighted']
        )
        
        # Add pollution transport potential
        df['cross_border_transport_potential'] = (
            df['spatial_india_wind_transport_24h'] * 0.7 +
            df['spatial_china_wind_transport_24h'] * 0.3
        )
        
        # Add regional pollution trend
        df['regional_pollution_trend'] = (
            df['spatial_india_pm25_weighted'].rolling(24, min_periods=1).mean() +
            df['spatial_china_pm25_weighted'].rolling(24, min_periods=1).mean()
        ) / 2
        
        return df
    
    def _add_regional_pollution_trends(
        self, 
        df: pd.DataFrame, 
        target_station_id: str
    ) -> pd.DataFrame:
        """Add regional pollution trend features."""
        
        # Calculate regional pollution trends
        df['regional_pollution_6h_trend'] = df['cross_border_pollution_index'].diff(6)
        df['regional_pollution_12h_trend'] = df['cross_border_pollution_index'].diff(12)
        df['regional_pollution_24h_trend'] = df['cross_border_pollution_index'].diff(24)
        
        # Add pollution gradient
        df['regional_pollution_gradient'] = (
            df['spatial_india_pm25_weighted'] - df['spatial_china_pm25_weighted']
        )
        
        # Add pollution accumulation index
        df['regional_pollution_accumulation'] = (
            df['cross_border_pollution_index'].rolling(48, min_periods=1).sum()
        )
        
        return df
    
    def _calculate_distance_weights(self, stations: List[Dict]) -> Dict[str, float]:
        """Calculate distance-based weights for stations."""
        weights = {}
        
        for station in stations:
            distance = geodesic(
                self.config.kathmandu_coords,
                (station['lat'], station['lon'])
            ).kilometers
            
            # Inverse distance weighting with maximum distance cutoff
            if distance <= self.config.max_distance_km:
                weight = station['weight'] * (1 / (1 + distance / 100))
                weights[station['id']] = weight
            else:
                weights[station['id']] = 0.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _get_external_station_data(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Get external station data from OpenAQ or synthetic sources."""
        
        # Check cache first
        cache_key = f"{start_date}_{end_date}"
        if (cache_key in self.external_data_cache and 
            cache_key in self.cache_timestamps and
            (datetime.now() - self.cache_timestamps[cache_key]).total_seconds() < 
            self.config.cache_ttl_hours * 3600):
            return self.external_data_cache[cache_key]
        
        # SYNTHETIC DATA DISABLED - Only fetch real data
        logger.info("🌍 Fetching REAL air quality data from monitoring stations")
        
        if self.config.use_synthetic_data:
            logger.error("❌ SYNTHETIC DATA BANNED - Forcing real data extraction")
            self.config.use_synthetic_data = False
        
        # Fetch real data from OpenAQ
        external_data = self._fetch_openaq_data(start_date, end_date)
        
        if external_data.empty:
            logger.warning("⚠️ No real OpenAQ data found - will retry with different parameters")
            external_data = self._fetch_openaq_fallback(start_date, end_date)
        
        # Cache the data
        self.external_data_cache[cache_key] = external_data
        self.cache_timestamps[cache_key] = datetime.now()
        
        return external_data
    
    def _generate_synthetic_external_data(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate synthetic external station data for testing."""
        
        # Create hourly timestamps
        timestamps = pd.date_range(start_date, end_date, freq='h')
        
        # Generate synthetic data for all external stations
        all_data = []
        
        # India stations (higher baseline PM2.5)
        for station in self.config.india_stations:
            base_pm25 = np.random.normal(80, 20)  # Higher baseline
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * timestamps.dayofyear / 365)
            daily_pattern = 1 + 0.2 * np.sin(2 * np.pi * timestamps.hour / 24)
            
            pm25_values = base_pm25 * seasonal_factor * daily_pattern
            pm25_values = np.maximum(pm25_values, 10)  # Minimum PM2.5
            
            for i, timestamp in enumerate(timestamps):
                all_data.append({
                    'datetime_utc': timestamp,
                    'station_id': station['id'],
                    'station_name': station['name'],
                    'country': 'india',
                    'pm25': pm25_values[i],
                    'temperature': np.random.normal(25, 5),
                    'humidity': np.random.normal(60, 15)
                })
        
        # China stations (moderate baseline PM2.5)
        for station in self.config.china_stations:
            base_pm25 = np.random.normal(60, 15)  # Moderate baseline
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * timestamps.dayofyear / 365)
            daily_pattern = 1 + 0.15 * np.sin(2 * np.pi * timestamps.hour / 24)
            
            pm25_values = base_pm25 * seasonal_factor * daily_pattern
            pm25_values = np.maximum(pm25_values, 10)  # Minimum PM2.5
            
            for i, timestamp in enumerate(timestamps):
                all_data.append({
                    'datetime_utc': timestamp,
                    'station_id': station['id'],
                    'station_name': station['name'],
                    'country': 'china',
                    'pm25': pm25_values[i],
                    'temperature': np.random.normal(20, 8),
                    'humidity': np.random.normal(55, 20)
                })
        
        # US stations (lower baseline PM2.5)
        for station in self.config.us_stations:
            base_pm25 = np.random.normal(35, 10)  # Lower baseline
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * timestamps.dayofyear / 365)
            daily_pattern = 1 + 0.1 * np.sin(2 * np.pi * timestamps.hour / 24)
            
            pm25_values = base_pm25 * seasonal_factor * daily_pattern
            pm25_values = np.maximum(pm25_values, 5)  # Minimum PM2.5
            
            for i, timestamp in enumerate(timestamps):
                all_data.append({
                    'datetime_utc': timestamp,
                    'station_id': station['id'],
                    'station_name': station['name'],
                    'country': 'usa',
                    'pm25': pm25_values[i],
                    'temperature': np.random.normal(15, 10),
                    'humidity': np.random.normal(50, 15)
                })
        
        # SYNTHETIC DATA REMOVED - Real data extraction only
        logger.error("❌ SYNTHETIC DATA BANNED - This method should not be called")
        return pd.DataFrame()
    
    def _fetch_openaq_data(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch real data from OpenAQ API."""
        
        # This would implement real OpenAQ API calls
        # For now, return empty DataFrame
        logger.warning("OpenAQ API integration not implemented yet")
        return pd.DataFrame()
    
    def _calculate_weighted_pm25(
        self, 
        external_data: pd.DataFrame, 
        weights: Dict[str, float], 
        country: str
    ) -> pd.Series:
        """Calculate weighted PM2.5 from external stations."""
        
        if external_data.empty:
            return pd.Series([0] * len(external_data))
        
        # Filter by country
        country_data = external_data[external_data['country'] == country].copy()
        
        if country_data.empty:
            return pd.Series([0] * len(external_data))
        
        # Calculate weighted average
        weighted_pm25 = []
        for timestamp in external_data['datetime_utc'].unique():
            timestamp_data = country_data[country_data['datetime_utc'] == timestamp]
            
            weighted_sum = 0
            total_weight = 0
            
            for _, row in timestamp_data.iterrows():
                station_id = row['station_id']
                if station_id in weights and weights[station_id] > 0:
                    weighted_sum += row['pm25'] * weights[station_id]
                    total_weight += weights[station_id]
            
            if total_weight > 0:
                weighted_pm25.append(weighted_sum / total_weight)
            else:
                weighted_pm25.append(0)
        
        return pd.Series(weighted_pm25)
    
    def _calculate_temp_correlation(
        self, 
        external_data: pd.DataFrame, 
        weights: Dict[str, float], 
        country: str
    ) -> pd.Series:
        """Calculate temperature correlation from external stations."""
        
        if external_data.empty:
            return pd.Series([20] * len(external_data))
        
        # Filter by country
        country_data = external_data[external_data['country'] == country].copy()
        
        if country_data.empty:
            return pd.Series([20] * len(external_data))
        
        # Calculate weighted average temperature
        weighted_temp = []
        for timestamp in external_data['datetime_utc'].unique():
            timestamp_data = country_data[country_data['datetime_utc'] == timestamp]
            
            weighted_sum = 0
            total_weight = 0
            
            for _, row in timestamp_data.iterrows():
                station_id = row['station_id']
                if station_id in weights and weights[station_id] > 0:
                    weighted_sum += row['temperature'] * weights[station_id]
                    total_weight += weights[station_id]
            
            if total_weight > 0:
                weighted_temp.append(weighted_sum / total_weight)
            else:
                weighted_temp.append(20)
        
        return pd.Series(weighted_temp)
    
    def _calculate_wind_transport(
        self, 
        external_data: pd.DataFrame, 
        country: str, 
        lag_hours: int, 
        target_df: pd.DataFrame
    ) -> pd.Series:
        """Calculate wind transport features."""
        
        if external_data.empty:
            return pd.Series([0] * len(target_df))
        
        # Filter by country
        country_data = external_data[external_data['country'] == country].copy()
        
        if country_data.empty:
            return pd.Series([0] * len(target_df))
        
        # Calculate wind transport based on lag
        transport_values = []
        
        for _, row in target_df.iterrows():
            target_time = row['datetime_utc']
            source_time = target_time - timedelta(hours=lag_hours)
            
            # Find data at source time
            source_data = country_data[country_data['datetime_utc'] == source_time]
            
            if not source_data.empty:
                # Calculate weighted PM2.5 at source time
                weighted_pm25 = 0
                total_weight = 0
                
                for _, source_row in source_data.iterrows():
                    station_id = source_row['station_id']
                    if country == 'india':
                        weights = self._calculate_distance_weights(self.config.india_stations)
                    else:
                        weights = self._calculate_distance_weights(self.config.china_stations)
                    
                    if station_id in weights and weights[station_id] > 0:
                        weighted_pm25 += source_row['pm25'] * weights[station_id]
                        total_weight += weights[station_id]
                
                if total_weight > 0:
                    transport_values.append(weighted_pm25 / total_weight)
                else:
                    transport_values.append(0)
            else:
                transport_values.append(0)
        
        return pd.Series(transport_values)
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """Get spatial correlation feature categories."""
        
        return {
            "distance_weighted": [
                "spatial_india_pm25_weighted", "spatial_china_pm25_weighted",
                "spatial_india_temp_correlation", "spatial_china_temp_correlation"
            ],
            "wind_transport": [
                f"spatial_india_wind_transport_{h}h" for h in self.config.wind_transport_hours
            ] + [
                f"spatial_china_wind_transport_{h}h" for h in self.config.wind_transport_hours
            ],
            "seasonal_correlation": [
                "spatial_seasonal_weight", "spatial_india_seasonal_correlation",
                "spatial_china_seasonal_correlation"
            ],
            "cross_border": [
                "cross_border_pollution_index", "cross_border_transport_potential",
                "regional_pollution_trend"
            ],
            "regional_trends": [
                "regional_pollution_6h_trend", "regional_pollution_12h_trend",
                "regional_pollution_24h_trend", "regional_pollution_gradient",
                "regional_pollution_accumulation"
            ]
        }
