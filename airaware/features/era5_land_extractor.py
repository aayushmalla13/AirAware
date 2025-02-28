"""ERA5-Land weather data extractor for enhanced PM2.5 forecasting."""

import logging
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ERA5LandExtractor:
    """Extract ERA5-Land weather data for air quality forecasting."""
    
    def __init__(self, cache_dir: str = "data/cache/era5_land"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = 6  # Cache for 6 hours
        
        # ERA5-Land variables for air quality
        self.variables = {
            '2m_temperature': 't2m',  # Temperature at 2m
            '2m_dewpoint_temperature': 'd2m',  # Dewpoint at 2m
            '10m_u_component_of_wind': 'u10',  # U-component of wind
            '10m_v_component_of_wind': 'v10',  # V-component of wind
            'surface_pressure': 'sp',  # Surface pressure
            'total_precipitation': 'tp',  # Total precipitation
            'boundary_layer_height': 'blh',  # Boundary layer height
            'surface_solar_radiation_downwards': 'ssrd',  # Solar radiation
            'surface_thermal_radiation_downwards': 'strd',  # Thermal radiation
            'volumetric_surface_soil_moisture_layer_1': 'swvl1',  # Soil moisture
        }
        
        # Default coordinates for Kathmandu area
        self.default_coords = {
            'lat': 27.7172,
            'lon': 85.3240
        }
    
    def extract_era5_data(
        self, 
        start_date: datetime, 
        end_date: datetime,
        lat: float = None,
        lon: float = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Extract ERA5-Land data for the specified time range and location."""
        
        if lat is None:
            lat = self.default_coords['lat']
        if lon is None:
            lon = self.default_coords['lon']
        
        # Check cache first
        if use_cache:
            cached_data = self._load_from_cache(start_date, end_date, lat, lon)
            if cached_data is not None:
                logger.info(f"Using cached ERA5-Land data for {lat:.2f}, {lon:.2f}")
                return cached_data
        
        logger.info(f"Extracting ERA5-Land data for {lat:.2f}, {lon:.2f} from {start_date} to {end_date}")
        
        try:
            # Fetch real ERA5 reanalysis from Open-Meteo (no API key required)
            data = self._fetch_opennem_era5(start_date, end_date, lat, lon)
            
            # Cache the data
            if use_cache and data is not None and not data.empty:
                self._save_to_cache(data, start_date, end_date, lat, lon)
            
            if data is None or data.empty:
                logger.warning("Open-Meteo ERA5 returned empty; using minimal fallback values")
                return self._generate_fallback_data(start_date, end_date)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to extract ERA5-Land data: {e}")
            # Return fallback data
            return self._generate_fallback_data(start_date, end_date)
    
    def _fetch_opennem_era5(self, start_date: datetime, end_date: datetime, lat: float, lon: float) -> Optional[pd.DataFrame]:
        """Fetch ERA5 reanalysis via Open-Meteo API (hourly).
        Docs: https://open-meteo.com/en/docs/era5
        """
        try:
            # Open-Meteo expects inclusive dates (YYYY-MM-DD)
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            base = "https://era5.open-meteo.com/v1/era5"
            hourly_vars = ",".join([
                "temperature_2m",
                "dewpoint_2m",
                "u_component_of_wind_10m",
                "v_component_of_wind_10m",
                "surface_pressure",
                "precipitation",
                "boundary_layer_height",
                "shortwave_radiation",
                "direct_normal_irradiance"
            ])
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_str,
                "end_date": end_str,
                "hourly": hourly_vars,
                "timezone": "UTC"
            }
            resp = requests.get(base, params=params, timeout=30)
            resp.raise_for_status()
            js = resp.json()
            hourly = js.get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                return pd.DataFrame()
            df = pd.DataFrame({
                "datetime_utc": pd.to_datetime(times),
                "t2m_celsius": hourly.get("temperature_2m"),
                "d2m_celsius": hourly.get("dewpoint_2m"),
                "u10": hourly.get("u_component_of_wind_10m"),
                "v10": hourly.get("v_component_of_wind_10m"),
                "sp": hourly.get("surface_pressure"),  # in Pa per docs
                "precip": hourly.get("precipitation"),
                "blh": hourly.get("boundary_layer_height"),
                "shortwave_radiation": hourly.get("shortwave_radiation"),
                "dni": hourly.get("direct_normal_irradiance")
            })
            # Derive
            df["latitude"] = lat
            df["longitude"] = lon
            df["wind_speed"] = np.sqrt(np.square(df["u10"].astype(float)) + np.square(df["v10"].astype(float)))
            df["wind_direction"] = np.degrees(np.arctan2(df["v10"].astype(float), df["u10"].astype(float)))
            df["pressure_hpa"] = df["sp"].astype(float) / 100.0
            df["relative_humidity"] = self._calculate_rh(df["t2m_celsius"].astype(float), df["d2m_celsius"].astype(float))
            # Keep only requested hourly range strictly between start and end timestamps
            df = df[(df["datetime_utc"] >= start_date.floor('h')) & (df["datetime_utc"] <= end_date.floor('h'))]
            logger.info(f"Fetched ERA5(Open-Meteo) records: {len(df)}")
            return df.reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Open-Meteo ERA5 fetch failed: {e}")
            return pd.DataFrame()
    
    def _calculate_rh(self, temp_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
        """Calculate relative humidity from temperature and dewpoint."""
        # Magnus formula
        es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
        e = 6.112 * np.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
        rh = (e / es) * 100
        return np.clip(rh, 0, 100)
    
    def _generate_fallback_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate fallback ERA5-Land data when extraction fails."""
        
        # Use lower-case 'h' to avoid pandas FutureWarning
        timestamps = pd.date_range(start_date, end_date, freq='h')
        
        data = []
        for timestamp in timestamps:
            data.append({
                'datetime_utc': timestamp,
                'latitude': self.default_coords['lat'],
                'longitude': self.default_coords['lon'],
                't2m_celsius': 20.0,
                'd2m_celsius': 15.0,
                'u10': 2.0,
                'v10': 1.0,
                'wind_speed': 2.2,
                'wind_direction': 180.0,
                'relative_humidity': 60.0,
                'pressure_hpa': 850.0,
                'precip': 0.0,
                'blh': 500.0,
                'ssrd': 400.0,
                'strd': 300.0,
                'swvl1': 0.3
            })
        
        return pd.DataFrame(data)
    
    def _get_cache_path(self, start_date: datetime, end_date: datetime, lat: float, lon: float) -> Path:
        """Get cache file path for the given parameters."""
        cache_key = f"era5_{lat:.2f}_{lon:.2f}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        return self.cache_dir / cache_key
    
    def _load_from_cache(self, start_date: datetime, end_date: datetime, lat: float, lon: float) -> Optional[pd.DataFrame]:
        """Load data from cache if available and not expired."""
        
        cache_path = self._get_cache_path(start_date, end_date, lat, lon)
        
        if not cache_path.exists():
            return None
        
        # Check if cache is expired
        cache_age = time.time() - cache_path.stat().st_mtime
        if cache_age > (self.cache_ttl_hours * 3600):
            cache_path.unlink()  # Remove expired cache
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded ERA5-Land data from cache: {len(df)} records")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(self, df: pd.DataFrame, start_date: datetime, end_date: datetime, lat: float, lon: float) -> None:
        """Save data to cache."""
        
        cache_path = self._get_cache_path(start_date, end_date, lat, lon)
        
        try:
            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached ERA5-Land data: {len(df)} records")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_enhanced_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced weather features derived from ERA5-Land data."""
        
        if df.empty:
            return df
        
        enhanced_df = df.copy()
        
        # Add derived meteorological features
        if 't2m_celsius' in enhanced_df.columns and 'd2m_celsius' in enhanced_df.columns:
            enhanced_df['relative_humidity'] = self._calculate_rh(
                enhanced_df['t2m_celsius'], enhanced_df['d2m_celsius']
            )
        
        if 'u10' in enhanced_df.columns and 'v10' in enhanced_df.columns:
            enhanced_df['wind_speed'] = np.sqrt(enhanced_df['u10']**2 + enhanced_df['v10']**2)
            enhanced_df['wind_direction'] = np.degrees(np.arctan2(enhanced_df['v10'], enhanced_df['u10']))
        
        if 'sp' in enhanced_df.columns:
            enhanced_df['pressure_hpa'] = enhanced_df['sp'] / 100
        
        # Add weather stability indices
        if 't2m_celsius' in enhanced_df.columns and 'blh' in enhanced_df.columns:
            # Temperature gradient (proxy for atmospheric stability)
            enhanced_df['temp_gradient'] = enhanced_df['t2m_celsius'] / (enhanced_df['blh'] / 1000)
            
            # Stability index (lower values = more stable)
            enhanced_df['stability_index'] = enhanced_df['blh'] / (enhanced_df['t2m_celsius'] + 273.15)
        
        # Add pollution potential features
        if 'wind_speed' in enhanced_df.columns and 'blh' in enhanced_df.columns:
            # Ventilation index (higher = better dispersion)
            enhanced_df['ventilation_index'] = enhanced_df['wind_speed'] * enhanced_df['blh']
            
            # Pollution potential (inverse of ventilation)
            enhanced_df['pollution_potential'] = 1.0 / (enhanced_df['ventilation_index'] + 1.0)
        
        # Add diurnal features
        if 'datetime_utc' in enhanced_df.columns:
            enhanced_df['hour'] = pd.to_datetime(enhanced_df['datetime_utc']).dt.hour
            enhanced_df['is_night'] = (enhanced_df['hour'] < 6) | (enhanced_df['hour'] > 18)
            enhanced_df['is_morning'] = (enhanced_df['hour'] >= 6) & (enhanced_df['hour'] < 12)
            enhanced_df['is_afternoon'] = (enhanced_df['hour'] >= 12) & (enhanced_df['hour'] < 18)
        
        logger.info(f"Added {len(enhanced_df.columns) - len(df.columns)} enhanced weather features")
        return enhanced_df
