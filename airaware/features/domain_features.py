"""Domain-specific feature engineering for air quality prediction."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DomainFeatureConfig(BaseModel):
    """Configuration for domain-specific feature generation."""
    air_quality_indices: bool = Field(True, description="Generate AQI-related features")
    emission_source_proxies: bool = Field(True, description="Generate emission source indicators")
    dispersion_modeling: bool = Field(True, description="Generate dispersion model features")
    seasonal_patterns: bool = Field(True, description="Generate Nepal-specific seasonal features")
    urban_effects: bool = Field(True, description="Generate urban microclimate features")


class DomainFeatureGenerator:
    """Generate domain-specific features for air quality prediction in Kathmandu Valley."""
    
    def __init__(self, config: Optional[DomainFeatureConfig] = None):
        self.config = config or DomainFeatureConfig()
        logger.info("DomainFeatureGenerator initialized")
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive domain-specific features."""
        
        if df.empty:
            logger.warning("Empty dataframe provided")
            return df
        
        logger.info(f"Generating domain-specific features for {len(df):,} records")
        
        df = df.copy()
        
        # Generate different types of domain features
        feature_generators = [
            self._add_air_quality_indices,
            self._add_emission_source_proxies,
            self._add_dispersion_features,
            self._add_seasonal_patterns,
            self._add_urban_effects,
            self._add_topographic_effects,
            self._add_temporal_patterns,
            self._add_threshold_features
        ]
        
        for generator in feature_generators:
            try:
                df = generator(df)
            except Exception as e:
                logger.warning(f"Failed to generate features with {generator.__name__}: {e}")
        
        # Log feature summary
        domain_features = [col for col in df.columns 
                          if any(col.startswith(prefix) for prefix in 
                                ['aqi_', 'emission_', 'dispersion_', 'seasonal_', 'urban_', 'topo_', 'temporal_', 'threshold_'])]
        logger.info(f"Generated {len(domain_features)} domain-specific features")
        
        return df
    
    def _add_air_quality_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add air quality index and health-related features."""
        
        if not self.config.air_quality_indices:
            return df
        
        logger.debug("Adding air quality index features")
        
        if 'pm25' not in df.columns:
            return df
        
        # AQI calculation (US EPA standard)
        def calculate_aqi(pm25_value):
            """Calculate AQI from PM2.5 concentration."""
            if pm25_value <= 12.0:
                return pm25_value * 50 / 12.0
            elif pm25_value <= 35.4:
                return 50 + (pm25_value - 12.0) * 50 / (35.4 - 12.0)
            elif pm25_value <= 55.4:
                return 100 + (pm25_value - 35.4) * 50 / (55.4 - 35.4)
            elif pm25_value <= 150.4:
                return 150 + (pm25_value - 55.4) * 100 / (150.4 - 55.4)
            elif pm25_value <= 250.4:
                return 200 + (pm25_value - 150.4) * 100 / (250.4 - 150.4)
            else:
                return 300 + (pm25_value - 250.4) * 100 / (350.4 - 250.4)
        
        df['aqi_us_epa'] = df['pm25'].apply(calculate_aqi)
        
        # AQI categories
        df['aqi_category'] = pd.cut(
            df['aqi_us_epa'],
            bins=[0, 50, 100, 150, 200, 300, float('inf')],
            labels=['Good', 'Moderate', 'USG', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
        ).astype(str)
        
        # Health risk indicators
        df['aqi_health_risk_low'] = (df['aqi_us_epa'] <= 50).astype(int)
        df['aqi_health_risk_moderate'] = ((df['aqi_us_epa'] > 50) & (df['aqi_us_epa'] <= 100)).astype(int)
        df['aqi_health_risk_high'] = (df['aqi_us_epa'] > 150).astype(int)
        
        # Exceedance indicators (WHO guidelines)
        df['aqi_who_exceedance'] = (df['pm25'] > 15.0).astype(int)  # WHO 24-hour guideline
        df['aqi_who_annual_exceedance'] = (df['pm25'] > 5.0).astype(int)  # WHO annual guideline
        
        # Relative pollution level
        if len(df) > 24:
            # Calculate relative to recent baseline (24-hour rolling mean)
            baseline = df['pm25'].rolling(window=24, min_periods=12).mean()
            df['aqi_relative_pollution'] = (df['pm25'] - baseline) / (baseline + 1)
        
        return df
    
    def _add_emission_source_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that proxy for different emission sources."""
        
        if not self.config.emission_source_proxies:
            return df
        
        logger.debug("Adding emission source proxy features")
        
        if 'datetime_utc' not in df.columns:
            return df
        
        # Convert to local time (Nepal is UTC+5:45)
        local_time = df['datetime_utc'] + pd.Timedelta(hours=5, minutes=45)
        hour = local_time.dt.hour
        day_of_week = local_time.dt.dayofweek
        
        # Traffic emission proxies
        # Morning rush hours (7-10 AM)
        df['emission_traffic_morning'] = ((hour >= 7) & (hour <= 10) & (day_of_week < 5)).astype(int)
        
        # Evening rush hours (5-8 PM)
        df['emission_traffic_evening'] = ((hour >= 17) & (hour <= 20) & (day_of_week < 5)).astype(int)
        
        # Weekend traffic patterns (different timing)
        df['emission_traffic_weekend'] = ((hour >= 10) & (hour <= 18) & (day_of_week >= 5)).astype(int)
        
        # Industrial emission proxies
        # Weekday working hours
        df['emission_industrial'] = ((hour >= 8) & (hour <= 18) & (day_of_week < 5)).astype(int)
        
        # Residential heating/cooking proxies
        # Morning cooking (6-9 AM)
        df['emission_cooking_morning'] = ((hour >= 6) & (hour <= 9)).astype(int)
        
        # Evening cooking (6-9 PM)
        df['emission_cooking_evening'] = ((hour >= 18) & (hour <= 21)).astype(int)
        
        # Winter heating (Nov-Feb, evening/night)
        month = local_time.dt.month
        winter_heating = ((month.isin([11, 12, 1, 2])) & 
                         ((hour >= 18) | (hour <= 7))).astype(int)
        df['emission_heating_winter'] = winter_heating
        
        # Agricultural burning proxy (spring: Mar-May)
        spring_burning = ((month.isin([3, 4, 5])) & 
                         (hour.isin([10, 11, 12, 13, 14, 15]))).astype(int)
        df['emission_agricultural_burning'] = spring_burning
        
        return df
    
    def _add_dispersion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add atmospheric dispersion modeling features."""
        
        if not self.config.dispersion_modeling:
            return df
        
        logger.debug("Adding atmospheric dispersion features")
        
        required_cols = ['wind_speed', 'blh']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for dispersion features: {missing_cols}")
            return df
        
        # Gaussian plume dispersion parameters
        # Pasquill-Gifford stability classes proxy
        hour = df['datetime_utc'].dt.hour if 'datetime_utc' in df.columns else 12
        
        # Simplified stability classification
        stability_conditions = [
            (df['wind_speed'] < 2) & (hour.between(10, 16)),  # A-B (unstable, day)
            (df['wind_speed'] < 2) & (~hour.between(6, 18)),  # F (stable, night)
            (df['wind_speed'].between(2, 3)) & (hour.between(6, 18)),  # C (slightly unstable)
            (df['wind_speed'].between(2, 3)) & (~hour.between(6, 18)),  # E (slightly stable)
            (df['wind_speed'].between(3, 5)),  # D (neutral)
            (df['wind_speed'] > 5)  # D (neutral, high wind)
        ]
        
        stability_values = [1, 6, 2, 5, 4, 4]  # Pasquill classes (1=A, 6=F)
        df['dispersion_stability_class'] = pd.Series(dtype=float)
        
        for condition, value in zip(stability_conditions, stability_values):
            df.loc[condition, 'dispersion_stability_class'] = value
        
        df['dispersion_stability_class'] = df['dispersion_stability_class'].fillna(4)  # Default to neutral
        
        # Dispersion coefficients (simplified)
        # Horizontal dispersion coefficient
        df['dispersion_sigma_y'] = 0.1 * np.sqrt(df['wind_speed'] + 0.1) * (df['dispersion_stability_class'] / 4.0)
        
        # Vertical dispersion coefficient
        df['dispersion_sigma_z'] = 0.05 * np.sqrt(df['blh'] / 1000.0) * (6.0 / df['dispersion_stability_class'])
        
        # Effective stack height (proxy for emission height)
        df['dispersion_effective_height'] = np.minimum(df['blh'] * 0.8, 200)  # Max 200m
        
        # Plume rise (buoyancy effect proxy)
        if 't2m_celsius' in df.columns:
            # Simple buoyancy calculation
            temp_gradient = np.maximum(0, 25 - df['t2m_celsius'])  # Assume source is warmer
            df['dispersion_plume_rise'] = temp_gradient * 2.0  # Simplified plume rise
        
        # Overall dispersion potential
        df['dispersion_potential'] = (
            df['wind_speed'] * 
            np.log(df['blh'] + 1) * 
            (1.0 / (df['dispersion_stability_class'] + 1))
        )
        
        return df
    
    def _add_seasonal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Nepal-specific seasonal patterns."""
        
        if not self.config.seasonal_patterns:
            return df
        
        logger.debug("Adding Nepal-specific seasonal patterns")
        
        if 'datetime_utc' not in df.columns:
            return df
        
        # Convert to local time for Nepal
        local_time = df['datetime_utc'] + pd.Timedelta(hours=5, minutes=45)
        month = local_time.dt.month
        
        # Nepal seasonal patterns
        # Pre-monsoon (March-May): dust storms, heating
        df['seasonal_pre_monsoon'] = month.isin([3, 4, 5]).astype(int)
        
        # Monsoon (June-September): rain, washing
        df['seasonal_monsoon'] = month.isin([6, 7, 8, 9]).astype(int)
        
        # Post-monsoon (October-November): clear, crop burning
        df['seasonal_post_monsoon'] = month.isin([10, 11]).astype(int)
        
        # Winter (December-February): heating, inversion
        df['seasonal_winter'] = month.isin([12, 1, 2]).astype(int)
        
        # Festival seasons (Dashain/Tihar: Sept-Nov)
        df['seasonal_festival'] = month.isin([9, 10, 11]).astype(int)
        
        # Agricultural seasons
        # Planting season (May-July)
        df['seasonal_planting'] = month.isin([5, 6, 7]).astype(int)
        
        # Harvest season (October-December)
        df['seasonal_harvest'] = month.isin([10, 11, 12]).astype(int)
        
        # Burning season (March-May, October-November)
        df['seasonal_burning'] = month.isin([3, 4, 5, 10, 11]).astype(int)
        
        return df
    
    def _add_urban_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add urban microclimate and heat island effects."""
        
        if not self.config.urban_effects:
            return df
        
        logger.debug("Adding urban microclimate features")
        
        if 't2m_celsius' not in df.columns:
            return df
        
        # Urban heat island intensity proxy
        # Assume temperature > 25°C indicates urban heating
        df['urban_heat_island_intensity'] = np.maximum(0, df['t2m_celsius'] - 22)
        
        # Urban heat island categories
        df['urban_heat_island_category'] = pd.cut(
            df['urban_heat_island_intensity'],
            bins=[0, 2, 5, 8, float('inf')],
            labels=['None', 'Weak', 'Moderate', 'Strong']
        ).astype(str)
        
        # Time-of-day urban effects
        if 'datetime_utc' in df.columns:
            hour = df['datetime_utc'].dt.hour
            
            # Peak urban heating (2-4 PM local time, 8-10 UTC)
            df['urban_peak_heating'] = hour.isin([8, 9, 10]).astype(int)
            
            # Urban cooling (evening/night)
            df['urban_cooling'] = hour.isin([13, 14, 15, 16, 17, 18]).astype(int)
        
        # Wind speed modification by urban roughness
        if 'wind_speed' in df.columns:
            # Urban areas typically reduce wind speed
            df['urban_wind_reduction'] = np.exp(-df['wind_speed'] / 3.0)
            
            # Canyon effect (low wind + high temperature)
            df['urban_canyon_effect'] = (
                (df['wind_speed'] < 2) & 
                (df['t2m_celsius'] > 25)
            ).astype(int)
        
        return df
    
    def _add_topographic_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add topographic effects specific to Kathmandu Valley."""
        
        logger.debug("Adding topographic effects")
        
        # Valley wind patterns
        if 'datetime_utc' in df.columns and 'wind_direction' in df.columns:
            local_time = df['datetime_utc'] + pd.Timedelta(hours=5, minutes=45)
            hour = local_time.dt.hour
            
            # Valley breeze (upslope, day): generally from south (180°)
            valley_breeze_condition = (
                (hour.between(8, 16)) &
                (df['wind_direction'].between(135, 225))
            )
            df['topo_valley_breeze'] = valley_breeze_condition.astype(int)
            
            # Mountain breeze (downslope, night): generally from north (0°/360°)
            mountain_breeze_condition = (
                ((hour >= 18) | (hour <= 6)) &
                ((df['wind_direction'] <= 45) | (df['wind_direction'] >= 315))
            )
            df['topo_mountain_breeze'] = mountain_breeze_condition.astype(int)
        
        # Valley inversion potential
        if 'blh' in df.columns and 't2m_celsius' in df.columns:
            # Low boundary layer + cold temperature = inversion potential
            df['topo_inversion_potential'] = (
                (df['blh'] < 300) & 
                (df['t2m_celsius'] < 15)
            ).astype(int)
        
        # Terrain channeling effect
        if 'wind_speed' in df.columns and 'wind_direction' in df.columns:
            # Kathmandu Valley main axis is roughly NW-SE
            main_axis_wind = (
                (df['wind_direction'].between(315, 45)) |  # NW-N-NE
                (df['wind_direction'].between(135, 225))   # SE-S-SW
            )
            
            # Enhanced wind speed along valley axis
            df['topo_valley_channeling'] = (
                main_axis_wind & (df['wind_speed'] > 2)
            ).astype(int)
        
        return df
    
    def _add_temporal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced temporal patterns specific to air quality."""
        
        logger.debug("Adding temporal patterns")
        
        if 'datetime_utc' not in df.columns:
            return df
        
        local_time = df['datetime_utc'] + pd.Timedelta(hours=5, minutes=45)
        hour = local_time.dt.hour
        
        # Bimodal daily pattern (typical for urban PM2.5)
        # Morning peak (7-10 AM)
        df['temporal_morning_peak'] = hour.isin([7, 8, 9, 10]).astype(int)
        
        # Evening peak (6-9 PM)
        df['temporal_evening_peak'] = hour.isin([18, 19, 20, 21]).astype(int)
        
        # Low pollution hours (midnight-6 AM)
        df['temporal_low_pollution'] = hour.isin([0, 1, 2, 3, 4, 5, 6]).astype(int)
        
        # Weekend pattern differences
        day_of_week = local_time.dt.dayofweek
        
        # Weekday vs weekend patterns
        df['temporal_weekday_pattern'] = (day_of_week < 5).astype(int)
        df['temporal_weekend_pattern'] = (day_of_week >= 5).astype(int)
        
        # School hours (affects traffic patterns)
        school_hours = (
            (hour.between(6, 8)) |  # Morning school traffic
            (hour.between(15, 17)) |  # Afternoon school traffic
            (hour.between(10, 15))   # School hours
        ) & (day_of_week < 5)
        
        df['temporal_school_hours'] = school_hours.astype(int)
        
        return df
    
    def _add_threshold_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add threshold-based features for critical conditions."""
        
        logger.debug("Adding threshold features")
        
        # PM2.5 threshold features
        if 'pm25' in df.columns:
            # WHO thresholds
            df['threshold_who_daily'] = (df['pm25'] > 15).astype(int)
            df['threshold_who_annual'] = (df['pm25'] > 5).astype(int)
            
            # Health alert levels
            df['threshold_unhealthy'] = (df['pm25'] > 55.4).astype(int)
            df['threshold_very_unhealthy'] = (df['pm25'] > 150.4).astype(int)
            df['threshold_hazardous'] = (df['pm25'] > 250.4).astype(int)
        
        # Meteorological thresholds
        if 'wind_speed' in df.columns:
            # Low wind conditions (poor dispersion)
            df['threshold_low_wind'] = (df['wind_speed'] < 1.5).astype(int)
            df['threshold_very_low_wind'] = (df['wind_speed'] < 0.5).astype(int)
        
        if 'blh' in df.columns:
            # Low boundary layer (poor mixing)
            df['threshold_low_blh'] = (df['blh'] < 200).astype(int)
            df['threshold_very_low_blh'] = (df['blh'] < 100).astype(int)
        
        if 't2m_celsius' in df.columns:
            # Temperature extremes
            df['threshold_hot'] = (df['t2m_celsius'] > 30).astype(int)
            df['threshold_cold'] = (df['t2m_celsius'] < 10).astype(int)
        
        # Combined critical conditions
        if all(col in df.columns for col in ['wind_speed', 'blh']):
            # Stagnation conditions
            df['threshold_stagnation'] = (
                (df['wind_speed'] < 1.5) & (df['blh'] < 300)
            ).astype(int)
        
        return df
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """Get domain-specific feature categories."""
        
        categories = {
            "air_quality_indices": [
                "aqi_us_epa", "aqi_category", "aqi_health_risk_low", "aqi_health_risk_moderate",
                "aqi_health_risk_high", "aqi_who_exceedance", "aqi_who_annual_exceedance",
                "aqi_relative_pollution"
            ],
            "emission_sources": [
                "emission_traffic_morning", "emission_traffic_evening", "emission_traffic_weekend",
                "emission_industrial", "emission_cooking_morning", "emission_cooking_evening",
                "emission_heating_winter", "emission_agricultural_burning"
            ],
            "dispersion_modeling": [
                "dispersion_stability_class", "dispersion_sigma_y", "dispersion_sigma_z",
                "dispersion_effective_height", "dispersion_plume_rise", "dispersion_potential"
            ],
            "seasonal_patterns": [
                "seasonal_pre_monsoon", "seasonal_monsoon", "seasonal_post_monsoon",
                "seasonal_winter", "seasonal_festival", "seasonal_planting",
                "seasonal_harvest", "seasonal_burning"
            ],
            "urban_effects": [
                "urban_heat_island_intensity", "urban_heat_island_category",
                "urban_peak_heating", "urban_cooling", "urban_wind_reduction",
                "urban_canyon_effect"
            ],
            "topographic_effects": [
                "topo_valley_breeze", "topo_mountain_breeze", "topo_inversion_potential",
                "topo_valley_channeling"
            ],
            "temporal_patterns": [
                "temporal_morning_peak", "temporal_evening_peak", "temporal_low_pollution",
                "temporal_weekday_pattern", "temporal_weekend_pattern", "temporal_school_hours"
            ],
            "threshold_features": [
                "threshold_who_daily", "threshold_who_annual", "threshold_unhealthy",
                "threshold_very_unhealthy", "threshold_hazardous", "threshold_low_wind",
                "threshold_very_low_wind", "threshold_low_blh", "threshold_very_low_blh",
                "threshold_hot", "threshold_cold", "threshold_stagnation"
            ]
        }
        
        return categories


