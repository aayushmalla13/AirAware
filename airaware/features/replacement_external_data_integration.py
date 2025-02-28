"""REAL External Data Integration - NO SYNTHETIC DATA"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json

from .real_data_extractor import RealDataExtractor, RealDataExtractorConfig

logger = logging.getLogger(__name__)


class RealExternalDataConfig(BaseModel):
    """Configuration for REAL external data integration - NO SYNTHETIC DATA"""
    
    # Cache settings for real data
    cache_real_data: bool = Field(True, description="Cache real air quality data")
    cache_ttl_hours: int = Field(2, description="Cache TTL for real data (hours)")
    
    # Data freshness requirements
    max_data_age_hours: int = Field(24, description="Maximum age of cached data")
    min_real_data_points: int = Field(10, description="Minimum real data points required")
    
    # Enable/disable real data sources
    enable_india_real_data: bool = Field(True, description="Enable real Indian data")
    enable_china_real_data: bool = Field(True, description="Enable real Chinese data")
    
    # Real data extractor settings
    real_data_config: RealDataExtractorConfig = Field(
        default_factory=RealDataExtractorConfig,
        description="Configuration for real data extraction"
    )


class RealExternalDataIntegrator:
    """Integrates REAL air quality data from external monitoring stations - NO SYNTHETIC DATA"""
    
    def __init__(self, config: Optional[RealExternalDataConfig] = None):
        self.config = config or RealExternalDataConfig()
        self.cache_dir = Path("data/cache/external_real")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.real_extractor = RealDataExtractor(self.config.real_data_config)
        
        logger.info("🌍 REAL External Data Integrator initialized")
        logger.info("❌ SYNTHIC DATA DISABLED - Using ONLY real air quality data")
        logger.info(f"🇮🇳 Real Indian stations: {len(self.config.real_data_config.india_real_stations)}")
        logger.info(f"🇨🇳 Real Chinese stations: {len(self.config.real_data_config.china_real_stations)}")
    
    def fetch_real_external_data(
        self, 
        start_date: datetime, 
        end_date: datetime,
        countries: List[str] = ["india", "china"]
    ) -> pd.DataFrame:
        """Fetch REAL external air quality data - NO SYNTHETIC DATA"""
        
        logger.info(f"🌍 Fetching REAL external data from {start_date} to {end_date}")
        logger.info("❌ SYNTHETIC DATA DISABLED - Only extracting real measurements")
        
        all_real_data = []
        
        for country in countries:
            country_real_data = self._fetch_country_real_data(country, start_date, end_date)
            if not country_real_data.empty:
                all_real_data.append(country_real_data)
                logger.info(f"✅ Real {country} data: {len(country_real_data)} records")
            else:
                logger.warning(f"⚠️ No real data found for {country}")
        
        if all_real_data:
            combined_df = pd.concat(all_real_data, ignore_index=True)
            
            # Sort and clean real data
            combined_df = combined_df.sort_values(['country', 'datetime_utc'])
            combined_df = self._clean_real_data(combined_df)
            
            logger.info(f"🌍 Total REAL external data: {len(combined_df)} records")
            logger.info(f"🌍 Data sources: {combined_df['source_api'].value_counts().to_dict()}")
            
            return combined_df
        else:
            logger.error("❌ NO REAL DATA COULD BE EXTRACTED")
            return pd.DataFrame()
    
    def _fetch_country_real_data(
        self, 
        country: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch real data for a specific country"""
        
        # Check cache first
        cache_file = self.cache_dir / f"real_{country}_{start_date.date()}_{end_date.date()}.parquet"
        if cache_file.exists() and self._is_cache_valid(cache_file):
            logger.info(f"📁 Loading cached real {country} data")
            cached_df = pd.read_parquet(cache_file)
            logger.info(f"📁 Cached real {country} data: {len(cached_df)} records")
            return cached_df
        
        # Extract fresh real data
        logger.info(f"🌐 Extracting fresh real {country} data from APIs...")
        
        if country == "india" and self.config.enable_india_real_data:
            real_data = self.real_extractor.extract_real_india_data(start_date, end_date)
        elif country == "china" and self.config.enable_china_real_data:
            real_data = self.real_extractor.extract_real_china_data(start_date, end_date)
        else:
            logger.warning(f"⚠️ Real data extraction disabled for {country}")
            return pd.DataFrame()
        
        # Cache real data if we have enough
        if len(real_data) >= self.config.min_real_data_points:
            real_data.to_parquet(cache_file)
            logger.info(f"💾 Cached real {country} data: {len(real_data)} records")
        
        return real_data
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cached real data is still valid"""
        
        if not self.config.cache_real_data:
            return False
        
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return cache_age.total_seconds() < (self.config.cache_ttl_hours * 3600)
    
    def _clean_real_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate real air quality data"""
        
        if df.empty:
            return df
        
        logger.info(f"🔍 Cleaning {len(df)} real air quality records")
        
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['datetime_utc', 'station_id'])
        
        # Remove records with invalid PM2.5 values
        df = df[(df['pm25'] >= 0) & (df['pm25'] <= 1000)]
        
        # Remove records that are too old
        max_age = datetime.now(timezone.utc) - timedelta(hours=self.config.max_data_age_hours * 2)
        df = df[df['datetime_utc'] >= max_age]
        
        # Remove extreme outliers using IQR method
        q1 = df['pm25'].quantile(0.25)
        q3 = df['pm25'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr  # More lenient than usual
        upper_bound = q3 + 3 * iqr
        df = df[(df['pm25'] >= lower_bound) & (df['pm25'] <= upper_bound)]
        
        cleaned_count = len(df)
        removed_count = initial_count - cleaned_count
        
        if removed_count > 0:
            logger.info(f"🧹 Removed {removed_count} invalid records, kept {cleaned_count} real records")
        
        logger.info(f"✅ Real data cleaning complete: {cleaned_count} valid records")
        
        return df
    
    def get_real_data_summary(self) -> Dict[str, Any]:
        """Get summary of real external data availability"""
        
        summary = {
            "data_type": "REAL_ONLY",
            "synthetic_disabled": True,
            "countries_enabled": [],
            "total_real_stations": 0,
            "cache_enabled": self.config.cache_real_data,
            "cache_ttl_hours": self.config.cache_ttl_hours
        }
        
        if self.config.enable_india_real_data:
            summary["countries_enabled"].append("india")
            summary["india_stations"] = len(self.config.real_data_config.india_real_stations)
            summary["total_real_stations"] += len(self.config.real_data_config.india_real_stations)
        
        if self.config.enable_china_real_data:
            summary["countries_enabled"].append("china")
            summary["china_stations"] = len(self.config.real_data_config.china_real_stations)
            summary["total_real_stations"] += len(self.config.real_data_config.china_real_stations)
        
        return summary
    
    def get_real_station_info(self, country: str = None) -> Dict[str, Any]:
        """Get information about real monitoring stations"""
        
        if country:
            station_info = self.real_extractor.get_real_station_info()
            return station_info.get(country, [])
        else:
            return self.real_extractor.get_real_station_info()
    
    def clear_real_data_cache(self) -> bool:
        """Clear cached real data"""
        
        try:
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("real_*.parquet"):
                    cache_file.unlink()
                logger.info("🗑️ Cleared all cached real data")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {e}")
            return False


def create_real_external_data_service():
    """Factory function to create a real external data service"""
    
    config = RealExternalDataConfig()
    service = RealExternalDataIntegrator(config)
    
    logger.info("🏭 Created REAL External Data Service")
    logger.info("❌ SYNTHETIC DATA: DISABLED")
    logger.info("🌍 REAL DATA ONLY: ENABLED")
    
    return service
