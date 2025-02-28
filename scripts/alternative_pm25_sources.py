#!/usr/bin/env python3
"""
Alternative PM2.5 Data Sources for Kathmandu Valley
This script explores multiple data sources to get comprehensive historical PM2.5 data
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlternativePM25Sources:
    """Alternative PM2.5 data sources for Kathmandu Valley"""
    
    def __init__(self):
        self.kathmandu_coords = {
            'lat': 27.7172,
            'lon': 85.324,
            'bbox': [85.0, 27.5, 85.6, 28.0]  # [west, south, east, north]
        }
    
    def test_purpleair_api(self) -> Dict:
        """Test PurpleAir API for Kathmandu Valley sensors"""
        logger.info("🔍 Testing PurpleAir API...")
        
        try:
            # PurpleAir API endpoint
            url = "https://api.purpleair.com/v1/sensors"
            params = {
                'fields': 'pm2.5_atm,pm2.5_cf_1,pm2.5_alt,latitude,longitude,name',
                'location_type': 0,  # Outdoor sensors
                'max_age': 3600,  # 1 hour max age
                'nwlng': self.kathmandu_coords['bbox'][0],  # west
                'nwlat': self.kathmandu_coords['bbox'][3],  # north
                'selng': self.kathmandu_coords['bbox'][2],  # east
                'selat': self.kathmandu_coords['bbox'][1],  # south
            }
            
            # PurpleAir requires API key
            api_key = os.getenv('PURPLEAIR_API_KEY')
            if not api_key:
                logger.warning("⚠️ PurpleAir API key not found in .env")
                return {'status': 'no_key', 'sensors': 0}
            
            headers = {'X-API-Key': api_key}
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                sensors = data.get('data', [])
                logger.info(f"✅ Found {len(sensors)} PurpleAir sensors in Kathmandu Valley")
                return {'status': 'success', 'sensors': len(sensors), 'data': sensors}
            else:
                logger.error(f"❌ PurpleAir API error: {response.status_code}")
                return {'status': 'error', 'sensors': 0}
                
        except Exception as e:
            logger.error(f"❌ PurpleAir API exception: {e}")
            return {'status': 'error', 'sensors': 0}
    
    def test_world_aqi_api(self) -> Dict:
        """Test World Air Quality Index API"""
        logger.info("🔍 Testing World AQI API...")
        
        try:
            # World AQI API
            url = "https://api.waqi.info/search/"
            params = {
                'token': os.getenv('WAQI_API_KEY', 'demo'),
                'keyword': 'Kathmandu'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                stations = data.get('data', [])
                logger.info(f"✅ Found {len(stations)} World AQI stations for Kathmandu")
                return {'status': 'success', 'stations': len(stations), 'data': stations}
            else:
                logger.error(f"❌ World AQI API error: {response.status_code}")
                return {'status': 'error', 'stations': 0}
                
        except Exception as e:
            logger.error(f"❌ World AQI API exception: {e}")
            return {'status': 'error', 'stations': 0}
    
    def generate_synthetic_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic synthetic PM2.5 data based on meteorological patterns"""
        logger.info("🔧 Generating synthetic historical PM2.5 data...")
        
        # Load ERA5 meteorological data
        try:
            era5_df = pd.read_parquet('data/interim/era5_hourly.parquet')
            logger.info(f"📊 Loaded {len(era5_df)} ERA5 records")
        except Exception as e:
            logger.error(f"❌ Failed to load ERA5 data: {e}")
            return pd.DataFrame()
        
        # Filter ERA5 data for Kathmandu Valley area
        era5_filtered = era5_df[
            (era5_df['latitude'] >= self.kathmandu_coords['bbox'][1]) &
            (era5_df['latitude'] <= self.kathmandu_coords['bbox'][3]) &
            (era5_df['longitude'] >= self.kathmandu_coords['bbox'][0]) &
            (era5_df['longitude'] <= self.kathmandu_coords['bbox'][2])
        ].copy()
        
        if len(era5_filtered) == 0:
            logger.error("❌ No ERA5 data found for Kathmandu Valley")
            return pd.DataFrame()
        
        # Generate PM2.5 based on meteorological patterns
        pm25_data = []
        
        for _, row in era5_filtered.iterrows():
            # Base PM2.5 level (typical for Kathmandu)
            base_pm25 = 45.0
            
            # Meteorological influences
            wind_speed = row.get('wind_speed', 2.0)
            temperature = row.get('temperature_celsius', 20.0)
            precipitation = row.get('precipitation_mm', 0.0)
            
            # Wind speed effect (higher wind = lower PM2.5)
            wind_factor = max(0.3, 1.0 - (wind_speed - 1.0) * 0.1)
            
            # Temperature effect (higher temp = higher PM2.5 due to thermal inversion)
            temp_factor = 1.0 + (temperature - 20.0) * 0.02
            
            # Precipitation effect (rain washes out particles)
            precip_factor = max(0.5, 1.0 - precipitation * 0.1)
            
            # Seasonal variation (higher in winter due to heating)
            month = row['datetime_utc'].month
            seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12)
            
            # Diurnal variation (higher at night due to inversion)
            hour = row['datetime_utc'].hour
            diurnal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * (hour - 6) / 24)
            
            # Calculate PM2.5
            pm25 = base_pm25 * wind_factor * temp_factor * precip_factor * seasonal_factor * diurnal_factor
            
            # Add some random variation
            pm25 += np.random.normal(0, 5.0)
            
            # Ensure realistic range
            pm25 = max(5.0, min(200.0, pm25))
            
            pm25_data.append({
                'datetime_utc': row['datetime_utc'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'pm25': pm25,
                'station_id': f"synthetic_{int(row['latitude']*100)}_{int(row['longitude']*100)}",
                'data_source': 'synthetic_meteorological',
                'quality': 'estimated'
            })
        
        df = pd.DataFrame(pm25_data)
        logger.info(f"✅ Generated {len(df)} synthetic PM2.5 records")
        return df
    
    def test_openaq_historical_endpoints(self) -> Dict:
        """Test various OpenAQ historical data endpoints"""
        logger.info("🔍 Testing OpenAQ historical endpoints...")
        
        api_key = os.getenv('OPENAQ_KEY')
        headers = {'X-API-Key': api_key} if api_key else {}
        
        endpoints_to_test = [
            'https://api.openaq.org/v3/sensors/7711/hours',
            'https://api.openaq.org/v3/sensors/7711/days',
            'https://api.openaq.org/v3/sensors/7711/measurements',
        ]
        
        results = {}
        
        for endpoint in endpoints_to_test:
            try:
                # Test with date range
                params = {
                    'date_from': '2024-01-01',
                    'date_to': '2024-12-31',
                    'limit': 10
                }
                
                response = requests.get(endpoint, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    results[endpoint] = {
                        'status': 'success',
                        'records': len(data.get('results', [])),
                        'sample': data.get('results', [])[:2] if data.get('results') else []
                    }
                else:
                    results[endpoint] = {
                        'status': 'error',
                        'code': response.status_code,
                        'message': response.text[:100]
                    }
                    
            except Exception as e:
                results[endpoint] = {
                    'status': 'exception',
                    'error': str(e)
                }
        
        return results
    
    def create_enhanced_dataset(self) -> pd.DataFrame:
        """Create enhanced dataset combining all available sources"""
        logger.info("🔧 Creating enhanced PM2.5 dataset...")
        
        # Load existing real data
        try:
            real_data = pd.read_parquet('data/interim/targets.parquet')
            logger.info(f"📊 Loaded {len(real_data)} real PM2.5 records")
        except Exception as e:
            logger.error(f"❌ Failed to load real data: {e}")
            real_data = pd.DataFrame()
        
        # Generate synthetic historical data
        synthetic_data = self.generate_synthetic_historical_data('2024-01-01', '2025-09-30')
        
        if len(synthetic_data) > 0:
            # Combine real and synthetic data
            combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
            
            # Remove duplicates based on datetime and location
            combined_data = combined_data.drop_duplicates(
                subset=['datetime_utc', 'latitude', 'longitude'], 
                keep='first'
            )
            
            # Sort by datetime
            combined_data = combined_data.sort_values('datetime_utc')
            
            logger.info(f"✅ Enhanced dataset: {len(combined_data)} total records")
            logger.info(f"   - Real data: {len(real_data)} records")
            logger.info(f"   - Synthetic data: {len(synthetic_data)} records")
            
            return combined_data
        else:
            logger.warning("⚠️ No synthetic data generated, returning real data only")
            return real_data

def main():
    """Main function to test alternative PM2.5 sources"""
    print("🌍 Alternative PM2.5 Data Sources for Kathmandu Valley")
    print("=" * 60)
    
    sources = AlternativePM25Sources()
    
    # Test various sources
    print("\n1️⃣ Testing PurpleAir API...")
    purpleair_result = sources.test_purpleair_api()
    print(f"   Status: {purpleair_result['status']}")
    print(f"   Sensors found: {purpleair_result['sensors']}")
    
    print("\n2️⃣ Testing World AQI API...")
    waqi_result = sources.test_world_aqi_api()
    print(f"   Status: {waqi_result['status']}")
    print(f"   Stations found: {waqi_result['stations']}")
    
    print("\n3️⃣ Testing OpenAQ historical endpoints...")
    openaq_results = sources.test_openaq_historical_endpoints()
    for endpoint, result in openaq_results.items():
        print(f"   {endpoint}: {result['status']}")
        if result['status'] == 'success':
            print(f"     Records: {result['records']}")
    
    print("\n4️⃣ Creating enhanced dataset...")
    enhanced_data = sources.create_enhanced_dataset()
    
    if len(enhanced_data) > 0:
        # Save enhanced dataset
        output_path = 'data/interim/enhanced_pm25_data.parquet'
        enhanced_data.to_parquet(output_path, index=False)
        print(f"✅ Enhanced dataset saved: {output_path}")
        print(f"   Total records: {len(enhanced_data)}")
        print(f"   Date range: {enhanced_data['datetime_utc'].min()} to {enhanced_data['datetime_utc'].max()}")
        print(f"   Data sources: {enhanced_data['data_source'].unique()}")
    else:
        print("❌ No enhanced dataset created")

if __name__ == "__main__":
    main()
