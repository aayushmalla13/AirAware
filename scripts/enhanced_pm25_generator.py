#!/usr/bin/env python3
"""
Enhanced PM2.5 Data Generator for Kathmandu Valley
Generates comprehensive historical PM2.5 data based on meteorological patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPM25Generator:
    """Generate comprehensive PM2.5 data for Kathmandu Valley"""
    
    def __init__(self):
        self.kathmandu_stations = [
            {'id': 5509787, 'name': 'Baluwatar (SC-02)', 'lat': 27.7249438, 'lon': 85.331062},
            {'id': 5506835, 'name': 'Gaushala Chowk (SC-01)', 'lat': 27.707763, 'lon': 85.343189},
            {'id': 5633032, 'name': 'Nakhipot (SC-08)', 'lat': 27.6510811, 'lon': 85.3178568}
        ]
        
        # Kathmandu Valley PM2.5 characteristics based on research
        self.base_pm25 = 45.0  # Typical annual average
        self.seasonal_patterns = {
            'winter': {'factor': 1.4, 'months': [11, 12, 1, 2]},  # Higher due to heating
            'spring': {'factor': 1.1, 'months': [3, 4, 5]},       # Moderate
            'monsoon': {'factor': 0.7, 'months': [6, 7, 8, 9]},   # Lower due to rain
            'autumn': {'factor': 1.0, 'months': [10]}             # Baseline
        }
    
    def load_meteorological_data(self) -> pd.DataFrame:
        """Load ERA5 meteorological data"""
        try:
            era5_df = pd.read_parquet('data/interim/era5_hourly.parquet')
            logger.info(f"📊 Loaded {len(era5_df)} ERA5 records")
            return era5_df
        except Exception as e:
            logger.error(f"❌ Failed to load ERA5 data: {e}")
            return pd.DataFrame()
    
    def load_real_pm25_data(self) -> pd.DataFrame:
        """Load existing real PM2.5 data"""
        try:
            real_df = pd.read_parquet('data/interim/targets.parquet')
            logger.info(f"📊 Loaded {len(real_df)} real PM2.5 records")
            return real_df
        except Exception as e:
            logger.error(f"❌ Failed to load real PM2.5 data: {e}")
            return pd.DataFrame()
    
    def calculate_pm25_from_meteorology(self, row: pd.Series) -> float:
        """Calculate PM2.5 based on meteorological conditions"""
        
        # Base PM2.5 level
        pm25 = self.base_pm25
        
        # Wind speed effect (higher wind = lower PM2.5 due to dispersion)
        wind_speed = row.get('wind_speed', 2.0)
        wind_factor = max(0.3, 1.0 - (wind_speed - 1.0) * 0.15)
        
        # Temperature effect (higher temp = higher PM2.5 due to thermal inversion)
        temperature = row.get('t2m_celsius', 20.0)
        temp_factor = 1.0 + (temperature - 20.0) * 0.02
        
        # Boundary layer height effect (lower BLH = higher PM2.5)
        blh = row.get('blh', 500.0)
        blh_factor = max(0.5, 1.0 - (500.0 - blh) / 1000.0)
        
        # Seasonal variation
        month = row['datetime_utc'].month
        seasonal_factor = 1.0
        for season, info in self.seasonal_patterns.items():
            if month in info['months']:
                seasonal_factor = info['factor']
                break
        
        # Diurnal variation (higher at night due to inversion)
        hour = row['datetime_utc'].hour
        diurnal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Weekend effect (slightly lower due to reduced traffic)
        weekday = row['datetime_utc'].weekday()
        weekend_factor = 0.95 if weekday >= 5 else 1.0
        
        # Calculate final PM2.5
        pm25 = pm25 * wind_factor * temp_factor * blh_factor * seasonal_factor * diurnal_factor * weekend_factor
        
        # Add realistic random variation
        pm25 += np.random.normal(0, 8.0)
        
        # Ensure realistic range for Kathmandu Valley
        pm25 = max(10.0, min(300.0, pm25))
        
        return pm25
    
    def generate_comprehensive_dataset(self, start_date: str = '2024-01-01', end_date: str = '2025-09-30') -> pd.DataFrame:
        """Generate comprehensive PM2.5 dataset"""
        logger.info("🔧 Generating comprehensive PM2.5 dataset...")
        
        # Load meteorological data
        era5_df = self.load_meteorological_data()
        if len(era5_df) == 0:
            logger.error("❌ No meteorological data available")
            return pd.DataFrame()
        
        # Load real PM2.5 data for calibration
        real_df = self.load_real_pm25_data()
        
        # Generate synthetic data for each station
        all_data = []
        
        for station in self.kathmandu_stations:
            logger.info(f"📍 Generating data for {station['name']}")
            
            # Filter ERA5 data for this station's area (approximate)
            station_era5 = era5_df[
                (era5_df['datetime_utc'] >= start_date) &
                (era5_df['datetime_utc'] <= end_date)
            ].copy()
            
            # Generate PM2.5 for each meteorological record
            station_data = []
            for _, row in station_era5.iterrows():
                pm25 = self.calculate_pm25_from_meteorology(row)
                
                station_data.append({
                    'station_id': station['id'],
                    'sensor_id': f"synthetic_{station['id']}",
                    'parameter': 'pm25',
                    'pm25': pm25,
                    'unit': 'µg/m³',
                    'datetime_utc': row['datetime_utc'],
                    'latitude': station['lat'],
                    'longitude': station['lon'],
                    'quality_flag': 'synthetic',
                    'data_source': 'synthetic_meteorological',
                    'quality': 'estimated'
                })
            
            all_data.extend(station_data)
            logger.info(f"   Generated {len(station_data)} records")
        
        # Combine with real data
        synthetic_df = pd.DataFrame(all_data)
        
        if len(real_df) > 0:
            # Calibrate synthetic data to match real data patterns
            synthetic_df = self.calibrate_to_real_data(synthetic_df, real_df)
            
            # Combine datasets
            combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
            
            # Remove duplicates (keep real data over synthetic)
            combined_df = combined_df.drop_duplicates(
                subset=['datetime_utc', 'station_id'], 
                keep='first'
            )
        else:
            combined_df = synthetic_df
        
        # Sort by datetime
        combined_df = combined_df.sort_values('datetime_utc')
        
        logger.info(f"✅ Generated comprehensive dataset: {len(combined_df)} records")
        logger.info(f"   - Real data: {len(real_df)} records")
        logger.info(f"   - Synthetic data: {len(synthetic_df)} records")
        logger.info(f"   - Combined: {len(combined_df)} records")
        
        return combined_df
    
    def calibrate_to_real_data(self, synthetic_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
        """Calibrate synthetic data to match real data patterns"""
        logger.info("🔧 Calibrating synthetic data to real data patterns...")
        
        # Calculate statistics from real data
        real_stats = real_df.groupby('station_id')['pm25'].agg(['mean', 'std']).reset_index()
        
        # Calibrate each station's synthetic data
        calibrated_data = []
        
        for station_id in synthetic_df['station_id'].unique():
            station_synthetic = synthetic_df[synthetic_df['station_id'] == station_id].copy()
            station_real_stats = real_stats[real_stats['station_id'] == station_id]
            
            if len(station_real_stats) > 0:
                real_mean = station_real_stats['mean'].iloc[0]
                real_std = station_real_stats['std'].iloc[0]
                
                # Scale synthetic data to match real data distribution
                synthetic_mean = station_synthetic['pm25'].mean()
                synthetic_std = station_synthetic['pm25'].std()
                
                if synthetic_std > 0:
                    # Z-score normalization then rescale
                    station_synthetic['pm25'] = (
                        (station_synthetic['pm25'] - synthetic_mean) / synthetic_std * real_std + real_mean
                    )
                
                logger.info(f"   Station {station_id}: calibrated to mean={real_mean:.1f}, std={real_std:.1f}")
            
            calibrated_data.append(station_synthetic)
        
        return pd.concat(calibrated_data, ignore_index=True)
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """Validate the generated dataset"""
        logger.info("🔍 Validating generated dataset...")
        
        validation_results = {
            'total_records': len(df),
            'date_range': (df['datetime_utc'].min(), df['datetime_utc'].max()),
            'stations': df['station_id'].nunique(),
            'data_sources': df['data_source'].value_counts().to_dict(),
            'pm25_stats': {
                'mean': df['pm25'].mean(),
                'std': df['pm25'].std(),
                'min': df['pm25'].min(),
                'max': df['pm25'].max(),
                'median': df['pm25'].median()
            },
            'quality_score': self.calculate_quality_score(df)
        }
        
        logger.info(f"✅ Validation complete:")
        logger.info(f"   Records: {validation_results['total_records']}")
        logger.info(f"   Date range: {validation_results['date_range'][0]} to {validation_results['date_range'][1]}")
        logger.info(f"   Stations: {validation_results['stations']}")
        logger.info(f"   PM2.5 mean: {validation_results['pm25_stats']['mean']:.1f} ± {validation_results['pm25_stats']['std']:.1f}")
        logger.info(f"   Quality score: {validation_results['quality_score']:.1f}%")
        
        return validation_results
    
    def calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate quality score for the dataset"""
        score = 100.0
        
        # Penalize for missing data
        missing_pct = df['pm25'].isna().sum() / len(df) * 100
        score -= missing_pct * 2
        
        # Penalize for unrealistic values
        unrealistic_pct = ((df['pm25'] < 5) | (df['pm25'] > 500)).sum() / len(df) * 100
        score -= unrealistic_pct * 1
        
        # Reward for good temporal coverage
        date_range = (df['datetime_utc'].max() - df['datetime_utc'].min()).days
        if date_range > 365:
            score += 10  # Bonus for >1 year of data
        
        return max(0, min(100, score))

def main():
    """Main function to generate enhanced PM2.5 dataset"""
    print("🌍 Enhanced PM2.5 Data Generator for Kathmandu Valley")
    print("=" * 60)
    
    generator = EnhancedPM25Generator()
    
    # Generate comprehensive dataset
    enhanced_data = generator.generate_comprehensive_dataset()
    
    if len(enhanced_data) > 0:
        # Validate dataset
        validation_results = generator.validate_dataset(enhanced_data)
        
        # Save enhanced dataset
        output_path = 'data/interim/enhanced_pm25_data.parquet'
        enhanced_data.to_parquet(output_path, index=False)
        
        print(f"\n✅ Enhanced PM2.5 dataset created!")
        print(f"📁 Saved to: {output_path}")
        print(f"📊 Total records: {len(enhanced_data):,}")
        print(f"📅 Date range: {validation_results['date_range'][0]} to {validation_results['date_range'][1]}")
        print(f"🏢 Stations: {validation_results['stations']}")
        print(f"📈 PM2.5 average: {validation_results['pm25_stats']['mean']:.1f} ± {validation_results['pm25_stats']['std']:.1f} μg/m³")
        print(f"⭐ Quality score: {validation_results['quality_score']:.1f}%")
        
        # Show data source breakdown
        print(f"\n📊 Data sources:")
        for source, count in validation_results['data_sources'].items():
            print(f"   {source}: {count:,} records")
        
    else:
        print("❌ Failed to generate enhanced dataset")

if __name__ == "__main__":
    main()
