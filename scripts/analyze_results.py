#!/usr/bin/env python3
"""
Data Analysis Script for Full-Scale OpenAQ Ingestion Results

This script provides comprehensive analysis of the collected air quality data
including data quality assessment, parameter coverage, temporal analysis,
and prediction readiness evaluation.

Usage:
    python analyze_results.py [data_directory]

Example:
    python analyze_results.py full_data
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class OpenAQDataAnalyzer:
    """Comprehensive analyzer for OpenAQ ingestion results"""
    
    def __init__(self, data_dir: str = "full_data"):
        """
        Initialize the analyzer.
        
        Args:
            data_dir: Directory containing the ingested data files
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} not found!")
            
        print(f"🔍 OpenAQ Data Analyzer initialized for: {self.data_dir}")
        
        # Find all data files
        self.locations_file = self.find_latest_file("all_locations_*.parquet")
        self.sensors_file = self.find_latest_file("all_sensors_*.parquet")
        self.measurement_files = list(self.data_dir.glob("measurements_*.parquet"))
        self.summary_file = self.find_latest_file("ingestion_summary_*.json")
        
        print(f"📊 Found {len(self.measurement_files)} measurement files")
        
    def find_latest_file(self, pattern: str) -> Path:
        """Find the latest file matching the pattern"""
        files = list(self.data_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        return max(files, key=lambda f: f.stat().st_mtime)
        
    def load_ingestion_summary(self) -> dict:
        """Load the ingestion summary JSON"""
        if not self.summary_file.exists():
            return {}
            
        with open(self.summary_file, 'r') as f:
            return json.load(f)
            
    def analyze_overview(self):
        """Provide high-level overview of the collected data"""
        print("\n" + "="*80)
        print("📋 DATA COLLECTION OVERVIEW")
        print("="*80)
        
        # Load summary if available
        summary = self.load_ingestion_summary()
        if summary:
            info = summary.get('ingestion_info', {})
            data_summary = summary.get('data_summary', {})
            
            print(f"⏱️  Collection completed: {info.get('end_time', 'Unknown')}")
            print(f"🕐 Duration: {info.get('duration_hours', 0):.2f} hours")
            print(f"🌍 Countries: {list(info.get('target_countries', {}).keys())}")
            print(f"📍 Total locations: {data_summary.get('total_locations', 0):,}")
            print(f"🔬 Total sensors: {data_summary.get('total_sensors', 0):,}")
            print(f"📊 Total measurements: {data_summary.get('total_measurements', 0):,}")
            print(f"💾 Data batches: {data_summary.get('batch_count', 0)}")
        
        # File size analysis
        total_size = 0
        for file in self.data_dir.iterdir():
            if file.is_file() and file.suffix == '.parquet':
                total_size += file.stat().st_size
                
        print(f"💿 Total data size: {total_size / (1024**3):.2f} GB")
        print(f"📁 Data files: {len(list(self.data_dir.glob('*.parquet')))} parquet files")
        
    def analyze_locations(self):
        """Analyze location data"""
        print("\n" + "="*80)
        print("📍 LOCATION ANALYSIS")
        print("="*80)
        
        locations_df = pd.read_parquet(self.locations_file)
        
        print(f"Total locations: {len(locations_df):,}")
        
        # Country breakdown
        if 'country' in locations_df.columns:
            country_counts = locations_df['country'].apply(
                lambda x: x.get('name') if isinstance(x, dict) else str(x)
            ).value_counts()
            
            print("\n🌍 Locations by country:")
            for country, count in country_counts.items():
                print(f"  {country}: {count:,} locations")
                
        # Geographic distribution
        if 'coordinates' in locations_df.columns:
            coords = locations_df['coordinates'].apply(
                lambda x: (x.get('latitude'), x.get('longitude')) 
                if isinstance(x, dict) else (None, None)
            )
            coords_df = pd.DataFrame(coords.tolist(), columns=['lat', 'lon'])
            coords_df = coords_df.dropna()
            
            print(f"\n🗺️  Geographic coverage:")
            print(f"  Latitude range: {coords_df['lat'].min():.3f} to {coords_df['lat'].max():.3f}")
            print(f"  Longitude range: {coords_df['lon'].min():.3f} to {coords_df['lon'].max():.3f}")
            
        # Mobile vs fixed sensors
        if 'isMobile' in locations_df.columns:
            mobile_count = locations_df['isMobile'].sum()
            fixed_count = len(locations_df) - mobile_count
            print(f"\n📱 Station types:")
            print(f"  Fixed stations: {fixed_count:,}")
            print(f"  Mobile stations: {mobile_count:,}")
            
    def analyze_sensors(self):
        """Analyze sensor data and parameters"""
        print("\n" + "="*80)
        print("🔬 SENSOR ANALYSIS")
        print("="*80)
        
        sensors_df = pd.read_parquet(self.sensors_file)
        
        print(f"Total sensors: {len(sensors_df):,}")
        
        # Parameter analysis
        if 'parameter' in sensors_df.columns:
            # Extract parameter info
            sensors_df['param_name'] = sensors_df['parameter'].apply(
                lambda x: x.get('name') if isinstance(x, dict) else str(x)
            )
            sensors_df['param_units'] = sensors_df['parameter'].apply(
                lambda x: x.get('units') if isinstance(x, dict) else str(x)
            )
            sensors_df['param_display'] = sensors_df['parameter'].apply(
                lambda x: x.get('displayName') if isinstance(x, dict) else str(x)
            )
            
            param_summary = sensors_df.groupby(['param_name', 'param_units']).agg({
                'sensor_id': 'count',
                'param_display': 'first'
            }).reset_index().sort_values('sensor_id', ascending=False)
            
            print("\n📊 Available parameters:")
            for _, row in param_summary.iterrows():
                display_name = row['param_display'] if row['param_display'] != 'None' else row['param_name']
                print(f"  {row['param_name']} ({row['param_units']}): {row['sensor_id']:,} sensors - {display_name}")
                
        # Country distribution of sensors
        if 'country' in sensors_df.columns:
            country_sensor_counts = sensors_df['country'].apply(
                lambda x: x.get('name') if isinstance(x, dict) else str(x)
            ).value_counts()
            
            print("\n🌍 Sensors by country:")
            for country, count in country_sensor_counts.items():
                print(f"  {country}: {count:,} sensors")
                
    def analyze_measurements_sample(self, sample_size: int = 100000):
        """Analyze a sample of measurements for data quality assessment"""
        print("\n" + "="*80)
        print("📊 MEASUREMENTS ANALYSIS (Sample)")
        print("="*80)
        
        # Load a sample from the first few measurement files
        sample_data = []
        current_size = 0
        
        for file in sorted(self.measurement_files)[:3]:  # First 3 files
            if current_size >= sample_size:
                break
                
            df = pd.read_parquet(file)
            remaining = sample_size - current_size
            
            if len(df) > remaining:
                df = df.sample(n=remaining, random_state=42)
                
            sample_data.append(df)
            current_size += len(df)
            
        if not sample_data:
            print("❌ No measurement data found!")
            return
            
        measurements_df = pd.concat(sample_data, ignore_index=True)
        print(f"Analyzing sample of {len(measurements_df):,} measurements")
        
        # Basic statistics
        print(f"\n📊 Sample statistics:")
        print(f"  Date range: {measurements_df['date'].min()} to {measurements_df['date'].max()}")
        print(f"  Unique sensors: {measurements_df['sensor_id'].nunique():,}")
        print(f"  Unique locations: {measurements_df['location_id'].nunique():,}")
        
        # Parameter distribution
        if 'parameter' in measurements_df.columns:
            param_counts = measurements_df['parameter'].value_counts()
            print(f"\n🧪 Parameter distribution (sample):")
            for param, count in param_counts.head(10).items():
                percentage = (count / len(measurements_df)) * 100
                print(f"  {param}: {count:,} ({percentage:.1f}%)")
                
        # Value statistics
        if 'value' in measurements_df.columns:
            print(f"\n📈 Value statistics:")
            print(f"  Mean: {measurements_df['value'].mean():.3f}")
            print(f"  Median: {measurements_df['value'].median():.3f}")
            print(f"  Std: {measurements_df['value'].std():.3f}")
            print(f"  Min: {measurements_df['value'].min():.3f}")
            print(f"  Max: {measurements_df['value'].max():.3f}")
            print(f"  Null values: {measurements_df['value'].isnull().sum():,}")
            
        # Data quality flags
        quality_columns = ['coordinates_lat', 'coordinates_lon', 'timezone']
        available_quality_cols = [col for col in quality_columns if col in measurements_df.columns]
        
        if available_quality_cols:
            print(f"\n🔍 Data quality assessment:")
            for col in available_quality_cols:
                null_count = measurements_df[col].isnull().sum()
                null_pct = (null_count / len(measurements_df)) * 100
                print(f"  {col}: {null_count:,} nulls ({null_pct:.1f}%)")
                
    def assess_prediction_readiness(self):
        """Assess the data's readiness for air quality prediction models"""
        print("\n" + "="*80)
        print("🤖 PREDICTION READINESS ASSESSMENT")
        print("="*80)
        
        # Load sensor data for parameter analysis
        sensors_df = pd.read_parquet(self.sensors_file)
        
        # Extract parameter information
        sensors_df['param_name'] = sensors_df['parameter'].apply(
            lambda x: x.get('name') if isinstance(x, dict) else str(x)
        )
        
        # Key air quality parameters for prediction
        key_parameters = {
            'pm25': ['pm25', 'PM2.5', 'pm2.5'],
            'pm10': ['pm10', 'PM10', 'pm1.0'],
            'no2': ['no2', 'NO2'],
            'o3': ['o3', 'O3', 'ozone'],
            'so2': ['so2', 'SO2'],
            'co': ['co', 'CO'],
            'temperature': ['temperature', 'temp'],
            'humidity': ['humidity', 'rh'],
            'pressure': ['pressure', 'bp']
        }
        
        print("🎯 Key parameters for air quality prediction:")
        
        param_availability = {}
        for key, variants in key_parameters.items():
            matching_sensors = sensors_df[
                sensors_df['param_name'].str.lower().isin([v.lower() for v in variants])
            ]
            sensor_count = len(matching_sensors)
            param_availability[key] = sensor_count
            
            status = "✅" if sensor_count > 0 else "❌"
            print(f"  {status} {key.upper()}: {sensor_count:,} sensors")
            
        # Spatial coverage assessment
        countries = sensors_df['country'].apply(
            lambda x: x.get('name') if isinstance(x, dict) else str(x)
        ).value_counts()
        
        print(f"\n🗺️  Spatial coverage:")
        print(f"  Countries: {len(countries)}")
        print(f"  Total locations: {sensors_df['location_id'].nunique():,}")
        
        # Temporal coverage (from sample)
        sample_file = self.measurement_files[0] if self.measurement_files else None
        if sample_file:
            sample_df = pd.read_parquet(sample_file)
            if 'date' in sample_df.columns:
                sample_df['date'] = pd.to_datetime(sample_df['date'])
                date_range = sample_df['date'].max() - sample_df['date'].min()
                print(f"  Temporal span: {date_range.days} days")
                
        # Prediction model recommendations
        print(f"\n🚀 Prediction model recommendations:")
        
        if param_availability.get('pm25', 0) > 100:
            print("  ✅ PM2.5 forecasting: Excellent data availability")
            
        if param_availability.get('temperature', 0) > 50 and param_availability.get('humidity', 0) > 50:
            print("  ✅ Weather-based prediction: Good meteorological data")
            
        if len(countries) >= 3:
            print("  ✅ Regional modeling: Multi-country coverage available")
            
        if sensors_df['location_id'].nunique() > 100:
            print("  ✅ Spatial interpolation: Dense sensor network")
            
        # Data completeness score
        available_key_params = sum(1 for count in param_availability.values() if count > 0)
        completeness_score = (available_key_params / len(key_parameters)) * 100
        
        print(f"\n📊 Data completeness score: {completeness_score:.1f}%")
        
        if completeness_score >= 80:
            print("  🟢 Excellent - Ready for comprehensive prediction models")
        elif completeness_score >= 60:
            print("  🟡 Good - Suitable for specific parameter prediction")
        elif completeness_score >= 40:
            print("  🟠 Fair - Limited prediction capabilities")
        else:
            print("  🔴 Poor - Significant data gaps")
            
    def generate_data_summary_report(self):
        """Generate a comprehensive data summary report"""
        report_file = self.data_dir / f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        print(f"\n📝 Generating comprehensive report: {report_file}")
        
        # Redirect print output to file
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = report_buffer = StringIO()
        
        try:
            self.analyze_overview()
            self.analyze_locations()
            self.analyze_sensors()
            self.analyze_measurements_sample()
            self.assess_prediction_readiness()
            
        finally:
            sys.stdout = old_stdout
            
        # Write report to file
        with open(report_file, 'w') as f:
            f.write("OpenAQ Data Analysis Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(report_buffer.getvalue())
            
        print(f"✅ Report saved to: {report_file}")
        
    def run_complete_analysis(self):
        """Run all analysis components"""
        print("🔍 Starting comprehensive data analysis...")
        
        self.analyze_overview()
        self.analyze_locations()
        self.analyze_sensors()
        self.analyze_measurements_sample()
        self.assess_prediction_readiness()
        self.generate_data_summary_report()
        
        print("\n🎉 Analysis complete!")


def main():
    """Main execution function"""
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/full_data"
    
    try:
        analyzer = OpenAQDataAnalyzer(data_dir)
        analyzer.run_complete_analysis()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure you've run the full ingestion script first!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()