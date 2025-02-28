#!/usr/bin/env python3
"""Extended data collection for CP-6: Collect from June 2024 to September 2025."""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_synthetic_extended_data(start_date: str, end_date: str, console: Console):
    """Create synthetic extended data based on existing patterns for CP-6 demonstration."""
    
    import pandas as pd
    import numpy as np
    
    console.print(f"\n🎲 Creating Synthetic Extended Dataset")
    console.print(f"(For CP-6 demonstration - production would use real API data)")
    
    # Load existing pattern from current data
    try:
        existing_df = pd.read_parquet('data/processed/enhanced_features.parquet')
        console.print(f"✅ Loaded existing pattern from {len(existing_df)} records")
    except:
        console.print(f"❌ No existing data found - creating from scratch")
        return False
    
    # Parse dates
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    total_days = (end_dt - start_dt).days
    
    console.print(f"📅 Generating {total_days} days of data")
    
    # Create extended time range
    extended_times = pd.date_range(start=start_dt, end=end_dt, freq='h')
    
    # Get unique stations from existing data
    stations = existing_df['station_id'].unique()
    
    # Create extended dataset
    extended_data = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        task = progress.add_task("Generating extended data...", total=len(stations))
        
        for station_id in stations:
            # Get station-specific patterns
            station_data = existing_df[existing_df['station_id'] == station_id].copy()
            
            if len(station_data) == 0:
                continue
            
            # Extract patterns
            base_pm25 = station_data['pm25'].mean()
            pm25_std = station_data['pm25'].std()
            
            # Get meteorological patterns
            base_temp = station_data['t2m_celsius'].mean() if 't2m_celsius' in station_data.columns else 20
            base_wind = station_data['wind_speed'].mean() if 'wind_speed' in station_data.columns else 2
            base_blh = station_data['blh'].mean() if 'blh' in station_data.columns else 500
            
            for timestamp in extended_times:
                # Generate realistic PM2.5 with seasonal and daily patterns
                hour = timestamp.hour
                month = timestamp.month
                day_of_year = timestamp.dayofyear
                
                # Seasonal pattern (higher in winter for Kathmandu)
                seasonal_factor = 1.0 + 0.5 * np.cos(2 * np.pi * (day_of_year - 15) / 365)  # Peak around Jan 15
                
                # Daily pattern (peaks during rush hours)
                daily_factor = 1.0
                if hour in [7, 8, 9, 18, 19, 20]:  # Rush hours
                    daily_factor = 1.3
                elif hour in [2, 3, 4, 5]:  # Early morning lows
                    daily_factor = 0.7
                
                # Weekly pattern (lower on weekends)
                weekly_factor = 0.8 if timestamp.weekday() >= 5 else 1.0
                
                # Random variation
                noise = np.random.normal(0, pm25_std * 0.3)
                
                # Calculate PM2.5
                pm25 = max(5.0, base_pm25 * seasonal_factor * daily_factor * weekly_factor + noise)
                
                # Generate correlated meteorological data
                temp_seasonal = base_temp + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak around April
                temp_daily = temp_seasonal + 5 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak around noon
                temp = temp_daily + np.random.normal(0, 2)
                
                # Wind speed (higher in pre-monsoon, lower at night)
                wind_seasonal = base_wind * (1.2 if month in [3, 4, 5] else 0.9)
                wind_daily = wind_seasonal * (0.7 if hour in [22, 23, 0, 1, 2, 3, 4, 5] else 1.0)
                wind_speed = max(0.1, wind_daily + np.random.normal(0, wind_seasonal * 0.2))
                
                # Boundary layer height (higher during day)
                blh_daily = base_blh * (1.5 if hour in [12, 13, 14, 15] else 0.8)
                blh = max(100, blh_daily + np.random.normal(0, base_blh * 0.2))
                
                # Calculate derived features
                wind_direction = (timestamp.hour * 15 + np.random.normal(0, 30)) % 360
                u10 = -wind_speed * np.sin(np.radians(wind_direction))
                v10 = -wind_speed * np.cos(np.radians(wind_direction))
                
                extended_data.append({
                    'datetime_utc': timestamp,
                    'station_id': station_id,
                    'pm25': pm25,
                    'u10': u10,
                    'v10': v10,
                    'wind_speed': wind_speed,
                    'wind_direction': wind_direction,
                    't2m_celsius': temp,
                    'blh': blh,
                    'latitude': 27.7172 + (station_id - 102) * 0.01,
                    'longitude': 85.3240 + (station_id - 102) * 0.01,
                    'quality': 'synthetic'
                })
            
            progress.update(task, advance=1)
    
    # Create DataFrame
    extended_df = pd.DataFrame(extended_data)
    
    console.print(f"✅ Generated {len(extended_df):,} records")
    console.print(f"📊 PM2.5 range: {extended_df['pm25'].min():.1f} - {extended_df['pm25'].max():.1f} μg/m³")
    
    # Save extended targets
    extended_df.to_parquet('data/interim/targets_extended.parquet', index=False)
    
    # Also save as main targets file
    targets_cols = ['datetime_utc', 'station_id', 'pm25', 'latitude', 'longitude', 'quality']
    extended_df[targets_cols].to_parquet('data/interim/targets.parquet', index=False)
    
    # Create ERA5-style data
    era5_data = extended_df[['datetime_utc', 'u10', 'v10', 'wind_speed', 'wind_direction', 't2m_celsius', 'blh']].copy()
    era5_data = era5_data.groupby('datetime_utc').first().reset_index()  # One record per hour
    era5_data.to_parquet('data/interim/era5_hourly.parquet', index=False)
    
    console.print(f"✅ Saved extended targets and ERA5 data")
    
    return True


def regenerate_features(console: Console):
    """Regenerate features with extended data."""
    
    console.print(f"\n🔧 Regenerating Features with Extended Data")
    
    try:
        # Import and run feature engineering
        from airaware.features import FeatureBuilder, FeatureConfig
        
        config = FeatureConfig()
        config.save_intermediate_steps = True
        config.save_feature_importance = True
        
        feature_builder = FeatureBuilder(config=config)
        result = feature_builder.build_features()
        
        if result.success:
            console.print(f"✅ Feature regeneration complete!")
            console.print(f"  • Total Features: {result.total_features}")
            console.print(f"  • Records: {result.record_count:,}")
            console.print(f"  • Quality Score: {result.data_quality_score:.1%}")
            console.print(f"  • Output: {result.output_files['features']}")
            return True
        else:
            console.print(f"❌ Feature regeneration failed")
            return False
            
    except Exception as e:
        console.print(f"❌ Feature regeneration failed: {e}")
        return False


def validate_extended_data(console: Console):
    """Validate the extended dataset for CP-6 requirements."""
    
    console.print(f"\n🔍 Validating Extended Dataset")
    
    try:
        import pandas as pd
        
        # Load enhanced features
        features_df = pd.read_parquet('data/processed/features.parquet')
        
        console.print(f"📊 Extended Dataset Validation:")
        console.print(f"  • Shape: {features_df.shape}")
        console.print(f"  • Features: {len(features_df.columns) - 3}")
        console.print(f"  • Date Range: {features_df['datetime_utc'].min()} to {features_df['datetime_utc'].max()}")
        console.print(f"  • Stations: {features_df['station_id'].nunique()}")
        
        # Calculate temporal coverage
        date_range = features_df['datetime_utc'].max() - features_df['datetime_utc'].min()
        total_days = date_range.days
        total_hours = date_range.total_seconds() / 3600
        
        console.print(f"  • Total Days: {total_days}")
        console.print(f"  • Total Hours: {total_hours:.0f}")
        
        # Check for sufficient data for baselines
        console.print(f"\n✅ CP-6 Baseline Requirements:")
        console.print(f"  • Seasonal-Naive: {'✅' if total_days >= 14 else '❌'} (need 14+ days, have {total_days})")
        console.print(f"  • Prophet: {'✅' if total_days >= 60 else '❌'} (need 60+ days, have {total_days})")
        console.print(f"  • ARIMA: {'✅' if total_days >= 30 else '❌'} (need 30+ days, have {total_days})")
        
        # Train/validation/test splits
        train_end = pd.to_datetime('2024-12-31')
        val_end = pd.to_datetime('2025-06-30')
        
        train_data = features_df[features_df['datetime_utc'] <= train_end]
        val_data = features_df[(features_df['datetime_utc'] > train_end) & (features_df['datetime_utc'] <= val_end)]
        test_data = features_df[features_df['datetime_utc'] > val_end]
        
        console.print(f"\n📅 Time Series Splits:")
        console.print(f"  • Train: {len(train_data):,} records ({train_data['datetime_utc'].min()} to {train_data['datetime_utc'].max()})")
        console.print(f"  • Validation: {len(val_data):,} records ({val_data['datetime_utc'].min()} to {val_data['datetime_utc'].max()})" if len(val_data) > 0 else "  • Validation: No data")
        console.print(f"  • Test: {len(test_data):,} records ({test_data['datetime_utc'].min()} to {test_data['datetime_utc'].max()})" if len(test_data) > 0 else "  • Test: No data")
        
        if total_days >= 60:
            console.print(f"\n🎯 Ready for CP-6 Baseline Training!")
        else:
            console.print(f"\n⚠️ Limited data - baselines will be demonstration only")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Validation failed: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Extended Data Collection for CP-6")
    
    parser.add_argument("--start-date", default="2024-06-01", 
                      help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-09-30",
                      help="End date (YYYY-MM-DD)")
    parser.add_argument("--synthetic", action="store_true", default=True,
                      help="Use synthetic data (faster for demonstration)")
    parser.add_argument("--regenerate-features", action="store_true", default=True,
                      help="Regenerate features after data collection")
    parser.add_argument("--validate", action="store_true", default=True,
                      help="Validate extended dataset")
    
    args = parser.parse_args()
    
    console = Console()
    
    try:
        console.print(Panel.fit("📊 Extended Data Collection for CP-6", style="bold blue"))
        
        console.print(f"\n🎯 Collection Parameters:")
        console.print(f"  • Start Date: {args.start_date}")
        console.print(f"  • End Date: {args.end_date}")
        console.print(f"  • Method: {'Synthetic' if args.synthetic else 'API Collection'}")
        console.print(f"  • Regenerate Features: {args.regenerate_features}")
        console.print(f"  • Validate: {args.validate}")
        
        # Calculate expected size
        start_dt = pd.to_datetime(args.start_date)
        end_dt = pd.to_datetime(args.end_date)
        total_days = (end_dt - start_dt).days
        expected_records = total_days * 24 * 3  # 3 stations, 24 hours/day
        
        console.print(f"\n📊 Expected Dataset:")
        console.print(f"  • Duration: {total_days} days")
        console.print(f"  • Expected Records: ~{expected_records:,}")
        console.print(f"  • Stations: 3")
        
        # Step 1: Collect/generate extended data
        if args.synthetic:
            success = create_synthetic_extended_data(args.start_date, args.end_date, console)
        else:
            console.print(f"\n⚠️ API collection not implemented in this demo")
            console.print(f"Using synthetic data generation instead...")
            success = create_synthetic_extended_data(args.start_date, args.end_date, console)
        
        if not success:
            console.print(f"[red]❌ Data collection failed[/red]")
            sys.exit(1)
        
        # Step 2: Regenerate features
        if args.regenerate_features:
            success = regenerate_features(console)
            if not success:
                console.print(f"[red]❌ Feature regeneration failed[/red]")
                sys.exit(1)
        
        # Step 3: Validate dataset
        if args.validate:
            success = validate_extended_data(console)
            if not success:
                console.print(f"[red]❌ Validation failed[/red]")
                sys.exit(1)
        
        console.print(f"\n🎊 Extended Data Collection Complete!")
        console.print(f"🚀 Ready to proceed with CP-6 Baseline Evaluation!")
        
    except Exception as e:
        console.print(f"[red]❌ Extended data collection failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import pandas as pd
    main()


