#!/usr/bin/env python3
"""
CLI for testing enhanced Prophet baseline with external regressors.

This script demonstrates the enhanced Prophet model with meteorological,
calendar, and pollution-related external regressors.
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from airaware.baselines import ProphetBaseline, ProphetConfig
from airaware.evaluation import ForecastingMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_enhanced_test_data(n_stations: int = 3, n_hours: int = 500) -> pd.DataFrame:
    """Generate synthetic test data with meteorological and calendar features."""
    
    logger.info(f"Generating enhanced test data: {n_stations} stations, {n_hours} hours")
    
    # Base time series with trend and seasonality
    base_time = pd.date_range(
        start=datetime.now() - timedelta(hours=n_hours),
        end=datetime.now(),
        freq='h'
    )
    
    data = []
    
    for station_id in range(n_stations):
        # Generate realistic PM2.5 patterns
        np.random.seed(42 + station_id)  # Reproducible but different per station
        
        # Base level with station-specific offset
        base_level = 30 + station_id * 10
        
        # Daily cycle (higher during day, lower at night)
        daily_cycle = 15 * np.sin(2 * np.pi * np.arange(len(base_time)) / 24)
        
        # Weekly cycle (weekend effect)
        weekly_cycle = 5 * np.sin(2 * np.pi * np.arange(len(base_time)) / (24 * 7))
        
        # Trend (gradual increase)
        trend = 0.01 * np.arange(len(base_time))
        
        # Weather effects
        wind_speed = np.random.normal(3, 1, len(base_time))
        wind_effect = -2 * wind_speed  # Wind disperses pollution
        
        temperature = np.random.normal(20, 5, len(base_time))
        temp_effect = 0.5 * (temperature - 20)  # Temperature affects pollution
        
        # Weekend effect
        weekend_effect = np.where(base_time.dayofweek >= 5, -5, 0)
        
        # Rush hour effect
        rush_hour_effect = np.where(
            np.isin(base_time.hour, [7, 8, 17, 18]), 5, 0
        )
        
        # Random noise
        noise = np.random.normal(0, 8, len(base_time))
        
        # Combine components
        pm25 = (base_level + daily_cycle + weekly_cycle + trend + 
                wind_effect + temp_effect + weekend_effect + rush_hour_effect + noise)
        pm25 = np.maximum(pm25, 5)  # Ensure positive values
        
        # Add some extreme events
        extreme_indices = np.random.choice(len(base_time), size=10, replace=False)
        pm25[extreme_indices] *= 2.5
        
        station_data = pd.DataFrame({
            'datetime_utc': base_time,
            'station_id': station_id,
            'pm25': pm25,
            'u10': np.random.normal(2, 1, len(base_time)),
            'v10': np.random.normal(1, 1, len(base_time)),
            't2m': temperature + 273.15,  # Convert to Kelvin
            'blh': np.random.normal(500, 200, len(base_time)),
            'precip': np.random.exponential(0.5, len(base_time))
        })
        
        data.append(station_data)
    
    combined_data = pd.concat(data, ignore_index=True)
    combined_data = combined_data.sort_values(['station_id', 'datetime_utc']).reset_index(drop=True)
    
    logger.info(f"Generated {len(combined_data)} data points with enhanced features")
    return combined_data

def test_enhanced_prophet(data: pd.DataFrame, 
                         config: ProphetConfig) -> Dict[str, any]:
    """Test enhanced Prophet baseline with external regressors."""
    
    logger.info("Testing enhanced Prophet baseline with external regressors")
    
    # Create enhanced Prophet model
    prophet_model = ProphetBaseline(config)
    
    # Split data for training and testing
    split_point = int(len(data) * 0.7)
    train_data = data.iloc[:split_point].copy()
    test_data = data.iloc[split_point:].copy()
    
    logger.info(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples")
    
    # Fit model
    prophet_model.fit(train_data, target_col='pm25', group_col='station_id')
    
    # Generate predictions
    timestamps = test_data['datetime_utc'].tolist()
    forecast = prophet_model.predict(timestamps, station_id=0)
    
    # Calculate metrics
    metrics_calculator = ForecastingMetrics()
    actuals = test_data['pm25'].tolist()
    predictions = forecast.predictions
    
    # Ensure same length
    min_length = min(len(actuals), len(predictions))
    actuals = actuals[:min_length]
    predictions = predictions[:min_length]
    
    metrics = metrics_calculator.evaluate_point_forecast(actuals, predictions)
    
    results = {
        'forecast': {
            'predictions': predictions,
            'timestamps': forecast.timestamps,
            'confidence_intervals': forecast.confidence_intervals,
            'trend': forecast.trend,
            'seasonal_components': forecast.seasonal_components,
            'forecast_metadata': forecast.forecast_metadata
        },
        'metrics': {
            metric_name: metric_result.value for metric_name, metric_result in metrics.items()
        },
        'config': {
            'include_meteorological': config.include_meteorological,
            'include_calendar_events': config.include_calendar_events,
            'include_pollution_features': config.include_pollution_features,
            'meteo_features': config.meteo_features,
            'pollution_features': config.pollution_features
        }
    }
    
    return results

def compare_prophet_configurations(data: pd.DataFrame) -> Dict[str, any]:
    """Compare different Prophet configurations."""
    
    logger.info("Comparing Prophet configurations")
    
    configurations = {
        'basic': ProphetConfig(
            include_meteorological=False,
            include_calendar_events=False,
            include_pollution_features=False
        ),
        'meteorological_only': ProphetConfig(
            include_meteorological=True,
            include_calendar_events=False,
            include_pollution_features=False
        ),
        'calendar_only': ProphetConfig(
            include_meteorological=False,
            include_calendar_events=True,
            include_pollution_features=False
        ),
        'pollution_only': ProphetConfig(
            include_meteorological=False,
            include_calendar_events=False,
            include_pollution_features=True
        ),
        'enhanced': ProphetConfig(
            include_meteorological=True,
            include_calendar_events=True,
            include_pollution_features=True
        )
    }
    
    results = {}
    
    for config_name, config in configurations.items():
        logger.info(f"Testing configuration: {config_name}")
        
        try:
            result = test_enhanced_prophet(data, config)
            results[config_name] = result
            logger.info(f"✅ {config_name} completed successfully")
        except Exception as e:
            logger.error(f"❌ {config_name} failed: {e}")
            results[config_name] = {'error': str(e)}
    
    return results

def analyze_regressor_importance(results: Dict[str, any]) -> Dict[str, any]:
    """Analyze the importance of different regressor types."""
    
    logger.info("Analyzing regressor importance")
    
    analysis = {}
    
    for config_name, result in results.items():
        if 'error' in result:
            continue
        
        metrics = result.get('metrics', {})
        analysis[config_name] = {
            'mae': metrics.get('MAE', float('inf')),
            'rmse': metrics.get('RMSE', float('inf')),
            'smape': metrics.get('SMAPE', float('inf'))
        }
    
    # Find best configuration
    if analysis:
        best_mae = min(analysis.items(), key=lambda x: x[1]['mae'])
        best_rmse = min(analysis.items(), key=lambda x: x[1]['rmse'])
        best_smape = min(analysis.items(), key=lambda x: x[1]['smape'])
        
        analysis['best_configurations'] = {
            'mae': best_mae[0],
            'rmse': best_rmse[0],
            'smape': best_smape[0]
        }
    
    return analysis

def main():
    """Main CLI function."""
    
    parser = argparse.ArgumentParser(description='Test enhanced Prophet baseline with external regressors')
    parser.add_argument('--output-dir', type=str, default='data/artifacts',
                       help='Output directory for results')
    parser.add_argument('--n-stations', type=int, default=3,
                       help='Number of stations in test data')
    parser.add_argument('--n-hours', type=int, default=500,
                       help='Number of hours in test data')
    parser.add_argument('--test-single', action='store_true',
                       help='Test only the enhanced configuration')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting enhanced Prophet baseline test")
    
    try:
        # Generate test data
        data = generate_enhanced_test_data(args.n_stations, args.n_hours)
        
        if args.test_single:
            # Test only enhanced configuration
            config = ProphetConfig(
                include_meteorological=True,
                include_calendar_events=True,
                include_pollution_features=True
            )
            
            results = test_enhanced_prophet(data, config)
            
            # Save results
            results_file = output_dir / 'enhanced_prophet_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_file}")
            
            # Print summary
            print("\n" + "="*60)
            print("ENHANCED PROPHET BASELINE TEST RESULTS")
            print("="*60)
            
            if 'metrics' in results:
                print(f"\nPerformance Metrics:")
                for metric, value in results['metrics'].items():
                    print(f"  {metric}: {value:.3f}")
            
            if 'forecast' in results and 'forecast_metadata' in results['forecast']:
                metadata = results['forecast']['forecast_metadata']
                print(f"\nModel Configuration:")
                print(f"  Model Type: {metadata.get('model_type', 'unknown')}")
                print(f"  Features Used: {metadata.get('features_used', [])}")
            
            print(f"\nDetailed results saved to: {results_file}")
            print("="*60)
            
        else:
            # Compare different configurations
            results = compare_prophet_configurations(data)
            
            # Analyze regressor importance
            analysis = analyze_regressor_importance(results)
            
            # Save results
            results_file = output_dir / 'prophet_configuration_comparison.json'
            
            # Convert datetime objects to strings for JSON serialization
            serializable_results = {}
            for config_name, result in results.items():
                if 'error' in result:
                    serializable_results[config_name] = result
                else:
                    serializable_results[config_name] = {
                        'metrics': result.get('metrics', {}),
                        'config': result.get('config', {}),
                        'forecast_metadata': result.get('forecast', {}).get('forecast_metadata', {})
                    }
            
            serializable_results['analysis'] = analysis
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_file}")
            
            # Print summary
            print("\n" + "="*60)
            print("PROPHET CONFIGURATION COMPARISON RESULTS")
            print("="*60)
            
            print(f"\nConfiguration Performance:")
            for config_name, result in results.items():
                if 'error' in result:
                    print(f"  {config_name}: ERROR - {result['error']}")
                else:
                    metrics = result.get('metrics', {})
                    print(f"  {config_name}:")
                    print(f"    MAE: {metrics.get('MAE', 'N/A'):.3f}")
                    print(f"    RMSE: {metrics.get('RMSE', 'N/A'):.3f}")
                    smape_val = metrics.get('SMAPE', 'N/A')
        if isinstance(smape_val, (int, float)):
            print(f"    SMAPE: {smape_val:.3f}")
        else:
            print(f"    SMAPE: {smape_val}")
            
            if 'best_configurations' in analysis:
                print(f"\nBest Configurations:")
                for metric, config in analysis['best_configurations'].items():
                    print(f"  {metric.upper()}: {config}")
            
            print(f"\nDetailed results saved to: {results_file}")
            print("="*60)
        
    except Exception as e:
        logger.error(f"Enhanced Prophet test failed: {e}")
        raise

if __name__ == '__main__':
    main()
