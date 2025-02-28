#!/usr/bin/env python3
"""
CLI for testing dynamic ensemble weighting.

This script demonstrates dynamic ensemble weighting with adaptive model weights
based on recent performance.
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

from airaware.baselines import (
    SeasonalNaiveForecaster,
    ProphetBaseline, 
    ARIMABaseline,
    DynamicEnsemble,
    EnsembleWeightingConfig
)
from airaware.evaluation import ForecastingMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_data(n_stations: int = 3, n_hours: int = 500) -> pd.DataFrame:
    """Generate synthetic test data for dynamic ensemble testing."""
    
    logger.info(f"Generating test data: {n_stations} stations, {n_hours} hours")
    
    # Base time series with trend and seasonality
    base_time = pd.date_range(
        start=datetime.now() - timedelta(hours=n_hours),
        end=datetime.now(),
        freq='H'
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
        
        # Random noise
        noise = np.random.normal(0, 8, len(base_time))
        
        # Combine components
        pm25 = base_level + daily_cycle + weekly_cycle + trend + noise
        pm25 = np.maximum(pm25, 5)  # Ensure positive values
        
        # Add some extreme events
        extreme_indices = np.random.choice(len(base_time), size=10, replace=False)
        pm25[extreme_indices] *= 2.5
        
        station_data = pd.DataFrame({
            'datetime_utc': base_time,
            'station_id': station_id,
            'pm25': pm25,
            'u10': np.random.normal(3, 1, len(base_time)),
            'v10': np.random.normal(2, 1, len(base_time)),
            't2m': np.random.normal(20, 5, len(base_time)),
            'blh': np.random.normal(500, 200, len(base_time))
        })
        
        data.append(station_data)
    
    combined_data = pd.concat(data, ignore_index=True)
    combined_data = combined_data.sort_values(['station_id', 'datetime_utc']).reset_index(drop=True)
    
    logger.info(f"Generated {len(combined_data)} data points")
    return combined_data

def simulate_performance_variation(data: pd.DataFrame, 
                                 model_name: str,
                                 base_error: float = 10.0) -> Dict[str, float]:
    """Simulate varying performance for a model over time."""
    
    # Simulate different performance patterns for different models
    if model_name == "seasonal_naive":
        # Seasonal naive performs well in stable periods, poorly during transitions
        time_factor = np.sin(2 * np.pi * np.arange(len(data)) / (24 * 7))  # Weekly pattern
        performance_factor = 0.8 + 0.4 * time_factor
    elif model_name == "prophet":
        # Prophet adapts well to trends but struggles with sudden changes
        trend_factor = np.cumsum(np.random.normal(0, 0.01, len(data)))
        performance_factor = 0.9 + 0.2 * np.tanh(trend_factor)
    elif model_name == "arima":
        # ARIMA performs consistently but with some volatility
        volatility = np.random.normal(0, 0.1, len(data))
        performance_factor = 0.85 + volatility
    else:
        performance_factor = np.ones(len(data))
    
    # Calculate MAE with performance variation
    mae = base_error * performance_factor
    mae = np.maximum(mae, 2.0)  # Minimum MAE
    
    # Calculate other metrics based on MAE
    rmse = mae * 1.2  # RMSE typically higher than MAE
    smape = mae * 0.8  # sMAPE typically lower than MAE
    
    return {
        'mae': float(np.mean(mae)),
        'rmse': float(np.mean(rmse)),
        'smape': float(np.mean(smape))
    }

def test_dynamic_ensemble(data: pd.DataFrame, 
                         config: EnsembleWeightingConfig) -> Dict[str, any]:
    """Test dynamic ensemble with performance tracking."""
    
    logger.info("Testing dynamic ensemble with performance tracking")
    
    # Define models
    models = {
        'seasonal_naive': SeasonalNaiveForecaster,
        'prophet': ProphetBaseline,
        'arima': ARIMABaseline
    }
    
    # Create dynamic ensemble
    ensemble = DynamicEnsemble(models, config)
    
    # Split data for training and testing
    split_point = int(len(data) * 0.7)
    train_data = data.iloc[:split_point].copy()
    test_data = data.iloc[split_point:].copy()
    
    logger.info(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples")
    
    # Fit ensemble
    ensemble.fit(train_data, target_col='pm25', group_col='station_id')
    
    # Test dynamic weighting over time
    results = {
        'weight_history': [],
        'performance_history': [],
        'predictions': [],
        'actual_values': []
    }
    
    # Generate predictions in batches to simulate real-time updates
    batch_size = 24  # 24-hour batches
    n_batches = len(test_data) // batch_size
    
    metrics_calculator = ForecastingMetrics()
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_data))
        batch_data = test_data.iloc[start_idx:end_idx]
        
        # Generate timestamps for prediction
        timestamps = batch_data['datetime_utc'].tolist()
        
        # Simulate performance data for each model
        performance_data = {}
        for model_name in models.keys():
            # Simulate performance variation
            model_performance = simulate_performance_variation(batch_data, model_name)
            performance_data[model_name] = model_performance
            
            # Update ensemble performance tracker
            ensemble.update_performance(
                model_name,
                timestamps[0],
                model_performance
            )
        
        # Generate ensemble prediction
        forecast = ensemble.predict(
            timestamps,
            station_id=0,  # Use first station
            update_weights=True,
            performance_data=performance_data
        )
        
        # Store results
        current_weights = ensemble.get_current_weights()
        results['weight_history'].append({
            'timestamp': timestamps[0],
            'weights': current_weights.copy()
        })
        
        results['performance_history'].append({
            'timestamp': timestamps[0],
            'performance': performance_data.copy()
        })
        
        results['predictions'].extend(forecast.predictions)
        results['actual_values'].extend(batch_data['pm25'].tolist())
        
        logger.info(f"Batch {batch_idx + 1}/{n_batches}: Weights = {current_weights}")
    
    # Calculate final ensemble performance
    if results['predictions'] and results['actual_values']:
        predictions = np.array(results['predictions'])
        actuals = np.array(results['actual_values'])
        
        ensemble_metrics = metrics_calculator.evaluate_point_forecast(
            actuals, predictions
        )
        
        results['ensemble_metrics'] = {
            metric_name: metric_result.value for metric_name, metric_result in ensemble_metrics.items()
        }
    
    return results

def analyze_weight_evolution(results: Dict[str, any]) -> Dict[str, any]:
    """Analyze how ensemble weights evolved over time."""
    
    logger.info("Analyzing weight evolution")
    
    weight_history = results['weight_history']
    performance_history = results['performance_history']
    
    if not weight_history:
        return {'error': 'No weight history available'}
    
    # Extract weight evolution
    timestamps = [entry['timestamp'] for entry in weight_history]
    model_names = list(weight_history[0]['weights'].keys())
    
    weight_evolution = {}
    for model_name in model_names:
        weight_evolution[model_name] = [
            entry['weights'][model_name] for entry in weight_history
        ]
    
    # Calculate weight statistics
    weight_stats = {}
    for model_name in model_names:
        weights = weight_evolution[model_name]
        weight_stats[model_name] = {
            'mean_weight': float(np.mean(weights)),
            'std_weight': float(np.std(weights)),
            'min_weight': float(np.min(weights)),
            'max_weight': float(np.max(weights)),
            'final_weight': float(weights[-1]) if weights else 0.0
        }
    
    # Analyze weight stability
    weight_changes = {}
    for model_name in model_names:
        weights = weight_evolution[model_name]
        if len(weights) > 1:
            changes = np.abs(np.diff(weights))
            weight_changes[model_name] = {
                'mean_change': float(np.mean(changes)),
                'max_change': float(np.max(changes)),
                'total_changes': int(np.sum(changes > 0.01))  # Significant changes
            }
    
    return {
        'weight_evolution': weight_evolution,
        'weight_stats': weight_stats,
        'weight_changes': weight_changes,
        'timestamps': timestamps
    }

def main():
    """Main CLI function."""
    
    parser = argparse.ArgumentParser(description='Test dynamic ensemble weighting')
    parser.add_argument('--output-dir', type=str, default='data/artifacts',
                       help='Output directory for results')
    parser.add_argument('--n-stations', type=int, default=3,
                       help='Number of stations in test data')
    parser.add_argument('--n-hours', type=int, default=500,
                       help='Number of hours in test data')
    parser.add_argument('--weighting-method', type=str, default='performance_based',
                       choices=['equal', 'performance_based', 'adaptive'],
                       help='Ensemble weighting method')
    parser.add_argument('--performance-window', type=int, default=168,
                       help='Performance window in hours')
    parser.add_argument('--adaptation-rate', type=float, default=0.1,
                       help='Weight adaptation rate')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting dynamic ensemble test")
    
    try:
        # Generate test data
        data = generate_test_data(args.n_stations, args.n_hours)
        
        # Configure ensemble weighting
        config = EnsembleWeightingConfig(
            weighting_method=args.weighting_method,
            performance_window_hours=args.performance_window,
            adaptation_rate=args.adaptation_rate,
            weight_smoothing=True,
            smoothing_factor=0.3
        )
        
        logger.info(f"Using weighting method: {config.weighting_method}")
        logger.info(f"Performance window: {config.performance_window_hours} hours")
        logger.info(f"Adaptation rate: {config.adaptation_rate}")
        
        # Test dynamic ensemble
        results = test_dynamic_ensemble(data, config)
        
        # Analyze weight evolution
        analysis = analyze_weight_evolution(results)
        
        # Save results
        results_file = output_dir / 'dynamic_ensemble_results.json'
        
        # Convert datetime objects to strings for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if key == 'weight_history':
                serializable_results[key] = [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'weights': entry['weights']
                    }
                    for entry in value
                ]
            elif key == 'performance_history':
                serializable_results[key] = [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'performance': entry['performance']
                    }
                    for entry in value
                ]
            else:
                serializable_results[key] = value
        
        # Add analysis results
        serializable_results['analysis'] = analysis
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("DYNAMIC ENSEMBLE TEST RESULTS")
        print("="*60)
        
        if 'ensemble_metrics' in results:
            print(f"\nEnsemble Performance:")
            for metric, value in results['ensemble_metrics'].items():
                print(f"  {metric.upper()}: {value:.3f}")
        
        if 'weight_stats' in analysis:
            print(f"\nWeight Statistics:")
            for model_name, stats in analysis['weight_stats'].items():
                print(f"  {model_name}:")
                print(f"    Mean Weight: {stats['mean_weight']:.3f}")
                print(f"    Final Weight: {stats['final_weight']:.3f}")
                print(f"    Weight Range: [{stats['min_weight']:.3f}, {stats['max_weight']:.3f}]")
        
        if 'weight_changes' in analysis:
            print(f"\nWeight Stability:")
            for model_name, changes in analysis['weight_changes'].items():
                print(f"  {model_name}:")
                print(f"    Mean Change: {changes['mean_change']:.3f}")
                print(f"    Total Significant Changes: {changes['total_changes']}")
        
        print(f"\nDetailed results saved to: {results_file}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Dynamic ensemble test failed: {e}")
        raise

if __name__ == '__main__':
    main()
