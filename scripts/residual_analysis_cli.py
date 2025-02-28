#!/usr/bin/env python3
"""
CLI for testing residual analysis and diagnostic plots.

This script demonstrates comprehensive residual analysis for baseline
forecasting models with statistical tests and diagnostic plots.
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
    DynamicEnsemble
)
from airaware.evaluation import (
    ResidualAnalyzer,
    ResidualAnalysisConfig,
    ForecastingMetrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_data(n_stations: int = 3, n_hours: int = 500) -> pd.DataFrame:
    """Generate synthetic test data for residual analysis."""
    
    logger.info(f"Generating test data: {n_stations} stations, {n_hours} hours")
    
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

def test_model_residuals(data: pd.DataFrame, 
                        model_class,
                        model_name: str,
                        config: Optional[ResidualAnalysisConfig] = None) -> Dict[str, any]:
    """Test residual analysis for a specific model."""
    
    logger.info(f"Testing residual analysis for {model_name}")
    
    # Create residual analyzer
    analyzer = ResidualAnalyzer(config)
    
    # Split data for training and testing
    split_point = int(len(data) * 0.7)
    train_data = data.iloc[:split_point].copy()
    test_data = data.iloc[split_point:].copy()
    
    logger.info(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples")
    
    # Create and fit model
    if model_name == "dynamic_ensemble":
        models = {
            'seasonal_naive': SeasonalNaiveForecaster,
            'prophet': ProphetBaseline,
            'arima': ARIMABaseline
        }
        model = DynamicEnsemble(models)
    else:
        model = model_class()
    
    model.fit(train_data, target_col='pm25', group_col='station_id')
    
    # Generate predictions
    timestamps = test_data['datetime_utc'].tolist()
    forecast = model.predict(timestamps, station_id=0)
    
    # Get actual values
    actuals = test_data['pm25'].tolist()
    predictions = forecast.predictions
    
    # Ensure same length
    min_length = min(len(actuals), len(predictions))
    actuals = actuals[:min_length]
    predictions = predictions[:min_length]
    timestamps = timestamps[:min_length]
    
    # Perform residual analysis
    results = analyzer.analyze_residuals(
        actuals=actuals,
        predictions=predictions,
        timestamps=timestamps,
        model_name=model_name
    )
    
    return results

def compare_model_residuals(data: pd.DataFrame, 
                          config: Optional[ResidualAnalysisConfig] = None) -> Dict[str, any]:
    """Compare residual analysis across different models."""
    
    logger.info("Comparing residual analysis across models")
    
    models_to_test = [
        (SeasonalNaiveForecaster, "seasonal_naive"),
        (ProphetBaseline, "prophet"),
        (ARIMABaseline, "arima"),
        (DynamicEnsemble, "dynamic_ensemble")
    ]
    
    results = {}
    
    for model_class, model_name in models_to_test:
        try:
            result = test_model_residuals(data, model_class, model_name, config)
            results[model_name] = result
            logger.info(f"✅ {model_name} residual analysis completed")
        except Exception as e:
            logger.error(f"❌ {model_name} residual analysis failed: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

def generate_comparison_report(results: Dict[str, any]) -> str:
    """Generate a comparison report across models."""
    
    report = """
RESIDUAL ANALYSIS COMPARISON REPORT
==================================

"""
    
    # Summary table
    report += "MODEL PERFORMANCE SUMMARY:\n"
    report += "-" * 80 + "\n"
    report += f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'sMAPE':<10} {'Residual Std':<15}\n"
    report += "-" * 80 + "\n"
    
    for model_name, result in results.items():
        if 'error' in result:
            report += f"{model_name:<20} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<15}\n"
        else:
            stats = result.get('basic_statistics', {})
            mae = stats.get('mae', 'N/A')
            rmse = stats.get('rmse', 'N/A')
            smape = stats.get('smape', 'N/A')
            residual_std = stats.get('residual_std', 'N/A')
            
            report += f"{model_name:<20} {mae:<10.3f} {rmse:<10.3f} {smape:<10.2f} {residual_std:<15.3f}\n"
    
    report += "\n"
    
    # Statistical tests summary
    report += "STATISTICAL TESTS SUMMARY:\n"
    report += "-" * 80 + "\n"
    
    for model_name, result in results.items():
        if 'error' in result:
            continue
        
        report += f"\n{model_name.upper()}:\n"
        tests = result.get('statistical_tests', {})
        
        if 'shapiro_wilk' in tests:
            sw = tests['shapiro_wilk']
            report += f"  - Normality (Shapiro-Wilk): {'✓' if sw['is_normal'] else '✗'} (p={sw['p_value']:.4f})\n"
        
        if 'ljung_box' in tests:
            lb = tests['ljung_box']
            report += f"  - Autocorrelation (Ljung-Box): {'✓' if lb['is_white_noise'] else '✗'}\n"
        
        if 'durbin_watson' in tests:
            dw = tests['durbin_watson']
            report += f"  - Autocorrelation (Durbin-Watson): {'✓' if dw['is_uncorrelated'] else '✗'} (DW={dw['statistic']:.3f})\n"
        
        if 'breusch_pagan' in tests:
            bp = tests['breusch_pagan']
            report += f"  - Heteroscedasticity (Breusch-Pagan): {'✓' if bp['is_homoscedastic'] else '✗'} (p={bp['p_value']:.4f})\n"
        
        if 'adf' in tests:
            adf = tests['adf']
            report += f"  - Stationarity (ADF): {'✓' if adf['is_stationary'] else '✗'} (p={adf['p_value']:.4f})\n"
    
    report += "\n"
    
    # Model diagnostics summary
    report += "MODEL DIAGNOSTICS SUMMARY:\n"
    report += "-" * 80 + "\n"
    
    for model_name, result in results.items():
        if 'error' in result:
            continue
        
        report += f"\n{model_name.upper()}:\n"
        diagnostics = result.get('model_diagnostics', {})
        
        nse = diagnostics.get('nash_sutcliffe', 'N/A')
        r2 = diagnostics.get('coefficient_of_determination', 'N/A')
        outliers = diagnostics.get('outlier_percentage', 'N/A')
        within_2std = diagnostics.get('residuals_within_2std', 'N/A')
        
        report += f"  - Nash-Sutcliffe Efficiency: {nse:.4f}\n"
        report += f"  - R²: {r2:.4f}\n"
        report += f"  - Outlier Percentage: {outliers:.2f}%\n"
        report += f"  - Residuals within 2σ: {within_2std:.2f}%\n"
    
    return report

def main():
    """Main CLI function."""
    
    parser = argparse.ArgumentParser(description='Test residual analysis and diagnostic plots')
    parser.add_argument('--output-dir', type=str, default='data/artifacts',
                       help='Output directory for results')
    parser.add_argument('--n-stations', type=int, default=3,
                       help='Number of stations in test data')
    parser.add_argument('--n-hours', type=int, default=500,
                       help='Number of hours in test data')
    parser.add_argument('--test-single', type=str, default=None,
                       choices=['seasonal_naive', 'prophet', 'arima', 'dynamic_ensemble'],
                       help='Test only a single model')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save diagnostic plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting residual analysis test")
    
    try:
        # Generate test data
        data = generate_test_data(args.n_stations, args.n_hours)
        
        # Configure residual analysis
        config = ResidualAnalysisConfig(
            save_plots=args.save_plots,
            plot_format='png',
            plot_dpi=300
        )
        
        if args.test_single:
            # Test only single model
            model_map = {
                'seasonal_naive': SeasonalNaiveForecaster,
                'prophet': ProphetBaseline,
                'arima': ARIMABaseline,
                'dynamic_ensemble': DynamicEnsemble
            }
            
            model_class = model_map[args.test_single]
            results = test_model_residuals(data, model_class, args.test_single, config)
            
            # Save results
            results_file = output_dir / f'residual_analysis_{args.test_single}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_file}")
            
            # Print summary
            print("\n" + "="*60)
            print(f"RESIDUAL ANALYSIS RESULTS: {args.test_single.upper()}")
            print("="*60)
            
            if 'basic_statistics' in results:
                stats = results['basic_statistics']
                print(f"\nPerformance Metrics:")
                print(f"  MAE: {stats.get('mae', 'N/A'):.3f}")
                print(f"  RMSE: {stats.get('rmse', 'N/A'):.3f}")
                print(f"  sMAPE: {stats.get('smape', 'N/A'):.2f}%")
                print(f"  Residual Std: {stats.get('residual_std', 'N/A'):.3f}")
            
            if 'statistical_tests' in results:
                tests = results['statistical_tests']
                print(f"\nStatistical Tests:")
                if 'shapiro_wilk' in tests:
                    sw = tests['shapiro_wilk']
                    print(f"  Normality (Shapiro-Wilk): {'✓' if sw['is_normal'] else '✗'} (p={sw['p_value']:.4f})")
                if 'ljung_box' in tests:
                    lb = tests['ljung_box']
                    print(f"  Autocorrelation (Ljung-Box): {'✓' if lb['is_white_noise'] else '✗'}")
            
            if 'plots' in results and 'diagnostic_plot' in results['plots']:
                print(f"\nDiagnostic plot saved to: {results['plots']['diagnostic_plot']}")
            
            print(f"\nDetailed results saved to: {results_file}")
            print("="*60)
            
        else:
            # Compare all models
            results = compare_model_residuals(data, config)
            
            # Generate comparison report
            report = generate_comparison_report(results)
            
            # Save results
            results_file = output_dir / 'residual_analysis_comparison.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            report_file = output_dir / 'residual_analysis_report.txt'
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Results saved to {results_file}")
            logger.info(f"Report saved to {report_file}")
            
            # Print summary
            print("\n" + "="*60)
            print("RESIDUAL ANALYSIS COMPARISON RESULTS")
            print("="*60)
            
            print(f"\nModel Performance Summary:")
            for model_name, result in results.items():
                if 'error' in result:
                    print(f"  {model_name}: ERROR - {result['error']}")
                else:
                    stats = result.get('basic_statistics', {})
                    print(f"  {model_name}:")
                    print(f"    MAE: {stats.get('mae', 'N/A'):.3f}")
                    print(f"    RMSE: {stats.get('rmse', 'N/A'):.3f}")
                    print(f"    sMAPE: {stats.get('smape', 'N/A'):.2f}%")
            
            print(f"\nDetailed results saved to: {results_file}")
            print(f"Comparison report saved to: {report_file}")
            print("="*60)
        
    except Exception as e:
        logger.error(f"Residual analysis test failed: {e}")
        raise

if __name__ == '__main__':
    main()
