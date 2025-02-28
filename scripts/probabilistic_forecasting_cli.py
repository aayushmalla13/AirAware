#!/usr/bin/env python3
"""
Probabilistic Forecasting CLI for AirAware

This script provides probabilistic forecasting capabilities with uncertainty quantification,
including conformal prediction, quantile regression, and ensemble uncertainty.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from airaware.baselines import (
    SeasonalNaiveForecaster,
    ProphetBaseline, 
    ARIMABaseline,
    ProbabilisticBaselineWrapper,
    UncertaintyConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Probabilistic Forecasting for AirAware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate probabilistic forecasts with conformal prediction
  python scripts/probabilistic_forecasting_cli.py forecast \\
    --data data/processed/features.parquet \\
    --model seasonal-naive \\
    --horizon 24 \\
    --uncertainty-method conformal \\
    --output results/probabilistic_forecast.json

  # Generate ensemble-based uncertainty quantification
  python scripts/probabilistic_forecasting_cli.py ensemble \\
    --data data/processed/features.parquet \\
    --model prophet \\
    --horizon 12 \\
    --n-bootstrap 50 \\
    --output results/ensemble_forecast.json

  # Compare uncertainty methods
  python scripts/probabilistic_forecasting_cli.py compare \\
    --data data/processed/features.parquet \\
    --models seasonal-naive prophet \\
    --horizon 24 \\
    --output-dir results/uncertainty_comparison/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Generate probabilistic forecast')
    
    forecast_parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to processed features data (Parquet file)'
    )
    
    forecast_parser.add_argument(
        '--model', '-m',
        choices=['seasonal-naive', 'prophet', 'arima'],
        required=True,
        help='Base model to use'
    )
    
    forecast_parser.add_argument(
        '--horizon',
        type=int,
        default=24,
        help='Forecast horizon in hours (default: 24)'
    )
    
    forecast_parser.add_argument(
        '--uncertainty-method',
        choices=['conformal', 'quantile', 'ensemble'],
        default='conformal',
        help='Uncertainty quantification method (default: conformal)'
    )
    
    forecast_parser.add_argument(
        '--conformal-alpha',
        type=float,
        default=0.1,
        help='Conformal prediction alpha level (default: 0.1 for 90% intervals)'
    )
    
    forecast_parser.add_argument(
        '--quantiles',
        nargs='+',
        type=float,
        default=[0.1, 0.25, 0.5, 0.75, 0.9],
        help='Quantiles for quantile regression (default: 0.1 0.25 0.5 0.75 0.9)'
    )
    
    forecast_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output file for probabilistic forecast (JSON format)'
    )
    
    # Ensemble command
    ensemble_parser = subparsers.add_parser('ensemble', help='Generate ensemble-based uncertainty')
    
    ensemble_parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to processed features data (Parquet file)'
    )
    
    ensemble_parser.add_argument(
        '--model', '-m',
        choices=['seasonal-naive', 'prophet', 'arima'],
        required=True,
        help='Base model to use'
    )
    
    ensemble_parser.add_argument(
        '--horizon',
        type=int,
        default=24,
        help='Forecast horizon in hours (default: 24)'
    )
    
    ensemble_parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=50,
        help='Number of bootstrap samples (default: 50)'
    )
    
    ensemble_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output file for ensemble forecast (JSON format)'
    )
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare uncertainty methods')
    
    compare_parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to processed features data (Parquet file)'
    )
    
    compare_parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=['seasonal-naive', 'prophet', 'arima'],
        default=['seasonal-naive', 'prophet'],
        help='Models to compare'
    )
    
    compare_parser.add_argument(
        '--horizon',
        type=int,
        default=24,
        help='Forecast horizon in hours (default: 24)'
    )
    
    compare_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results/uncertainty_comparison',
        help='Output directory for comparison results'
    )
    
    # Common arguments
    for subparser in [forecast_parser, ensemble_parser, compare_parser]:
        subparser.add_argument(
            '--sample-size',
            type=int,
            help='Sample size for faster processing'
        )
        
        subparser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
    
    return parser

def get_model_class(model_name: str):
    """Get model class by name."""
    model_map = {
        'seasonal-naive': SeasonalNaiveForecaster,
        'prophet': ProphetBaseline,
        'arima': ARIMABaseline
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model_map[model_name]

def load_data(data_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load and validate data."""
    logger.info(f"Loading data from {data_path}")
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = pd.read_parquet(data_path)
    
    # Sample data if requested
    if sample_size and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} records for probabilistic forecasting")
    
    # Validate required columns
    required_cols = ['datetime_utc', 'pm25', 'station_id']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Loaded {len(data)} records with {data['station_id'].nunique()} stations")
    return data

def create_uncertainty_config(args) -> UncertaintyConfig:
    """Create uncertainty configuration from arguments."""
    config = UncertaintyConfig()
    
    if hasattr(args, 'conformal_alpha'):
        config.conformal_alpha = args.conformal_alpha
    
    if hasattr(args, 'quantiles'):
        config.quantiles = args.quantiles
    
    if hasattr(args, 'n_bootstrap'):
        config.n_bootstrap_samples = args.n_bootstrap
    
    return config

def run_forecast_command(args):
    """Run probabilistic forecast generation command."""
    # Load data
    data = load_data(args.data, args.sample_size)
    
    # Get model class
    model_class = get_model_class(args.model)
    
    # Create uncertainty configuration
    config = create_uncertainty_config(args)
    
    # Create probabilistic wrapper
    base_model = model_class()
    probabilistic_model = ProbabilisticBaselineWrapper(base_model, config)
    
    # Fit model
    logger.info(f"Fitting probabilistic {args.model} model...")
    probabilistic_model.fit(data, target_col='pm25', group_col='station_id')
    
    # Generate forecast
    logger.info(f"Generating {args.horizon}h probabilistic forecast...")
    forecast_start = data['datetime_utc'].max() + pd.Timedelta(hours=1)
    timestamps = [forecast_start + pd.Timedelta(hours=i) for i in range(args.horizon)]
    
    # Generate probabilistic forecast
    prob_forecast = probabilistic_model.predict_probabilistic(timestamps)
    
    # Prepare output
    output_data = {
        'model': args.model,
        'uncertainty_method': args.uncertainty_method,
        'horizon': args.horizon,
        'timestamps': [ts.isoformat() for ts in timestamps],
        'predictions': prob_forecast.predictions,
        'quantiles': prob_forecast.quantiles,
        'confidence_intervals': prob_forecast.confidence_intervals,
        'uncertainty_method_used': prob_forecast.uncertainty_method,
        'calibration_score': prob_forecast.calibration_score
    }
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 PROBABILISTIC FORECAST COMPLETE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Uncertainty method: {args.uncertainty_method}")
    print(f"Horizon: {args.horizon}h")
    print(f"Predictions: {len(prob_forecast.predictions)}")
    print(f"Quantiles: {list(prob_forecast.quantiles.keys())}")
    print(f"Output: {args.output}")
    print("=" * 60)

def run_ensemble_command(args):
    """Run ensemble-based uncertainty quantification command."""
    # Load data
    data = load_data(args.data, args.sample_size)
    
    # Get model class
    model_class = get_model_class(args.model)
    
    # Create uncertainty configuration
    config = create_uncertainty_config(args)
    config.ensemble_method = "bootstrap"
    
    # Create ensemble uncertainty
    ensemble_uncertainty = EnsembleUncertainty(config)
    
    # Fit ensemble
    logger.info(f"Fitting {args.n_bootstrap} bootstrap models for {args.model}...")
    ensemble_uncertainty.fit_ensemble(
        data, 
        model_class, 
        target_col='pm25', 
        group_col='station_id'
    )
    
    # Generate ensemble forecast
    logger.info(f"Generating {args.horizon}h ensemble forecast...")
    forecast_start = data['datetime_utc'].max() + pd.Timedelta(hours=1)
    timestamps = [forecast_start + pd.Timedelta(hours=i) for i in range(args.horizon)]
    
    ensemble_results = ensemble_uncertainty.predict_ensemble(timestamps)
    
    # Prepare output
    output_data = {
        'model': args.model,
        'ensemble_method': 'bootstrap',
        'n_bootstrap_samples': args.n_bootstrap,
        'horizon': args.horizon,
        'timestamps': [ts.isoformat() for ts in timestamps],
        'ensemble_results': ensemble_results
    }
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 ENSEMBLE FORECAST COMPLETE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Ensemble method: bootstrap")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Horizon: {args.horizon}h")
    print(f"Output: {args.output}")
    print("=" * 60)

def run_compare_command(args):
    """Run uncertainty method comparison command."""
    # Load data
    data = load_data(args.data, args.sample_size)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for model_name in args.models:
        logger.info(f"Comparing uncertainty methods for {model_name}...")
        
        # Get model class
        model_class = get_model_class(model_name)
        
        # Test different uncertainty methods
        uncertainty_methods = ['conformal', 'quantile']
        
        model_results = {}
        
        for method in uncertainty_methods:
            try:
                # Create configuration
                config = UncertaintyConfig()
                config.conformal_alpha = 0.1
                
                # Create probabilistic wrapper
                base_model = model_class()
                probabilistic_model = ProbabilisticBaselineWrapper(base_model, config)
                
                # Fit model
                probabilistic_model.fit(data, target_col='pm25', group_col='station_id')
                
                # Generate forecast
                forecast_start = data['datetime_utc'].max() + pd.Timedelta(hours=1)
                timestamps = [forecast_start + pd.Timedelta(hours=i) for i in range(args.horizon)]
                
                prob_forecast = probabilistic_model.predict_probabilistic(timestamps)
                
                model_results[method] = {
                    'predictions': prob_forecast.predictions,
                    'quantiles': prob_forecast.quantiles,
                    'confidence_intervals': prob_forecast.confidence_intervals
                }
                
                logger.info(f"✅ {model_name} with {method} uncertainty completed")
                
            except Exception as e:
                logger.error(f"❌ {model_name} with {method} failed: {e}")
                model_results[method] = {'error': str(e)}
        
        results[model_name] = model_results
    
    # Save comparison results
    comparison_file = Path(args.output_dir) / "uncertainty_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 UNCERTAINTY COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Models: {', '.join(args.models)}")
    print(f"Horizon: {args.horizon}h")
    print(f"Output directory: {args.output_dir}")
    print(f"Comparison file: {comparison_file}")
    print("=" * 60)

def main():
    """Main function."""
    
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.command == 'forecast':
            run_forecast_command(args)
        elif args.command == 'ensemble':
            run_ensemble_command(args)
        elif args.command == 'compare':
            run_compare_command(args)
        else:
            parser.print_help()
        
        logger.info("Probabilistic forecasting completed successfully!")
        
    except Exception as e:
        logger.error(f"Probabilistic forecasting failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
