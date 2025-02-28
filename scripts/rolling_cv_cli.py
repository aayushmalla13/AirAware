#!/usr/bin/env python3
"""
Rolling Origin Cross-Validation CLI for AirAware Baseline Models

This script provides comprehensive rolling-origin cross-validation evaluation
for baseline forecasting models, ensuring proper temporal ordering and robust
performance assessment.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from airaware.evaluation import RollingOriginCV, RollingCVConfig
from airaware.baselines import (
    SeasonalNaiveForecaster,
    ProphetBaseline, 
    ARIMABaseline,
    BaselineEnsemble
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
        description="Rolling Origin Cross-Validation for AirAware Baseline Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick rolling CV with default settings
  python scripts/rolling_cv_cli.py evaluate --data data/processed/features.parquet

  # Comprehensive rolling CV with custom parameters
  python scripts/rolling_cv_cli.py evaluate --data data/processed/features.parquet \\
    --initial-train-size 336 --step-size 48 --max-origins 15 --horizons 6 12 24 48

  # Parallel evaluation with prediction saving
  python scripts/rolling_cv_cli.py evaluate --data data/processed/features.parquet \\
    --n-jobs 4 --save-predictions --output results/rolling_cv_results.json

  # Compare specific models only
  python scripts/rolling_cv_cli.py evaluate --data data/processed/features.parquet \\
    --models seasonal-naive prophet --quick
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run rolling-origin cross-validation')
    
    # Data parameters
    eval_parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to processed features data (Parquet file)'
    )
    
    # Model selection
    eval_parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=['seasonal-naive', 'prophet', 'arima', 'ensemble'],
        default=['seasonal-naive', 'prophet', 'arima', 'ensemble'],
        help='Models to evaluate (default: all)'
    )
    
    # Rolling CV parameters
    eval_parser.add_argument(
        '--initial-train-size',
        type=int,
        default=168,
        help='Initial training window size in hours (default: 168)'
    )
    
    eval_parser.add_argument(
        '--step-size',
        type=int,
        default=24,
        help='Step size between origins in hours (default: 24)'
    )
    
    eval_parser.add_argument(
        '--max-origins',
        type=int,
        default=10,
        help='Maximum number of origins (default: 10)'
    )
    
    eval_parser.add_argument(
        '--horizons',
        nargs='+',
        type=int,
        default=[6, 12, 24],
        help='Forecast horizons in hours (default: 6 12 24)'
    )
    
    eval_parser.add_argument(
        '--gap-size',
        type=int,
        default=0,
        help='Gap between train and test in hours (default: 0)'
    )
    
    eval_parser.add_argument(
        '--min-train-size',
        type=int,
        default=72,
        help='Minimum training window size in hours (default: 72)'
    )
    
    # Performance parameters
    eval_parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (default: 1)'
    )
    
    eval_parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    # Output parameters
    eval_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for results (JSON format)'
    )
    
    eval_parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save individual predictions for analysis'
    )
    
    eval_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    eval_parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick evaluation with reduced parameters'
    )
    
    return parser

def get_model_class(model_name: str):
    """Get model class by name."""
    model_map = {
        'seasonal-naive': SeasonalNaiveForecaster,
        'prophet': ProphetBaseline,
        'arima': ARIMABaseline,
        'ensemble': BaselineEnsemble
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model_map[model_name]

def load_data(data_path: str) -> pd.DataFrame:
    """Load and validate data."""
    logger.info(f"Loading data from {data_path}")
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = pd.read_parquet(data_path)
    
    # Validate required columns
    required_cols = ['datetime_utc', 'pm25', 'station_id']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Loaded {len(data)} records with {data['station_id'].nunique()} stations")
    return data

def create_rolling_cv_config(args) -> RollingCVConfig:
    """Create rolling CV configuration from arguments."""
    
    if args.quick:
        # Quick evaluation settings
        config = RollingCVConfig(
            initial_train_size=72,
            step_size=24,
            max_origins=3,
            horizons=[6, 12],
            min_train_size=48,
            n_jobs=1,
            random_state=args.random_state,
            save_predictions=args.save_predictions
        )
        logger.info("Using quick evaluation settings")
    else:
        # Full evaluation settings
        config = RollingCVConfig(
            initial_train_size=args.initial_train_size,
            step_size=args.step_size,
            max_origins=args.max_origins,
            horizons=args.horizons,
            gap_size=args.gap_size,
            min_train_size=args.min_train_size,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            save_predictions=args.save_predictions
        )
    
    return config

def run_rolling_cv_evaluation(data: pd.DataFrame, 
                            models: Dict[str, Any], 
                            config: RollingCVConfig) -> Dict[str, Any]:
    """Run rolling-origin cross-validation evaluation."""
    
    logger.info("Starting rolling-origin cross-validation evaluation")
    
    # Initialize rolling CV
    rolling_cv = RollingOriginCV(config)
    
    # Run evaluation
    results = rolling_cv.compare_models(
        data=data,
        models=models,
        target_col='pm25',
        group_col='station_id',
        datetime_col='datetime_utc'
    )
    
    return results

def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    
    logger.info(f"Saving results to {output_path}")
    
    # Convert results to serializable format
    serializable_results = {}
    
    for model_name, result in results.items():
        serializable_results[model_name] = {
            'model_name': result.model_name,
            'config': {
                'initial_train_size': result.config.initial_train_size,
                'step_size': result.config.step_size,
                'max_origins': result.config.max_origins,
                'horizons': result.config.horizons,
                'gap_size': result.config.gap_size,
                'min_train_size': result.config.min_train_size,
                'n_jobs': result.config.n_jobs,
                'random_state': result.config.random_state,
                'save_predictions': result.config.save_predictions
            },
            'origins': [origin.isoformat() for origin in result.origins],
            'train_sizes': result.train_sizes,
            'test_sizes': result.test_sizes,
            'metrics_by_horizon': result.metrics_by_horizon,
            'metrics_by_origin': {
                str(k): v for k, v in result.metrics_by_origin.items()
            },
            'execution_time': result.execution_time,
            'n_origins_completed': result.n_origins_completed,
            'errors': result.errors
        }
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_path}")

def print_summary_report(results: Dict[str, Any]):
    """Print comprehensive summary report."""
    
    print("\n" + "=" * 80)
    print("ROLLING-ORIGIN CROSS-VALIDATION SUMMARY REPORT")
    print("=" * 80)
    
    # Configuration summary
    if results:
        first_result = next(iter(results.values()))
        config = first_result.config
        
        print(f"\nConfiguration:")
        print(f"  • Initial train size: {config.initial_train_size} hours")
        print(f"  • Step size: {config.step_size} hours")
        print(f"  • Max origins: {config.max_origins}")
        print(f"  • Horizons: {config.horizons} hours")
        print(f"  • Gap size: {config.gap_size} hours")
        print(f"  • Parallel jobs: {config.n_jobs}")
    
    # Model performance summary
    print(f"\nModel Performance Summary:")
    print("-" * 50)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  • Origins completed: {result.n_origins_completed}")
        print(f"  • Execution time: {result.execution_time:.2f}s")
        print(f"  • Errors: {len(result.errors)}")
        
        if result.errors:
            print(f"  • Error details: {result.errors[:2]}...")
        
        # Average metrics by horizon
        for horizon in result.config.horizons:
            if horizon in result.metrics_by_horizon:
                metrics = result.metrics_by_horizon[horizon]
                if metrics:
                    print(f"  • Horizon {horizon}h:")
                    for metric_name, values in metrics.items():
                        if values:
                            avg_value = np.mean(values)
                            std_value = np.std(values)
                            print(f"    - {metric_name.upper()}: {avg_value:.3f} ± {std_value:.3f}")
    
    # Best model by horizon
    print(f"\nBest Model by Horizon:")
    print("-" * 30)
    
    if results:
        horizons = next(iter(results.values())).config.horizons
        
        for horizon in horizons:
            best_model = None
            best_mae = float('inf')
            
            for model_name, result in results.items():
                if (horizon in result.metrics_by_horizon and 
                    'mae' in result.metrics_by_horizon[horizon] and
                    result.metrics_by_horizon[horizon]['mae']):
                    
                    avg_mae = np.mean(result.metrics_by_horizon[horizon]['mae'])
                    if avg_mae < best_mae:
                        best_mae = avg_mae
                        best_model = model_name
            
            if best_model:
                print(f"  • {horizon}h: {best_model.upper()} (MAE: {best_mae:.3f})")
    
    print("\n" + "=" * 80)

def main():
    """Main function."""
    
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if args.command != 'evaluate':
        parser.print_help()
        return
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load data
        data = load_data(args.data)
        
        # Get model classes
        models = {name: get_model_class(name) for name in args.models}
        
        # Create configuration
        config = create_rolling_cv_config(args)
        
        # Run evaluation
        results = run_rolling_cv_evaluation(data, models, config)
        
        # Print summary report
        print_summary_report(results)
        
        # Save results if requested
        if args.output:
            save_results(results, args.output)
        
        logger.info("Rolling-origin cross-validation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
