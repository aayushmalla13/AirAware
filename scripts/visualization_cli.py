#!/usr/bin/env python3
"""
Forecast Visualization Dashboard CLI for AirAware

This script provides comprehensive visualization tools for forecasting results,
including interactive plots, performance comparisons, and diagnostic visualizations.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from airaware.visualization import ForecastDashboard, DashboardConfig
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
        description="Forecast Visualization Dashboard for AirAware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate forecast comparison plots
  python scripts/visualization_cli.py plot --data data/processed/features.parquet \\
    --models seasonal-naive prophet --output-dir plots/

  # Create interactive dashboard with rolling CV results
  python scripts/visualization_cli.py dashboard --data data/processed/features.parquet \\
    --cv-results results/rolling_cv_results.json --output plots/dashboard.html

  # Generate all visualizations
  python scripts/visualization_cli.py all --data data/processed/features.parquet \\
    --models seasonal-naive prophet arima ensemble --output-dir data/artifacts/plots
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Generate forecast comparison plots')
    
    plot_parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to processed features data (Parquet file)'
    )
    
    plot_parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=['seasonal-naive', 'prophet', 'arima', 'ensemble'],
        default=['seasonal-naive', 'prophet'],
        help='Models to visualize (default: seasonal-naive prophet)'
    )
    
    plot_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/artifacts/plots',
        help='Output directory for plots (default: data/artifacts/plots)'
    )
    
    plot_parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Sample size for visualization (default: 1000)'
    )
    
    plot_parser.add_argument(
        '--horizon',
        type=int,
        default=24,
        help='Forecast horizon in hours (default: 24)'
    )
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Create comprehensive dashboard')
    
    dashboard_parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to processed features data (Parquet file)'
    )
    
    dashboard_parser.add_argument(
        '--cv-results',
        type=str,
        help='Path to rolling CV results JSON file'
    )
    
    dashboard_parser.add_argument(
        '--output',
        type=str,
        default='data/artifacts/forecast_dashboard.html',
        help='Output HTML file path (default: data/artifacts/forecast_dashboard.html)'
    )
    
    dashboard_parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=['seasonal-naive', 'prophet', 'arima', 'ensemble'],
        default=['seasonal-naive', 'prophet'],
        help='Models to include in dashboard'
    )
    
    # All command
    all_parser = subparsers.add_parser('all', help='Generate all visualizations')
    
    all_parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to processed features data (Parquet file)'
    )
    
    all_parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=['seasonal-naive', 'prophet', 'arima', 'ensemble'],
        default=['seasonal-naive', 'prophet', 'arima', 'ensemble'],
        help='Models to visualize'
    )
    
    all_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/artifacts/plots',
        help='Output directory for all plots'
    )
    
    all_parser.add_argument(
        '--cv-results',
        type=str,
        help='Path to rolling CV results JSON file'
    )
    
    # Common arguments
    for subparser in [plot_parser, dashboard_parser, all_parser]:
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
        'arima': ARIMABaseline,
        'ensemble': BaselineEnsemble
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
        logger.info(f"Sampled {sample_size} records for visualization")
    
    # Validate required columns
    required_cols = ['datetime_utc', 'pm25', 'station_id']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Loaded {len(data)} records with {data['station_id'].nunique()} stations")
    return data

def generate_forecasts(data: pd.DataFrame, 
                      models: Dict[str, Any], 
                      horizon: int = 24) -> Dict[str, Any]:
    """Generate forecasts for all models."""
    logger.info(f"Generating forecasts for {len(models)} models with {horizon}h horizon")
    
    forecasts = {}
    
    # Prepare training data (use most recent data)
    train_data = data.tail(500).copy()  # Use last 500 points for training
    
    # Generate forecast start time
    forecast_start = train_data['datetime_utc'].max() + pd.Timedelta(hours=1)
    
    for model_name, model_class in models.items():
        try:
            logger.info(f"Fitting {model_name} model...")
            
            # Fit model
            model = model_class()
            model.fit(train_data, target_col='pm25', group_col='station_id')
            
            # Generate forecast
            forecast = model.forecast(forecast_start, horizon)
            forecasts[model_name] = forecast
            
            logger.info(f"✅ {model_name}: Generated {len(forecast.predictions)} predictions")
            
        except Exception as e:
            logger.error(f"❌ {model_name}: Failed to generate forecast - {e}")
    
    return forecasts

def load_cv_results(cv_results_path: str) -> Optional[Dict[str, Any]]:
    """Load rolling CV results from JSON file."""
    if not cv_results_path or not Path(cv_results_path).exists():
        logger.warning(f"CV results file not found: {cv_results_path}")
        return None
    
    logger.info(f"Loading CV results from {cv_results_path}")
    
    with open(cv_results_path, 'r') as f:
        cv_data = json.load(f)
    
    # Convert back to proper objects (simplified)
    return cv_data

def run_plot_command(args):
    """Run plot generation command."""
    # Load data
    data = load_data(args.data, args.sample_size)
    
    # Get model classes
    models = {name: get_model_class(name) for name in args.models}
    
    # Generate forecasts
    forecasts = generate_forecasts(data, models, args.horizon)
    
    if not forecasts:
        logger.error("No forecasts generated. Exiting.")
        return
    
    # Create dashboard
    config = DashboardConfig(output_dir=args.output_dir)
    dashboard = ForecastDashboard(config)
    
    # Generate plots
    logger.info("Generating visualization plots...")
    saved_files = dashboard.save_plots(
        data=data.tail(200),  # Use recent data for context
        forecasts=forecasts,
        actuals=None
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 FORECAST VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Models visualized: {', '.join(forecasts.keys())}")
    print(f"Forecast horizon: {args.horizon}h")
    print(f"Output directory: {args.output_dir}")
    print("\nGenerated files:")
    for plot_type, file_path in saved_files.items():
        print(f"  • {plot_type}: {file_path}")
    print("=" * 60)

def run_dashboard_command(args):
    """Run dashboard creation command."""
    # Load data
    data = load_data(args.data)
    
    # Get model classes
    models = {name: get_model_class(name) for name in args.models}
    
    # Generate forecasts
    forecasts = generate_forecasts(data, models)
    
    if not forecasts:
        logger.error("No forecasts generated. Exiting.")
        return
    
    # Load CV results if provided
    cv_results = load_cv_results(args.cv_results)
    
    # Create dashboard
    config = DashboardConfig()
    dashboard = ForecastDashboard(config)
    
    # Create comprehensive dashboard
    logger.info("Creating comprehensive dashboard...")
    
    if hasattr(dashboard, 'create_dashboard_html'):
        dashboard_path = dashboard.create_dashboard_html(
            data=data.tail(200),
            forecasts=forecasts,
            cv_results=cv_results,
            actuals=None,
            output_path=args.output
        )
        
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE DASHBOARD CREATED")
        print("=" * 60)
        print(f"Dashboard: {dashboard_path}")
        print(f"Models: {', '.join(forecasts.keys())}")
        if cv_results:
            print(f"CV Results: Included")
        print("=" * 60)
    else:
        logger.warning("Dashboard HTML creation not available")

def run_all_command(args):
    """Run all visualizations command."""
    # Load data
    data = load_data(args.data)
    
    # Get model classes
    models = {name: get_model_class(name) for name in args.models}
    
    # Generate forecasts
    forecasts = generate_forecasts(data, models)
    
    if not forecasts:
        logger.error("No forecasts generated. Exiting.")
        return
    
    # Load CV results if provided
    cv_results = load_cv_results(args.cv_results)
    
    # Create dashboard
    config = DashboardConfig(output_dir=args.output_dir)
    dashboard = ForecastDashboard(config)
    
    # Generate all plots
    logger.info("Generating all visualizations...")
    saved_files = dashboard.save_plots(
        data=data.tail(200),
        forecasts=forecasts,
        cv_results=cv_results,
        actuals=None,
        output_dir=args.output_dir
    )
    
    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("🎨 ALL VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print(f"Models: {', '.join(forecasts.keys())}")
    print(f"Output directory: {args.output_dir}")
    if cv_results:
        print(f"CV Results: Included")
    print("\nGenerated files:")
    for plot_type, file_path in saved_files.items():
        print(f"  • {plot_type}: {file_path}")
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
        if args.command == 'plot':
            run_plot_command(args)
        elif args.command == 'dashboard':
            run_dashboard_command(args)
        elif args.command == 'all':
            run_all_command(args)
        else:
            parser.print_help()
        
        logger.info("Visualization completed successfully!")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
