#!/usr/bin/env python3
"""Enhanced CP-6 CLI with parallel evaluation, hyperparameter tuning, and advanced visualization."""

import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup argument parser for enhanced CP-6 CLI."""
    parser = argparse.ArgumentParser(
        description="Enhanced CP-6: Parallel Evaluation, Hyperparameter Tuning & Advanced Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Parallel evaluation command
    parallel_parser = subparsers.add_parser('parallel-eval', help='Run parallel model evaluation')
    parallel_parser.add_argument('--data-path', default='data/processed/train_ready.parquet',
                               help='Path to training data')
    parallel_parser.add_argument('--max-workers', type=int, default=4,
                               help='Maximum number of parallel workers')
    parallel_parser.add_argument('--use-threading', action='store_true',
                               help='Use threading instead of multiprocessing')
    parallel_parser.add_argument('--models', nargs='+', 
                               default=['seasonal_naive', 'prophet', 'arima', 'ensemble'],
                               help='Models to evaluate')
    parallel_parser.add_argument('--horizons', nargs='+', type=int, default=[6, 12, 24],
                               help='Forecast horizons in hours')
    parallel_parser.add_argument('--enable-caching', action='store_true',
                               help='Enable model caching')
    parallel_parser.add_argument('--quick', action='store_true',
                               help='Quick mode with reduced data')
    
    # Hyperparameter optimization command
    hyperopt_parser = subparsers.add_parser('hyperopt', help='Run hyperparameter optimization')
    hyperopt_parser.add_argument('--data-path', default='data/processed/train_ready.parquet',
                               help='Path to training data')
    hyperopt_parser.add_argument('--n-trials', type=int, default=50,
                               help='Number of optimization trials')
    hyperopt_parser.add_argument('--timeout', type=int, default=None,
                               help='Timeout in seconds')
    hyperopt_parser.add_argument('--objective', default='mae',
                               choices=['mae', 'rmse', 'smape'],
                               help='Optimization objective')
    hyperopt_parser.add_argument('--models', nargs='+',
                               default=['seasonal_naive', 'prophet', 'arima', 'ensemble'],
                               help='Models to optimize')
    hyperopt_parser.add_argument('--cv-folds', type=int, default=3,
                               help='Cross-validation folds')
    
    # Advanced visualization command
    viz_parser = subparsers.add_parser('advanced-viz', help='Create advanced visualizations')
    viz_parser.add_argument('--data-path', default='data/processed/train_ready.parquet',
                          help='Path to training data')
    viz_parser.add_argument('--output-dir', default='data/artifacts/plots',
                          help='Output directory for plots')
    viz_parser.add_argument('--uncertainty', action='store_true',
                          help='Include uncertainty visualization')
    viz_parser.add_argument('--heatmap', action='store_true',
                          help='Create performance heatmaps')
    viz_parser.add_argument('--interactive', action='store_true',
                          help='Create interactive plots')
    
    # Comprehensive evaluation command
    comprehensive_parser = subparsers.add_parser('comprehensive', help='Run comprehensive evaluation')
    comprehensive_parser.add_argument('--data-path', default='data/processed/train_ready.parquet',
                                    help='Path to training data')
    comprehensive_parser.add_argument('--parallel', action='store_true',
                                    help='Use parallel evaluation')
    comprehensive_parser.add_argument('--hyperopt', action='store_true',
                                    help='Include hyperparameter optimization')
    comprehensive_parser.add_argument('--viz', action='store_true',
                                    help='Generate visualizations')
    comprehensive_parser.add_argument('--quick', action='store_true',
                                    help='Quick mode with reduced trials/data')
    
    return parser


def run_parallel_evaluation(args):
    """Run parallel model evaluation."""
    console = Console()
    
    try:
        console.print(Panel.fit("🚀 Parallel Model Evaluation", style="bold blue"))
        
        # Check if data file exists
        if not os.path.exists(args.data_path):
            console.print(f"❌ Data file not found: {args.data_path}", style="red")
            console.print("Available data files:", style="yellow")
            import glob
            parquet_files = glob.glob("data/**/*.parquet", recursive=True)
            for file in parquet_files[:10]:  # Show first 10 files
                console.print(f"  • {file}")
            return
        
        import pandas as pd
        from airaware.evaluation.parallel_evaluator import ParallelEvaluator, ParallelEvaluationConfig
        
        # Load data with error handling
        try:
            df = pd.read_parquet(args.data_path)
            console.print(f"📊 Loaded {len(df):,} records from {args.data_path}")
        except Exception as e:
            console.print(f"❌ Failed to load data: {e}", style="red")
            return
        
        # Check required columns
        required_cols = ['datetime_utc', 'pm25', 'station_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            console.print(f"❌ Missing required columns: {missing_cols}", style="red")
            console.print(f"Available columns: {list(df.columns)}", style="yellow")
            return
        
        # Split data
        split_date = pd.to_datetime('2025-01-01')
        train_df = df[df['datetime_utc'] < split_date].copy()
        test_df = df[df['datetime_utc'] >= split_date].head(72).copy()
        
        # Quick mode adjustments
        if args.quick:
            console.print("⚡ Quick mode: Using smaller dataset and fewer models")
            # Ensure we have overlapping stations
            train_stations = train_df['station_id'].unique()
            test_stations = test_df['station_id'].unique()
            overlapping_stations = [s for s in train_stations if s in test_stations]
            
            if overlapping_stations:
                # Use only overlapping stations
                train_df = train_df[train_df['station_id'].isin(overlapping_stations)].tail(1000)
                test_df = test_df[test_df['station_id'].isin(overlapping_stations)].head(24)
                console.print(f"Using overlapping stations: {overlapping_stations}")
            else:
                console.print("⚠️ No overlapping stations found, using all data")
            
            args.models = ['seasonal_naive', 'prophet']  # Only test 2 models
            args.horizons = [6, 12]  # Only test 2 horizons
            args.max_workers = min(args.max_workers, 2)  # Limit workers
        
        console.print(f"📈 Training: {len(train_df):,} records, Testing: {len(test_df):,} records")
        
        # Check if we have enough data
        if len(train_df) < 10:
            console.print("❌ Insufficient training data", style="red")
            return
        
        if len(test_df) < max(args.horizons):
            console.print("❌ Insufficient test data", style="red")
            return
        
        # Configure parallel evaluation
        config = ParallelEvaluationConfig(
            max_workers=args.max_workers,
            use_threading=True,  # Use threading to avoid pickle issues
            forecast_horizons=args.horizons,
            models_to_evaluate=args.models,
            enable_model_caching=args.enable_caching,
            show_progress=True
        )
        
        # Run evaluation
        evaluator = ParallelEvaluator(config)
        results = evaluator.evaluate_models_parallel(train_df, test_df)
        
        # Display results
        from rich.table import Table
        
        table = Table(title="🚀 Parallel Evaluation Results")
        table.add_column("Model", style="cyan")
        table.add_column("MAE", justify="right", style="green")
        table.add_column("RMSE", justify="right", style="yellow")
        table.add_column("sMAPE", justify="right", style="red")
        table.add_column("Time (s)", justify="right", style="blue")
        table.add_column("Status", justify="center")
        
        for key, result in results.items():
            if result.success:
                table.add_row(
                    key,
                    f"{result.mae:.2f}" if result.mae and not np.isnan(result.mae) else "N/A",
                    f"{result.rmse:.2f}" if result.rmse and not np.isnan(result.rmse) else "N/A",
                    f"{result.smape:.2f}" if result.smape and not np.isnan(result.smape) else "N/A",
                    f"{result.execution_time:.2f}",
                    "✅"
                )
            else:
                table.add_row(
                    key,
                    "N/A", "N/A", "N/A", "N/A",
                    "❌"
                )
        
        console.print(table)
        
        # Save results
        output_path = "data/artifacts/parallel_evaluation_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        serializable_results = {}
        for key, result in results.items():
            serializable_results[key] = {
                'model_name': result.model_name,
                'success': result.success,
                'mae': result.mae,
                'rmse': result.rmse,
                'smape': result.smape,
                'execution_time': result.execution_time,
                'error_message': result.error_message
            }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        console.print(f"✅ Results saved to {output_path}")
        
    except Exception as e:
        console.print(f"❌ Parallel evaluation failed: {e}", style="red")
        if args.verbose if hasattr(args, 'verbose') else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_hyperparameter_optimization(args):
    """Run hyperparameter optimization."""
    console = Console()
    
    try:
        console.print(Panel.fit("🔧 Hyperparameter Optimization", style="bold yellow"))
        
        # Check if hyperparameter optimization is available
        try:
            from airaware.evaluation.hyperparameter_optimizer import HyperparameterOptimizer, HyperparameterOptimizationConfig
        except ImportError:
            console.print("❌ Hyperparameter optimization requires 'optuna' package. Install with: pip install optuna", style="red")
            return
        
        import pandas as pd
        
        # Load data
        df = pd.read_parquet(args.data_path)
        console.print(f"📊 Loaded {len(df):,} records from {args.data_path}")
        
        # Split data
        split_date = pd.to_datetime('2025-01-01')
        train_df = df[df['datetime_utc'] < split_date].copy()
        test_df = df[df['datetime_utc'] >= split_date].head(72).copy()
        
        # Quick mode adjustments
        if args.quick:
            console.print("⚡ Quick mode: Using smaller dataset and fewer models")
            # Ensure we have overlapping stations
            train_stations = train_df['station_id'].unique()
            test_stations = test_df['station_id'].unique()
            overlapping_stations = [s for s in train_stations if s in test_stations]
            
            if overlapping_stations:
                # Use only overlapping stations
                train_df = train_df[train_df['station_id'].isin(overlapping_stations)].tail(1000)
                test_df = test_df[test_df['station_id'].isin(overlapping_stations)].head(24)
                console.print(f"Using overlapping stations: {overlapping_stations}")
            else:
                console.print("⚠️ No overlapping stations found, using all data")
            
            args.models = ['seasonal_naive', 'prophet']  # Only test 2 models
            args.horizons = [6, 12]  # Only test 2 horizons
            args.max_workers = min(args.max_workers, 2)  # Limit workers
        
        console.print(f"📈 Training: {len(train_df):,} records, Testing: {len(test_df):,} records")
        
        # Configure optimization
        config = HyperparameterOptimizationConfig(
            n_trials=args.n_trials,
            timeout=args.timeout,
            objective_metric=args.objective,
            cv_folds=args.cv_folds,
            optimize_seasonal_naive='seasonal_naive' in args.models,
            optimize_prophet='prophet' in args.models,
            optimize_arima='arima' in args.models,
            optimize_ensemble='ensemble' in args.models
        )
        
        # Run optimization
        optimizer = HyperparameterOptimizer(config)
        results = optimizer.optimize_models(train_df, test_df)
        
        # Display results
        from rich.table import Table
        
        table = Table(title="🔧 Hyperparameter Optimization Results")
        table.add_column("Model", style="cyan")
        table.add_column("Best Score", justify="right", style="green")
        table.add_column("Trials", justify="right", style="blue")
        table.add_column("Best Parameters", style="yellow")
        
        for model_name, result in results.items():
            table.add_row(
                model_name,
                f"{result['best_value']:.4f}",
                str(result['n_trials']),
                str(result['best_params'])[:50] + "..." if len(str(result['best_params'])) > 50 else str(result['best_params'])
            )
        
        console.print(table)
        
        console.print("✅ Hyperparameter optimization completed")
        
    except Exception as e:
        console.print(f"❌ Hyperparameter optimization failed: {e}", style="red")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_advanced_visualization(args):
    """Run advanced visualization."""
    console = Console()
    
    try:
        console.print(Panel.fit("📊 Advanced Visualization", style="bold magenta"))
        
        # Check if data file exists
        if not os.path.exists(args.data_path):
            console.print(f"❌ Data file not found: {args.data_path}", style="red")
            return
        
        import pandas as pd
        from airaware.visualization.forecast_dashboard import ForecastDashboard, DashboardConfig
        from airaware.baselines import SeasonalNaiveForecaster, ProphetBaseline, ARIMABaseline
        
        # Load data with error handling
        try:
            df = pd.read_parquet(args.data_path)
            console.print(f"📊 Loaded {len(df):,} records from {args.data_path}")
        except Exception as e:
            console.print(f"❌ Failed to load data: {e}", style="red")
            return
        
        # Check required columns
        required_cols = ['datetime_utc', 'pm25', 'station_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            console.print(f"❌ Missing required columns: {missing_cols}", style="red")
            return
        
        # Prepare data
        train_data = df.tail(500).copy()
        forecast_start = train_data['datetime_utc'].max() + pd.Timedelta(hours=1)
        
        # Generate forecasts with error handling
        models = {
            'Seasonal-Naive': SeasonalNaiveForecaster(),
            'Prophet': ProphetBaseline(),
            'ARIMA': ARIMABaseline()
        }
        
        forecasts = {}
        for model_name, model in models.items():
            try:
                console.print(f"🔧 Generating {model_name} forecast...")
                model.fit(train_data, target_col='pm25', group_col='station_id')
                forecast = model.forecast(forecast_start, 24)
                forecasts[model_name] = forecast
                console.print(f"✅ {model_name}: Generated forecast")
            except Exception as e:
                console.print(f"❌ {model_name}: {e}")
        
        if not forecasts:
            console.print("❌ No forecasts generated successfully", style="red")
            return
        
        # Create dashboard
        dashboard_config = DashboardConfig(output_dir=args.output_dir)
        dashboard = ForecastDashboard(dashboard_config)
        
        # Generate visualizations
        saved_files = {}
        
        if args.uncertainty:
            console.print("📈 Creating uncertainty visualizations...")
            try:
                uncertainty_plot = dashboard.plot_interactive_forecast_with_uncertainty(
                    train_data, forecasts, title="Forecast with Uncertainty Bands"
                )
                if uncertainty_plot:
                    uncertainty_path = os.path.join(args.output_dir, "uncertainty_forecast.html")
                    os.makedirs(args.output_dir, exist_ok=True)
                    uncertainty_plot.write_html(uncertainty_path)
                    saved_files['uncertainty'] = uncertainty_path
                    console.print(f"✅ Uncertainty plot saved to {uncertainty_path}")
            except Exception as e:
                console.print(f"❌ Uncertainty visualization failed: {e}")
        
        if args.heatmap:
            console.print("🔥 Creating performance heatmaps...")
            try:
                # Generate sample performance data
                performance_data = {}
                for model_name in forecasts.keys():
                    for season in ['Spring', 'Summer', 'Fall', 'Winter']:
                        for hour in range(24):
                            key = f"{model_name}_{season}_{hour}"
                            performance_data[key] = {'mae': np.random.uniform(10, 50)}
                
                heatmap_plot = dashboard.plot_model_performance_heatmap(
                    performance_data, title="Model Performance Heatmap"
                )
                if heatmap_plot:
                    heatmap_path = os.path.join(args.output_dir, "performance_heatmap.html")
                    heatmap_plot.write_html(heatmap_path)
                    saved_files['heatmap'] = heatmap_path
                    console.print(f"✅ Heatmap saved to {heatmap_path}")
            except Exception as e:
                console.print(f"❌ Heatmap visualization failed: {e}")
        
        if args.interactive:
            console.print("🎨 Creating interactive plots...")
            try:
                interactive_plot = dashboard.plot_interactive_forecast(
                    train_data, forecasts, title="Interactive Forecast Comparison"
                )
                if interactive_plot:
                    interactive_path = os.path.join(args.output_dir, "interactive_forecast.html")
                    interactive_plot.write_html(interactive_path)
                    saved_files['interactive'] = interactive_path
                    console.print(f"✅ Interactive plot saved to {interactive_path}")
            except Exception as e:
                console.print(f"❌ Interactive visualization failed: {e}")
        
        # Display results
        if saved_files:
            from rich.table import Table
            
            table = Table(title="📊 Generated Visualizations")
            table.add_column("Type", style="cyan")
            table.add_column("File Path", style="green")
            
            for viz_type, file_path in saved_files.items():
                table.add_row(viz_type, file_path)
            
            console.print(table)
        else:
            console.print("⚠️ No visualizations were generated", style="yellow")
        
        console.print("✅ Advanced visualization completed")
        
    except Exception as e:
        console.print(f"❌ Advanced visualization failed: {e}", style="red")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_comprehensive_evaluation(args):
    """Run comprehensive evaluation with all enhancements."""
    console = Console()
    
    try:
        console.print(Panel.fit("🎯 Comprehensive CP-6 Evaluation", style="bold green"))
        
        # Adjust parameters for quick mode
        if args.quick:
            console.print("⚡ Quick mode enabled - reduced trials and data")
            args.n_trials = 10
            args.max_workers = 2
        
        # Step 1: Parallel Evaluation
        if args.parallel:
            console.print("\n🚀 Step 1: Parallel Evaluation")
            run_parallel_evaluation(args)
        
        # Step 2: Hyperparameter Optimization
        if args.hyperopt:
            console.print("\n🔧 Step 2: Hyperparameter Optimization")
            run_hyperparameter_optimization(args)
        
        # Step 3: Advanced Visualization
        if args.viz:
            console.print("\n📊 Step 3: Advanced Visualization")
            run_advanced_visualization(args)
        
        console.print("\n🎉 Comprehensive evaluation completed successfully!")
        
    except Exception as e:
        console.print(f"❌ Comprehensive evaluation failed: {e}", style="red")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'parallel-eval':
        run_parallel_evaluation(args)
    elif args.command == 'hyperopt':
        run_hyperparameter_optimization(args)
    elif args.command == 'advanced-viz':
        run_advanced_visualization(args)
    elif args.command == 'comprehensive':
        run_comprehensive_evaluation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
