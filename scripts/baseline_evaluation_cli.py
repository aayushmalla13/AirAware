#!/usr/bin/env python3
"""Comprehensive CLI for CP-6 baseline evaluation and comparison."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from airaware.evaluation.baseline_evaluator import BaselineEvaluator, BaselineEvaluationConfig


def run_comprehensive_evaluation(args):
    """Run comprehensive baseline evaluation."""
    console = Console()
    
    try:
        console.print(Panel.fit("🚀 CP-6: Comprehensive Baseline Evaluation", style="bold blue"))
        
        # Load data
        console.print(f"📊 Loading dataset from {args.data_path}")
        
        import pandas as pd
        df = pd.read_parquet(args.data_path)
        
        console.print(f"✅ Loaded {len(df):,} records")
        console.print(f"  • Date range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}")
        console.print(f"  • Stations: {df['station_id'].nunique()}")
        console.print(f"  • Features: {len(df.columns) - 3}")
        
        # Configure evaluation
        config = BaselineEvaluationConfig(
            train_end_date=args.train_end_date,
            validation_end_date=args.validation_end_date,
            forecast_horizons=args.horizons,
            evaluate_seasonal_naive=args.seasonal_naive,
            evaluate_prophet=args.prophet,
            evaluate_arima=args.arima,
            evaluate_ensemble=args.ensemble,
            save_predictions=args.save_predictions,
            save_visualizations=args.save_plots
        )
        
        console.print(f"🎯 Evaluation Configuration:")
        console.print(f"  • Train End: {config.train_end_date}")
        console.print(f"  • Validation End: {config.validation_end_date}")
        console.print(f"  • Horizons: {config.forecast_horizons}")
        console.print(f"  • Models: ", end="")
        models = []
        if config.evaluate_seasonal_naive: models.append("Seasonal-Naive")
        if config.evaluate_prophet: models.append("Prophet")
        if config.evaluate_arima: models.append("ARIMA")
        if config.evaluate_ensemble: models.append("Ensemble")
        console.print(", ".join(models))
        
        # Run evaluation
        evaluator = BaselineEvaluator(config)
        result = evaluator.evaluate_all_baselines(df, target_col='pm25', group_col='station_id')
        
        # Save results
        if args.output_file:
            output_path = evaluator.save_results(result, args.output_file)
            console.print(f"📁 Detailed results saved to {output_path}")
        
        # Display success criteria analysis
        console.print(f"\n🎯 SUCCESS CRITERIA ANALYSIS:")
        console.print("=" * 50)
        
        if 'success_criteria' in result.summary_metrics:
            criteria = result.summary_metrics['success_criteria']
            
            overall_success = True
            for horizon in ['6h', '12h', '24h']:
                if horizon in criteria:
                    improvement = criteria[horizon]['best_improvement_pct']
                    meets_10 = criteria[horizon]['meets_10pct_target']
                    meets_15 = criteria[horizon]['meets_15pct_target']
                    
                    status_10 = "✅ PASS" if meets_10 else "❌ FAIL"
                    status_15 = "✅ PASS" if meets_15 else "❌ FAIL"
                    
                    console.print(f"  {horizon:>4}: {improvement:+6.1f}% | 10% target: {status_10} | 15% target: {status_15}")
                    
                    if not meets_10:
                        overall_success = False
                else:
                    console.print(f"  {horizon:>4}: No data")
                    overall_success = False
            
            console.print(f"\n🏆 OVERALL SUCCESS: {'✅ PASS' if overall_success else '❌ FAIL'}")
            
            if overall_success:
                console.print("🎊 Baselines meet the ≥10% improvement criteria!")
            else:
                console.print("⚠️  Some baselines need improvement to meet success criteria.")
        
        # Display champion model
        if result.champion_model:
            console.print(f"\n🥇 CHAMPION MODEL: {result.champion_model['model_name']}")
            console.print(f"   Average MAE: {result.champion_model['average_mae']:.2f} μg/m³")
        
        console.print(f"\n✅ Comprehensive baseline evaluation complete!")
        
        return result
        
    except Exception as e:
        console.print(f"[red]❌ Evaluation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def quick_benchmark(args):
    """Run quick benchmark of all models."""
    console = Console()
    
    try:
        console.print(Panel.fit("⚡ Quick Baseline Benchmark", style="bold yellow"))
        
        import pandas as pd
        from airaware.baselines import SeasonalNaiveForecaster, ProphetBaseline, ARIMABaseline
        
        # Load data
        df = pd.read_parquet(args.data_path)
        
        # Quick split
        split_date = pd.to_datetime('2025-01-01')
        train_df = df[df['datetime_utc'] < split_date].copy()
        test_df = df[df['datetime_utc'] >= split_date].head(72).copy()  # 72 hours
        
        console.print(f"📊 Quick test: {len(train_df):,} train, {len(test_df):,} test records")
        
        # Test timestamps
        test_timestamps = test_df['datetime_utc'].head(24).tolist()
        
        models = [
            ("Seasonal-Naive", SeasonalNaiveForecaster()),
            ("Prophet", ProphetBaseline()),
            ("ARIMA", ARIMABaseline())
        ]
        
        results = []
        
        for model_name, model in models:
            try:
                console.print(f"🔧 Testing {model_name}...")
                
                # Fit and predict
                model.fit(train_df, target_col='pm25', group_col='station_id')
                forecast = model.predict(test_timestamps, station_id=101)
                
                # Calculate simple MAE
                actual_values = test_df.head(24)['pm25'].tolist()
                predictions = forecast.predictions
                
                mae = sum(abs(a - p) for a, p in zip(actual_values, predictions)) / len(actual_values)
                
                results.append((model_name, mae, len(predictions)))
                console.print(f"  ✅ {model_name}: MAE = {mae:.2f} μg/m³")
                
            except Exception as e:
                console.print(f"  ❌ {model_name}: {e}")
                results.append((model_name, float('inf'), 0))
        
        # Display results
        from rich.table import Table
        
        table = Table(title="Quick Benchmark Results")
        table.add_column("Model", style="cyan")
        table.add_column("MAE (μg/m³)", justify="right", style="green")
        table.add_column("Predictions", justify="right")
        table.add_column("Status", justify="center")
        
        for model_name, mae, n_pred in results:
            status = "✅" if mae < float('inf') else "❌"
            mae_str = f"{mae:.2f}" if mae < float('inf') else "FAILED"
            
            table.add_row(model_name, mae_str, str(n_pred), status)
        
        console.print(table)
        
        # Find best
        valid_results = [(name, mae) for name, mae, _ in results if mae < float('inf')]
        if valid_results:
            best_model, best_mae = min(valid_results, key=lambda x: x[1])
            console.print(f"\n🏆 Best Model: {best_model} (MAE: {best_mae:.2f})")
        
        console.print(f"\n⚡ Quick benchmark complete!")
        
    except Exception as e:
        console.print(f"[red]❌ Quick benchmark failed: {e}[/red]")
        sys.exit(1)


def display_data_status(args):
    """Display current data status for baseline evaluation."""
    console = Console()
    
    try:
        console.print(Panel.fit("📊 Data Status for Baseline Evaluation", style="bold green"))
        
        import pandas as pd
        
        # Load and analyze data
        df = pd.read_parquet(args.data_path)
        
        console.print(f"📈 Dataset Overview:")
        console.print(f"  • Total Records: {len(df):,}")
        console.print(f"  • Features: {len(df.columns) - 3}")
        console.print(f"  • Stations: {df['station_id'].nunique()}")
        console.print(f"  • Date Range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}")
        
        # Calculate duration
        duration = df['datetime_utc'].max() - df['datetime_utc'].min()
        console.print(f"  • Duration: {duration.days} days ({duration.days/30:.1f} months)")
        
        # Time series splits
        train_end = pd.to_datetime('2024-12-31')
        val_end = pd.to_datetime('2025-06-30')
        
        train_data = df[df['datetime_utc'] <= train_end]
        val_data = df[(df['datetime_utc'] > train_end) & (df['datetime_utc'] <= val_end)]
        test_data = df[df['datetime_utc'] > val_end]
        
        console.print(f"\n📅 Time Series Splits:")
        console.print(f"  • Training: {len(train_data):,} records ({train_data['datetime_utc'].min()} to {train_data['datetime_utc'].max()})")
        console.print(f"  • Validation: {len(val_data):,} records ({val_data['datetime_utc'].min()} to {val_data['datetime_utc'].max()})" if len(val_data) > 0 else "  • Validation: No data")
        console.print(f"  • Test: {len(test_data):,} records ({test_data['datetime_utc'].min()} to {test_data['datetime_utc'].max()})" if len(test_data) > 0 else "  • Test: No data")
        
        # PM2.5 analysis
        console.print(f"\n🎯 PM2.5 Target Analysis:")
        console.print(f"  • Range: {df['pm25'].min():.1f} - {df['pm25'].max():.1f} μg/m³")
        console.print(f"  • Mean: {df['pm25'].mean():.1f} μg/m³")
        console.print(f"  • Std: {df['pm25'].std():.1f} μg/m³")
        console.print(f"  • Missing: {df['pm25'].isnull().sum()} ({df['pm25'].isnull().mean()*100:.1f}%)")
        
        # Data quality
        console.print(f"\n🔍 Data Quality:")
        missing_by_station = df.groupby('station_id')['pm25'].apply(lambda x: x.isnull().mean() * 100)
        console.print(f"  • Missing by Station: {missing_by_station.min():.1f}% - {missing_by_station.max():.1f}%")
        
        temporal_coverage = len(df) / (duration.total_seconds() / 3600 / df['station_id'].nunique())
        console.print(f"  • Temporal Coverage: {temporal_coverage*100:.1f}%")
        
        # Readiness assessment
        console.print(f"\n✅ Baseline Evaluation Readiness:")
        
        checks = [
            ("Sufficient training data", len(train_data) >= 1000, f"{len(train_data):,} records"),
            ("Has test data", len(test_data) >= 100, f"{len(test_data):,} records"),
            ("Multiple stations", df['station_id'].nunique() >= 2, f"{df['station_id'].nunique()} stations"),
            ("Low missing data", df['pm25'].isnull().mean() < 0.1, f"{df['pm25'].isnull().mean()*100:.1f}% missing"),
            ("Sufficient duration", duration.days >= 30, f"{duration.days} days")
        ]
        
        all_passed = True
        for check_name, passed, details in checks:
            status = "✅" if passed else "❌"
            console.print(f"  {status} {check_name}: {details}")
            if not passed:
                all_passed = False
        
        if all_passed:
            console.print(f"\n🎊 Data is ready for comprehensive baseline evaluation!")
        else:
            console.print(f"\n⚠️  Some data quality issues detected - evaluation may be limited.")
        
    except Exception as e:
        console.print(f"[red]❌ Data status check failed: {e}[/red]")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="CP-6 Baseline Evaluation CLI")
    
    # Global arguments
    parser.add_argument("--data-path", default="data/processed/features.parquet",
                       help="Path to processed features dataset")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Comprehensive evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Run comprehensive baseline evaluation")
    eval_parser.add_argument("--train-end-date", default="2024-12-31",
                           help="End date for training data")
    eval_parser.add_argument("--validation-end-date", default="2025-06-30", 
                           help="End date for validation data")
    eval_parser.add_argument("--horizons", nargs="+", type=int, default=[6, 12, 24],
                           help="Forecast horizons to evaluate")
    eval_parser.add_argument("--seasonal-naive", action="store_true", default=True,
                           help="Evaluate seasonal naive model")
    eval_parser.add_argument("--prophet", action="store_true", default=True,
                           help="Evaluate Prophet model")
    eval_parser.add_argument("--arima", action="store_true", default=True,
                           help="Evaluate ARIMA model")
    eval_parser.add_argument("--ensemble", action="store_true", default=True,
                           help="Evaluate ensemble model")
    eval_parser.add_argument("--output-file", default="data/artifacts/baseline_evaluation.json",
                           help="Output file for results")
    eval_parser.add_argument("--save-predictions", action="store_true", default=True,
                           help="Save detailed predictions")
    eval_parser.add_argument("--save-plots", action="store_true", default=True,
                           help="Save evaluation plots")
    
    # Quick benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run quick baseline benchmark")
    
    # Data status command
    status_parser = subparsers.add_parser("status", help="Display data status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command handler
    if args.command == "evaluate":
        run_comprehensive_evaluation(args)
    elif args.command == "benchmark":
        quick_benchmark(args)
    elif args.command == "status":
        display_data_status(args)


if __name__ == "__main__":
    main()


