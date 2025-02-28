#!/usr/bin/env python3
"""Quick enhanced evaluation with sampling for fast benchmarking."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def quick_enhanced_benchmark(args):
    """Run super-fast benchmark with enhanced models using sampling."""
    console = Console()
    
    try:
        console.print(Panel.fit("⚡ Quick Enhanced Benchmark (Sampled)", style="bold yellow"))
        
        import pandas as pd
        from airaware.baselines import SeasonalNaiveForecaster, ProphetBaseline, ARIMABaseline, BaselineEnsemble
        
        # Load data
        df = pd.read_parquet(args.data_path)
        
        # Quick split with sampling - use a date within our data range
        split_date = pd.to_datetime('2025-09-22').tz_localize('UTC')
        train_df_full = df[df['datetime_utc'] < split_date].copy()
        test_df = df[df['datetime_utc'] >= split_date].head(24).copy()  # Small test set
        
        # Sample training data for speed
        sample_size = min(args.sample_size, len(train_df_full))
        train_df = train_df_full.sample(n=sample_size, random_state=42)
        
        console.print(f"📊 Quick test: {len(train_df):,} train (sampled from {len(train_df_full):,}), {len(test_df):,} test")
        
        # Test timestamps
        test_timestamps = test_df['datetime_utc'].head(12).tolist()  # 12 hour forecast
        
        models = [
            ("Seasonal-Naive", SeasonalNaiveForecaster()),
            ("Prophet (Real)", ProphetBaseline()),
            ("ARIMA (Auto)", ARIMABaseline()),
            ("Ensemble", BaselineEnsemble())
        ]
        
        results = []
        
        for model_name, model in models:
            try:
                console.print(f"🔧 Testing {model_name}...")
                
                # Fit and predict
                if args.verbose:
                    console.print(f"  • Fitting on {len(train_df):,} records...")
                
                model.fit(train_df, target_col='pm25', group_col='station_id')
                
                if args.verbose:
                    console.print(f"  • Predicting {len(test_timestamps)} steps...")
                
                forecast = model.predict(test_timestamps, station_id=101)
                
                # Calculate simple MAE
                actual_values = test_df.head(12)['pm25'].tolist()
                predictions = forecast.predictions
                
                if len(actual_values) == len(predictions):
                    mae = sum(abs(a - p) for a, p in zip(actual_values, predictions)) / len(actual_values)
                    
                    results.append((model_name, mae, len(predictions), "✅"))
                    console.print(f"  ✅ {model_name}: MAE = {mae:.2f} μg/m³")
                else:
                    results.append((model_name, float('inf'), 0, "❌"))
                    console.print(f"  ❌ {model_name}: Length mismatch")
                
            except Exception as e:
                console.print(f"  ❌ {model_name}: {str(e)[:50]}...")
                results.append((model_name, float('inf'), 0, "❌"))
        
        # Display results
        from rich.table import Table
        
        table = Table(title="⚡ Quick Enhanced Benchmark Results")
        table.add_column("Model", style="cyan")
        table.add_column("MAE (μg/m³)", justify="right", style="green")
        table.add_column("Predictions", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Speed", justify="center")
        
        for model_name, mae, n_pred, status in results:
            mae_str = f"{mae:.2f}" if mae < float('inf') else "FAILED"
            
            # Assign speed based on model type
            if "Seasonal" in model_name:
                speed = "🚀"
            elif "Prophet" in model_name or "ARIMA" in model_name:
                speed = "🐌" if sample_size > 1000 else "⚡"
            else:
                speed = "🔄"
            
            table.add_row(model_name, mae_str, str(n_pred), status, speed)
        
        console.print(table)
        
        # Find best
        valid_results = [(name, mae) for name, mae, _, status in results if mae < float('inf') and status == "✅"]
        if valid_results:
            best_model, best_mae = min(valid_results, key=lambda x: x[1])
            console.print(f"\n🏆 Best Model: {best_model} (MAE: {best_mae:.2f})")
            
            # Performance insights
            console.print(f"\n💡 Performance Insights:")
            console.print(f"  • Training sample: {sample_size:,} records ({sample_size/len(train_df_full)*100:.1f}% of full data)")
            console.print(f"  • Test horizon: 12 hours")
            console.print(f"  • Best MAE: {best_mae:.2f} μg/m³")
            
            if best_mae < 5.0:
                console.print(f"  • 🎊 Excellent performance! (<5 μg/m³)")
            elif best_mae < 10.0:
                console.print(f"  • ✅ Good performance! (<10 μg/m³)")
            else:
                console.print(f"  • ⚠️ Needs improvement (>10 μg/m³)")
        
        console.print(f"\n⚡ Quick enhanced benchmark complete!")
        console.print(f"💡 For full evaluation, use: python scripts/baseline_evaluation_cli.py evaluate")
        
    except Exception as e:
        console.print(f"[red]❌ Quick benchmark failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_quick_visualizations(args):
    """Create quick visualizations from evaluation results."""
    console = Console()
    
    try:
        console.print(Panel.fit("📊 Quick Visualization Creation", style="bold green"))
        
        import json
        import pandas as pd
        from airaware.visualization import ForecastDashboard
        
        # Load evaluation results
        if not Path(args.results_file).exists():
            console.print(f"❌ Results file not found: {args.results_file}")
            console.print("💡 Run evaluation first: python scripts/baseline_evaluation_cli.py evaluate")
            return
        
        with open(args.results_file, 'r') as f:
            results = json.load(f)
        
        console.print(f"✅ Loaded results from {args.results_file}")
        
        # Create dashboard
        dashboard = ForecastDashboard()
        
        # Create sample forecast data for visualization
        console.print("📈 Creating sample visualizations...")
        
        # Mock data for quick visualization demo
        sample_actual = pd.DataFrame({
            'datetime_utc': pd.date_range('2025-07-01', periods=100, freq='h'),
            'station_id': [101] * 100,
            'pm25': 20 + 10 * np.sin(np.arange(100) * 0.1) + np.random.normal(0, 2, 100)
        })
        
        sample_forecasts = {
            'prophet': pd.DataFrame({
                'datetime_utc': sample_actual['datetime_utc'],
                'predictions': sample_actual['pm25'] + np.random.normal(0, 1, 100),
                'station_id': sample_actual['station_id']
            }),
            'seasonal_naive': pd.DataFrame({
                'datetime_utc': sample_actual['datetime_utc'],
                'predictions': sample_actual['pm25'] + np.random.normal(0, 3, 100),
                'station_id': sample_actual['station_id']
            })
        }
        
        # Create visualizations
        import numpy as np
        plots = dashboard.create_comprehensive_dashboard(
            sample_actual, sample_forecasts, results, args.output_dir
        )
        
        console.print(f"✅ Created {len(plots)} visualizations:")
        for plot_name, plot_path in plots.items():
            console.print(f"  • {plot_name}: {plot_path}")
        
        console.print(f"\n📁 All plots saved to: {args.output_dir}")
        
    except Exception as e:
        console.print(f"[red]❌ Visualization creation failed: {e}[/red]")
        import traceback
        traceback.print_exc()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Quick Enhanced CP-6 Evaluation")
    
    # Global arguments
    parser.add_argument("--data-path", default="data/processed/features.parquet",
                       help="Path to processed features dataset")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Quick benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run quick enhanced benchmark")
    bench_parser.add_argument("--sample-size", type=int, default=500,
                            help="Training sample size for speed")
    bench_parser.add_argument("--verbose", action="store_true",
                            help="Verbose output")
    
    # Quick visualization command
    viz_parser = subparsers.add_parser("visualize", help="Create quick visualizations")
    viz_parser.add_argument("--results-file", default="data/artifacts/baseline_evaluation.json",
                          help="Results file from evaluation")
    viz_parser.add_argument("--output-dir", default="data/artifacts/visualizations",
                          help="Output directory for plots")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command handler
    if args.command == "benchmark":
        quick_enhanced_benchmark(args)
    elif args.command == "visualize":
        create_quick_visualizations(args)


if __name__ == "__main__":
    main()

