#!/usr/bin/env python3
"""Feature Engineering CLI for AirAware PM₂.₅ nowcasting pipeline."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from airaware.features import FeatureBuilder, FeatureConfig, FeatureValidator


def run_feature_pipeline_command(args):
    """Run complete feature engineering pipeline."""
    console = Console()
    
    try:
        console.print(Panel.fit("🔧 Feature Engineering Pipeline", style="bold blue"))
        
        # Initialize feature builder
        config = FeatureConfig()
        
        # Apply command-line overrides
        if args.min_variance:
            config.min_feature_variance = args.min_variance
        
        if args.max_correlation:
            config.max_correlation_threshold = args.max_correlation
        
        config.save_intermediate_steps = args.save_intermediate
        config.save_feature_importance = args.feature_importance
        
        feature_builder = FeatureBuilder(config=config)
        
        console.print(f"\n🚀 Starting Feature Engineering Pipeline")
        console.print(f"Configuration:")
        console.print(f"  • Min Feature Variance: {config.min_feature_variance}")
        console.print(f"  • Max Correlation: {config.max_correlation_threshold}")
        console.print(f"  • Save Intermediate: {config.save_intermediate_steps}")
        console.print(f"  • Feature Importance: {config.save_feature_importance}")
        
        # Run feature building
        result = feature_builder.build_features()
        
        if result.success:
            # Display results
            console.print(f"\n✅ Feature Engineering Complete!")
            
            # Create results table
            results_table = Table(title="Feature Engineering Results")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="white")
            
            results_table.add_row("Total Features", str(result.total_features))
            results_table.add_row("Selected Features", str(result.selected_features))
            results_table.add_row("Records", f"{result.record_count:,}")
            results_table.add_row("Stations", str(len(result.stations_processed)))
            results_table.add_row("Quality Score", f"{result.data_quality_score:.1%}")
            results_table.add_row("Processing Time", f"{result.processing_time_minutes:.1f} min")
            
            console.print(results_table)
            
            # Feature categories
            if result.feature_categories:
                categories_table = Table(title="Feature Categories")
                categories_table.add_column("Category", style="cyan")
                categories_table.add_column("Count", style="green")
                
                for category, count in result.feature_categories.items():
                    categories_table.add_row(category.title(), str(count))
                
                console.print(categories_table)
            
            # Output files
            console.print(f"\n📁 Output Files:")
            for file_type, path in result.output_files.items():
                console.print(f"  • {file_type.title()}: {path}")
            
            # Print summary
            summary = feature_builder.get_build_summary(result)
            console.print(Panel(summary, title="Feature Engineering Summary", border_style="green"))
            
        else:
            console.print(f"[red]❌ Feature engineering failed[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]❌ Feature engineering failed: {e}[/red]")
        sys.exit(1)


def validate_features_command(args):
    """Validate existing feature set."""
    console = Console()
    
    try:
        console.print(Panel.fit("🔍 Feature Validation", style="bold blue"))
        
        # Check if features file exists
        features_path = Path("data/processed/features.parquet")
        
        if not features_path.exists():
            console.print(f"[red]❌ Features file not found: {features_path}[/red]")
            console.print("Run feature engineering first: python scripts/feature_engineering_cli.py run")
            sys.exit(1)
        
        # Load features
        import pandas as pd
        features_df = pd.read_parquet(features_path)
        
        console.print(f"✅ Loaded {len(features_df):,} records with {len(features_df.columns)} columns")
        
        # Initialize validator
        validator = FeatureValidator(
            correlation_threshold=args.correlation_threshold,
            missing_threshold=args.missing_threshold,
            variance_threshold=args.variance_threshold
        )
        
        # Run validation
        console.print("\n🔍 Running feature validation...")
        
        metrics = validator.validate_features(features_df, target_col='pm25')
        
        # Display validation results
        validation_table = Table(title="Feature Validation Results")
        validation_table.add_column("Metric", style="cyan")
        validation_table.add_column("Value", style="white")
        
        validation_table.add_row("Total Features", str(metrics.total_features))
        validation_table.add_row("Numeric Features", str(metrics.numeric_features))
        validation_table.add_row("Categorical Features", str(metrics.categorical_features))
        validation_table.add_row("Data Quality Score", f"{metrics.data_quality_score:.1%}")
        validation_table.add_row("Missing Values Rate", f"{metrics.missing_values_rate:.1%}")
        validation_table.add_row("Infinite Values", str(metrics.infinite_values_count))
        validation_table.add_row("Constant Features", str(metrics.constant_features))
        validation_table.add_row("Duplicate Features", str(metrics.duplicate_features))
        validation_table.add_row("High Correlation Pairs", str(metrics.highly_correlated_pairs))
        
        console.print(validation_table)
        
        # Display quality issues
        if metrics.quality_issues:
            console.print(f"\n⚠️ Quality Issues:")
            for issue in metrics.quality_issues:
                console.print(f"  • {issue}")
        else:
            console.print(f"\n✅ No major quality issues detected")
        
        # Generate and display full report
        if args.detailed:
            report = validator.generate_validation_report(metrics)
            console.print(Panel(report, title="Detailed Validation Report", border_style="yellow"))
            
            # Get recommendations
            recommendations = validator.get_feature_recommendations(metrics)
            
            if any(recommendations.values()):
                console.print(f"\n💡 Feature Recommendations:")
                
                for action, features in recommendations.items():
                    if features:
                        console.print(f"\n{action.title()}:")
                        for feature in features:
                            console.print(f"  • {feature}")
        
        # Save validation report
        if args.save_report:
            report_path = Path("data/processed/feature_validation_report.json")
            
            with open(report_path, 'w') as f:
                json.dump(metrics.model_dump(), f, indent=2, default=str)
            
            console.print(f"\n📄 Validation report saved: {report_path}")
        
    except Exception as e:
        console.print(f"[red]❌ Feature validation failed: {e}[/red]")
        sys.exit(1)


def analyze_features_command(args):
    """Analyze feature importance and statistics."""
    console = Console()
    
    try:
        console.print(Panel.fit("📊 Feature Analysis", style="bold blue"))
        
        # Load features
        features_path = Path("data/processed/features.parquet")
        
        if not features_path.exists():
            console.print(f"[red]❌ Features file not found: {features_path}[/red]")
            sys.exit(1)
        
        import pandas as pd
        features_df = pd.read_parquet(features_path)
        
        console.print(f"✅ Loaded {len(features_df):,} records")
        
        # Basic feature analysis
        feature_cols = [col for col in features_df.columns 
                       if col not in ['datetime_utc', 'station_id', 'pm25']]
        
        numeric_cols = features_df[feature_cols].select_dtypes(include=['number']).columns
        categorical_cols = features_df[feature_cols].select_dtypes(exclude=['number']).columns
        
        # Feature type analysis
        type_table = Table(title="Feature Type Analysis")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", style="green")
        type_table.add_column("Examples", style="white")
        
        type_table.add_row("Numeric", str(len(numeric_cols)), 
                          ", ".join(numeric_cols[:3].tolist()) + ("..." if len(numeric_cols) > 3 else ""))
        type_table.add_row("Categorical", str(len(categorical_cols)),
                          ", ".join(categorical_cols[:3].tolist()) + ("..." if len(categorical_cols) > 3 else ""))
        
        console.print(type_table)
        
        # Target correlation analysis (top correlations)
        if 'pm25' in features_df.columns and len(numeric_cols) > 0:
            console.print(f"\n🎯 Top Target Correlations:")
            
            correlations = features_df[numeric_cols + ['pm25']].corr()['pm25'].abs().sort_values(ascending=False)
            correlations = correlations.drop('pm25')  # Remove self-correlation
            
            top_correlations = correlations.head(10)
            
            corr_table = Table(title="Top 10 Feature-Target Correlations")
            corr_table.add_column("Feature", style="cyan")
            corr_table.add_column("Correlation", style="green")
            
            for feature, corr in top_correlations.items():
                corr_table.add_row(feature, f"{corr:.3f}")
            
            console.print(corr_table)
        
        # Missing values analysis
        missing_analysis = features_df[feature_cols].isnull().sum()
        missing_features = missing_analysis[missing_analysis > 0].sort_values(ascending=False)
        
        if len(missing_features) > 0:
            console.print(f"\n❓ Missing Values Analysis:")
            
            missing_table = Table(title="Features with Missing Values")
            missing_table.add_column("Feature", style="cyan")
            missing_table.add_column("Missing Count", style="red")
            missing_table.add_column("Missing %", style="red")
            
            for feature, count in missing_features.head(10).items():
                pct = (count / len(features_df)) * 100
                missing_table.add_row(feature, str(count), f"{pct:.1f}%")
            
            console.print(missing_table)
        else:
            console.print(f"\n✅ No missing values in feature set")
        
        # Feature categories analysis
        console.print(f"\n📋 Feature Categories:")
        
        categories = {
            "Lag Features": len([col for col in feature_cols if col.startswith('lag_')]),
            "Rolling Features": len([col for col in feature_cols if col.startswith('rolling_') or col.startswith('met_rolling_')]),
            "Calendar Features": len([col for col in feature_cols if col.startswith('calendar_')]),
            "Cyclical Features": len([col for col in feature_cols if col.startswith('cyclical_') or col.startswith('seasonal_')]),
            "Wind Features": len([col for col in feature_cols if col.startswith('wind_')]),
            "Temperature Features": len([col for col in feature_cols if col.startswith('temp_')]),
            "Stability Features": len([col for col in feature_cols if col.startswith('stability_')]),
            "Other Met Features": len([col for col in feature_cols if any(col.startswith(p) for p in ['comfort_', 'pollution_', 'bl_', 'weather_', 'met_lag_'])]),
            "Other Features": len([col for col in feature_cols if not any(col.startswith(p) for p in ['lag_', 'rolling_', 'met_rolling_', 'calendar_', 'cyclical_', 'seasonal_', 'wind_', 'temp_', 'stability_', 'comfort_', 'pollution_', 'bl_', 'weather_', 'met_lag_'])])
        }
        
        cat_table = Table(title="Feature Categories")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", style="green")
        cat_table.add_column("Percentage", style="white")
        
        total_features = len(feature_cols)
        
        for category, count in categories.items():
            if count > 0:
                pct = (count / total_features) * 100
                cat_table.add_row(category, str(count), f"{pct:.1f}%")
        
        console.print(cat_table)
        
    except Exception as e:
        console.print(f"[red]❌ Feature analysis failed: {e}[/red]")
        sys.exit(1)


def status_command(args):
    """Show feature engineering status."""
    console = Console()
    
    try:
        console.print(Panel.fit("📊 Feature Engineering Status", style="bold blue"))
        
        # Check for required input files
        input_files = {
            "OpenAQ Targets": Path("data/interim/targets.parquet"),
            "ERA5 Meteorological": Path("data/interim/era5_hourly.parquet"),
            "IMERG Precipitation": Path("data/interim/imerg_hourly.parquet")
        }
        
        # Check for output files
        output_files = {
            "Joined Data": Path("data/processed/joined_data.parquet"),
            "Features": Path("data/processed/features.parquet"),
            "Feature Report": Path("data/processed/feature_selection_report.json"),
            "Validation Report": Path("data/processed/feature_validation_report.json")
        }
        
        # Input status table
        input_table = Table(title="Input Data Status")
        input_table.add_column("Data Source", style="cyan")
        input_table.add_column("Status", style="white")
        input_table.add_column("Records", style="green")
        
        for name, path in input_files.items():
            if path.exists():
                try:
                    import pandas as pd
                    df = pd.read_parquet(path)
                    input_table.add_row(name, "✅ Available", f"{len(df):,}")
                except Exception:
                    input_table.add_row(name, "⚠️ Error reading", "Unknown")
            else:
                input_table.add_row(name, "❌ Missing", "0")
        
        console.print(input_table)
        
        # Output status table
        output_table = Table(title="Feature Engineering Output Status")
        output_table.add_column("Output", style="cyan")
        output_table.add_column("Status", style="white")
        output_table.add_column("Size", style="green")
        
        for name, path in output_files.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                output_table.add_row(name, "✅ Available", f"{size_mb:.1f} MB")
            else:
                output_table.add_row(name, "❌ Missing", "0 MB")
        
        console.print(output_table)
        
        # Overall readiness assessment
        required_inputs = [input_files["OpenAQ Targets"], input_files["ERA5 Meteorological"]]
        inputs_ready = all(path.exists() for path in required_inputs)
        features_ready = output_files["Features"].exists()
        
        if features_ready:
            console.print("\n🎯 [bold green]Features ready for model training![/bold green]")
        elif inputs_ready:
            console.print("\n⚙️ [bold yellow]Input data ready - run feature engineering[/bold yellow]")
            console.print("Command: python scripts/feature_engineering_cli.py run")
        else:
            console.print("\n⚠️ [bold red]Input data missing - run ETL pipelines first[/bold red]")
        
    except Exception as e:
        console.print(f"[red]❌ Status check failed: {e}[/red]")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline for AirAware")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run feature pipeline
    run_parser = subparsers.add_parser("run", help="Run complete feature engineering pipeline")
    run_parser.add_argument("--min-variance", type=float, default=0.001,
                           help="Minimum feature variance threshold")
    run_parser.add_argument("--max-correlation", type=float, default=0.95,
                           help="Maximum correlation threshold")
    run_parser.add_argument("--save-intermediate", action="store_true",
                           help="Save intermediate processing steps")
    run_parser.add_argument("--feature-importance", action="store_true", default=True,
                           help="Generate feature importance analysis")
    
    # Validate features
    validate_parser = subparsers.add_parser("validate", help="Validate existing feature set")
    validate_parser.add_argument("--correlation-threshold", type=float, default=0.95,
                                help="Correlation threshold for validation")
    validate_parser.add_argument("--missing-threshold", type=float, default=0.5,
                                help="Missing values threshold")
    validate_parser.add_argument("--variance-threshold", type=float, default=0.001,
                                help="Variance threshold")
    validate_parser.add_argument("--detailed", action="store_true",
                                help="Show detailed validation report")
    validate_parser.add_argument("--save-report", action="store_true",
                                help="Save validation report to file")
    
    # Analyze features
    analyze_parser = subparsers.add_parser("analyze", help="Analyze feature importance and statistics")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show feature engineering status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command handler
    if args.command == "run":
        run_feature_pipeline_command(args)
    elif args.command == "validate":
        validate_features_command(args)
    elif args.command == "analyze":
        analyze_features_command(args)
    elif args.command == "status":
        status_command(args)


if __name__ == "__main__":
    main()


