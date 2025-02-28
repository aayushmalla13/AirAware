#!/usr/bin/env python3
"""Enhanced ETL Demonstration - showcasing CP-4 improvements."""

import json
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from airaware.etl.era5_etl import ERA5ETL
from airaware.etl.met_data_validator import MeteorologicalDataValidator
from airaware.etl.data_completeness import DataCompletenessAnalyzer
from airaware.etl.performance_optimizer import ETLPerformanceOptimizer
from airaware.etl.error_recovery import ETLErrorRecovery


def demonstrate_enhanced_era5_processing():
    """Demonstrate enhanced ERA5 ETL capabilities."""
    console = Console()
    
    console.print(Panel.fit("🚀 Enhanced ERA5 ETL Demonstration", style="bold blue"))
    
    # Initialize enhanced ETL
    etl = ERA5ETL(use_local=True)
    
    console.print("\n📊 Loading existing ERA5 data for demonstration...")
    
    # Check if we have processed data
    data_file = Path("data/interim/era5_hourly.parquet")
    if not data_file.exists():
        console.print("[red]❌ No ERA5 data found. Run ERA5 ETL first.[/red]")
        return
    
    # Load data for analysis
    import pandas as pd
    df = pd.read_parquet(data_file)
    
    console.print(f"✅ Loaded {len(df)} ERA5 records for analysis")
    
    # Demonstrate enhanced data quality validation
    console.print(Panel.fit("1️⃣ Advanced Meteorological Data Validation", style="bold green"))
    
    validator = MeteorologicalDataValidator()
    quality_metrics = validator.validate_era5_data(df)
    
    # Create quality metrics table
    quality_table = Table(title="ERA5 Data Quality Metrics")
    quality_table.add_column("Metric", style="cyan")
    quality_table.add_column("Score", style="white")
    quality_table.add_column("Status", style="green")
    
    quality_table.add_row("Temporal Consistency", f"{quality_metrics.temporal_consistency:.1%}", 
                         "✅ Excellent" if quality_metrics.temporal_consistency > 0.95 else "⚠️ Fair")
    quality_table.add_row("Spatial Consistency", f"{quality_metrics.spatial_consistency:.1%}", 
                         "✅ Excellent" if quality_metrics.spatial_consistency > 0.95 else "⚠️ Fair")
    quality_table.add_row("Physical Realism", f"{quality_metrics.physical_realism:.1%}", 
                         "✅ Excellent" if quality_metrics.physical_realism > 0.95 else "⚠️ Fair")
    quality_table.add_row("Overall Quality", f"{quality_metrics.overall_quality_score:.1%}", 
                         "✅ Excellent" if quality_metrics.overall_quality_score > 0.9 else "⚠️ Fair")
    
    console.print(quality_table)
    
    # Generate quality report
    quality_report = validator.generate_quality_report(quality_metrics, "ERA5")
    console.print(Panel(quality_report, title="Detailed Quality Report", border_style="blue"))
    
    # Demonstrate data completeness analysis
    console.print(Panel.fit("2️⃣ Advanced Data Completeness Analysis", style="bold yellow"))
    
    analyzer = DataCompletenessAnalyzer()
    start_date = df['datetime_utc'].min()
    end_date = df['datetime_utc'].max()
    
    completeness_metrics = analyzer.analyze_era5_completeness(df, start_date, end_date)
    
    # Create completeness table
    completeness_table = Table(title="ERA5 Data Completeness Analysis")
    completeness_table.add_column("Metric", style="cyan")
    completeness_table.add_column("Value", style="white")
    
    completeness_table.add_row("Data Quality Grade", completeness_metrics.data_quality_grade)
    completeness_table.add_row("Completeness", f"{completeness_metrics.completeness_percentage:.1f}%")
    completeness_table.add_row("Expected Records", f"{completeness_metrics.total_expected_records:,}")
    completeness_table.add_row("Actual Records", f"{completeness_metrics.total_actual_records:,}")
    completeness_table.add_row("Gaps Detected", str(completeness_metrics.gaps_detected))
    completeness_table.add_row("Longest Gap (hours)", f"{completeness_metrics.longest_gap_hours:.1f}")
    
    console.print(completeness_table)
    
    # Generate completeness report
    completeness_report = analyzer.generate_completeness_report(completeness_metrics, "ERA5")
    console.print(Panel(completeness_report, title="Detailed Completeness Report", border_style="yellow"))
    
    # Demonstrate performance optimization
    console.print(Panel.fit("3️⃣ Performance Optimization Analysis", style="bold magenta"))
    
    optimizer = ETLPerformanceOptimizer()
    
    # Test optimized parquet writing
    test_output = Path("data/artifacts/test_optimized.parquet")
    perf_result = optimizer.optimize_parquet_writing(df, test_output)
    
    # Performance table
    perf_table = Table(title="Performance Optimization Results")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="white")
    
    if perf_result["success"]:
        perf_table.add_row("Write Duration", f"{perf_result['duration_seconds']:.2f}s")
        perf_table.add_row("File Size", f"{perf_result['file_size_mb']:.1f} MB")
        perf_table.add_row("Throughput", f"{perf_result['throughput']:.0f} records/sec")
        perf_table.add_row("Compression", "Snappy (optimized)")
        perf_table.add_row("Schema Optimization", "✅ Applied")
    
    console.print(perf_table)
    
    # Clean up test file
    if test_output.exists():
        test_output.unlink()
    
    # Demonstrate error recovery system
    console.print(Panel.fit("4️⃣ Error Recovery & Resilience System", style="bold red"))
    
    error_recovery = ETLErrorRecovery()
    
    # Simulate recording an error
    test_context = {
        "operation": "demo_test",
        "timestamp": datetime.now().isoformat(),
        "test_mode": True
    }
    
    try:
        # Simulate an error
        raise ConnectionError("Simulated network error for demonstration")
    except Exception as e:
        error_id = error_recovery.record_error("era5_download", e, test_context)
        console.print(f"📝 Recorded demo error: {error_id}")
    
    # Show recovery status
    recovery_status = error_recovery.get_recovery_status()
    
    recovery_table = Table(title="Error Recovery System Status")
    recovery_table.add_column("Metric", style="cyan")
    recovery_table.add_column("Value", style="white")
    
    recovery_table.add_row("Total Errors", str(recovery_status["total_errors"]))
    recovery_table.add_row("Recovered Errors", str(recovery_status["recovered_errors"]))
    recovery_table.add_row("Pending Errors", str(recovery_status["pending_errors"]))
    recovery_table.add_row("Manual Interventions", str(recovery_status["manual_interventions_required"]))
    recovery_table.add_row("Recovery Rate", f"{recovery_status['recovery_rate']:.1%}")
    
    console.print(recovery_table)
    
    # Show error summary
    error_summary = error_recovery.get_error_summary()
    if error_summary["errors_by_type"]:
        console.print(f"\n📊 Error Types: {error_summary['errors_by_type']}")
    
    # Demonstration summary
    console.print(Panel.fit("✨ CP-4 Enhancement Demonstration Complete!", style="bold green"))
    
    summary_text = Text()
    summary_text.append("🎯 Enhanced Capabilities Demonstrated:\n\n", style="bold")
    summary_text.append("✅ Advanced meteorological data validation\n", style="green")
    summary_text.append("✅ Comprehensive data completeness analysis\n", style="green")
    summary_text.append("✅ Performance optimization with monitoring\n", style="green")
    summary_text.append("✅ Intelligent error recovery system\n", style="green")
    summary_text.append("✅ Production-grade quality assurance\n", style="green")
    
    console.print(Panel(summary_text, title="Enhancement Summary", border_style="green"))


def demonstrate_data_insights():
    """Generate advanced data insights from processed ERA5 data."""
    console = Console()
    
    console.print(Panel.fit("📈 Advanced Data Insights", style="bold cyan"))
    
    data_file = Path("data/interim/era5_hourly.parquet")
    if not data_file.exists():
        console.print("[red]❌ No ERA5 data available for insights.[/red]")
        return
    
    import pandas as pd
    import numpy as np
    
    df = pd.read_parquet(data_file)
    
    # Generate insights table
    insights_table = Table(title="ERA5 Meteorological Insights")
    insights_table.add_column("Variable", style="cyan")
    insights_table.add_column("Mean", style="white")
    insights_table.add_column("Min", style="blue")
    insights_table.add_column("Max", style="red")
    insights_table.add_column("Std Dev", style="yellow")
    
    variables = ["t2m_celsius", "wind_speed", "wind_direction", "blh"]
    
    for var in variables:
        if var in df.columns:
            values = df[var].dropna()
            if len(values) > 0:
                insights_table.add_row(
                    var,
                    f"{values.mean():.2f}",
                    f"{values.min():.2f}",
                    f"{values.max():.2f}",
                    f"{values.std():.2f}"
                )
    
    console.print(insights_table)
    
    # Generate weather insights
    if "t2m_celsius" in df.columns:
        avg_temp = df["t2m_celsius"].mean()
        temp_range = df["t2m_celsius"].max() - df["t2m_celsius"].min()
        
        weather_insights = f"""
🌡️  Temperature Analysis:
   • Average: {avg_temp:.1f}°C
   • Range: {temp_range:.1f}°C
   • Season: {"Summer" if avg_temp > 25 else "Spring/Autumn" if avg_temp > 15 else "Winter"}

💨 Wind Analysis:
   • Average Speed: {df['wind_speed'].mean():.1f} m/s
   • Predominant Direction: {df['wind_direction'].mode().iloc[0]:.0f}° 
   • Wind Class: {"Light" if df['wind_speed'].mean() < 3 else "Moderate" if df['wind_speed'].mean() < 7 else "Strong"}

🏔️  Atmospheric Conditions:
   • Boundary Layer Height: {df['blh'].mean():.0f} m
   • Stability: {"Stable" if df['blh'].mean() < 500 else "Mixed" if df['blh'].mean() < 1000 else "Unstable"}
        """
        
        console.print(Panel(weather_insights, title="Weather Insights", border_style="cyan"))


if __name__ == "__main__":
    try:
        demonstrate_enhanced_era5_processing()
        demonstrate_data_insights()
    except Exception as e:
        console = Console()
        console.print(f"[red]❌ Demo failed: {e}[/red]")
        sys.exit(1)


