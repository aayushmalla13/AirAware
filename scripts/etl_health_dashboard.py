#!/usr/bin/env python3
"""ETL Health Dashboard for monitoring CP-4 pipeline status."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from airaware.etl.openaq_etl import OpenAQETL
from airaware.etl.era5_etl import ERA5ETL
from airaware.etl.imerg_etl import IMERGETL


def create_etl_health_dashboard():
    """Create a comprehensive ETL health dashboard."""
    
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=5)
    )
    
    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right")
    )
    
    layout["left"].split_column(
        Layout(name="status"),
        Layout(name="quality")
    )
    
    layout["right"].split_column(
        Layout(name="performance"),
        Layout(name="alerts")
    )
    
    return layout


def update_header(layout):
    """Update header with current timestamp."""
    header_text = Text()
    header_text.append("🌍 AirAware ETL Health Dashboard", style="bold blue")
    header_text.append(f"\nLast Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
    
    layout["header"].update(Panel(header_text, border_style="blue"))


def update_pipeline_status(layout):
    """Update pipeline status information."""
    try:
        # Get status from all pipelines
        openaq_etl = OpenAQETL()
        era5_etl = ERA5ETL()
        imerg_etl = IMERGETL()
        
        openaq_status = openaq_etl.get_etl_status()
        era5_status = era5_etl.get_era5_status()
        imerg_status = imerg_etl.get_imerg_status()
        
        # Create status table
        status_table = Table(title="Pipeline Status", show_header=True)
        status_table.add_column("Pipeline", style="cyan")
        status_table.add_column("Status", style="white")
        status_table.add_column("Records", style="green")
        status_table.add_column("Last Update", style="yellow")
        
        # OpenAQ status
        openaq_ready = openaq_status["total_records"] > 0
        status_table.add_row(
            "OpenAQ PM₂.₅",
            "🟢 Active" if openaq_ready else "🔴 Inactive",
            f"{openaq_status['total_records']:,}",
            openaq_status.get("last_updated", "Never")[:16] if openaq_status.get("last_updated") else "Never"
        )
        
        # ERA5 status
        era5_ready = era5_status["era5_file_exists"]
        status_table.add_row(
            "ERA5 Meteorological",
            "🟢 Active" if era5_ready else "🔴 Inactive",
            f"{era5_status['record_count']:,}",
            era5_status.get("latest_data", "Never")[:16] if era5_status.get("latest_data") else "Never"
        )
        
        # IMERG status
        imerg_ready = imerg_status["imerg_file_exists"]
        status_table.add_row(
            "IMERG Precipitation",
            "🟢 Active" if imerg_ready else "🟡 Optional",
            f"{imerg_status['record_count']:,}",
            imerg_status.get("latest_data", "Never")[:16] if imerg_status.get("latest_data") else "Never"
        )
        
        layout["status"].update(Panel(status_table, border_style="green"))
        
    except Exception as e:
        error_text = Text(f"❌ Status Error: {str(e)}", style="red")
        layout["status"].update(Panel(error_text, border_style="red"))


def update_data_quality(layout):
    """Update data quality metrics."""
    try:
        # Load ERA5 data for quality analysis
        import pandas as pd
        
        era5_file = Path("data/interim/era5_hourly.parquet")
        if era5_file.exists():
            df = pd.read_parquet(era5_file)
            
            from airaware.etl.met_data_validator import MeteorologicalDataValidator
            validator = MeteorologicalDataValidator()
            
            quality_metrics = validator.validate_era5_data(df)
            
            # Create quality table
            quality_table = Table(title="Data Quality Metrics", show_header=True)
            quality_table.add_column("Metric", style="cyan")
            quality_table.add_column("Score", style="white")
            quality_table.add_column("Grade", style="green")
            
            quality_table.add_row("Temporal Consistency", f"{quality_metrics.temporal_consistency:.1%}", 
                                 "A" if quality_metrics.temporal_consistency > 0.95 else "B")
            quality_table.add_row("Spatial Consistency", f"{quality_metrics.spatial_consistency:.1%}", 
                                 "A" if quality_metrics.spatial_consistency > 0.95 else "B")
            quality_table.add_row("Physical Realism", f"{quality_metrics.physical_realism:.1%}", 
                                 "A" if quality_metrics.physical_realism > 0.95 else "B")
            quality_table.add_row("Overall Quality", f"{quality_metrics.overall_quality_score:.1%}", 
                                 "A" if quality_metrics.overall_quality_score > 0.9 else "B")
            
            # Add summary
            summary_text = Text()
            summary_text.append(f"\n📊 Records Analyzed: {quality_metrics.total_records:,}\n", style="dim")
            summary_text.append(f"🔍 Missing Values: {quality_metrics.missing_values}\n", style="dim")
            summary_text.append(f"⚠️  Outliers: {quality_metrics.outlier_records}\n", style="dim")
            
            combined_content = Table.grid()
            combined_content.add_row(quality_table)
            combined_content.add_row(summary_text)
            
            layout["quality"].update(Panel(combined_content, border_style="yellow"))
        else:
            layout["quality"].update(Panel("📈 No quality data available", border_style="dim"))
            
    except Exception as e:
        error_text = Text(f"❌ Quality Error: {str(e)}", style="red")
        layout["quality"].update(Panel(error_text, border_style="red"))


def update_performance_metrics(layout):
    """Update performance metrics."""
    try:
        from airaware.etl.performance_optimizer import ETLPerformanceOptimizer
        
        optimizer = ETLPerformanceOptimizer()
        summary = optimizer.get_performance_summary()
        
        if summary.get("no_data"):
            perf_text = Text("📊 No performance data available\nRun ETL operations to see metrics", style="dim")
            layout["performance"].update(Panel(perf_text, border_style="dim"))
        else:
            # Create performance table
            perf_table = Table(title="Performance Metrics", show_header=True)
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value", style="white")
            
            perf_table.add_row("Total Operations", f"{summary['total_operations']:,}")
            perf_table.add_row("Avg Duration", f"{summary['avg_duration_seconds']:.2f}s")
            perf_table.add_row("Avg Throughput", f"{summary['avg_throughput']:.0f} rec/s")
            perf_table.add_row("Total Records", f"{summary['total_records_processed']:,}")
            perf_table.add_row("Total Data Size", f"{summary['total_data_size_mb']:.1f} MB")
            
            layout["performance"].update(Panel(perf_table, border_style="magenta"))
            
    except Exception as e:
        error_text = Text(f"❌ Performance Error: {str(e)}", style="red")
        layout["performance"].update(Panel(error_text, border_style="red"))


def update_system_alerts(layout):
    """Update system alerts and warnings."""
    alerts = []
    
    # Check data freshness
    try:
        import pandas as pd
        
        era5_file = Path("data/interim/era5_hourly.parquet")
        if era5_file.exists():
            df = pd.read_parquet(era5_file)
            last_data = pd.to_datetime(df['datetime_utc'].max())
            hours_ago = (datetime.now() - last_data).total_seconds() / 3600
            
            if hours_ago > 48:
                alerts.append("🔴 ERA5 data is > 48 hours old")
            elif hours_ago > 24:
                alerts.append("🟡 ERA5 data is > 24 hours old")
            else:
                alerts.append("🟢 ERA5 data is fresh")
        else:
            alerts.append("🔴 No ERA5 data available")
            
    except Exception:
        alerts.append("⚠️ Error checking data freshness")
    
    # Check disk space
    try:
        import shutil
        
        data_dir = Path("data")
        if data_dir.exists():
            total, used, free = shutil.disk_usage(data_dir)
            free_gb = free / (1024**3)
            
            if free_gb < 1:
                alerts.append("🔴 Low disk space (< 1GB)")
            elif free_gb < 5:
                alerts.append("🟡 Disk space warning (< 5GB)")
            else:
                alerts.append(f"🟢 Disk space OK ({free_gb:.1f}GB free)")
    except Exception:
        alerts.append("⚠️ Error checking disk space")
    
    # Check error recovery status
    try:
        from airaware.etl.error_recovery import ETLErrorRecovery
        
        error_recovery = ETLErrorRecovery()
        status = error_recovery.get_recovery_status()
        
        if status["manual_interventions_required"] > 0:
            alerts.append(f"🔴 {status['manual_interventions_required']} manual interventions needed")
        elif status["pending_errors"] > 0:
            alerts.append(f"🟡 {status['pending_errors']} pending errors")
        else:
            alerts.append("🟢 No errors pending")
            
    except Exception:
        alerts.append("⚠️ Error checking error recovery")
    
    # Create alerts display
    alerts_text = Text()
    alerts_text.append("System Health Alerts\n", style="bold")
    
    for alert in alerts:
        alerts_text.append(f"{alert}\n", style="white")
    
    layout["alerts"].update(Panel(alerts_text, border_style="red"))


def update_footer(layout):
    """Update footer with system information."""
    footer_text = Text()
    footer_text.append("💡 Commands: ", style="bold")
    footer_text.append("python scripts/era5_etl_cli.py status", style="cyan")
    footer_text.append(" | ", style="dim")
    footer_text.append("python scripts/unified_etl_cli.py status", style="cyan")
    footer_text.append(" | ", style="dim")
    footer_text.append("Ctrl+C to exit", style="red")
    
    layout["footer"].update(Panel(footer_text, border_style="blue"))


def run_dashboard():
    """Run the live ETL health dashboard."""
    console = Console()
    layout = create_etl_health_dashboard()
    
    try:
        with Live(layout, console=console, refresh_per_second=0.5, screen=True):
            while True:
                update_header(layout)
                update_pipeline_status(layout)
                update_data_quality(layout)
                update_performance_metrics(layout)
                update_system_alerts(layout)
                update_footer(layout)
                
                time.sleep(5)  # Update every 5 seconds
                
    except KeyboardInterrupt:
        console.print("\n👋 ETL Health Dashboard stopped")


if __name__ == "__main__":
    run_dashboard()


