#!/usr/bin/env python3
"""Extended data collection script for CP-6: Collect data from June 2024 to September 2025."""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from airaware.data_pipeline.auto_updater import AutoDataUpdater, UpdateConfig


def collect_extended_data(args):
    """Collect extended historical data for realistic baseline evaluation."""
    console = Console()
    
    try:
        console.print(Panel.fit("📊 Extended Data Collection for CP-6", style="bold blue"))
        
        # Configure data collection
        config = UpdateConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            auto_extend_to_latest=args.auto_extend,
            batch_size_days=args.batch_size,
            update_openaq=args.openaq,
            update_era5=args.era5,
            update_imerg=args.imerg,
            regenerate_features=args.features
        )
        
        console.print(f"\n🎯 Data Collection Configuration:")
        console.print(f"  • Start Date: {config.start_date}")
        console.print(f"  • End Date: {config.end_date}")
        console.print(f"  • Auto Extend: {'Yes' if config.auto_extend_to_latest else 'No'}")
        console.print(f"  • Batch Size: {config.batch_size_days} days")
        console.print(f"  • Data Sources:")
        console.print(f"    - OpenAQ: {'✅' if config.update_openaq else '❌'}")
        console.print(f"    - ERA5: {'✅' if config.update_era5 else '❌'}")
        console.print(f"    - IMERG: {'✅' if config.update_imerg else '❌'}")
        console.print(f"  • Regenerate Features: {'Yes' if config.regenerate_features else 'No'}")
        
        # Calculate expected duration
        start_dt = datetime.strptime(config.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(config.end_date, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days
        
        console.print(f"\n📅 Collection Scope:")
        console.print(f"  • Total Days: {total_days}")
        console.print(f"  • Estimated Duration: {total_days // config.batch_size_days} batches")
        console.print(f"  • Expected Records: ~{total_days * 24 * 3:,} (3 stations × 24h)")
        
        if not args.force and total_days > 100:
            confirm = console.input(f"\n⚠️  This will collect {total_days} days of data. Continue? [y/N]: ")
            if confirm.lower() != 'y':
                console.print("Collection cancelled.")
                return
        
        # Initialize auto updater
        console.print(f"\n🚀 Starting Extended Data Collection")
        
        updater = AutoDataUpdater(config)
        
        # Perform extended data collection
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Collecting extended data...", total=100)
            
            # Check current status
            status = updater.get_update_status()
            current_data = status["current_data"]
            
            console.print(f"\n📊 Current Data Status:")
            console.print(f"  • Has Data: {'Yes' if current_data['has_data'] else 'No'}")
            if current_data['has_data']:
                console.print(f"  • Latest Date: {current_data['latest_date']}")
                console.print(f"  • Total Records: {current_data['total_records']:,}")
                console.print(f"  • Stations: {current_data['stations_count']}")
                console.print(f"  • Quality Score: {current_data['quality_score']:.1%}")
            
            progress.update(task, advance=10)
            
            # Perform the extended collection
            result = updater.extend_historical_data(config.end_date)
            
            progress.update(task, advance=90)
        
        # Display results
        if result.success:
            console.print(f"\n✅ Extended Data Collection Complete!")
            
            from rich.table import Table
            
            results_table = Table(title="Collection Results")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="white")
            
            results_table.add_row("Start Date", result.start_date)
            results_table.add_row("End Date", result.end_date)
            results_table.add_row("Records Added", f"{result.records_added:,}")
            results_table.add_row("Sources Updated", ", ".join(result.sources_updated))
            results_table.add_row("Features Regenerated", "Yes" if result.features_regenerated else "No")
            results_table.add_row("Duration", f"{result.update_duration_minutes:.1f} minutes")
            results_table.add_row("Quality Score", f"{result.data_quality_score:.1%}")
            results_table.add_row("Next Update Due", result.next_update_due)
            
            console.print(results_table)
            
            # Show file locations
            console.print(f"\n📁 Updated Data Files:")
            
            data_files = [
                "data/interim/targets.parquet",
                "data/interim/era5_hourly.parquet", 
                "data/interim/imerg_hourly.parquet",
                "data/processed/enhanced_features.parquet"
            ]
            
            for file_path in data_files:
                path = Path(file_path)
                if path.exists():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    console.print(f"  ✅ {file_path}: {size_mb:.1f} MB")
                else:
                    console.print(f"  ❌ {file_path}: Not found")
            
            # Validation
            console.print(f"\n🔍 Data Validation:")
            
            if Path("data/processed/enhanced_features.parquet").exists():
                import pandas as pd
                
                df = pd.read_parquet("data/processed/enhanced_features.parquet")
                
                console.print(f"  • Final Dataset Shape: {df.shape}")
                console.print(f"  • Date Range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}")
                console.print(f"  • Stations: {df['station_id'].nunique()}")
                console.print(f"  • Features: {len(df.columns) - 3}")
                console.print(f"  • PM2.5 Range: {df['pm25'].min():.1f} - {df['pm25'].max():.1f} μg/m³")
                
                # Calculate data coverage
                expected_hours = (df['datetime_utc'].max() - df['datetime_utc'].min()).total_seconds() / 3600
                actual_hours = len(df) / df['station_id'].nunique()
                coverage = (actual_hours / expected_hours) * 100
                
                console.print(f"  • Temporal Coverage: {coverage:.1f}%")
                
                console.print(f"\n🎯 Ready for CP-6 Baseline Evaluation!")
                
        else:
            console.print(f"[red]❌ Extended data collection failed[/red]")
            console.print(f"Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]❌ Extended data collection failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def check_data_status(args):
    """Check current data status and coverage."""
    console = Console()
    
    try:
        console.print(Panel.fit("📊 Data Status Check", style="bold blue"))
        
        # Initialize updater to check status
        config = UpdateConfig()
        updater = AutoDataUpdater(config)
        
        status = updater.get_update_status()
        current_data = status["current_data"]
        
        from rich.table import Table
        
        # Current data status
        status_table = Table(title="Current Data Status")
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Value", style="white")
        
        status_table.add_row("Has Data", "Yes" if current_data['has_data'] else "No")
        
        if current_data['has_data']:
            status_table.add_row("Earliest Date", current_data['earliest_date'])
            status_table.add_row("Latest Date", current_data['latest_date'])
            status_table.add_row("Total Records", f"{current_data['total_records']:,}")
            status_table.add_row("Stations Count", str(current_data['stations_count']))
            status_table.add_row("Quality Score", f"{current_data['quality_score']:.1%}")
        
        status_table.add_row("Needs Update", "Yes" if status['needs_update'] else "No")
        status_table.add_row("Next Update", status['next_scheduled_update'])
        status_table.add_row("Auto Extend", "Yes" if status['auto_extend_enabled'] else "No")
        
        console.print(status_table)
        
        # Data gaps analysis
        if current_data['data_gaps']:
            console.print(f"\n⚠️ Data Gaps Detected:")
            
            gaps_table = Table(title="Data Gaps")
            gaps_table.add_column("Start", style="yellow")
            gaps_table.add_column("End", style="yellow")
            gaps_table.add_column("Duration (hours)", style="red")
            
            for gap in current_data['data_gaps'][:10]:  # Show top 10 gaps
                gaps_table.add_row(gap['start'], gap['end'], f"{gap['hours']:.1f}")
            
            console.print(gaps_table)
            
            if len(current_data['data_gaps']) > 10:
                console.print(f"  ... and {len(current_data['data_gaps']) - 10} more gaps")
        else:
            console.print(f"\n✅ No significant data gaps detected")
        
        # Recommendations
        console.print(f"\n💡 Recommendations:")
        
        if status['needs_update']:
            console.print(f"  • Run extended data collection to update")
            console.print(f"  • Command: python scripts/extended_data_collection.py collect")
        else:
            console.print(f"  • Data is current and ready for baseline evaluation")
            console.print(f"  • Proceed to CP-6 baseline training")
        
    except Exception as e:
        console.print(f"[red]❌ Status check failed: {e}[/red]")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Extended Data Collection for CP-6")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect extended historical data")
    collect_parser.add_argument("--start-date", default="2024-06-01",
                              help="Start date for collection (YYYY-MM-DD)")
    collect_parser.add_argument("--end-date", default="2025-09-30",
                              help="End date for collection (YYYY-MM-DD)")
    collect_parser.add_argument("--batch-size", type=int, default=7,
                              help="Batch size in days")
    collect_parser.add_argument("--auto-extend", action="store_true", default=True,
                              help="Auto extend to latest available data")
    collect_parser.add_argument("--openaq", action="store_true", default=True,
                              help="Collect OpenAQ data")
    collect_parser.add_argument("--era5", action="store_true", default=True,
                              help="Collect ERA5 data")
    collect_parser.add_argument("--imerg", action="store_true", default=False,
                              help="Collect IMERG data")
    collect_parser.add_argument("--features", action="store_true", default=True,
                              help="Regenerate features")
    collect_parser.add_argument("--force", action="store_true",
                              help="Force collection without confirmation")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check current data status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command handler
    if args.command == "collect":
        collect_extended_data(args)
    elif args.command == "status":
        check_data_status(args)


if __name__ == "__main__":
    main()


