#!/usr/bin/env python3
"""Unified ETL CLI for AirAware PM₂.₅ nowcasting pipeline.

Orchestrates OpenAQ, ERA5, and IMERG ETL pipelines with coordinated execution.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from airaware.etl.openaq_etl import OpenAQETL
from airaware.etl.era5_etl import ERA5ETL
from airaware.etl.imerg_etl import IMERGETL


def run_unified_etl_command(args):
    """Run unified ETL pipeline for all data sources."""
    console = Console()
    
    try:
        console.print(Panel.fit("🌍 AirAware Unified ETL Pipeline", style="bold blue"))
        
        # Parse dates
        start_date = None
        end_date = None
        
        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        
        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        
        console.print(f"\n🚀 Starting Unified ETL Pipeline")
        if start_date and end_date:
            console.print(f"Date range: {start_date.date()} to {end_date.date()}")
        console.print(f"Sources: OpenAQ, ERA5" + (", IMERG" if args.include_imerg else ""))
        console.print(f"Force refresh: {args.force}")
        
        results = {}
        
        # 1. OpenAQ ETL
        if not args.skip_openaq:
            console.print(Panel.fit("1️⃣ OpenAQ PM₂.₅ Data", style="bold green"))
            
            openaq_etl = OpenAQETL(
                use_local=args.use_local,
                cache_ttl_hours=args.cache_ttl,
                max_workers=args.workers
            )
            
            openaq_results = openaq_etl.run_etl_pipeline(
                start_date=start_date,
                end_date=end_date,
                force_refresh=args.force
            )
            
            results["openaq"] = openaq_results
            console.print(f"✅ OpenAQ: {openaq_results.get('success_rate', 0):.1f}% success")
        
        # 2. ERA5 ETL
        if not args.skip_era5:
            console.print(Panel.fit("2️⃣ ERA5 Meteorological Data", style="bold yellow"))
            
            era5_etl = ERA5ETL(
                use_local=args.use_local,
                cache_ttl_hours=args.cache_ttl,
                max_workers=min(args.workers, 2)  # ERA5 rate limits
            )
            
            era5_results = era5_etl.run_etl(
                start_date=start_date,
                end_date=end_date,
                force=args.force
            )
            
            results["era5"] = era5_results
            success_rate = (era5_results['dates_processed'] - era5_results['dates_failed']) / max(era5_results['dates_processed'], 1) * 100
            console.print(f"✅ ERA5: {success_rate:.1f}% success")
        
        # 3. IMERG ETL (optional)
        if args.include_imerg and not args.skip_imerg:
            console.print(Panel.fit("3️⃣ IMERG Precipitation Data", style="bold cyan"))
            
            imerg_etl = IMERGETL(
                use_local=args.use_local,
                cache_ttl_hours=args.cache_ttl
            )
            
            imerg_results = imerg_etl.run_etl(
                start_date=start_date,
                end_date=end_date,
                force=args.force
            )
            
            results["imerg"] = imerg_results
            success_rate = (imerg_results['timestamps_processed'] - imerg_results['files_failed']) / max(imerg_results['timestamps_processed'], 1) * 100
            console.print(f"✅ IMERG: {success_rate:.1f}% success")
        
        # Summary
        console.print(Panel.fit("📊 Unified ETL Summary", style="bold blue"))
        
        summary_table = Table(title="ETL Pipeline Results")
        summary_table.add_column("Data Source", style="cyan")
        summary_table.add_column("Status", style="white")
        summary_table.add_column("Records", style="green")
        summary_table.add_column("Output", style="yellow")
        
        for source, result in results.items():
            if source == "openaq":
                status = "✅ Complete" if result.get("success", False) else "❌ Failed"
                records = f"{result.get('total_records', 0):,}"
                output = "data/interim/targets.parquet"
            elif source == "era5":
                status = "✅ Complete" if result.get("success", False) else "❌ Failed"
                records = f"{result.get('total_records', 0):,}"
                output = "data/interim/era5_hourly.parquet"
            elif source == "imerg":
                status = "✅ Complete" if result.get("success", False) else "❌ Failed"
                records = f"{result.get('total_records', 0):,}"
                output = "data/interim/imerg_hourly.parquet"
            
            summary_table.add_row(source.upper(), status, records, output)
        
        console.print(summary_table)
        
        # Check for readiness for next phase
        targets_ready = results.get("openaq", {}).get("success", False)
        era5_ready = results.get("era5", {}).get("success", False)
        
        if targets_ready and era5_ready:
            console.print("\n🎯 [bold green]Ready for CP-5: Joiner & Feature Builder![/bold green]")
        else:
            console.print("\n⚠️ [bold yellow]Some data sources failed - check logs before proceeding[/bold yellow]")
        
    except Exception as e:
        console.print(f"[red]❌ Unified ETL failed: {e}[/red]")
        sys.exit(1)


def status_command(args):
    """Show unified ETL status."""
    console = Console()
    
    try:
        console.print(Panel.fit("📊 Unified ETL Status", style="bold blue"))
        
        # Get status from all pipelines
        openaq_etl = OpenAQETL()
        era5_etl = ERA5ETL()
        imerg_etl = IMERGETL()
        
        openaq_status = openaq_etl.get_etl_status()
        era5_status = era5_etl.get_era5_status()
        imerg_status = imerg_etl.get_imerg_status()
        
        # Create unified status table
        table = Table(title="Data Pipeline Overview")
        table.add_column("Data Source", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Records", style="green")
        table.add_column("Latest Data", style="yellow")
        
        # OpenAQ status
        openaq_ready = openaq_status["total_records"] > 0
        table.add_row(
            "OpenAQ PM₂.₅",
            "✅ Ready" if openaq_ready else "❌ No Data",
            f"{openaq_status['total_records']:,}",
            openaq_status.get("last_updated", "Never")
        )
        
        # ERA5 status
        era5_ready = era5_status["era5_file_exists"]
        table.add_row(
            "ERA5 Meteorological",
            "✅ Ready" if era5_ready else "❌ No Data",
            f"{era5_status['record_count']:,}",
            era5_status.get("latest_data", "Never")
        )
        
        # IMERG status
        imerg_ready = imerg_status["imerg_file_exists"]
        table.add_row(
            "IMERG Precipitation",
            "✅ Ready" if imerg_ready else "❌ No Data",
            f"{imerg_status['record_count']:,}",
            imerg_status.get("latest_data", "Never")
        )
        
        console.print(table)
        
        # Readiness assessment
        if openaq_ready and era5_ready:
            console.print("\n🎯 [bold green]Pipeline ready for feature engineering![/bold green]")
        else:
            console.print("\n⚠️ [bold yellow]Run ETL pipelines to prepare data[/bold yellow]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to get unified status: {e}[/red]")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="AirAware Unified ETL Pipeline")
    
    # Global options
    parser.add_argument("--use-local", action="store_true", default=True,
                       help="Use local data when available")
    parser.add_argument("--cache-ttl", type=int, default=24,
                       help="Cache TTL in hours (default: 24)")
    parser.add_argument("--workers", type=int, default=3,
                       help="Number of parallel workers (default: 3)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Unified ETL run command
    etl_parser = subparsers.add_parser("run", help="Run unified ETL pipeline")
    etl_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    etl_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    etl_parser.add_argument("--force", action="store_true",
                           help="Force refresh all data (ignore cache)")
    etl_parser.add_argument("--include-imerg", action="store_true",
                           help="Include IMERG precipitation data")
    etl_parser.add_argument("--skip-openaq", action="store_true",
                           help="Skip OpenAQ ETL")
    etl_parser.add_argument("--skip-era5", action="store_true",
                           help="Skip ERA5 ETL")
    etl_parser.add_argument("--skip-imerg", action="store_true",
                           help="Skip IMERG ETL")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show unified ETL status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command handler
    if args.command == "run":
        run_unified_etl_command(args)
    elif args.command == "status":
        status_command(args)


if __name__ == "__main__":
    main()


