#!/usr/bin/env python3
"""ERA5 ETL CLI for AirAware PM₂.₅ nowcasting pipeline.

Meteorological data ETL with reuse-or-fetch logic and spatial processing.
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

from airaware.etl.era5_etl import ERA5ETL


def run_era5_etl_command(args):
    """Run ERA5 ETL pipeline."""
    console = Console()
    
    try:
        console.print(Panel.fit("🌤️ ERA5 Meteorological ETL", style="bold blue"))
        
        # Parse dates
        start_date = None
        end_date = None
        
        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        
        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        
        # Initialize ETL
        etl = ERA5ETL(
            use_local=args.use_local,
            cache_ttl_hours=args.cache_ttl,
            max_workers=args.workers
        )
        
        # Run pipeline
        console.print(f"\n🚀 Starting ERA5 ETL Pipeline")
        if start_date and end_date:
            console.print(f"Date range: {start_date.date()} to {end_date.date()}")
        console.print(f"Variables: u10, v10, t2m, blh")
        console.print(f"Force refresh: {args.force}")
        console.print(f"Use local: {args.use_local}")
        
        results = etl.run_etl(
            start_date=start_date,
            end_date=end_date,
            force=args.force
        )
        
        # Display results
        console.print(f"\n✅ ERA5 ETL Pipeline Complete!")
        console.print(f"📊 Results:")
        console.print(f"  • Processed: {results['dates_processed']} days")
        console.print(f"  • Reused: {results['dates_reused']} days")
        console.print(f"  • Downloaded: {results['dates_downloaded']} days")
        console.print(f"  • Failed: {results['dates_failed']} days")
        console.print(f"  • Total records: {results['total_records']:,}")
        console.print(f"  • Duration: {results['duration_minutes']:.1f} minutes")
        
        if results.get('output_file'):
            console.print(f"  • Output: {results['output_file']}")
        
        success_rate = (results['dates_processed'] - results['dates_failed']) / max(results['dates_processed'], 1) * 100
        
        console.print(Panel.fit(
            f"📊 Results\n\n"
            f"• Processed: {results['dates_processed']} days\n"
            f"• Reused: {results['dates_reused']} days\n"
            f"• Downloaded: {results['dates_downloaded']} days\n"
            f"• Failed: {results['dates_failed']} days\n"
            f"• Success rate: {success_rate:.1f}%",
            title="ERA5 ETL Complete!"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ ERA5 ETL failed: {e}[/red]")
        sys.exit(1)


def status_command(args):
    """Show ERA5 ETL status."""
    console = Console()
    
    try:
        console.print(Panel.fit("📊 ERA5 ETL Status", style="bold blue"))
        
        etl = ERA5ETL()
        status = etl.get_era5_status()
        
        # Create status table
        table = Table(title="ERA5 ETL Overview")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("ERA5 File Exists", "✅" if status["era5_file_exists"] else "❌")
        table.add_row("Output Path", status["era5_file_path"])
        table.add_row("Raw NetCDF Files", str(status["raw_netcdf_files"]))
        table.add_row("Record Count", f"{status['record_count']:,}")
        
        if status["date_range"]:
            table.add_row("Date Range", status["date_range"])
        
        if status["latest_data"]:
            table.add_row("Latest Data", status["latest_data"])
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Failed to get ERA5 status: {e}[/red]")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="ERA5 Meteorological ETL Pipeline")
    
    # Global options
    parser.add_argument("--use-local", action="store_true", default=True,
                       help="Use local data when available")
    parser.add_argument("--cache-ttl", type=int, default=24,
                       help="Cache TTL in hours (default: 24)")
    parser.add_argument("--workers", type=int, default=2,
                       help="Number of parallel workers (default: 2)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ETL run command
    etl_parser = subparsers.add_parser("run", help="Run ERA5 ETL pipeline")
    etl_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    etl_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    etl_parser.add_argument("--force", action="store_true",
                           help="Force refresh all data (ignore cache)")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show ERA5 ETL status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command handler
    if args.command == "run":
        run_era5_etl_command(args)
    elif args.command == "status":
        status_command(args)


if __name__ == "__main__":
    main()


