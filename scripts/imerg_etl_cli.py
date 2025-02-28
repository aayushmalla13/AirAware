#!/usr/bin/env python3
"""IMERG ETL CLI for AirAware PM₂.₅ nowcasting pipeline.

Precipitation data ETL with JWT authentication and temporal resampling.
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

from airaware.etl.imerg_etl import IMERGETL


def run_imerg_etl_command(args):
    """Run IMERG ETL pipeline."""
    console = Console()
    
    try:
        console.print(Panel.fit("🌧️ IMERG Precipitation ETL", style="bold blue"))
        
        # Parse dates
        start_date = None
        end_date = None
        
        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        
        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        
        # Initialize ETL
        etl = IMERGETL(
            use_local=args.use_local,
            cache_ttl_hours=args.cache_ttl
        )
        
        # Run pipeline
        console.print(f"\n🚀 Starting IMERG ETL Pipeline")
        if start_date and end_date:
            console.print(f"Date range: {start_date.date()} to {end_date.date()}")
        console.print(f"Temporal resolution: 30min → 1H")
        console.print(f"Force refresh: {args.force}")
        console.print(f"Use local: {args.use_local}")
        
        results = etl.run_etl(
            start_date=start_date,
            end_date=end_date,
            force=args.force
        )
        
        # Display results
        console.print(f"\n✅ IMERG ETL Pipeline Complete!")
        console.print(f"📊 Results:")
        console.print(f"  • Processed: {results['timestamps_processed']} files")
        console.print(f"  • Reused: {results['files_reused']} files")
        console.print(f"  • Downloaded: {results['files_downloaded']} files")
        console.print(f"  • Failed: {results['files_failed']} files")
        console.print(f"  • Total records: {results['total_records']:,}")
        console.print(f"  • Duration: {results['duration_minutes']:.1f} minutes")
        
        if results.get('output_file'):
            console.print(f"  • Output: {results['output_file']}")
        
        success_rate = (results['timestamps_processed'] - results['files_failed']) / max(results['timestamps_processed'], 1) * 100
        
        console.print(Panel.fit(
            f"📊 Results\n\n"
            f"• Processed: {results['timestamps_processed']} files\n"
            f"• Reused: {results['files_reused']} files\n"
            f"• Downloaded: {results['files_downloaded']} files\n"
            f"• Failed: {results['files_failed']} files\n"
            f"• Success rate: {success_rate:.1f}%",
            title="IMERG ETL Complete!"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ IMERG ETL failed: {e}[/red]")
        sys.exit(1)


def status_command(args):
    """Show IMERG ETL status."""
    console = Console()
    
    try:
        console.print(Panel.fit("📊 IMERG ETL Status", style="bold blue"))
        
        etl = IMERGETL()
        status = etl.get_imerg_status()
        
        # Create status table
        table = Table(title="IMERG ETL Overview")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("IMERG File Exists", "✅" if status["imerg_file_exists"] else "❌")
        table.add_row("Output Path", status["imerg_file_path"])
        table.add_row("Raw HDF5 Files", str(status["raw_hdf5_files"]))
        table.add_row("Record Count", f"{status['record_count']:,}")
        
        if status["date_range"]:
            table.add_row("Date Range", status["date_range"])
        
        if status["latest_data"]:
            table.add_row("Latest Data", status["latest_data"])
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Failed to get IMERG status: {e}[/red]")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="IMERG Precipitation ETL Pipeline")
    
    # Global options
    parser.add_argument("--use-local", action="store_true", default=True,
                       help="Use local data when available")
    parser.add_argument("--cache-ttl", type=int, default=24,
                       help="Cache TTL in hours (default: 24)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ETL run command
    etl_parser = subparsers.add_parser("run", help="Run IMERG ETL pipeline")
    etl_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    etl_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    etl_parser.add_argument("--force", action="store_true",
                           help="Force refresh all data (ignore cache)")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show IMERG ETL status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command handler
    if args.command == "run":
        run_imerg_etl_command(args)
    elif args.command == "status":
        status_command(args)


if __name__ == "__main__":
    main()


