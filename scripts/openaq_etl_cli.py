#!/usr/bin/env python3
"""OpenAQ ETL CLI for AirAware PM₂.₅ nowcasting pipeline.

Production-grade ETL with reuse-or-fetch logic, partitioned storage,
and comprehensive monitoring.
"""

import argparse
import sys
from datetime import datetime, timedelta
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
from airaware.etl.manifest_manager import ManifestManager
from airaware.etl.quality_monitor import QualityMonitor
from airaware.etl.lineage_tracker import DataLineageTracker


def run_etl_command(args):
    """Run OpenAQ ETL pipeline."""
    console = Console()
    
    try:
        console.print(Panel.fit("🚀 OpenAQ ETL Pipeline", style="bold blue"))
        
        # Parse dates
        start_date = None
        end_date = None
        
        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        
        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        
        # Initialize ETL
        etl = OpenAQETL(
            use_local=args.use_local,
            cache_ttl_hours=args.cache_ttl,
            max_workers=args.workers
        )
        
        # Run pipeline
        results = etl.run_etl_pipeline(
            start_date=start_date,
            end_date=end_date,
            force_refresh=args.force,
            station_ids=args.stations
        )
        
        # Display results
        console.print(Panel.fit(
            f"[bold green]ETL Complete![/bold green]\n\n"
            f"• Processed: {results['stations_processed']} stations\n"
            f"• Reused: {results['results']['reused_partitions']} partitions\n"
            f"• Downloaded: {results['results']['downloaded_partitions']} partitions\n"
            f"• Failed: {results['results']['failed_partitions']} partitions\n"
            f"• Success rate: {results['results']['success_rate']:.1%}",
            title="📊 Results",
            style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ ETL pipeline failed: {e}[/red]")
        sys.exit(1)


def status_command(args):
    """Show ETL status and statistics."""
    console = Console()
    
    try:
        etl = OpenAQETL()
        status = etl.get_etl_status()
        
        console.print(Panel.fit("📊 OpenAQ ETL Status", style="bold blue"))
        
        # Overview table
        table = Table(title="ETL Overview")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Stations Configured", str(status["stations_configured"]))
        table.add_row("ETL Artifacts", f"{status['etl_artifacts']:,}")
        table.add_row("Total Size", f"{status['total_size_mb']:.1f} MB")
        table.add_row("Total Records", f"{status['total_records']:,}")
        table.add_row("Last Updated", status["last_updated"])
        
        console.print(table)
        
        # By source breakdown
        if status["by_source"]:
            source_table = Table(title="By Data Source")
            source_table.add_column("Source", style="cyan")
            source_table.add_column("Artifacts", style="white")
            
            for source, count in status["by_source"].items():
                source_table.add_row(source, str(count))
            
            console.print(source_table)
        
        # By stage breakdown
        if status["by_stage"]:
            stage_table = Table(title="By ETL Stage")
            stage_table.add_column("Stage", style="cyan")
            stage_table.add_column("Artifacts", style="white")
            
            for stage, count in status["by_stage"].items():
                stage_table.add_row(stage, str(count))
            
            console.print(stage_table)
        
    except Exception as e:
        console.print(f"[red]❌ Failed to get ETL status: {e}[/red]")
        sys.exit(1)


def validate_command(args):
    """Validate ETL data and manifest integrity."""
    console = Console()
    
    try:
        console.print(Panel.fit("🔍 ETL Data Validation", style="bold blue"))
        
        manifest_manager = ManifestManager()
        
        # Validate manifest integrity
        console.print("[blue]Checking manifest integrity...[/blue]")
        integrity_ok = manifest_manager.validate_manifest_integrity()
        
        if integrity_ok:
            console.print("[green]✅ Manifest integrity: OK[/green]")
        else:
            console.print("[red]❌ Manifest integrity: ISSUES FOUND[/red]")
        
        # Cleanup stale artifacts if requested
        if args.cleanup:
            console.print("[blue]Cleaning up stale artifacts...[/blue]")
            removed_count = manifest_manager.cleanup_stale_artifacts()
            console.print(f"[green]✅ Cleaned up {removed_count} stale references[/green]")
        
        # Get ETL statistics
        stats = manifest_manager.get_etl_stats()
        
        console.print(f"\n[bold]ETL Statistics:[/bold]")
        console.print(f"• Total artifacts: {stats.get('total_etl_artifacts', 0)}")
        console.print(f"• Total size: {stats.get('total_etl_size_mb', 0):.1f} MB")
        console.print(f"• Total records: {stats.get('total_records', 0):,}")
        
    except Exception as e:
        console.print(f"[red]❌ Validation failed: {e}[/red]")
        sys.exit(1)


def list_partitions_command(args):
    """List available data partitions."""
    console = Console()
    
    try:
        manifest_manager = ManifestManager()
        etl = OpenAQETL()
        
        console.print(Panel.fit("📋 Data Partitions", style="bold blue"))
        
        # Get stations from config
        stations = etl.stations
        
        if args.station:
            stations = [s for s in stations if s["station_id"] == args.station]
        
        for station in stations:
            station_id = station["station_id"]
            station_name = station["name"]
            
            partitions = manifest_manager.get_date_partitions(station_id)
            
            if partitions:
                console.print(f"\n[bold cyan]Station {station_id}: {station_name}[/bold cyan]")
                console.print(f"Available partitions: {len(partitions)}")
                
                # Group by year for better display
                by_year = {}
                for partition in partitions:
                    year = partition.split('-')[0]
                    if year not in by_year:
                        by_year[year] = []
                    by_year[year].append(partition)
                
                for year, year_partitions in sorted(by_year.items()):
                    console.print(f"  {year}: {', '.join(sorted(year_partitions))}")
            else:
                console.print(f"\n[yellow]Station {station_id}: {station_name} - No partitions found[/yellow]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to list partitions: {e}[/red]")
        sys.exit(1)


def export_command(args):
    """Export ETL catalog or data."""
    console = Console()
    
    try:
        manifest_manager = ManifestManager()
        
        if args.catalog:
            output_path = args.output or "data/artifacts/etl_catalog.json"
            manifest_manager.export_etl_catalog(output_path)
            console.print(f"[green]✅ ETL catalog exported to: {output_path}[/green]")
        
        # Could add other export formats here (CSV, etc.)
        
    except Exception as e:
        console.print(f"[red]❌ Export failed: {e}[/red]")
        sys.exit(1)


def quality_command(args):
    """Handle quality monitoring commands."""
    console = Console()
    
    try:
        quality_monitor = QualityMonitor()
        
        if args.quality_action == "report":
            console.print(Panel.fit("📊 Data Quality Report", style="bold blue"))
            report = quality_monitor.generate_quality_report(args.days)
            console.print(report)
            
        elif args.quality_action == "alerts":
            console.print(Panel.fit("🚨 Quality Alerts", style="bold red"))
            alerts = quality_monitor.get_active_alerts(
                station_id=args.station,
                severity=args.severity
            )
            
            if not alerts:
                console.print("[green]✅ No active alerts[/green]")
            else:
                alert_table = Table(title="Active Quality Alerts")
                alert_table.add_column("Time", style="cyan")
                alert_table.add_column("Severity", style="red")
                alert_table.add_column("Station", style="yellow") 
                alert_table.add_column("Message", style="white")
                
                for alert in alerts[:20]:  # Show top 20
                    severity_style = {
                        "low": "blue",
                        "medium": "yellow", 
                        "high": "red",
                        "critical": "bold red"
                    }.get(alert.severity, "white")
                    
                    alert_table.add_row(
                        alert.timestamp.strftime("%m-%d %H:%M"),
                        f"[{severity_style}]{alert.severity.upper()}[/{severity_style}]",
                        str(alert.station_id) if alert.station_id else "All",
                        alert.message
                    )
                
                console.print(alert_table)
                
    except Exception as e:
        console.print(f"[red]❌ Quality monitoring failed: {e}[/red]")
        sys.exit(1)


def lineage_command(args):
    """Handle lineage tracking commands."""
    console = Console()
    
    try:
        lineage_tracker = DataLineageTracker()
        
        if args.lineage_action == "report":
            console.print(Panel.fit("📋 Data Lineage Report", style="bold green"))
            report = lineage_tracker.generate_lineage_report(
                station_id=args.station,
                days_back=args.days
            )
            console.print(report)
            
    except Exception as e:
        console.print(f"[red]❌ Lineage tracking failed: {e}[/red]")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="OpenAQ ETL Pipeline for AirAware")
    
    # Global options
    parser.add_argument("--use-local", action="store_true", default=True,
                       help="Use local data when available")
    parser.add_argument("--cache-ttl", type=int, default=24,
                       help="Cache TTL in hours (default: 24)")
    parser.add_argument("--workers", type=int, default=3,
                       help="Number of parallel workers (default: 3)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ETL run command
    etl_parser = subparsers.add_parser("run", help="Run ETL pipeline")
    etl_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    etl_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    etl_parser.add_argument("--force", action="store_true",
                           help="Force refresh all data (ignore cache)")
    etl_parser.add_argument("--stations", type=int, nargs="+",
                           help="Specific station IDs to process")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show ETL status")
    
    # Validation command
    validate_parser = subparsers.add_parser("validate", help="Validate ETL data")
    validate_parser.add_argument("--cleanup", action="store_true",
                                help="Clean up stale artifact references")
    
    # List partitions command
    list_parser = subparsers.add_parser("partitions", help="List data partitions")
    list_parser.add_argument("--station", type=int, help="Specific station ID")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export ETL data/catalog")
    export_parser.add_argument("--catalog", action="store_true",
                              help="Export ETL catalog")
    export_parser.add_argument("--output", help="Output file path")
    
    # Quality monitoring command
    quality_parser = subparsers.add_parser("quality", help="Quality monitoring operations")
    quality_subparsers = quality_parser.add_subparsers(dest="quality_action")
    
    # Quality report
    report_parser = quality_subparsers.add_parser("report", help="Generate quality report")
    report_parser.add_argument("--days", type=int, default=7, help="Days to look back")
    report_parser.add_argument("--station", type=int, help="Specific station ID")
    
    # Quality alerts
    alerts_parser = quality_subparsers.add_parser("alerts", help="Show active alerts")
    alerts_parser.add_argument("--severity", choices=["low", "medium", "high", "critical"], help="Filter by severity")
    alerts_parser.add_argument("--station", type=int, help="Specific station ID")
    
    # Lineage tracking command
    lineage_parser = subparsers.add_parser("lineage", help="Data lineage operations")
    lineage_subparsers = lineage_parser.add_subparsers(dest="lineage_action")
    
    # Lineage report
    lineage_report_parser = lineage_subparsers.add_parser("report", help="Generate lineage report")
    lineage_report_parser.add_argument("--days", type=int, default=7, help="Days to look back")
    lineage_report_parser.add_argument("--station", type=int, help="Specific station ID")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command handler
    if args.command == "run":
        run_etl_command(args)
    elif args.command == "status":
        status_command(args)
    elif args.command == "validate":
        validate_command(args)
    elif args.command == "partitions":
        list_partitions_command(args)
    elif args.command == "export":
        export_command(args)
    elif args.command == "quality":
        quality_command(args)
    elif args.command == "lineage":
        lineage_command(args)


if __name__ == "__main__":
    main()
