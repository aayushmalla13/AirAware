#!/usr/bin/env python3
"""Advanced configuration management CLI for AirAware.

Enhanced CP-2 tool for configuration validation, versioning, and health monitoring.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from airaware.config import load_data_config, load_app_config
from airaware.config.validation import ConfigValidator
from airaware.config.versioning import ConfigVersionManager


def display_validation_results(console: Console, results, health_score: float):
    """Display validation results in a rich table."""
    table = Table(title="🔍 Configuration Health Check Results")
    
    table.add_column("Check", style="white", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Message", style="white")
    table.add_column("Recommendation", style="blue")
    
    for result in results:
        # Style status based on result
        if result.status == "pass":
            status = "[green]✅ PASS[/green]"
        elif result.status == "warning":
            status = "[yellow]⚠️ WARNING[/yellow]"
        else:
            status = "[red]❌ FAIL[/red]"
        
        # Truncate long messages
        message = result.message[:50] + "..." if len(result.message) > 50 else result.message
        recommendation = result.recommendation[:60] + "..." if len(result.recommendation) > 60 else result.recommendation
        
        table.add_row(
            result.check_name.replace("_", " ").title(),
            status,
            message,
            recommendation
        )
    
    console.print(table)
    
    # Health score summary
    score_color = "green" if health_score >= 0.8 else "yellow" if health_score >= 0.6 else "red"
    health_text = f"[{score_color}]Health Score: {health_score:.1%}[/{score_color}]"
    
    console.print(f"\n{health_text}")


def display_version_history(console: Console, versions):
    """Display configuration version history."""
    if not versions:
        console.print("[yellow]No version history found[/yellow]")
        return
    
    table = Table(title="📚 Configuration Version History")
    
    table.add_column("Version", style="cyan", no_wrap=True)
    table.add_column("Date", style="white")
    table.add_column("Stations", justify="right", style="blue")
    table.add_column("Avg Quality", justify="right", style="green")
    table.add_column("Status", justify="center")
    table.add_column("Description", style="white")
    
    for version in versions[:10]:  # Show last 10 versions
        status = "[bold green]ACTIVE[/bold green]" if version.is_active else ""
        date_str = version.timestamp.strftime("%Y-%m-%d %H:%M")
        
        table.add_row(
            version.version,
            date_str,
            str(version.station_count),
            f"{version.avg_quality_score:.3f}",
            status,
            version.description
        )
    
    console.print(table)


def validate_command(args):
    """Run configuration validation."""
    console = Console()
    
    try:
        # Load configuration
        console.print("[bold blue]Loading configuration...[/bold blue]")
        config = load_data_config(args.config_file)
        
        # Run validation
        console.print("[bold blue]Running health checks...[/bold blue]")
        validator = ConfigValidator()
        results = validator.validate_configuration(config)
        health_score = validator.get_health_score()
        summary = validator.get_summary()
        
        # Display results
        display_validation_results(console, results, health_score)
        
        # Summary panel
        status_color = "green" if summary["status"] == "healthy" else "yellow"
        summary_text = f"""
[bold]Overall Status:[/bold] [{status_color}]{summary["status"].upper()}[/{status_color}]
[bold]Checks Run:[/bold] {summary["total_checks"]}
[bold]Passed:[/bold] [green]{summary["pass_count"]}[/green]
[bold]Warnings:[/bold] [yellow]{summary["warning_count"]}[/yellow]  
[bold]Failed:[/bold] [red]{summary["fail_count"]}[/red]
"""
        
        console.print(Panel(summary_text, title="🎯 Validation Summary", border_style=status_color))
        
        # Exit with error code if unhealthy
        if summary["status"] != "healthy":
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]❌ Validation failed: {e}[/red]")
        sys.exit(1)


def version_command(args):
    """Handle version management commands."""
    console = Console()
    version_manager = ConfigVersionManager()
    
    if args.version_action == "list":
        # List version history
        versions = version_manager.get_version_history()
        display_version_history(console, versions)
        
    elif args.version_action == "create":
        # Create new version
        try:
            config = load_data_config(args.config_file)
            description = args.description or f"Manual version creation at {args.timestamp or 'now'}"
            version = version_manager.create_version(config, description, args.user)
            console.print(f"[green]✅ Created version {version}: {description}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to create version: {e}[/red]")
            sys.exit(1)
            
    elif args.version_action == "compare":
        # Compare two versions
        if not args.version1 or not args.version2:
            console.print("[red]❌ Must specify --version1 and --version2 for comparison[/red]")
            sys.exit(1)
            
        try:
            comparison = version_manager.compare_versions(args.version1, args.version2)
            
            console.print(f"[bold blue]📊 Comparing {args.version1} vs {args.version2}[/bold blue]")
            
            diff = comparison["differences"]
            console.print(f"Station count change: {diff['station_count_change']:+d}")
            console.print(f"Quality score change: {diff['quality_score_change']:+.3f}")
            console.print(f"Time difference: {diff['time_difference']:.1f} hours")
            
        except Exception as e:
            console.print(f"[red]❌ Comparison failed: {e}[/red]")
            sys.exit(1)


def status_command(args):
    """Show comprehensive configuration status."""
    console = Console()
    
    try:
        # Load configurations
        data_config = load_data_config(args.config_file)
        app_config = load_app_config()
        
        # Run quick validation
        validator = ConfigValidator()
        results = validator.validate_configuration(data_config)
        health_score = validator.get_health_score()
        
        # Version info
        version_manager = ConfigVersionManager()
        active_version = version_manager.get_active_version()
        
        # Create status display
        console.print(Panel.fit("🎯 AirAware Configuration Status", style="bold blue"))
        
        # Basic config info
        console.print(f"\n[bold]Application:[/bold] {app_config.app_name} v{app_config.version}")
        console.print(f"[bold]Stations Configured:[/bold] {len(data_config.stations)} primary + {len(data_config.backup_stations)} backup")
        console.print(f"[bold]Quality Threshold:[/bold] ≥{data_config.min_quality_score:.2f}")
        console.print(f"[bold]Missingness Threshold:[/bold] ≤{data_config.max_missingness_pct:.1f}%")
        
        # Version info
        if active_version:
            console.print(f"[bold]Active Version:[/bold] {active_version.version} ({active_version.timestamp.strftime('%Y-%m-%d %H:%M')})")
        else:
            console.print("[bold]Active Version:[/bold] [yellow]None (consider creating one)[/yellow]")
        
        # Health score
        score_color = "green" if health_score >= 0.8 else "yellow" if health_score >= 0.6 else "red"
        console.print(f"[bold]Health Score:[/bold] [{score_color}]{health_score:.1%}[/{score_color}]")
        
        # Quick health summary
        warning_count = sum(1 for r in results if r.status == "warning")
        fail_count = sum(1 for r in results if r.status == "fail")
        
        if fail_count > 0:
            console.print(f"[red]⚠️ {fail_count} critical issues found[/red]")
        elif warning_count > 0:
            console.print(f"[yellow]⚠️ {warning_count} warnings found[/yellow]")
        else:
            console.print("[green]✅ Configuration is healthy[/green]")
        
        # Station summary table
        table = Table(title="Station Summary")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Quality", justify="right", style="green")
        table.add_column("Missingness", justify="right", style="yellow")
        table.add_column("Distance", justify="right", style="blue")
        
        for station in data_config.stations:
            table.add_row(
                str(station.station_id),
                station.name[:30] + "..." if len(station.name) > 30 else station.name,
                f"{station.quality_score:.3f}",
                f"{station.missingness_pct:.1f}%",
                f"{station.distance_km:.1f}km"
            )
        
        console.print(f"\n{table}")
        
    except Exception as e:
        console.print(f"[red]❌ Status check failed: {e}[/red]")
        sys.exit(1)


def main():
    """Main configuration manager CLI."""
    parser = argparse.ArgumentParser(description="AirAware Configuration Manager")
    parser.add_argument("--config-file", default="configs/data.yaml",
                       help="Configuration file path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration health")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Manage configuration versions")
    version_parser.add_argument("version_action", choices=["list", "create", "compare"],
                               help="Version action to perform")
    version_parser.add_argument("--description", help="Version description")
    version_parser.add_argument("--user", help="User creating the version")
    version_parser.add_argument("--version1", help="First version for comparison")
    version_parser.add_argument("--version2", help="Second version for comparison")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show configuration status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command handler
    if args.command == "validate":
        validate_command(args)
    elif args.command == "version":
        version_command(args)
    elif args.command == "status":
        status_command(args)


if __name__ == "__main__":
    main()


