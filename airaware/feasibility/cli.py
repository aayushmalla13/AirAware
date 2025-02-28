"""Command-line interface for AirAware feasibility assessment."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .acceptance import AcceptanceCriteria, FeasibilityValidator

# Load environment variables
load_dotenv()

console = Console()
app = typer.Typer(
    name="feasibility",
    help="AirAware feasibility assessment toolkit",
    add_completion=False,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)


def print_banner() -> None:
    """Print AirAware feasibility banner."""
    banner = Text("🌍 AirAware Feasibility Assessment", style="bold blue")
    console.print(Panel(banner, title="Data Availability Checker", expand=False))


def check_api_keys(
    require_openaq: bool = True,
    require_cds: bool = True,
    require_earthdata: bool = False,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Check and validate API keys."""
    openaq_key = os.getenv("OPENAQ_KEY")
    cds_key = os.getenv("CDSAPI_KEY")
    earthdata_token = os.getenv("EARTHDATA_TOKEN")
    
    missing_keys = []
    
    if require_openaq and not openaq_key:
        missing_keys.append("OPENAQ_KEY")
    
    if require_cds and not cds_key:
        missing_keys.append("CDSAPI_KEY")
    
    if require_earthdata and not earthdata_token:
        missing_keys.append("EARTHDATA_TOKEN")
    
    if missing_keys:
        console.print("[red]❌ Missing required API keys:[/red]")
        for key in missing_keys:
            console.print(f"  • {key}")
        console.print("\n[yellow]Set keys in .env file or environment variables[/yellow]")
        return None, None, None
    
    # Mask keys for display
    def mask_key(key: Optional[str]) -> str:
        if not key:
            return "Not set"
        return f"{'*' * min(8, len(key))}..."
    
    console.print("[green]✅ API Keys Status:[/green]")
    console.print(f"  • OPENAQ_KEY: {mask_key(openaq_key)}")
    console.print(f"  • CDSAPI_KEY: {mask_key(cds_key)}")
    console.print(f"  • EARTHDATA_TOKEN: {mask_key(earthdata_token)}")
    
    return openaq_key, cds_key, earthdata_token


def print_openaq_results(result) -> None:
    """Print OpenAQ validation results."""
    status = "✅ PASS" if result.openaq_pass else "❌ FAIL"
    console.print(f"\n[bold]OpenAQ PM₂.₅ Stations: {status}[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Status", style="green" if result.openaq_pass else "red")
    
    table.add_row("Stations Found", str(result.pm25_stations_found), "")
    table.add_row("Stations Qualified", str(result.pm25_stations_qualified), "")
    table.add_row("Minimum Required", str(result.criteria_used.min_pm25_stations), "")
    
    console.print(table)
    
    if result.best_stations:
        console.print("\n[bold]Top Qualifying Stations:[/bold]")
        station_table = Table(show_header=True, header_style="bold magenta")
        station_table.add_column("ID", style="cyan")
        station_table.add_column("Name", style="white")
        station_table.add_column("Distance (km)", style="yellow")
        station_table.add_column("Quality Score", style="green")
        station_table.add_column("Missingness %", style="red")
        
        for station in result.best_stations[:5]:
            station_table.add_row(
                str(station.station_id),
                station.station_name[:30] + "..." if len(station.station_name) > 30 else station.station_name,
                f"{station.distance_km:.1f}",
                f"{station.data_quality_score:.2f}",
                f"{station.missingness_pct:.1f}%",
            )
        
        console.print(station_table)
    
    if result.openaq_issues:
        console.print("\n[red]Issues:[/red]")
        for issue in result.openaq_issues:
            console.print(f"  • {issue}")


def print_era5_results(result) -> None:
    """Print ERA5 validation results."""
    status = "✅ PASS" if result.era5_pass else "❌ FAIL"
    console.print(f"\n[bold]ERA5 Meteorological Data: {status}[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Variable", style="cyan")
    table.add_column("Available", style="green")
    table.add_column("Description", style="white")
    
    required_vars = ["u10", "v10", "t2m", "blh"]
    descriptions = {
        "u10": "10m U wind component",
        "v10": "10m V wind component", 
        "t2m": "2m temperature",
        "blh": "Boundary layer height",
    }
    
    for var in required_vars:
        available = "✅" if var in result.era5_variables_available else "❌"
        table.add_row(var, available, descriptions.get(var, ""))
    
    console.print(table)
    
    if result.era5_sample_results:
        success_count = sum(1 for r in result.era5_sample_results if r.overall_ok)
        console.print(f"\n[bold]Sample validation:[/bold] {success_count}/{len(result.era5_sample_results)} days successful")
        console.print(f"[bold]Hourly data OK:[/bold] {result.era5_hourly_ok}")
    
    if result.era5_issues:
        console.print("\n[red]Issues:[/red]")
        for issue in result.era5_issues:
            console.print(f"  • {issue}")


def print_imerg_results(result) -> None:
    """Print IMERG validation results."""
    if not result.criteria_used.require_imerg:
        console.print("\n[dim]IMERG precipitation data: Optional (skipped)[/dim]")
        return
    
    status = "✅ PASS" if result.imerg_pass else "❌ FAIL"
    console.print(f"\n[bold]IMERG Precipitation Data: {status}[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Availability", f"{result.imerg_availability_pct:.1f}%")
    table.add_row("Required", f"{result.criteria_used.min_imerg_availability_pct:.1f}%")
    
    if result.imerg_sample_results:
        success_count = sum(1 for r in result.imerg_sample_results if r.data_available)
        table.add_row("Sample Days", f"{success_count}/{len(result.imerg_sample_results)}")
    
    console.print(table)
    
    if result.imerg_issues:
        console.print("\n[red]Issues:[/red]")
        for issue in result.imerg_issues:
            console.print(f"  • {issue}")


def print_summary(result) -> None:
    """Print overall summary."""
    console.print("\n" + "="*60)
    
    if result.overall_pass:
        console.print("[bold green]🎉 FEASIBILITY ASSESSMENT: PASS[/bold green]")
        console.print("\n[green]AirAware can proceed with development using the identified data sources.[/green]")
    else:
        console.print("[bold red]❌ FEASIBILITY ASSESSMENT: FAIL[/bold red]")
        console.print("\n[red]Address the issues below before proceeding with development.[/red]")
    
    console.print(f"\n[bold]Summary:[/bold] {result.summary_message}")
    
    if result.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in result.recommendations:
            console.print(f"  • {rec}")


@app.command()
def check_openaq(
    use_local: bool = typer.Option(False, "--use-local", help="Use only local data"),
    quick_scan: bool = typer.Option(False, "--quick-scan", help="Fast scan using manifest"),
) -> None:
    """Check OpenAQ PM₂.₅ station availability."""
    print_banner()
    
    if not use_local:
        openaq_key, _, _ = check_api_keys(require_cds=False, require_earthdata=False)
        if not openaq_key:
            raise typer.Exit(1)
    else:
        openaq_key = None
        console.print("[yellow]📁 Using local data only[/yellow]")
    
    # Initialize validator
    validator = FeasibilityValidator(
        openaq_api_key=openaq_key,
        use_local=use_local,
    )
    
    # Run OpenAQ validation only
    criteria = AcceptanceCriteria()
    openaq_results = validator.validate_openaq_requirements(criteria)
    
    # Create minimal result for display
    from types import SimpleNamespace
    result = SimpleNamespace()
    result.openaq_pass = openaq_results["pass"]
    result.pm25_stations_found = openaq_results["stations_found"]
    result.pm25_stations_qualified = openaq_results["stations_qualified"]
    result.best_stations = openaq_results["best_stations"]
    result.openaq_issues = openaq_results["issues"]
    result.criteria_used = criteria
    
    print_openaq_results(result)
    
    status = "PASS" if result.openaq_pass else "FAIL"
    console.print(f"\n[bold]RESULT: {status}[/bold]")


@app.command()
def check_era5(
    date: str = typer.Option("2024-12-01", "--date", help="Date to check (YYYY-MM-DD)"),
    use_local: bool = typer.Option(False, "--use-local", help="Use only local data"),
) -> None:
    """Check ERA5 meteorological data availability."""
    print_banner()
    
    if not use_local:
        _, cds_key, _ = check_api_keys(require_openaq=False, require_earthdata=False)
        if not cds_key:
            raise typer.Exit(1)
    else:
        cds_key = None
        console.print("[yellow]📁 Using local data only[/yellow]")
    
    # Parse date
    try:
        check_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        console.print(f"[red]Invalid date format: {date}. Use YYYY-MM-DD[/red]")
        raise typer.Exit(1)
    
    # Initialize ERA5 checker
    from .met.era5_check import ERA5Checker
    checker = ERA5Checker(cds_api_key=cds_key, use_local=use_local)
    
    # Check specific date
    console.print(f"[bold]Checking ERA5 data for {date}...[/bold]")
    result = checker.check_era5_day(check_date)
    
    # Print results
    console.print(f"\n[bold]ERA5 Validation Results:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Check", style="cyan")
    table.add_column("Result", style="white")
    table.add_column("Details", style="yellow")
    
    table.add_row("Variables OK", "✅" if result.ok_vars else "❌", f"{len(result.variables_found)}/4 found")
    table.add_row("Time Steps OK", "✅" if result.ok_time24 else "❌", f"{result.actual_time_steps}/24 hours")
    table.add_row("Overall", "✅" if result.overall_ok else "❌", "")
    
    console.print(table)
    
    if result.validation_errors:
        console.print("\n[red]Errors:[/red]")
        for error in result.validation_errors:
            console.print(f"  • {error}")
    
    console.print(f"\nok_vars={result.ok_vars}")
    console.print(f"ok_time24={result.ok_time24}")


@app.command()
def check_all(
    use_local: bool = typer.Option(False, "--use-local", help="Use only local data"),
    quick_scan: bool = typer.Option(False, "--quick-scan", help="Fast scan using manifest"),
    require_imerg: bool = typer.Option(False, "--require-imerg", help="Require IMERG precipitation data"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Save results to JSON file"),
) -> None:
    """Run comprehensive feasibility assessment."""
    print_banner()
    
    if not use_local:
        openaq_key, cds_key, earthdata_token = check_api_keys(
            require_earthdata=require_imerg
        )
        if not openaq_key or not cds_key:
            raise typer.Exit(1)
    else:
        openaq_key = cds_key = earthdata_token = None
        console.print("[yellow]📁 Using local data only[/yellow]")
    
    # Setup criteria
    criteria = AcceptanceCriteria(require_imerg=require_imerg)
    
    # Initialize validator
    console.print("\n[bold]Initializing feasibility validator...[/bold]")
    validator = FeasibilityValidator(
        openaq_api_key=openaq_key,
        cds_api_key=cds_key,
        earthdata_token=earthdata_token,
        use_local=use_local,
    )
    
    # Run full assessment
    console.print("[bold]Running comprehensive assessment...[/bold]")
    with console.status("[bold green]Validating data sources..."):
        result = validator.run_full_assessment(criteria)
    
    # Print detailed results
    print_openaq_results(result)
    print_era5_results(result)
    print_imerg_results(result)
    print_summary(result)
    
    # Save to JSON if requested
    if output_json:
        output_path = Path(output_json)
        with open(output_path, 'w') as f:
            # Convert to dict for JSON serialization
            result_dict = result.model_dump()
            json.dump(result_dict, f, indent=2, default=str)
        console.print(f"\n[green]Results saved to {output_path}[/green]")


@app.command()
def show_manifest(
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
) -> None:
    """Show local data manifest summary."""
    print_banner()
    
    manifest_path = Path(data_dir) / "artifacts" / "local_data_manifest.json"
    
    if not manifest_path.exists():
        console.print(f"[red]Manifest not found at {manifest_path}[/red]")
        console.print("[yellow]Run data scanner first: python -m airaware.utils.data_scanner[/yellow]")
        raise typer.Exit(1)
    
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        console.print(f"[bold]Local Data Manifest: {manifest_path}[/bold]")
        
        # Overall stats
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Total Files", str(manifest.get("total_files", 0)))
        table.add_row("Total Size", f"{manifest.get('total_size_mb', 0):.1f} MB")
        table.add_row("Last Scan", manifest.get("scan_timestamp", "Unknown"))
        table.add_row("Scan Mode", "Quick" if manifest.get("quick_mode") else "Full")
        
        console.print(table)
        
        # By dataset type
        summary = manifest.get("summary", {})
        by_type = summary.get("by_type", {})
        
        if by_type:
            console.print("\n[bold]By Dataset Type:[/bold]")
            type_table = Table(show_header=True, header_style="bold magenta")
            type_table.add_column("Dataset Type", style="cyan")
            type_table.add_column("Files", style="white")
            type_table.add_column("Size (MB)", style="yellow")
            
            for dataset_type, stats in by_type.items():
                type_table.add_row(
                    dataset_type,
                    str(stats.get("files", 0)),
                    f"{stats.get('size_mb', 0):.1f}"
                )
            
            console.print(type_table)
        
        # Enhanced file listing for OpenAQ data
        openaq_files = [
            f for f in manifest.get("files", [])
            if ("openaq_targets" in f.get("dataset_types", []) or
                "nepal" in f.get("path", "").lower())
        ]
        
        if openaq_files:
            console.print("\n[bold]OpenAQ Data Files:[/bold]")
            file_table = Table(show_header=True, header_style="bold magenta")
            file_table.add_column("File", style="cyan")
            file_table.add_column("Size (MB)", style="white")
            file_table.add_column("Type", style="yellow")
            
            for file_info in openaq_files[:10]:  # Show first 10
                file_path = file_info.get("path", "")
                file_size = file_info.get("size_mb", 0)
                
                # Determine file type
                if "sensor" in file_path.lower():
                    file_type = "Sensors"
                elif "location" in file_path.lower():
                    file_type = "Locations"
                elif "measurement" in file_path.lower():
                    file_type = "Measurements"
                else:
                    file_type = "Unknown"
                
                file_table.add_row(
                    Path(file_path).name,
                    f"{file_size:.2f}",
                    file_type
                )
            
            console.print(file_table)
            
            if len(openaq_files) > 10:
                console.print(f"[dim]... and {len(openaq_files) - 10} more files[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error reading manifest: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def test_local_data(
    data_dir: str = typer.Option("data", "--data-dir", help="Data directory"),
) -> None:
    """Test and validate local data with enhanced analysis."""
    print_banner()
    
    console.print("[bold]🧪 Testing Local Data Availability...[/bold]")
    
    # Test OpenAQ client with local data
    from ..openaq_client import OpenAQClient
    
    try:
        client = OpenAQClient(use_local=True)
        summary = client._analyze_local_data()
        
        console.print(f"\n[bold]Local OpenAQ Analysis:[/bold]")
        
        if summary.stations_found:
            station_table = Table(show_header=True, header_style="bold magenta")
            station_table.add_column("Station ID", style="cyan")
            station_table.add_column("Measurements", style="white")
            station_table.add_column("Quality Score", style="green")
            
            for station_id in summary.stations_found:
                measurements = summary.total_measurements.get(station_id, 0)
                quality = summary.data_quality.get(station_id, 0.0)
                
                station_table.add_row(
                    str(station_id),
                    f"{measurements:,}",
                    f"{quality:.2f}"
                )
            
            console.print(station_table)
            
            # Test distance calculation
            test_stations = client._get_local_stations(27.7172, 85.3240, 25.0)
            console.print(f"\n[bold]Stations within 25km of Kathmandu:[/bold] {len(test_stations)}")
            
            if test_stations:
                console.print("✅ Local data validation successful!")
                console.print(f"📊 Ready for feasibility assessment with {len(test_stations)} stations")
            else:
                console.print("⚠️  No stations found within radius")
        else:
            console.print("[yellow]No local OpenAQ stations detected[/yellow]")
            console.print("💡 Run full data ingestion to collect station data")
            
    except Exception as e:
        console.print(f"[red]Error testing local data: {e}[/red]")
        raise typer.Exit(1)


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
