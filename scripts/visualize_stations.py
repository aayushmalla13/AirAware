#!/usr/bin/env python3
"""Visualization tool for station selection and temporal coverage.

Enhanced CP-2 auto-enhancement: adds temporal coverage visualization 
and geographic diversity analysis.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from airaware.config import load_data_config


def plot_temporal_coverage(stations, output_dir="data/artifacts"):
    """Create temporal coverage visualization for selected stations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, station in enumerate(stations[:5]):  # Limit to 5 stations for clarity
        if station.first_measurement and station.last_measurement:
            start_date = station.first_measurement
            end_date = station.last_measurement
            
            # Create date range
            dates = pd.date_range(start_date, end_date, freq='D')
            
            # Plot as horizontal bar
            ax.barh(i, (end_date - start_date).days, 
                   left=start_date, height=0.6, 
                   color=colors[i % len(colors)], alpha=0.7,
                   label=f"{station.name} (Q={station.quality_score:.2f})")
    
    # Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('Station')
    ax.set_title('Temporal Data Coverage by Station')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set y-axis labels
    station_names = [s.name[:20] + '...' if len(s.name) > 20 else s.name 
                     for s in stations[:5]]
    ax.set_yticks(range(len(station_names)))
    ax.set_yticklabels(station_names)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "station_temporal_coverage.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return output_path


def plot_geographic_diversity(stations, output_dir="data/artifacts"):
    """Create geographic distribution plot for selected stations."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract coordinates
    lats = [s.latitude for s in stations]
    lons = [s.longitude for s in stations]
    qualities = [s.quality_score for s in stations]
    
    # Create scatter plot with quality score as color and size
    scatter = ax.scatter(lons, lats, c=qualities, s=[q*200 for q in qualities], 
                        alpha=0.7, cmap='viridis')
    
    # Add station labels
    for station in stations:
        ax.annotate(f"{station.name}\n(Q={station.quality_score:.2f})", 
                   (station.longitude, station.latitude),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, ha='left')
    
    # Add Kathmandu center
    center_lat, center_lon = 27.7172, 85.3240
    ax.scatter(center_lon, center_lat, c='red', s=100, marker='x', 
              label='Kathmandu Center')
    
    # Formatting
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude') 
    ax.set_title('Geographic Distribution of Selected Stations')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Quality Score')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "station_geographic_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return output_path


def analyze_coverage_gaps(stations):
    """Analyze temporal gaps in data coverage."""
    console = Console()
    
    console.print("\n[bold blue]📊 Temporal Coverage Analysis[/bold blue]")
    
    for station in stations:
        if station.first_measurement and station.last_measurement:
            span_days = (station.last_measurement - station.first_measurement).days
            expected_measurements = span_days * 24  # Hourly data
            actual_measurements = station.data_span_days * 24  # Approximate
            
            coverage_rate = (actual_measurements / expected_measurements) * 100 if expected_measurements > 0 else 0
            
            console.print(f"\n[cyan]{station.name}[/cyan]:")
            console.print(f"  • Data span: {span_days} days")
            console.print(f"  • Expected measurements: {expected_measurements:,}")
            console.print(f"  • Estimated actual: {actual_measurements:,}")
            console.print(f"  • Coverage rate: {coverage_rate:.1f}%")
            console.print(f"  • Quality score: {station.quality_score:.2f}")


def main():
    """Main visualization workflow."""
    parser = argparse.ArgumentParser(description="Visualize station selection and coverage")
    parser.add_argument("--output-dir", default="data/artifacts",
                       help="Output directory for plots")
    parser.add_argument("--show-plots", action="store_true",
                       help="Display plots interactively")
    
    args = parser.parse_args()
    
    console = Console()
    
    try:
        # Load configuration
        console.print("[bold green]📈 Loading Station Configuration...[/bold green]")
        config = load_data_config()
        
        if not config.stations:
            console.print("[red]❌[/red] No stations found in configuration")
            return
        
        # Generate visualizations
        console.print(f"[blue]🎨[/blue] Creating visualizations for {len(config.stations)} stations...")
        
        # Plot temporal coverage
        temporal_plot = plot_temporal_coverage(config.stations, args.output_dir)
        console.print(f"[green]✅[/green] Temporal coverage plot saved: {temporal_plot}")
        
        # Plot geographic distribution  
        geo_plot = plot_geographic_diversity(config.stations, args.output_dir)
        console.print(f"[green]✅[/green] Geographic distribution plot saved: {geo_plot}")
        
        # Analyze coverage gaps
        analyze_coverage_gaps(config.stations)
        
        # Display plots if requested
        if args.show_plots:
            plt.show()
        
        console.print(f"\n[bold green]🎉 Visualization complete![/bold green]")
        console.print(f"Plots saved to: {args.output_dir}")
        
    except Exception as e:
        console.print(f"[red]❌[/red] Error during visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


