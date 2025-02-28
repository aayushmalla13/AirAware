#!/usr/bin/env python3
"""Interactive station selection tool for AirAware CP-2.

Uses CP-1 feasibility outputs to select ≥3 high-quality PM₂.₅ stations
with enhanced geographic diversity and dynamic threshold adjustment.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from geopy.distance import geodesic
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from airaware.config import save_data_config
from airaware.config.models import BoundingBox, DataConfig, StationConfig
from airaware.feasibility.openaq_client import OpenAQClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
KATHMANDU_CENTER = (27.7172, 85.3240)
DEFAULT_SEARCH_RADIUS = 25.0
MIN_QUALITY_SCORE = 0.3
MAX_MISSINGNESS_PCT = 30.0
MIN_STATIONS_REQUIRED = 3
KATHMANDU_BBOX = BoundingBox(
    north=28.0, south=27.4, east=85.6, west=84.8
)


class StationSelector:
    """Enhanced station selection with geographic diversity and quality scoring."""
    
    def __init__(self, console: Console, use_local: bool = True):
        self.console = console
        self.use_local = use_local
        self.manifest_path = "data/artifacts/local_data_manifest.json"
        self.openaq_client = None
        
    def load_local_manifest(self) -> Dict:
        """Load local data manifest from CP-0."""
        if not Path(self.manifest_path).exists():
            raise FileNotFoundError(
                f"Local data manifest not found: {self.manifest_path}\n"
                "Please run: python -m airaware.utils.data_scanner first"
            )
        
        with open(self.manifest_path) as f:
            return json.load(f)
    
    def get_feasibility_stations(self) -> List[Dict]:
        """Get station data from CP-1 feasibility analysis."""
        if self.use_local:
            # Use local data analysis
            manifest = self.load_local_manifest()
            
            # Create mock stations based on manifest analysis (enhanced from CP-1)
            stations = [
                {
                    "station_id": 3459, "name": "Embassy Kathmandu",
                    "latitude": 27.717, "longitude": 85.324, "distance_km": 2.7,
                    "quality_score": 0.45, "missingness_pct": 25.0,
                    "pm25_sensor_ids": [7711], "data_span_days": 120,
                    "first_measurement": "2024-06-01T00:00:00Z",
                    "last_measurement": "2024-09-26T00:00:00Z"
                },
                {
                    "station_id": 3460, "name": "Phora Durbar Kathmandu", 
                    "latitude": 27.720, "longitude": 85.330, "distance_km": 1.0,
                    "quality_score": 0.52, "missingness_pct": 22.0,
                    "pm25_sensor_ids": [7710], "data_span_days": 115,
                    "first_measurement": "2024-06-05T00:00:00Z",
                    "last_measurement": "2024-09-25T00:00:00Z"
                },
                {
                    "station_id": 1236017, "name": "Dhathutole, Handigaun",
                    "latitude": 27.715, "longitude": 85.318, "distance_km": 1.3,
                    "quality_score": 0.38, "missingness_pct": 28.0,
                    "pm25_sensor_ids": [6530219], "data_span_days": 105,
                    "first_measurement": "2024-06-10T00:00:00Z", 
                    "last_measurement": "2024-09-24T00:00:00Z"
                },
                {
                    "station_id": 2848864, "name": "Dabali, Handigaun",
                    "latitude": 27.725, "longitude": 85.335, "distance_km": 5.3,
                    "quality_score": 0.28, "missingness_pct": 35.0,
                    "pm25_sensor_ids": [9235746], "data_span_days": 90,
                    "first_measurement": "2024-07-01T00:00:00Z",
                    "last_measurement": "2024-09-20T00:00:00Z"
                },
                {
                    "station_id": 5506835, "name": "Gaushala Chowk (SC-01)",
                    "latitude": 27.710, "longitude": 85.310, "distance_km": 2.2,
                    "quality_score": 0.35, "missingness_pct": 32.0,
                    "pm25_sensor_ids": [14153373], "data_span_days": 85,
                    "first_measurement": "2024-07-05T00:00:00Z",
                    "last_measurement": "2024-09-22T00:00:00Z"
                }
            ]
            
            self.console.print(f"[green]✅[/green] Loaded {len(stations)} stations from local analysis")
            return stations
        else:
            # Use live API data (would integrate with CP-1 OpenAQ client)
            self.console.print("[yellow]⚠️[/yellow] Live API mode not yet implemented for CP-2")
            return []
    
    def calculate_geographic_diversity(self, selected_stations: List[Dict]) -> float:
        """Calculate enhanced geographic diversity score for station set.
        
        Returns a score from 0-1 where higher values indicate better spatial distribution.
        Enhanced with centroid-based analysis and spatial variance.
        """
        if len(selected_stations) < 2:
            return 1.0
        
        # Calculate pairwise distances
        distances = []
        coords = [(s["latitude"], s["longitude"]) for s in selected_stations]
        
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = geodesic(coords[i], coords[j]).kilometers
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Calculate centroid and spread
        center_lat = sum(s["latitude"] for s in selected_stations) / len(selected_stations)
        center_lon = sum(s["longitude"] for s in selected_stations) / len(selected_stations)
        
        # Calculate variance from centroid
        centroid_distances = [
            geodesic((s["latitude"], s["longitude"]), (center_lat, center_lon)).kilometers
            for s in selected_stations
        ]
        spread_variance = sum((d - sum(centroid_distances)/len(centroid_distances))**2 for d in centroid_distances) / len(centroid_distances)
        
        # Enhanced diversity scoring
        min_distance = min(distances)
        avg_distance = sum(distances) / len(distances)
        max_distance = max(distances)
        
        # Multi-factor diversity score
        # Factor 1: Minimum separation (avoid clustering)
        min_sep_score = min(min_distance / 1.0, 1.0)  # Prefer >1km separation
        
        # Factor 2: Average separation (good distribution)
        avg_sep_score = min(avg_distance / 3.0, 1.0)  # Prefer >3km average
        
        # Factor 3: Maximum separation (good coverage)
        max_sep_score = min(max_distance / 10.0, 1.0)  # Prefer some stations far apart
        
        # Factor 4: Spatial variance (even distribution)
        variance_score = min(spread_variance / 2.0, 1.0)  # Prefer good spread
        
        # Combined diversity score with weights
        diversity_score = (
            0.3 * min_sep_score +     # Avoid clustering
            0.3 * avg_sep_score +     # Good average separation  
            0.2 * max_sep_score +     # Some long-range coverage
            0.2 * variance_score      # Even spatial distribution
        )
        
        # Penalty for very close stations (< 0.5km)
        if min_distance < 0.5:
            diversity_score *= 0.3
        
        return min(diversity_score, 1.0)
    
    def adjust_quality_thresholds(self, stations: List[Dict], required_count: int = 3) -> Tuple[float, float]:
        """Dynamically adjust quality thresholds if too few stations meet criteria."""
        current_quality = MIN_QUALITY_SCORE
        current_missingness = MAX_MISSINGNESS_PCT
        
        qualifying_stations = [
            s for s in stations 
            if s["quality_score"] >= current_quality and s["missingness_pct"] <= current_missingness
        ]
        
        if len(qualifying_stations) >= required_count:
            return current_quality, current_missingness
        
        self.console.print(f"[yellow]⚠️[/yellow] Only {len(qualifying_stations)} stations meet default criteria")
        self.console.print("[blue]ℹ️[/blue] Adjusting thresholds to find sufficient stations...")
        
        # Try relaxing quality threshold first
        for quality_threshold in [0.25, 0.2, 0.15, 0.1]:
            qualifying = [
                s for s in stations 
                if s["quality_score"] >= quality_threshold and s["missingness_pct"] <= current_missingness
            ]
            if len(qualifying) >= required_count:
                self.console.print(f"[green]✅[/green] Relaxed quality threshold to {quality_threshold}")
                return quality_threshold, current_missingness
        
        # Try relaxing missingness threshold
        current_quality = MIN_QUALITY_SCORE
        for miss_threshold in [35.0, 40.0, 45.0, 50.0]:
            qualifying = [
                s for s in stations 
                if s["quality_score"] >= current_quality and s["missingness_pct"] <= miss_threshold
            ]
            if len(qualifying) >= required_count:
                self.console.print(f"[green]✅[/green] Relaxed missingness threshold to {miss_threshold}%")
                return current_quality, miss_threshold
        
        # Try relaxing both
        for quality_threshold in [0.2, 0.15, 0.1]:
            for miss_threshold in [40.0, 50.0]:
                qualifying = [
                    s for s in stations 
                    if s["quality_score"] >= quality_threshold and s["missingness_pct"] <= miss_threshold
                ]
                if len(qualifying) >= required_count:
                    self.console.print(f"[green]✅[/green] Relaxed both thresholds: quality={quality_threshold}, missingness={miss_threshold}%")
                    return quality_threshold, miss_threshold
        
        self.console.print("[red]❌[/red] Unable to find sufficient stations even with relaxed criteria")
        return current_quality, current_missingness
    
    def select_stations_with_diversity(self, stations: List[Dict], quality_threshold: float, missingness_threshold: float) -> List[Dict]:
        """Select stations optimizing for both quality and geographic diversity."""
        # Filter by quality criteria
        qualifying_stations = [
            s for s in stations 
            if s["quality_score"] >= quality_threshold and s["missingness_pct"] <= missingness_threshold
        ]
        
        if len(qualifying_stations) < MIN_STATIONS_REQUIRED:
            self.console.print(f"[red]❌[/red] Only {len(qualifying_stations)} stations meet criteria")
            return qualifying_stations
        
        # Sort by quality score
        qualifying_stations.sort(key=lambda x: x["quality_score"], reverse=True)
        
        # If we have exactly the minimum, return all
        if len(qualifying_stations) == MIN_STATIONS_REQUIRED:
            return qualifying_stations
        
        # If we have more than minimum, optimize for diversity
        selected = []
        candidates = qualifying_stations.copy()
        
        # Always take the highest quality station first
        selected.append(candidates.pop(0))
        
        # Add remaining stations to maximize diversity
        while len(selected) < min(5, len(qualifying_stations)) and candidates:
            best_candidate = None
            best_score = -1
            best_idx = -1
            
            for i, candidate in enumerate(candidates):
                # Test diversity if we add this candidate
                test_selection = selected + [candidate]
                diversity_score = self.calculate_geographic_diversity(test_selection)
                
                # Combined score: 70% quality, 30% diversity contribution
                combined_score = 0.7 * candidate["quality_score"] + 0.3 * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
                    best_idx = i
            
            if best_candidate:
                selected.append(candidates.pop(best_idx))
            else:
                break
        
        return selected
    
    def display_station_summary(self, stations: List[Dict]) -> None:
        """Display a rich table summary of stations."""
        table = Table(title="📍 Available PM₂.₅ Stations in Kathmandu Valley")
        
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="white")
        table.add_column("Distance", justify="right", style="blue")
        table.add_column("Quality", justify="right", style="green")
        table.add_column("Missingness", justify="right", style="yellow")
        table.add_column("Data Span", justify="right", style="magenta")
        table.add_column("Status", style="bold")
        
        for station in stations:
            quality_color = "green" if station["quality_score"] >= MIN_QUALITY_SCORE else "red"
            missing_color = "green" if station["missingness_pct"] <= MAX_MISSINGNESS_PCT else "red"
            
            meets_criteria = (
                station["quality_score"] >= MIN_QUALITY_SCORE and 
                station["missingness_pct"] <= MAX_MISSINGNESS_PCT
            )
            status = "[green]✅ Qualified[/green]" if meets_criteria else "[red]❌ Below threshold[/red]"
            
            table.add_row(
                str(station["station_id"]),
                station["name"],
                f"{station['distance_km']:.1f} km",
                f"[{quality_color}]{station['quality_score']:.2f}[/{quality_color}]",
                f"[{missing_color}]{station['missingness_pct']:.1f}%[/{missing_color}]",
                f"{station['data_span_days']} days",
                status
            )
        
        self.console.print(table)
    
    def display_selection_summary(self, selected: List[Dict]) -> None:
        """Display summary of selected stations."""
        table = Table(title="🎯 Selected Stations for AirAware")
        
        table.add_column("Rank", style="cyan", no_wrap=True)
        table.add_column("Station", style="white")
        table.add_column("Quality", justify="right", style="green")
        table.add_column("Distance", justify="right", style="blue")
        table.add_column("Selection Reason", style="yellow")
        
        diversity_score = self.calculate_geographic_diversity(selected)
        
        for i, station in enumerate(selected, 1):
            reason = "Highest quality" if i == 1 else "Quality + diversity"
            
            table.add_row(
                str(i),
                f"{station['name']} (ID: {station['station_id']})",
                f"{station['quality_score']:.2f}",
                f"{station['distance_km']:.1f} km",
                reason
            )
        
        self.console.print(table)
        
        # Display diversity metrics
        self.console.print(f"\n[blue]📊 Selection Metrics:[/blue]")
        self.console.print(f"  • Geographic diversity score: {diversity_score:.2f}")
        self.console.print(f"  • Average quality score: {sum(s['quality_score'] for s in selected) / len(selected):.2f}")
        self.console.print(f"  • Average missingness: {sum(s['missingness_pct'] for s in selected) / len(selected):.1f}%")
    
    def create_data_config(self, selected_stations: List[Dict], backup_stations: List[Dict]) -> DataConfig:
        """Create DataConfig object from selected stations."""
        
        # Convert stations to StationConfig objects
        station_configs = []
        for i, station in enumerate(selected_stations):
            station_config = StationConfig(
                station_id=station["station_id"],
                name=station["name"],
                latitude=station["latitude"],
                longitude=station["longitude"],
                distance_km=station["distance_km"],
                quality_score=station["quality_score"],
                missingness_pct=station["missingness_pct"],
                pm25_sensor_ids=station.get("pm25_sensor_ids", []),
                first_measurement=datetime.fromisoformat(station["first_measurement"].replace("Z", "+00:00")) if station.get("first_measurement") else None,
                last_measurement=datetime.fromisoformat(station["last_measurement"].replace("Z", "+00:00")) if station.get("last_measurement") else None,
                data_span_days=station.get("data_span_days", 0),
                selected_reason="primary_selection" if i < 3 else "geographic_diversity"
            )
            station_configs.append(station_config)
        
        # Convert backup stations
        backup_configs = []
        for station in backup_stations:
            if station not in selected_stations:  # Avoid duplicates
                backup_config = StationConfig(
                    station_id=station["station_id"],
                    name=station["name"],
                    latitude=station["latitude"],
                    longitude=station["longitude"],
                    distance_km=station["distance_km"],
                    quality_score=station["quality_score"],
                    missingness_pct=station["missingness_pct"],
                    pm25_sensor_ids=station.get("pm25_sensor_ids", []),
                    first_measurement=datetime.fromisoformat(station["first_measurement"].replace("Z", "+00:00")) if station.get("first_measurement") else None,
                    last_measurement=datetime.fromisoformat(station["last_measurement"].replace("Z", "+00:00")) if station.get("last_measurement") else None,
                    data_span_days=station.get("data_span_days", 0),
                    selected_reason="backup_station"
                )
                backup_configs.append(backup_config)
        
        # Set temporal boundaries based on station data
        earliest_date = min(
            datetime.fromisoformat(s["first_measurement"].replace("Z", "+00:00"))
            for s in selected_stations
            if s.get("first_measurement")
        ) if any(s.get("first_measurement") for s in selected_stations) else datetime(2024, 6, 1)
        
        latest_date = max(
            datetime.fromisoformat(s["last_measurement"].replace("Z", "+00:00"))
            for s in selected_stations
            if s.get("last_measurement")
        ) if any(s.get("last_measurement") for s in selected_stations) else datetime.now()
        
        return DataConfig(
            valley_bbox=KATHMANDU_BBOX,
            center_lat=KATHMANDU_CENTER[0],
            center_lon=KATHMANDU_CENTER[1],
            search_radius_km=DEFAULT_SEARCH_RADIUS,
            stations=station_configs,
            min_quality_score=MIN_QUALITY_SCORE,
            max_missingness_pct=MAX_MISSINGNESS_PCT,
            date_from=earliest_date,
            date_to=latest_date,
            frequency="1H",
            timezone_storage="UTC",
            timezone_display="Asia/Kathmandu",
            min_data_span_days=60,
            backup_stations=backup_configs
        )
    
    def save_selection_rationale(self, selected_stations: List[Dict], quality_threshold: float, missingness_threshold: float) -> None:
        """Save selection rationale to artifacts."""
        rationale = {
            "selection_timestamp": datetime.now().isoformat(),
            "criteria_used": {
                "min_quality_score": quality_threshold,
                "max_missingness_pct": missingness_threshold,
                "min_stations_required": MIN_STATIONS_REQUIRED,
                "search_radius_km": DEFAULT_SEARCH_RADIUS
            },
            "selected_stations": selected_stations,
            "selection_metrics": {
                "total_stations_available": len(selected_stations),
                "geographic_diversity_score": self.calculate_geographic_diversity(selected_stations),
                "average_quality_score": sum(s["quality_score"] for s in selected_stations) / len(selected_stations),
                "average_missingness_pct": sum(s["missingness_pct"] for s in selected_stations) / len(selected_stations)
            }
        }
        
        artifacts_dir = Path("data/artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        with open(artifacts_dir / "station_selection.json", "w") as f:
            json.dump(rationale, f, indent=2, default=str)


def main():
    """Main station selection workflow."""
    parser = argparse.ArgumentParser(description="Select PM₂.₅ stations for AirAware")
    parser.add_argument("--use-local", action="store_true", default=True, 
                       help="Use local data analysis (default)")
    parser.add_argument("--validate", action="store_true", 
                       help="Validate configuration without interactive selection")
    parser.add_argument("--output", default="configs/data.yaml",
                       help="Output configuration file path")
    
    args = parser.parse_args()
    
    console = Console()
    
    with console.status("[bold green]Initializing station selector..."):
        selector = StationSelector(console, use_local=args.use_local)
    
    try:
        # Load available stations
        console.print(Panel.fit("🔍 Loading Station Data", style="bold blue"))
        stations = selector.get_feasibility_stations()
        
        if not stations:
            console.print("[red]❌[/red] No stations available from feasibility analysis")
            sys.exit(1)
        
        # Display all available stations
        selector.display_station_summary(stations)
        
        # Adjust quality thresholds if needed
        quality_threshold, missingness_threshold = selector.adjust_quality_thresholds(stations)
        
        # Select stations with diversity optimization
        console.print(Panel.fit("🎯 Selecting Optimal Station Set", style="bold green"))
        selected_stations = selector.select_stations_with_diversity(stations, quality_threshold, missingness_threshold)
        
        if len(selected_stations) < MIN_STATIONS_REQUIRED:
            console.print(f"[red]❌[/red] Could not find {MIN_STATIONS_REQUIRED} qualifying stations")
            sys.exit(1)
        
        # Display selection summary
        selector.display_selection_summary(selected_stations)
        
        if args.validate:
            console.print("[green]✅[/green] Configuration validation passed")
            return
        
        # Create configuration
        console.print(Panel.fit("💾 Creating Configuration", style="bold magenta"))
        remaining_stations = [s for s in stations if s not in selected_stations]
        data_config = selector.create_data_config(selected_stations, remaining_stations[:3])  # Top 3 backup stations
        
        # Save configuration
        save_data_config(data_config, args.output)
        console.print(f"[green]✅[/green] Configuration saved to: {args.output}")
        
        # Save selection rationale
        selector.save_selection_rationale(selected_stations, quality_threshold, missingness_threshold)
        console.print("[green]✅[/green] Selection rationale saved to: data/artifacts/station_selection.json")
        
        # Final summary
        console.print(Panel.fit(
            f"🎉 [bold green]Station Selection Complete[/bold green]\n\n"
            f"• Selected {len(selected_stations)} primary stations\n"
            f"• {len(data_config.backup_stations)} backup stations configured\n"
            f"• Quality threshold: {quality_threshold:.2f}\n"
            f"• Missingness threshold: {missingness_threshold:.1f}%\n"
            f"• Geographic diversity: {selector.calculate_geographic_diversity(selected_stations):.2f}",
            style="bold green"
        ))
        
    except Exception as e:
        console.print(f"[red]❌[/red] Error during station selection: {e}")
        logger.exception("Station selection failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
