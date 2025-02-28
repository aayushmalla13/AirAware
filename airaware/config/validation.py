"""Advanced configuration validation and health checks for AirAware."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from geopy.distance import geodesic
from pydantic import BaseModel

from .models import DataConfig, StationConfig

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Result of a configuration validation check."""
    check_name: str
    status: str  # "pass", "warning", "fail"
    message: str
    details: Dict = {}
    recommendation: str = ""


class ConfigValidator:
    """Advanced configuration validation with health checks."""
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
    
    def validate_configuration(self, config: DataConfig) -> List[ValidationResult]:
        """Run comprehensive configuration validation."""
        self.validation_results = []
        
        # Run all validation checks
        self._check_station_quality_distribution(config)
        self._check_geographic_coverage(config)
        self._check_temporal_coverage(config)
        self._check_redundancy_and_backup(config)
        self._check_data_freshness(config)
        self._check_station_clustering(config)
        self._check_quality_thresholds(config)
        
        return self.validation_results
    
    def _add_result(self, check_name: str, status: str, message: str, 
                    details: Dict = None, recommendation: str = ""):
        """Add a validation result."""
        result = ValidationResult(
            check_name=check_name,
            status=status,
            message=message,
            details=details or {},
            recommendation=recommendation
        )
        self.validation_results.append(result)
    
    def _check_station_quality_distribution(self, config: DataConfig):
        """Check if station quality scores are well distributed."""
        qualities = [s.quality_score for s in config.stations]
        
        if not qualities:
            self._add_result(
                "quality_distribution", "fail",
                "No stations found",
                recommendation="Add stations to configuration"
            )
            return
        
        avg_quality = sum(qualities) / len(qualities)
        min_quality = min(qualities)
        max_quality = max(qualities)
        quality_range = max_quality - min_quality
        
        # Check for good quality distribution
        if quality_range < 0.1:
            self._add_result(
                "quality_distribution", "warning",
                f"Low quality score range ({quality_range:.3f})",
                {"avg_quality": avg_quality, "range": quality_range},
                "Consider including stations with more diverse quality scores"
            )
        elif avg_quality < 0.4:
            self._add_result(
                "quality_distribution", "warning", 
                f"Low average quality score ({avg_quality:.3f})",
                {"avg_quality": avg_quality},
                "Consider raising quality thresholds or finding better stations"
            )
        else:
            self._add_result(
                "quality_distribution", "pass",
                f"Good quality distribution (avg={avg_quality:.3f}, range={quality_range:.3f})",
                {"avg_quality": avg_quality, "range": quality_range}
            )
    
    def _check_geographic_coverage(self, config: DataConfig):
        """Check geographic coverage and distribution."""
        if len(config.stations) < 3:
            self._add_result(
                "geographic_coverage", "fail",
                f"Insufficient stations ({len(config.stations)}) for good coverage",
                recommendation="Add more stations to improve geographic coverage"
            )
            return
        
        # Calculate coverage area using convex hull approximation
        lats = [s.latitude for s in config.stations]
        lons = [s.longitude for s in config.stations]
        
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        
        # Calculate distances from valley center
        center = (config.center_lat, config.center_lon)
        distances = [
            geodesic(center, (s.latitude, s.longitude)).kilometers
            for s in config.stations
        ]
        
        max_distance = max(distances)
        coverage_efficiency = (lat_range * lon_range) / (config.search_radius_km ** 2) * 100
        
        if max_distance < config.search_radius_km * 0.2:  # More lenient threshold
            self._add_result(
                "geographic_coverage", "warning",
                f"Stations clustered near center (max distance: {max_distance:.1f}km)",
                {"max_distance": max_distance, "coverage_efficiency": coverage_efficiency},
                "Consider adding stations farther from center for better coverage"
            )
        elif coverage_efficiency < 20:
            self._add_result(
                "geographic_coverage", "warning",
                f"Low geographic coverage efficiency ({coverage_efficiency:.1f}%)",
                {"coverage_efficiency": coverage_efficiency},
                "Stations may be too clustered - consider more distributed selection"
            )
        else:
            self._add_result(
                "geographic_coverage", "pass",
                f"Good geographic coverage (efficiency: {coverage_efficiency:.1f}%)",
                {"coverage_efficiency": coverage_efficiency, "max_distance": max_distance}
            )
    
    def _check_temporal_coverage(self, config: DataConfig):
        """Check temporal data coverage across stations."""
        now = datetime.now()
        recent_threshold = now - timedelta(days=7)
        old_threshold = now - timedelta(days=365)
        
        recent_stations = 0
        old_stations = 0
        total_span_days = 0
        
        for station in config.stations:
            if station.last_measurement and station.last_measurement.replace(tzinfo=None) > recent_threshold:
                recent_stations += 1
            if station.first_measurement and station.first_measurement.replace(tzinfo=None) < old_threshold:
                old_stations += 1
            total_span_days += station.data_span_days
        
        avg_span_days = total_span_days / len(config.stations) if config.stations else 0
        
        if recent_stations == 0:
            self._add_result(
                "temporal_coverage", "fail",
                "No stations with recent data (within 7 days)",
                {"recent_stations": recent_stations, "avg_span_days": avg_span_days},
                "Check station data freshness and connectivity"
            )
        elif recent_stations < len(config.stations) * 0.7:
            self._add_result(
                "temporal_coverage", "warning",
                f"Only {recent_stations}/{len(config.stations)} stations have recent data",
                {"recent_stations": recent_stations, "avg_span_days": avg_span_days},
                "Some stations may be offline or delayed"
            )
        elif avg_span_days < 60:
            self._add_result(
                "temporal_coverage", "warning",
                f"Short average data span ({avg_span_days:.0f} days)",
                {"avg_span_days": avg_span_days},
                "Stations may not have sufficient historical data for modeling"
            )
        else:
            self._add_result(
                "temporal_coverage", "pass",
                f"Good temporal coverage ({recent_stations} recent, {avg_span_days:.0f} day avg span)",
                {"recent_stations": recent_stations, "avg_span_days": avg_span_days}
            )
    
    def _check_redundancy_and_backup(self, config: DataConfig):
        """Check station redundancy and backup configuration."""
        primary_count = len(config.stations)
        backup_count = len(config.backup_stations)
        
        if backup_count == 0:
            self._add_result(
                "redundancy_backup", "warning",
                "No backup stations configured",
                {"primary_count": primary_count, "backup_count": backup_count},
                "Add backup stations for system resilience"
            )
        elif backup_count < primary_count * 0.5:
            self._add_result(
                "redundancy_backup", "warning",
                f"Low backup ratio ({backup_count}/{primary_count})",
                {"primary_count": primary_count, "backup_count": backup_count},
                "Consider adding more backup stations"
            )
        else:
            self._add_result(
                "redundancy_backup", "pass",
                f"Good redundancy ({backup_count} backup stations for {primary_count} primary)",
                {"primary_count": primary_count, "backup_count": backup_count}
            )
    
    def _check_data_freshness(self, config: DataConfig):
        """Check data freshness across all stations."""
        now = datetime.now()
        stale_threshold = now - timedelta(hours=6)  # Data older than 6 hours is stale
        
        stale_stations = []
        
        for station in config.stations:
            if station.last_measurement:
                last_measurement_naive = station.last_measurement.replace(tzinfo=None)
                if last_measurement_naive < stale_threshold:
                    hours_old = (now - last_measurement_naive).total_seconds() / 3600
                    stale_stations.append((station.name, hours_old))
        
        if len(stale_stations) == len(config.stations):
            self._add_result(
                "data_freshness", "fail",
                "All stations have stale data",
                {"stale_stations": len(stale_stations)},
                "Check data pipeline and station connectivity"
            )
        elif len(stale_stations) > 0:
            self._add_result(
                "data_freshness", "warning",
                f"{len(stale_stations)} stations have stale data",
                {"stale_stations": stale_stations},
                "Monitor data pipeline for delays"
            )
        else:
            self._add_result(
                "data_freshness", "pass",
                "All stations have fresh data",
                {"fresh_stations": len(config.stations)}
            )
    
    def _check_station_clustering(self, config: DataConfig):
        """Check for excessive station clustering."""
        if len(config.stations) < 2:
            return
        
        min_distances = []
        for i, station1 in enumerate(config.stations):
            distances_to_others = []
            for j, station2 in enumerate(config.stations):
                if i != j:
                    dist = geodesic(
                        (station1.latitude, station1.longitude),
                        (station2.latitude, station2.longitude)
                    ).kilometers
                    distances_to_others.append(dist)
            
            if distances_to_others:
                min_distances.append(min(distances_to_others))
        
        avg_min_distance = sum(min_distances) / len(min_distances)
        clustered_stations = [d for d in min_distances if d < 0.5]  # Very close stations
        
        if len(clustered_stations) > len(config.stations) * 0.5:
            self._add_result(
                "station_clustering", "warning",
                f"Many stations are very close together (avg min distance: {avg_min_distance:.2f}km)",
                {"avg_min_distance": avg_min_distance, "clustered_count": len(clustered_stations)},
                "Consider spreading stations more geographically"
            )
        elif avg_min_distance < 1.0:
            self._add_result(
                "station_clustering", "warning",
                f"Stations may be too clustered (avg min distance: {avg_min_distance:.2f}km)",
                {"avg_min_distance": avg_min_distance},
                "Verify geographic diversity meets requirements"
            )
        else:
            self._add_result(
                "station_clustering", "pass",
                f"Good station distribution (avg min distance: {avg_min_distance:.2f}km)",
                {"avg_min_distance": avg_min_distance}
            )
    
    def _check_quality_thresholds(self, config: DataConfig):
        """Check if quality thresholds are appropriate."""
        if config.min_quality_score < 0.2:
            self._add_result(
                "quality_thresholds", "warning",
                f"Very low quality threshold ({config.min_quality_score})",
                {"min_quality_score": config.min_quality_score},
                "Consider raising quality threshold for better data reliability"
            )
        elif config.max_missingness_pct > 40:
            self._add_result(
                "quality_thresholds", "warning",
                f"High missingness threshold ({config.max_missingness_pct}%)",
                {"max_missingness_pct": config.max_missingness_pct},
                "Consider lowering missingness threshold for more complete data"
            )
        else:
            self._add_result(
                "quality_thresholds", "pass",
                f"Appropriate quality thresholds (quality≥{config.min_quality_score}, miss≤{config.max_missingness_pct}%)",
                {"min_quality_score": config.min_quality_score, "max_missingness_pct": config.max_missingness_pct}
            )
    
    def get_health_score(self) -> float:
        """Calculate overall configuration health score (0-1)."""
        if not self.validation_results:
            return 0.0
        
        scores = []
        for result in self.validation_results:
            if result.status == "pass":
                scores.append(1.0)
            elif result.status == "warning":
                scores.append(0.7)
            else:  # fail
                scores.append(0.0)
        
        return sum(scores) / len(scores)
    
    def get_summary(self) -> Dict:
        """Get validation summary with health score."""
        pass_count = sum(1 for r in self.validation_results if r.status == "pass")
        warning_count = sum(1 for r in self.validation_results if r.status == "warning")
        fail_count = sum(1 for r in self.validation_results if r.status == "fail")
        
        return {
            "health_score": self.get_health_score(),
            "total_checks": len(self.validation_results),
            "pass_count": pass_count,
            "warning_count": warning_count,
            "fail_count": fail_count,
            "status": "healthy" if fail_count == 0 and warning_count <= 2 else "needs_attention"
        }
