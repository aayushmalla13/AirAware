"""Tests for advanced configuration validation."""

import pytest
from datetime import datetime, timedelta

from airaware.config.models import BoundingBox, DataConfig, StationConfig
from airaware.config.validation import ConfigValidator


class TestConfigValidator:
    """Test configuration validation and health checks."""
    
    def create_test_config(self, station_count=3, quality_scores=None, distances=None):
        """Create test configuration with specified parameters."""
        if quality_scores is None:
            quality_scores = [0.5, 0.4, 0.6]
        if distances is None:
            distances = [1.0, 2.0, 3.0]
        
        bbox = BoundingBox(north=28.0, south=27.4, east=85.6, west=84.8)
        
        stations = []
        for i in range(station_count):
            quality = quality_scores[i] if i < len(quality_scores) else 0.5
            distance = distances[i] if i < len(distances) else 1.0
            
            station = StationConfig(
                station_id=1000 + i,
                name=f"Test Station {i+1}",
                latitude=27.7 + i * 0.01,
                longitude=85.3 + i * 0.01,
                distance_km=distance,
                quality_score=quality,
                missingness_pct=20.0,
                data_span_days=120,
                first_measurement=datetime.now() - timedelta(days=120),
                last_measurement=datetime.now() - timedelta(hours=1)
            )
            stations.append(station)
        
        return DataConfig(
            valley_bbox=bbox,
            stations=stations,
            date_from=datetime(2024, 1, 1),
            date_to=datetime(2024, 12, 31)
        )
    
    def test_valid_configuration_validation(self):
        """Test validation of a healthy configuration."""
        config = self.create_test_config()
        validator = ConfigValidator()
        
        results = validator.validate_configuration(config)
        health_score = validator.get_health_score()
        
        assert len(results) > 0
        assert health_score > 0.0
        assert health_score <= 1.0
        
        # Should have mostly pass results for a good config
        pass_count = sum(1 for r in results if r.status == "pass")
        assert pass_count >= len(results) * 0.5  # At least half should pass
    
    def test_insufficient_stations_validation(self):
        """Test validation with insufficient stations."""
        config = self.create_test_config(station_count=1)
        validator = ConfigValidator()
        
        results = validator.validate_configuration(config)
        
        # Should flag geographic coverage issue
        coverage_results = [r for r in results if r.check_name == "geographic_coverage"]
        assert len(coverage_results) > 0
        assert coverage_results[0].status == "fail"
    
    def test_low_quality_stations_validation(self):
        """Test validation with low quality stations."""
        config = self.create_test_config(quality_scores=[0.1, 0.15, 0.2])
        validator = ConfigValidator()
        
        results = validator.validate_configuration(config)
        
        # Should flag quality distribution issue
        quality_results = [r for r in results if r.check_name == "quality_distribution"]
        assert len(quality_results) > 0
        assert quality_results[0].status == "warning"
    
    def test_clustered_stations_validation(self):
        """Test validation with clustered stations."""
        config = self.create_test_config(distances=[0.1, 0.2, 0.3])  # Very close
        validator = ConfigValidator()
        
        results = validator.validate_configuration(config)
        
        # Should flag clustering issue
        clustering_results = [r for r in results if r.check_name == "station_clustering"]
        assert len(clustering_results) > 0
        assert clustering_results[0].status == "warning"
    
    def test_stale_data_validation(self):
        """Test validation with stale data."""
        config = self.create_test_config()
        
        # Make all stations have old data
        for station in config.stations:
            station.last_measurement = datetime.now() - timedelta(days=2)
        
        validator = ConfigValidator()
        results = validator.validate_configuration(config)
        
        # Should flag data freshness issue
        freshness_results = [r for r in results if r.check_name == "data_freshness"]
        assert len(freshness_results) > 0
        assert freshness_results[0].status in ["fail", "warning"]
    
    def test_health_score_calculation(self):
        """Test health score calculation."""
        validator = ConfigValidator()
        
        # Simulate results
        validator.validation_results = [
            type('Result', (), {'status': 'pass'})(),
            type('Result', (), {'status': 'warning'})(),
            type('Result', (), {'status': 'fail'})(),
        ]
        
        health_score = validator.get_health_score()
        
        # Expected: (1.0 + 0.7 + 0.0) / 3 = 0.567
        assert abs(health_score - 0.567) < 0.01
    
    def test_validation_summary(self):
        """Test validation summary generation."""
        config = self.create_test_config()
        validator = ConfigValidator()
        
        results = validator.validate_configuration(config)
        summary = validator.get_summary()
        
        assert "health_score" in summary
        assert "total_checks" in summary
        assert "pass_count" in summary
        assert "warning_count" in summary
        assert "fail_count" in summary
        assert "status" in summary
        
        # Counts should add up
        total = summary["pass_count"] + summary["warning_count"] + summary["fail_count"]
        assert total == summary["total_checks"]
        assert summary["status"] in ["healthy", "needs_attention"]


