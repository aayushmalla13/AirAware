"""Tests for configuration versioning system."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from airaware.config.models import BoundingBox, DataConfig, StationConfig
from airaware.config.versioning import ConfigVersionManager


class TestConfigVersionManager:
    """Test configuration version management."""
    
    def create_test_config(self, station_count=3, quality_offset=0.0):
        """Create test configuration."""
        bbox = BoundingBox(north=28.0, south=27.4, east=85.6, west=84.8)
        
        stations = []
        for i in range(station_count):
            station = StationConfig(
                station_id=1000 + i,
                name=f"Test Station {i+1}",
                latitude=27.7 + i * 0.01,
                longitude=85.3 + i * 0.01,
                distance_km=1.0 + i,
                quality_score=0.5 + quality_offset,
                missingness_pct=20.0
            )
            stations.append(station)
        
        return DataConfig(
            valley_bbox=bbox,
            stations=stations,
            date_from=datetime(2024, 1, 1),
            date_to=datetime(2024, 12, 31)
        )
    
    def test_version_creation(self):
        """Test creating configuration versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "test_versions.json"
            manager = ConfigVersionManager(str(version_file))
            
            config = self.create_test_config()
            
            # Create first version
            version1 = manager.create_version(config, "Initial configuration", "test_user")
            assert version1 == "v001"
            
            # Create second version with different config
            config2 = self.create_test_config(station_count=4)
            version2 = manager.create_version(config2, "Added station", "test_user")
            assert version2 == "v002"
            
            # Verify versions file exists and contains data
            assert version_file.exists()
            
            with open(version_file) as f:
                data = json.load(f)
                assert "versions" in data
                assert len(data["versions"]) == 2
    
    def test_version_history(self):
        """Test retrieving version history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "test_versions.json"
            manager = ConfigVersionManager(str(version_file))
            
            config1 = self.create_test_config()
            config2 = self.create_test_config(station_count=4)
            
            manager.create_version(config1, "Version 1")
            manager.create_version(config2, "Version 2")
            
            history = manager.get_version_history()
            assert len(history) == 2
            
            # Should be sorted by timestamp, most recent first
            assert history[0].version == "v002"
            assert history[1].version == "v001"
    
    def test_active_version_tracking(self):
        """Test active version tracking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "test_versions.json"
            manager = ConfigVersionManager(str(version_file))
            
            config1 = self.create_test_config()
            config2 = self.create_test_config(station_count=4)
            
            manager.create_version(config1, "Version 1")
            manager.create_version(config2, "Version 2")
            
            active = manager.get_active_version()
            assert active is not None
            assert active.version == "v002"
            assert active.is_active is True
    
    def test_config_hash_calculation(self):
        """Test configuration hash calculation for change detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "test_versions.json"
            manager = ConfigVersionManager(str(version_file))
            
            config1 = self.create_test_config()
            config2 = self.create_test_config()  # Identical config
            
            hash1 = manager._calculate_config_hash(config1)
            hash2 = manager._calculate_config_hash(config2)
            
            # Identical configs should have same hash
            assert hash1 == hash2
            
            # Different config should have different hash
            config3 = self.create_test_config(station_count=4)
            hash3 = manager._calculate_config_hash(config3)
            assert hash1 != hash3
    
    def test_duplicate_version_prevention(self):
        """Test that duplicate configurations don't create new versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "test_versions.json"
            manager = ConfigVersionManager(str(version_file))
            
            config = self.create_test_config()
            
            # Create first version
            version1 = manager.create_version(config, "Version 1")
            
            # Try to create identical version
            version2 = manager.create_version(config, "Version 1 duplicate")
            
            # Should return same version number
            assert version1 == version2
            
            # Should only have one version
            history = manager.get_version_history()
            assert len(history) == 1
    
    def test_version_comparison(self):
        """Test comparing different versions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "test_versions.json"
            manager = ConfigVersionManager(str(version_file))
            
            # Create two different configs
            config1 = self.create_test_config(station_count=3, quality_offset=0.0)
            config2 = self.create_test_config(station_count=4, quality_offset=0.1)
            
            manager.create_version(config1, "Version 1")
            manager.create_version(config2, "Version 2")
            
            comparison = manager.compare_versions("v001", "v002")
            
            assert "version_1" in comparison
            assert "version_2" in comparison
            assert "differences" in comparison
            
            diff = comparison["differences"]
            assert diff["station_count_change"] == 1  # Added 1 station
            assert diff["quality_score_change"] > 0   # Quality increased
    
    def test_invalid_version_comparison(self):
        """Test error handling for invalid version comparison."""
        with tempfile.TemporaryDirectory() as temp_dir:
            version_file = Path(temp_dir) / "test_versions.json"
            manager = ConfigVersionManager(str(version_file))
            
            with pytest.raises(ValueError):
                manager.compare_versions("v999", "v998")  # Non-existent versions


