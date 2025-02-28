"""Tests for configuration loader utilities."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from airaware.config.loader import (
    get_config_dir,
    load_app_config,
    load_data_config,
    load_yaml_config,
    save_app_config,
    save_data_config
)
from airaware.config.models import AppConfig, BoundingBox, DataConfig, StationConfig


class TestConfigLoader:
    """Test configuration loading utilities."""
    
    def test_get_config_dir(self):
        """Test config directory detection."""
        # Should find configs/ directory in project root
        config_dir = get_config_dir()
        assert config_dir.name == "configs"
        assert config_dir.exists()
    
    def test_load_yaml_config(self):
        """Test YAML configuration loading."""
        # Should be able to load existing app.yaml
        config_dict = load_yaml_config("app")
        assert isinstance(config_dict, dict)
        assert "app_name" in config_dict
    
    def test_load_yaml_config_missing_file(self):
        """Test error handling for missing config file."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config("nonexistent")
    
    def test_load_data_config(self):
        """Test data configuration loading."""
        # Should be able to load existing data.yaml
        data_config = load_data_config()
        assert isinstance(data_config, DataConfig)
        assert len(data_config.stations) >= 3
        assert data_config.valley_bbox.north > data_config.valley_bbox.south
    
    def test_load_app_config(self):
        """Test application configuration loading."""
        # Should be able to load existing app.yaml
        app_config = load_app_config()
        assert isinstance(app_config, AppConfig)
        assert app_config.app_name == "AirAware"
        assert app_config.api_port > 0
    
    def test_load_config_from_custom_path(self):
        """Test loading configuration from custom file path."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_config = {
                "app_name": "TestApp",
                "version": "0.1.0",
                "api_port": 9000
            }
            yaml.dump(test_config, f)
            temp_path = f.name
        
        try:
            app_config = load_app_config(temp_path)
            assert app_config.app_name == "TestApp"
            assert app_config.version == "0.1.0"
            assert app_config.api_port == 9000
        finally:
            os.unlink(temp_path)
    
    def test_save_and_load_data_config(self):
        """Test saving and loading data configuration."""
        # Create test configuration
        bbox = BoundingBox(north=28.0, south=27.4, east=85.6, west=84.8)
        
        stations = [
            StationConfig(
                station_id=1, name="Test Station 1", latitude=27.7, longitude=85.3,
                distance_km=1.0, quality_score=0.5, missingness_pct=20.0
            ),
            StationConfig(
                station_id=2, name="Test Station 2", latitude=27.8, longitude=85.4,
                distance_km=2.0, quality_score=0.4, missingness_pct=25.0
            ),
            StationConfig(
                station_id=3, name="Test Station 3", latitude=27.6, longitude=85.2,
                distance_km=3.0, quality_score=0.6, missingness_pct=15.0
            )
        ]
        
        from datetime import datetime
        test_config = DataConfig(
            valley_bbox=bbox,
            stations=stations,
            date_from=datetime(2024, 1, 1),
            date_to=datetime(2024, 12, 31)
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_data_config(test_config, temp_path)
            
            # Load it back
            loaded_config = load_data_config(temp_path)
            
            assert loaded_config.valley_bbox.north == test_config.valley_bbox.north
            assert len(loaded_config.stations) == len(test_config.stations)
            assert loaded_config.stations[0].name == "Test Station 1"
            assert loaded_config.date_from == test_config.date_from
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_and_load_app_config(self):
        """Test saving and loading application configuration."""
        # Create test configuration
        test_config = AppConfig(
            app_name="TestAirAware",
            version="2.0.0",
            api_port=9001
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            save_app_config(test_config, temp_path)
            
            # Load it back
            loaded_config = load_app_config(temp_path)
            
            assert loaded_config.app_name == "TestAirAware"
            assert loaded_config.version == "2.0.0"
            assert loaded_config.api_port == 9001
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_invalid_yaml_content(self):
        """Test error handling for invalid YAML content."""
        # Create file with invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            temp_path = f.name
        
        try:
            with pytest.raises((ValueError, yaml.YAMLError)):
                load_app_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_invalid_config_validation(self):
        """Test error handling for configuration validation failures."""
        # Create config with invalid data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            invalid_config = {
                "app_name": "TestApp",
                "api_port": -1  # Invalid port
            }
            yaml.dump(invalid_config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_app_config(temp_path)
        finally:
            os.unlink(temp_path)
