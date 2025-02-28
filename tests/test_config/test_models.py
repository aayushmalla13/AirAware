"""Tests for configuration models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from airaware.config.models import (
    AppConfig, 
    BoundingBox, 
    DataConfig, 
    StationConfig, 
    UIConfig
)


class TestBoundingBox:
    """Test BoundingBox validation."""
    
    def test_valid_bounding_box(self):
        """Test valid bounding box creation."""
        bbox = BoundingBox(north=28.0, south=27.0, east=85.5, west=84.5)
        assert bbox.north == 28.0
        assert bbox.south == 27.0
        assert bbox.east == 85.5
        assert bbox.west == 84.5
    
    def test_invalid_north_south(self):
        """Test that north must be greater than south."""
        with pytest.raises(ValidationError):
            BoundingBox(north=27.0, south=28.0, east=85.5, west=84.5)
    
    def test_invalid_east_west(self):
        """Test that east must be greater than west."""
        with pytest.raises(ValidationError):
            BoundingBox(north=28.0, south=27.0, east=84.5, west=85.5)
    
    def test_latitude_bounds(self):
        """Test latitude bounds validation."""
        with pytest.raises(ValidationError):
            BoundingBox(north=91.0, south=27.0, east=85.5, west=84.5)
        
        with pytest.raises(ValidationError):
            BoundingBox(north=28.0, south=-91.0, east=85.5, west=84.5)
    
    def test_longitude_bounds(self):
        """Test longitude bounds validation."""
        with pytest.raises(ValidationError):
            BoundingBox(north=28.0, south=27.0, east=181.0, west=84.5)
        
        with pytest.raises(ValidationError):
            BoundingBox(north=28.0, south=27.0, east=85.5, west=-181.0)


class TestStationConfig:
    """Test StationConfig validation."""
    
    def test_valid_station_config(self):
        """Test valid station configuration."""
        station = StationConfig(
            station_id=3459,
            name="Embassy Kathmandu",
            latitude=27.717,
            longitude=85.324,
            distance_km=2.7,
            quality_score=0.45,
            missingness_pct=25.0,
            pm25_sensor_ids=[7711],
            data_span_days=120
        )
        
        assert station.station_id == 3459
        assert station.name == "Embassy Kathmandu"
        assert station.quality_score == 0.45
        assert station.missingness_pct == 25.0
    
    def test_empty_name_validation(self):
        """Test that station name cannot be empty."""
        with pytest.raises(ValidationError):
            StationConfig(
                station_id=1,
                name="",
                latitude=27.717,
                longitude=85.324,
                distance_km=2.7,
                quality_score=0.45,
                missingness_pct=25.0
            )
    
    def test_quality_score_bounds(self):
        """Test quality score bounds validation."""
        with pytest.raises(ValidationError):
            StationConfig(
                station_id=1,
                name="Test Station",
                latitude=27.717,
                longitude=85.324,
                distance_km=2.7,
                quality_score=-0.1,  # Invalid
                missingness_pct=25.0
            )
        
        with pytest.raises(ValidationError):
            StationConfig(
                station_id=1,
                name="Test Station",
                latitude=27.717,
                longitude=85.324,
                distance_km=2.7,
                quality_score=1.1,  # Invalid
                missingness_pct=25.0
            )
    
    def test_missingness_bounds(self):
        """Test missingness percentage bounds validation."""
        with pytest.raises(ValidationError):
            StationConfig(
                station_id=1,
                name="Test Station",
                latitude=27.717,
                longitude=85.324,
                distance_km=2.7,
                quality_score=0.5,
                missingness_pct=-1.0  # Invalid
            )
        
        with pytest.raises(ValidationError):
            StationConfig(
                station_id=1,
                name="Test Station",
                latitude=27.717,
                longitude=85.324,
                distance_km=2.7,
                quality_score=0.5,
                missingness_pct=101.0  # Invalid
            )


class TestDataConfig:
    """Test DataConfig validation."""
    
    def test_valid_data_config(self):
        """Test valid data configuration."""
        bbox = BoundingBox(north=28.0, south=27.4, east=85.6, west=84.8)
        
        stations = [
            StationConfig(
                station_id=1, name="Station 1", latitude=27.7, longitude=85.3,
                distance_km=1.0, quality_score=0.5, missingness_pct=20.0
            ),
            StationConfig(
                station_id=2, name="Station 2", latitude=27.8, longitude=85.4,
                distance_km=2.0, quality_score=0.4, missingness_pct=25.0
            ),
            StationConfig(
                station_id=3, name="Station 3", latitude=27.6, longitude=85.2,
                distance_km=3.0, quality_score=0.6, missingness_pct=15.0
            )
        ]
        
        config = DataConfig(
            valley_bbox=bbox,
            stations=stations,
            date_from=datetime(2024, 1, 1),
            date_to=datetime(2024, 12, 31)
        )
        
        assert len(config.stations) == 3
        assert config.center_lat == 27.7172
        assert config.timezone_storage == "UTC"
    
    def test_insufficient_stations(self):
        """Test that at least 3 stations are required."""
        bbox = BoundingBox(north=28.0, south=27.4, east=85.6, west=84.8)
        
        stations = [
            StationConfig(
                station_id=1, name="Station 1", latitude=27.7, longitude=85.3,
                distance_km=1.0, quality_score=0.5, missingness_pct=20.0
            ),
            StationConfig(
                station_id=2, name="Station 2", latitude=27.8, longitude=85.4,
                distance_km=2.0, quality_score=0.4, missingness_pct=25.0
            )
        ]
        
        with pytest.raises(ValidationError):
            DataConfig(
                valley_bbox=bbox,
                stations=stations,
                date_from=datetime(2024, 1, 1),
                date_to=datetime(2024, 12, 31)
            )
    
    def test_low_quality_stations(self):
        """Test validation of station quality criteria."""
        bbox = BoundingBox(north=28.0, south=27.4, east=85.6, west=84.8)
        
        # All stations below quality threshold
        stations = [
            StationConfig(
                station_id=1, name="Station 1", latitude=27.7, longitude=85.3,
                distance_km=1.0, quality_score=0.1, missingness_pct=20.0
            ),
            StationConfig(
                station_id=2, name="Station 2", latitude=27.8, longitude=85.4,
                distance_km=2.0, quality_score=0.2, missingness_pct=25.0
            ),
            StationConfig(
                station_id=3, name="Station 3", latitude=27.6, longitude=85.2,
                distance_km=3.0, quality_score=0.1, missingness_pct=15.0
            )
        ]
        
        with pytest.raises(ValidationError):
            DataConfig(
                valley_bbox=bbox,
                stations=stations,
                min_quality_score=0.3,
                date_from=datetime(2024, 1, 1),
                date_to=datetime(2024, 12, 31)
            )
    
    def test_invalid_date_range(self):
        """Test that date_to must be after date_from."""
        bbox = BoundingBox(north=28.0, south=27.4, east=85.6, west=84.8)
        
        stations = [
            StationConfig(
                station_id=1, name="Station 1", latitude=27.7, longitude=85.3,
                distance_km=1.0, quality_score=0.5, missingness_pct=20.0
            ),
            StationConfig(
                station_id=2, name="Station 2", latitude=27.8, longitude=85.4,
                distance_km=2.0, quality_score=0.4, missingness_pct=25.0
            ),
            StationConfig(
                station_id=3, name="Station 3", latitude=27.6, longitude=85.2,
                distance_km=3.0, quality_score=0.6, missingness_pct=15.0
            )
        ]
        
        with pytest.raises(ValidationError):
            DataConfig(
                valley_bbox=bbox,
                stations=stations,
                date_from=datetime(2024, 12, 31),
                date_to=datetime(2024, 1, 1)  # Invalid: before date_from
            )


class TestUIConfig:
    """Test UIConfig validation."""
    
    def test_valid_ui_config(self):
        """Test valid UI configuration."""
        ui_config = UIConfig(
            language_default="en",
            forecast_horizons=[6, 12, 24],
            uncertainty_levels=[0.8, 0.9, 0.95]
        )
        
        assert ui_config.language_default == "en"
        assert ui_config.forecast_horizons == [6, 12, 24]
        assert ui_config.timezone_display == "Asia/Kathmandu"
    
    def test_default_values(self):
        """Test that default values are applied correctly."""
        ui_config = UIConfig()
        
        assert ui_config.language_default == "auto"
        assert ui_config.language_options == ["en", "ne"]
        assert ui_config.refresh_interval_minutes == 60


class TestAppConfig:
    """Test AppConfig validation."""
    
    def test_valid_app_config(self):
        """Test valid application configuration."""
        app_config = AppConfig(
            app_name="AirAware",
            version="1.0.0",
            api_port=8000
        )
        
        assert app_config.app_name == "AirAware"
        assert app_config.version == "1.0.0"
        assert app_config.api_port == 8000
    
    def test_invalid_port(self):
        """Test port number validation."""
        with pytest.raises(ValidationError):
            AppConfig(api_port=0)  # Invalid port
        
        with pytest.raises(ValidationError):
            AppConfig(api_port=70000)  # Invalid port
    
    def test_default_ui_config(self):
        """Test that UI config defaults are applied."""
        app_config = AppConfig()
        
        assert isinstance(app_config.ui, UIConfig)
        assert app_config.ui.language_default == "auto"
        assert app_config.enable_cors is True


