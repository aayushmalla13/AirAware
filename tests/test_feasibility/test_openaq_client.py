"""Tests for OpenAQ client functionality."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from airaware.feasibility.openaq_client import (
    CoverageAnalysis,
    LocalDataSummary,
    OpenAQClient,
    StationInfo,
)


class TestStationInfo:
    """Test StationInfo model."""
    
    def test_station_creation(self) -> None:
        """Test station info creation."""
        station = StationInfo(
            station_id=123,
            station_name="Test Station",
            latitude=27.7172,
            longitude=85.3240,
            distance_km=5.2,
            pm25_sensor_ids=[456, 789],
        )
        
        assert station.station_id == 123
        assert station.station_name == "Test Station"
        assert station.latitude == 27.7172
        assert station.longitude == 85.3240
        assert station.distance_km == 5.2
        assert station.pm25_sensor_ids == [456, 789]


class TestCoverageAnalysis:
    """Test CoverageAnalysis model."""
    
    def test_coverage_creation(self) -> None:
        """Test coverage analysis creation."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        
        coverage = CoverageAnalysis(
            station_id=123,
            sensor_id=456,
            date_range=(start_date, end_date),
            expected_measurements=24,
            actual_measurements=20,
            missing_measurements=4,
            missingness_pct=16.7,
            coverage_score=0.83,
        )
        
        assert coverage.station_id == 123
        assert coverage.sensor_id == 456
        assert coverage.date_range == (start_date, end_date)
        assert coverage.expected_measurements == 24
        assert coverage.actual_measurements == 20
        assert coverage.missing_measurements == 4
        assert coverage.missingness_pct == 16.7
        assert coverage.coverage_score == 0.83


class TestOpenAQClient:
    """Test OpenAQ client functionality."""
    
    def test_initialization_with_api_key(self) -> None:
        """Test client initialization with API key."""
        client = OpenAQClient(api_key="test_key")
        
        assert client.use_local is False
        assert client.max_workers == 5
        assert client.client is not None
    
    def test_initialization_local_only(self) -> None:
        """Test client initialization in local-only mode."""
        client = OpenAQClient(use_local=True)
        
        assert client.use_local is True
        assert client.client is None
    
    def test_distance_calculation(self) -> None:
        """Test great circle distance calculation."""
        client = OpenAQClient(use_local=True)
        
        # Test Kathmandu to nearby point (approximately)
        distance = client._calculate_distance(
            27.7172, 85.3240,  # Kathmandu
            27.7000, 85.3000   # Nearby point
        )
        
        assert isinstance(distance, float)
        assert 0 < distance < 10  # Should be less than 10km
    
    def test_distance_calculation_same_point(self) -> None:
        """Test distance calculation for same point."""
        client = OpenAQClient(use_local=True)
        
        distance = client._calculate_distance(
            27.7172, 85.3240,
            27.7172, 85.3240,
        )
        
        assert distance == 0.0
    
    def test_load_local_manifest_missing_file(self) -> None:
        """Test loading manifest when file doesn't exist."""
        client = OpenAQClient(
            use_local=True,
            data_manifest_path=Path("nonexistent_manifest.json"),
        )
        
        manifest = client._load_local_manifest()
        assert manifest == {}
    
    def test_load_local_manifest_valid_file(self) -> None:
        """Test loading valid manifest file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_manifest = {
                "total_files": 5,
                "files": [
                    {"path": "openaq/station_123.parquet", "dataset_types": ["openaq_targets"]}
                ]
            }
            json.dump(test_manifest, f)
            manifest_path = Path(f.name)
        
        try:
            client = OpenAQClient(
                use_local=True,
                data_manifest_path=manifest_path,
            )
            
            manifest = client._load_local_manifest()
            assert manifest["total_files"] == 5
            assert len(manifest["files"]) == 1
            
        finally:
            manifest_path.unlink()
    
    def test_analyze_local_data_empty_manifest(self) -> None:
        """Test local data analysis with empty manifest."""
        client = OpenAQClient(
            use_local=True,
            data_manifest_path=Path("nonexistent.json"),
        )
        
        summary = client._analyze_local_data()
        
        assert isinstance(summary, LocalDataSummary)
        assert summary.stations_found == []
        assert summary.total_measurements == {}
    
    def test_analyze_local_data_with_stations(self) -> None:
        """Test local data analysis with station data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_manifest = {
                "files": [
                    {
                        "path": "openaq/station_123/data.parquet",
                        "dataset_types": ["openaq_targets"],
                        "schema_validation": {
                            "validated": True,
                            "num_rows": 1000,
                        }
                    },
                    {
                        "path": "openaq/station_456/data.parquet", 
                        "dataset_types": ["openaq_targets"],
                        "schema_validation": {
                            "validated": True,
                            "num_rows": 2000,
                        }
                    }
                ]
            }
            json.dump(test_manifest, f)
            manifest_path = Path(f.name)
        
        try:
            client = OpenAQClient(
                use_local=True,
                data_manifest_path=manifest_path,
            )
            
            summary = client._analyze_local_data()
            
            assert 123 in summary.stations_found
            assert 456 in summary.stations_found
            assert summary.total_measurements[123] == 1000
            assert summary.total_measurements[456] == 2000
            
        finally:
            manifest_path.unlink()
    
    def test_get_local_stations(self) -> None:
        """Test getting stations from local data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_manifest = {
                "files": [
                    {
                        "path": "openaq/station_123/data.parquet",
                        "dataset_types": ["openaq_targets"],
                        "schema_validation": {"validated": True, "num_rows": 500}
                    }
                ]
            }
            json.dump(test_manifest, f)
            manifest_path = Path(f.name)
        
        try:
            client = OpenAQClient(
                use_local=True,
                data_manifest_path=manifest_path,
            )
            
            stations = client._get_local_stations(
                lat=27.7172,
                lon=85.3240,
                radius_km=25.0,
            )
            
            assert len(stations) >= 1
            station = stations[0]
            assert station.station_id == 123
            assert station.total_measurements == 500
            assert station.distance_km <= 25.0
            
        finally:
            manifest_path.unlink()
    
    @patch('airaware.feasibility.openaq_client.OpenAQClient._analyze_local_sensor_coverage')
    def test_analyze_local_sensor_coverage(self, mock_analyze: Mock) -> None:
        """Test local sensor coverage analysis."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        # Mock return value
        mock_coverage = CoverageAnalysis(
            station_id=0,
            sensor_id=123,
            date_range=(start_date, end_date),
            expected_measurements=744,  # 31 days * 24 hours
            actual_measurements=700,
            missing_measurements=44,
            missingness_pct=5.9,
            coverage_score=0.94,
        )
        mock_analyze.return_value = mock_coverage
        
        client = OpenAQClient(use_local=True)
        
        result = client.get_sensor_coverage(
            sensor_id=123,
            start_date=start_date,
            end_date=end_date,
        )
        
        assert result.sensor_id == 123
        assert result.actual_measurements == 700
        assert result.missingness_pct == 5.9
        mock_analyze.assert_called_once()
    
    @patch('airaware.utils.api_helpers.APIClient.get')
    def test_get_locations_pm25_api_success(self, mock_get: Mock) -> None:
        """Test getting PM2.5 locations via API."""
        # Mock API response
        mock_response = Mock()
        mock_response.success = True
        mock_response.data = {
            "results": [
                {
                    "id": 123,
                    "name": "Test Station",
                    "coordinates": {"latitude": 27.7, "longitude": 85.3},
                    "sensors": [
                        {"id": 456, "parameter": {"id": 2, "name": "pm25"}}
                    ]
                }
            ]
        }
        mock_get.return_value = mock_response
        
        client = OpenAQClient(api_key="test_key")
        stations = client.get_locations_pm25()
        
        assert len(stations) == 1
        station = stations[0]
        assert station.station_id == 123
        assert station.station_name == "Test Station"
        assert station.pm25_sensor_ids == [456]
        assert station.distance_km < 50  # Should be within reasonable range
    
    @patch('airaware.utils.api_helpers.APIClient.get')
    def test_get_locations_pm25_api_failure(self, mock_get: Mock) -> None:
        """Test API failure handling."""
        # Mock API failure
        mock_response = Mock()
        mock_response.success = False
        mock_response.error = "API Error"
        mock_get.return_value = mock_response
        
        client = OpenAQClient(api_key="test_key")
        
        with pytest.raises(Exception, match="Failed to fetch locations"):
            client.get_locations_pm25()
    
    def test_analyze_stations_parallel_empty_list(self) -> None:
        """Test parallel analysis with empty station list."""
        client = OpenAQClient(use_local=True)
        
        result = client.analyze_stations_parallel([])
        assert result == []
    
    @patch('airaware.feasibility.openaq_client.OpenAQClient.get_sensor_coverage')
    def test_analyze_stations_parallel(self, mock_coverage: Mock) -> None:
        """Test parallel station analysis."""
        # Mock coverage analysis
        mock_coverage.return_value = CoverageAnalysis(
            station_id=0,
            sensor_id=456,
            date_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
            expected_measurements=744,
            actual_measurements=700,
            missing_measurements=44,
            missingness_pct=5.9,
            coverage_score=0.94,
        )
        
        # Create test stations
        stations = [
            StationInfo(
                station_id=123,
                station_name="Test Station",
                latitude=27.7,
                longitude=85.3,
                distance_km=5.0,
                pm25_sensor_ids=[456],
            )
        ]
        
        client = OpenAQClient(use_local=True, max_workers=1)
        result = client.analyze_stations_parallel(stations)
        
        assert len(result) == 1
        station = result[0]
        assert station.data_quality_score == 0.94
        assert station.missingness_pct == 5.9
        assert station.total_measurements == 700


class TestLocalDataSummary:
    """Test LocalDataSummary model."""
    
    def test_empty_summary(self) -> None:
        """Test empty summary creation."""
        summary = LocalDataSummary()
        
        assert summary.stations_found == []
        assert summary.date_ranges == {}
        assert summary.total_measurements == {}
        assert summary.data_quality == {}
    
    def test_summary_with_data(self) -> None:
        """Test summary with data."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        summary = LocalDataSummary(
            stations_found=[123, 456],
            date_ranges={123: (start_date, end_date)},
            total_measurements={123: 1000, 456: 2000},
            data_quality={123: 0.95, 456: 0.87},
        )
        
        assert summary.stations_found == [123, 456]
        assert summary.date_ranges[123] == (start_date, end_date)
        assert summary.total_measurements[123] == 1000
        assert summary.data_quality[123] == 0.95


