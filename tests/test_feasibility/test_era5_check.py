"""Tests for ERA5 data availability checker."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from airaware.feasibility.met.era5_check import (
    ERA5Checker,
    ERA5ValidationResult,
    LocalERA5Summary,
    KATHMANDU_BBOX,
    REQUIRED_VARIABLES,
)


class TestERA5ValidationResult:
    """Test ERA5ValidationResult model."""
    
    def test_result_creation(self) -> None:
        """Test validation result creation."""
        date = datetime(2024, 12, 1)
        
        result = ERA5ValidationResult(
            date_checked=date,
            bbox=KATHMANDU_BBOX,
            variables_requested=["u10", "v10"],
            variables_found=["u10", "v10"],
            missing_variables=[],
            time_dimension_ok=True,
            expected_time_steps=24,
            actual_time_steps=24,
            ok_vars=True,
            ok_time24=True,
            overall_ok=True,
        )
        
        assert result.date_checked == date
        assert result.variables_found == ["u10", "v10"]
        assert result.missing_variables == []
        assert result.ok_vars is True
        assert result.ok_time24 is True
        assert result.overall_ok is True


class TestLocalERA5Summary:
    """Test LocalERA5Summary model."""
    
    def test_empty_summary(self) -> None:
        """Test empty summary creation."""
        summary = LocalERA5Summary()
        
        assert summary.files_found == []
        assert summary.date_coverage == []
        assert summary.variables_available == []
        assert summary.total_size_mb == 0.0
    
    def test_summary_with_data(self) -> None:
        """Test summary with data."""
        dates = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        
        summary = LocalERA5Summary(
            files_found=["era5_20240101.nc", "era5_20240102.nc"],
            date_coverage=dates,
            variables_available=["u10", "v10", "t2m"],
            total_size_mb=150.5,
        )
        
        assert len(summary.files_found) == 2
        assert len(summary.date_coverage) == 2
        assert "u10" in summary.variables_available
        assert summary.total_size_mb == 150.5


class TestERA5Checker:
    """Test ERA5Checker functionality."""
    
    def test_initialization_local_only(self) -> None:
        """Test checker initialization in local-only mode."""
        checker = ERA5Checker(use_local=True)
        
        assert checker.use_local is True
        assert checker.cds_client is None
    
    def test_initialization_missing_key(self) -> None:
        """Test initialization without API key."""
        checker = ERA5Checker(cds_api_key=None)
        
        assert checker.cds_client is None
    
    @patch('cdsapi.Client')
    def test_setup_cds_client_success(self, mock_client_class: Mock) -> None:
        """Test successful CDS client setup."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        checker = ERA5Checker(cds_api_key="test_key")
        
        assert checker.cds_client == mock_client
        mock_client_class.assert_called_once_with(
            key="test_key",
            verify=True,
            timeout=300,
            retry_max=3,
        )
    
    @patch('cdsapi.Client')
    def test_setup_cds_client_licence_error(self, mock_client_class: Mock) -> None:
        """Test CDS client setup with licence error."""
        mock_client_class.side_effect = Exception("licence not accepted")
        
        with pytest.raises(ValueError, match="ERA5 licence not accepted"):
            ERA5Checker(cds_api_key="test_key")
    
    @patch('cdsapi.Client')
    def test_setup_cds_client_auth_error(self, mock_client_class: Mock) -> None:
        """Test CDS client setup with authentication error."""
        mock_client_class.side_effect = Exception("authentication failed")
        
        with pytest.raises(ValueError, match="CDS API authentication failed"):
            ERA5Checker(cds_api_key="test_key")
    
    def test_load_local_manifest_missing_file(self) -> None:
        """Test loading manifest when file doesn't exist."""
        checker = ERA5Checker(
            use_local=True,
            data_manifest_path=Path("nonexistent.json"),
        )
        
        manifest = checker._load_local_manifest()
        assert manifest == {}
    
    def test_load_local_manifest_valid_file(self) -> None:
        """Test loading valid manifest file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_manifest = {
                "files": [
                    {"path": "era5_hourly.parquet", "dataset_types": ["era5_hourly"]}
                ]
            }
            json.dump(test_manifest, f)
            manifest_path = Path(f.name)
        
        try:
            checker = ERA5Checker(
                use_local=True,
                data_manifest_path=manifest_path,
            )
            
            manifest = checker._load_local_manifest()
            assert len(manifest["files"]) == 1
            
        finally:
            manifest_path.unlink()
    
    def test_analyze_local_era5_data_empty_manifest(self) -> None:
        """Test local ERA5 analysis with empty manifest."""
        checker = ERA5Checker(
            use_local=True,
            data_manifest_path=Path("nonexistent.json"),
        )
        
        summary = checker._analyze_local_era5_data()
        
        assert isinstance(summary, LocalERA5Summary)
        assert summary.files_found == []
        assert summary.variables_available == []
    
    def test_analyze_local_era5_data_with_files(self) -> None:
        """Test local ERA5 analysis with files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_manifest = {
                "files": [
                    {
                        "path": "era5/era5_hourly_20240101.parquet",
                        "dataset_types": ["era5_hourly"],
                        "size_mb": 50.0,
                        "schema_validation": {
                            "validated": True,
                            "columns": ["u10", "v10", "t2m", "blh", "ts_utc"],
                        }
                    },
                    {
                        "path": "era5/era5_hourly_20240102.parquet",
                        "dataset_types": ["era5_hourly"],
                        "size_mb": 48.5,
                        "schema_validation": {
                            "validated": True,
                            "columns": ["u10", "v10", "t2m"],
                        }
                    }
                ]
            }
            json.dump(test_manifest, f)
            manifest_path = Path(f.name)
        
        try:
            checker = ERA5Checker(
                use_local=True,
                data_manifest_path=manifest_path,
            )
            
            summary = checker._analyze_local_era5_data()
            
            assert len(summary.files_found) == 2
            assert summary.total_size_mb == 98.5
            # Should find all required variables across files
            for var in ["u10", "v10", "t2m"]:
                assert var in summary.variables_available
            
        finally:
            manifest_path.unlink()
    
    def test_check_local_era5_day_with_data(self) -> None:
        """Test local ERA5 day check with available data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_manifest = {
                "files": [
                    {
                        "path": "era5/era5_20240101.nc",
                        "dataset_types": ["era5_hourly"],
                        "schema_validation": {
                            "validated": True,
                            "columns": ["u10", "v10", "t2m", "blh"],
                        }
                    }
                ]
            }
            json.dump(test_manifest, f)
            manifest_path = Path(f.name)
        
        try:
            checker = ERA5Checker(
                use_local=True,
                data_manifest_path=manifest_path,
            )
            
            # Mock the analyze method to return coverage for the test date
            with patch.object(checker, '_analyze_local_era5_data') as mock_analyze:
                mock_summary = LocalERA5Summary(
                    date_coverage=[datetime(2024, 1, 1)],
                    variables_available=["u10", "v10", "t2m", "blh"],
                )
                mock_analyze.return_value = mock_summary
                
                result = checker._check_local_era5_day(
                    date=datetime(2024, 1, 1),
                    bbox=KATHMANDU_BBOX,
                    variables=["u10", "v10", "t2m", "blh"],
                )
                
                assert result.ok_vars is True
                assert result.ok_time24 is True
                assert result.overall_ok is True
                assert len(result.variables_found) == 4
                assert result.missing_variables == []
            
        finally:
            manifest_path.unlink()
    
    def test_check_local_era5_day_missing_date(self) -> None:
        """Test local ERA5 day check with missing date."""
        checker = ERA5Checker(
            use_local=True,
            data_manifest_path=Path("nonexistent.json"),
        )
        
        result = checker._check_local_era5_day(
            date=datetime(2024, 1, 1),
            bbox=KATHMANDU_BBOX,
            variables=["u10", "v10", "t2m", "blh"],
        )
        
        assert result.ok_vars is False
        assert result.ok_time24 is False
        assert result.overall_ok is False
        assert len(result.validation_errors) > 0
        assert "not found in local data" in result.validation_errors[0]
    
    @patch('airaware.feasibility.met.era5_check.ERA5Checker._download_era5_day')
    @patch('airaware.feasibility.met.era5_check.ERA5Checker._validate_era5_file')
    def test_check_era5_day_api_success(self, mock_validate: Mock, mock_download: Mock) -> None:
        """Test ERA5 day check via API with success."""
        # Mock successful validation
        mock_result = ERA5ValidationResult(
            date_checked=datetime(2024, 12, 1),
            bbox=KATHMANDU_BBOX,
            variables_requested=["u10", "v10", "t2m", "blh"],
            variables_found=["u10", "v10", "t2m", "blh"],
            missing_variables=[],
            time_dimension_ok=True,
            expected_time_steps=24,
            actual_time_steps=24,
            ok_vars=True,
            ok_time24=True,
            overall_ok=True,
        )
        mock_validate.return_value = mock_result
        
        with patch('cdsapi.Client'):
            checker = ERA5Checker(cds_api_key="test_key")
            
            result = checker.check_era5_day(datetime(2024, 12, 1))
            
            assert result.ok_vars is True
            assert result.ok_time24 is True
            assert result.overall_ok is True
            mock_download.assert_called_once()
            mock_validate.assert_called_once()
    
    @patch('cdsapi.Client')
    def test_check_era5_day_api_error(self, mock_client_class: Mock) -> None:
        """Test ERA5 day check with API error."""
        mock_client = Mock()
        mock_client.retrieve.side_effect = Exception("CDS Error")
        mock_client_class.return_value = mock_client
        
        checker = ERA5Checker(cds_api_key="test_key")
        
        result = checker.check_era5_day(datetime(2024, 12, 1))
        
        assert result.ok_vars is False
        assert result.ok_time24 is False
        assert result.overall_ok is False
        assert len(result.validation_errors) > 0
        assert "CDS Error" in result.validation_errors[0]
    
    @patch('xarray.open_dataset')
    def test_validate_era5_file_success(self, mock_open: Mock) -> None:
        """Test successful ERA5 file validation."""
        # Mock xarray dataset
        mock_ds = Mock()
        mock_ds.data_vars.keys.return_value = ["u10", "v10", "t2m", "blh"]
        mock_ds.time = range(24)  # 24 time steps
        mock_ds.dims = {"time": 24, "lat": 10, "lon": 10}
        mock_open.return_value = mock_ds
        
        checker = ERA5Checker(use_local=True)
        
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            test_file = Path(f.name)
        
        try:
            result = checker._validate_era5_file(
                file_path=test_file,
                date=datetime(2024, 12, 1),
                bbox=KATHMANDU_BBOX,
                expected_variables=["u10", "v10", "t2m", "blh"],
            )
            
            assert result.ok_vars is True
            assert result.ok_time24 is True
            assert result.overall_ok is True
            assert result.variables_found == ["u10", "v10", "t2m", "blh"]
            assert result.missing_variables == []
            
        finally:
            test_file.unlink()
    
    @patch('xarray.open_dataset')
    def test_validate_era5_file_missing_variables(self, mock_open: Mock) -> None:
        """Test ERA5 file validation with missing variables."""
        # Mock dataset with missing variables
        mock_ds = Mock()
        mock_ds.data_vars.keys.return_value = ["u10", "v10"]  # Missing t2m, blh
        mock_ds.time = range(24)
        mock_open.return_value = mock_ds
        
        checker = ERA5Checker(use_local=True)
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            test_file = Path(f.name)
        
        try:
            result = checker._validate_era5_file(
                file_path=test_file,
                date=datetime(2024, 12, 1),
                bbox=KATHMANDU_BBOX,
                expected_variables=["u10", "v10", "t2m", "blh"],
            )
            
            assert result.ok_vars is False
            assert result.variables_found == ["u10", "v10"]
            assert result.missing_variables == ["t2m", "blh"]
            assert "Missing variables" in result.validation_errors[0]
            
        finally:
            test_file.unlink()
    
    @patch('xarray.open_dataset')
    def test_validate_era5_file_wrong_time_steps(self, mock_open: Mock) -> None:
        """Test ERA5 file validation with wrong time steps."""
        # Mock dataset with wrong time dimension
        mock_ds = Mock()
        mock_ds.data_vars.keys.return_value = ["u10", "v10", "t2m", "blh"]
        mock_ds.time = range(12)  # Only 12 hours instead of 24
        mock_open.return_value = mock_ds
        
        checker = ERA5Checker(use_local=True)
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            test_file = Path(f.name)
        
        try:
            result = checker._validate_era5_file(
                file_path=test_file,
                date=datetime(2024, 12, 1),
                bbox=KATHMANDU_BBOX,
                expected_variables=["u10", "v10", "t2m", "blh"],
            )
            
            assert result.ok_time24 is False
            assert result.actual_time_steps == 12
            assert "Expected 24 time steps, got 12" in result.validation_errors[0]
            
        finally:
            test_file.unlink()
    
    @patch('airaware.feasibility.met.era5_check.ERA5Checker.check_era5_day')
    def test_check_era5_availability(self, mock_check_day: Mock) -> None:
        """Test ERA5 availability check over date range."""
        # Mock daily checks
        mock_result = ERA5ValidationResult(
            date_checked=datetime(2024, 12, 1),
            bbox=KATHMANDU_BBOX,
            variables_requested=["u10", "v10", "t2m", "blh"],
            variables_found=["u10", "v10", "t2m", "blh"],
            missing_variables=[],
            time_dimension_ok=True,
            expected_time_steps=24,
            actual_time_steps=24,
            ok_vars=True,
            ok_time24=True,
            overall_ok=True,
        )
        mock_check_day.return_value = mock_result
        
        checker = ERA5Checker(use_local=True)
        
        start_date = datetime(2024, 12, 1)
        end_date = datetime(2024, 12, 3)
        
        results = checker.check_era5_availability(
            start_date=start_date,
            end_date=end_date,
            sample_days=2,
        )
        
        assert len(results) == 2
        assert all(r.overall_ok for r in results)
        assert mock_check_day.call_count == 2


class TestConstants:
    """Test module constants."""
    
    def test_kathmandu_bbox(self) -> None:
        """Test Kathmandu bounding box definition."""
        assert "north" in KATHMANDU_BBOX
        assert "south" in KATHMANDU_BBOX
        assert "east" in KATHMANDU_BBOX
        assert "west" in KATHMANDU_BBOX
        
        # Verify reasonable coordinates for Kathmandu
        assert 27.5 < KATHMANDU_BBOX["south"] < KATHMANDU_BBOX["north"] < 28.0
        assert 85.0 < KATHMANDU_BBOX["west"] < KATHMANDU_BBOX["east"] < 86.0
    
    def test_required_variables(self) -> None:
        """Test required ERA5 variables definition."""
        expected_vars = ["u10", "v10", "t2m", "blh"]
        
        assert len(REQUIRED_VARIABLES) == 4
        for var in expected_vars:
            assert var in REQUIRED_VARIABLES
            assert isinstance(REQUIRED_VARIABLES[var], str)
            assert len(REQUIRED_VARIABLES[var]) > 0


