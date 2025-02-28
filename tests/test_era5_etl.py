"""Tests for ERA5 ETL pipeline."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import xarray as xr
import numpy as np

from airaware.etl.era5_etl import ERA5ETL


class TestERA5ETL:
    """Test cases for ERA5 ETL pipeline."""
    
    @pytest.fixture
    def era5_etl(self):
        """Create ERA5ETL instance for testing."""
        return ERA5ETL(
            raw_data_dir="test_data/raw/era5",
            interim_data_dir="test_data/interim",
            use_local=True
        )
    
    def test_generate_date_range(self, era5_etl):
        """Test date range generation."""
        start_date = datetime(2024, 6, 1)
        end_date = datetime(2024, 6, 3)
        
        dates = era5_etl.generate_date_range(start_date, end_date)
        
        assert len(dates) == 3
        assert dates[0] == datetime(2024, 6, 1)
        assert dates[1] == datetime(2024, 6, 2)
        assert dates[2] == datetime(2024, 6, 3)
    
    def test_get_netcdf_path(self, era5_etl):
        """Test NetCDF file path generation."""
        date = datetime(2024, 6, 1)
        path = era5_etl.get_netcdf_path(date)
        
        expected_path = Path("test_data/raw/era5/era5_20240601.nc")
        assert path == expected_path
    
    @patch('xarray.open_dataset')
    def test_check_file_exists_and_valid_success(self, mock_open_dataset, era5_etl, tmp_path):
        """Test file validation when file is valid."""
        # Mock xarray dataset
        mock_ds = Mock()
        mock_ds.variables = ["u10", "v10", "t2m", "blh", "time", "lat", "lon"]
        mock_ds.dims = {"time": 24}
        mock_ds.sizes = {"time": 24}
        
        mock_open_dataset.return_value.__enter__.return_value = mock_ds
        
        # Create test file
        test_file = tmp_path / "test.nc"
        test_file.touch()
        
        result = era5_etl.check_file_exists_and_valid(test_file)
        assert result is True
    
    @patch('xarray.open_dataset')
    def test_check_file_exists_and_valid_missing_vars(self, mock_open_dataset, era5_etl, tmp_path):
        """Test file validation when variables are missing."""
        # Mock xarray dataset with missing variables
        mock_ds = Mock()
        mock_ds.variables = ["u10", "v10"]  # Missing t2m and blh
        mock_ds.dims = {"time": 24}
        mock_ds.sizes = {"time": 24}
        
        mock_open_dataset.return_value.__enter__.return_value = mock_ds
        
        # Create test file
        test_file = tmp_path / "test.nc"
        test_file.touch()
        
        result = era5_etl.check_file_exists_and_valid(test_file)
        assert result is False
    
    @patch('xarray.open_dataset')
    def test_process_netcdf_to_hourly(self, mock_open_dataset, era5_etl):
        """Test NetCDF to hourly DataFrame processing."""
        # Create mock dataset
        mock_ds = Mock()
        
        # Mock coordinates and data
        times = pd.date_range('2024-06-01', periods=24, freq='h')
        lats = np.linspace(28.0, 27.4, 10)
        lons = np.linspace(84.8, 85.6, 10)
        
        # Mock DataArrays
        mock_time_data = xr.DataArray(times, dims=['time'])
        mock_lat_data = xr.DataArray(lats, dims=['latitude'])
        mock_lon_data = xr.DataArray(lons, dims=['longitude'])
        
        # Mock weather variables with realistic data
        shape = (24, 10, 10)  # time, lat, lon
        mock_u10 = xr.DataArray(np.random.normal(5, 2, shape), dims=['time', 'latitude', 'longitude'])
        mock_v10 = xr.DataArray(np.random.normal(3, 2, shape), dims=['time', 'latitude', 'longitude'])
        mock_t2m = xr.DataArray(np.random.normal(298, 5, shape), dims=['time', 'latitude', 'longitude'])
        mock_blh = xr.DataArray(np.random.normal(1000, 200, shape), dims=['time', 'latitude', 'longitude'])
        
        # Configure mock dataset
        mock_ds.time.values = times.values
        mock_ds.sel.return_value.mean.return_value = mock_ds
        
        # Mock the mean calculations
        mock_ds.u10.isel.return_value.values = 5.0
        mock_ds.v10.isel.return_value.values = 3.0
        mock_ds.t2m.isel.return_value.values = 298.0
        mock_ds.blh.isel.return_value.values = 1000.0
        
        mock_open_dataset.return_value.__enter__.return_value = mock_ds
        
        # Create test file path
        test_file = Path("test.nc")
        
        result_df = era5_etl.process_netcdf_to_hourly(test_file)
        
        assert not result_df.empty
        assert len(result_df) == 24
        assert 'datetime_utc' in result_df.columns
        assert 'u10' in result_df.columns
        assert 'v10' in result_df.columns
        assert 't2m' in result_df.columns
        assert 'blh' in result_df.columns
        assert 'wind_speed' in result_df.columns
        assert 'wind_direction' in result_df.columns
        assert 't2m_celsius' in result_df.columns
    
    @patch.object(ERA5ETL, 'download_era5_day')
    @patch.object(ERA5ETL, 'process_netcdf_to_hourly')
    @patch.object(ERA5ETL, 'get_netcdf_path')
    def test_run_etl_success(self, mock_get_path, mock_process, mock_download, era5_etl, tmp_path):
        """Test successful ETL run."""
        # Mock file path
        test_file = tmp_path / "era5_20240601.nc"
        test_file.touch()
        mock_get_path.return_value = test_file
        
        # Mock download success
        mock_download.return_value = True
        
        # Mock processing
        mock_df = pd.DataFrame({
            'datetime_utc': pd.date_range('2024-06-01', periods=24, freq='h'),
            'u10': np.random.normal(5, 2, 24),
            'v10': np.random.normal(3, 2, 24),
            't2m': np.random.normal(298, 5, 24),
            'blh': np.random.normal(1000, 200, 24),
            'wind_speed': np.random.normal(6, 2, 24),
            'wind_direction': np.random.normal(180, 90, 24),
            't2m_celsius': np.random.normal(25, 5, 24)
        })
        mock_process.return_value = mock_df
        
        # Mock lineage tracker and manifest manager
        era5_etl.lineage_tracker = Mock()
        era5_etl.manifest_manager = Mock()
        
        # Create output directory
        era5_etl.interim_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Run ETL
        start_date = datetime(2024, 6, 1)
        end_date = datetime(2024, 6, 1)
        
        results = era5_etl.run_etl(start_date, end_date)
        
        assert results["success"] is True
        assert results["dates_processed"] == 1
        assert results["total_records"] == 24
        assert "output_file" in results
    
    def test_get_era5_status_no_file(self, era5_etl):
        """Test status when no ERA5 file exists."""
        status = era5_etl.get_era5_status()
        
        assert status["era5_file_exists"] is False
        assert status["record_count"] == 0
        assert status["latest_data"] is None
        assert status["date_range"] is None
    
    @patch('pandas.read_parquet')
    def test_get_era5_status_with_file(self, mock_read_parquet, era5_etl, tmp_path):
        """Test status when ERA5 file exists."""
        # Create mock DataFrame
        mock_df = pd.DataFrame({
            'datetime_utc': pd.date_range('2024-06-01', periods=48, freq='h'),
            'u10': np.random.normal(5, 2, 48),
            'v10': np.random.normal(3, 2, 48),
            't2m': np.random.normal(298, 5, 48),
            'blh': np.random.normal(1000, 200, 48)
        })
        mock_read_parquet.return_value = mock_df
        
        # Create fake file
        era5_etl.interim_data_dir.mkdir(parents=True, exist_ok=True)
        test_file = era5_etl.interim_data_dir / "era5_hourly.parquet"
        test_file.touch()
        
        status = era5_etl.get_era5_status()
        
        assert status["era5_file_exists"] is True
        assert status["record_count"] == 48
        assert status["latest_data"] is not None
        assert status["date_range"] is not None


