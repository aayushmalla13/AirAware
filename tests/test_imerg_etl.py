"""Tests for IMERG ETL pipeline."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import h5py

from airaware.etl.imerg_etl import IMERGETL


class TestIMERGETL:
    """Test cases for IMERG ETL pipeline."""
    
    @pytest.fixture
    def imerg_etl(self):
        """Create IMERGETL instance for testing."""
        return IMERGETL(
            raw_data_dir="test_data/raw/imerg",
            interim_data_dir="test_data/interim",
            use_local=True
        )
    
    def test_generate_half_hourly_timestamps(self, imerg_etl):
        """Test half-hourly timestamp generation."""
        start_date = datetime(2024, 6, 1, 0, 0)
        end_date = datetime(2024, 6, 1, 2, 0)
        
        timestamps = imerg_etl.generate_half_hourly_timestamps(start_date, end_date)
        
        expected_timestamps = [
            datetime(2024, 6, 1, 0, 0),
            datetime(2024, 6, 1, 0, 30),
            datetime(2024, 6, 1, 1, 0),
            datetime(2024, 6, 1, 1, 30),
            datetime(2024, 6, 1, 2, 0)
        ]
        
        assert len(timestamps) == 5
        assert timestamps == expected_timestamps
    
    def test_get_hdf5_filename(self, imerg_etl):
        """Test HDF5 filename generation."""
        timestamp = datetime(2024, 6, 1, 12, 30)
        filename = imerg_etl.get_hdf5_filename(timestamp)
        
        expected = "3B-HHR.MS.MRG.3IMERG.20240601-S123000-E125959.0000.V07B.HDF5"
        assert filename == expected
    
    def test_get_hdf5_path(self, imerg_etl):
        """Test HDF5 file path generation."""
        timestamp = datetime(2024, 6, 1, 12, 30)
        path = imerg_etl.get_hdf5_path(timestamp)
        
        expected_path = Path("test_data/raw/imerg/3B-HHR.MS.MRG.3IMERG.20240601-S123000-E125959.0000.V07B.HDF5")
        assert path == expected_path
    
    @patch('h5py.File')
    def test_check_hdf5_valid_success(self, mock_h5py_file, imerg_etl, tmp_path):
        """Test HDF5 validation when file is valid."""
        # Mock h5py file
        mock_file = Mock()
        mock_file.__contains__ = Mock(return_value=True)
        
        # Mock precipitation dataset
        mock_precip = Mock()
        mock_precip.shape = (1800, 3600)  # Valid 2D shape
        mock_file.__getitem__ = Mock(return_value=mock_precip)
        
        mock_h5py_file.return_value.__enter__.return_value = mock_file
        
        # Create test file
        test_file = tmp_path / "test.HDF5"
        test_file.touch()
        
        result = imerg_etl.check_hdf5_valid(test_file)
        assert result is True
    
    @patch('h5py.File')
    def test_check_hdf5_valid_missing_dataset(self, mock_h5py_file, imerg_etl, tmp_path):
        """Test HDF5 validation when precipitation dataset is missing."""
        # Mock h5py file without precipitation dataset
        mock_file = Mock()
        mock_file.__contains__ = Mock(return_value=False)
        
        mock_h5py_file.return_value.__enter__.return_value = mock_file
        
        # Create test file
        test_file = tmp_path / "test.HDF5"
        test_file.touch()
        
        result = imerg_etl.check_hdf5_valid(test_file)
        assert result is False
    
    @patch('h5py.File')
    def test_process_hdf5_to_precip(self, mock_h5py_file, imerg_etl, tmp_path):
        """Test HDF5 to precipitation processing."""
        # Mock h5py file
        mock_file = Mock()
        
        # Create mock precipitation data (1800x3600 grid)
        precip_data = np.random.uniform(0, 10, (1800, 3600))
        # Add some negative values (missing data indicator)
        precip_data[0:100, 0:100] = -9999.9
        
        # Mock coordinate bounds
        lat_bounds = np.array([[i-0.05, i+0.05] for i in np.linspace(90, -90, 1800)])
        lon_bounds = np.array([[i-0.05, i+0.05] for i in np.linspace(-180, 180, 3600)])
        
        # Configure mock file
        mock_file.__getitem__.side_effect = lambda key: {
            'Grid/precipitationCal': precip_data,
            'Grid/lat_bnds': lat_bounds,
            'Grid/lon_bnds': lon_bounds
        }[key]
        
        mock_h5py_file.return_value.__enter__.return_value = mock_file
        
        # Create test file path
        test_file = tmp_path / "test.HDF5"
        test_file.touch()
        
        timestamp = datetime(2024, 6, 1, 12, 30)
        result = imerg_etl.process_hdf5_to_precip(test_file, timestamp)
        
        assert result is not None
        assert 'datetime_utc' in result
        assert 'precipitation_mm_30min' in result
        assert 'valid_pixels' in result
        assert 'total_pixels' in result
        assert result['datetime_utc'] == timestamp
        assert isinstance(result['precipitation_mm_30min'], float)
        assert result['valid_pixels'] > 0
        assert result['total_pixels'] > 0
    
    def test_resample_to_hourly(self, imerg_etl):
        """Test resampling from 30-min to hourly data."""
        # Create test data with 4 half-hourly records (2 hours)
        half_hourly_data = [
            {
                'datetime_utc': datetime(2024, 6, 1, 0, 0),
                'precipitation_mm_30min': 2.5,
                'valid_pixels': 100,
                'total_pixels': 100
            },
            {
                'datetime_utc': datetime(2024, 6, 1, 0, 30),
                'precipitation_mm_30min': 1.5,
                'valid_pixels': 100,
                'total_pixels': 100
            },
            {
                'datetime_utc': datetime(2024, 6, 1, 1, 0),
                'precipitation_mm_30min': 3.0,
                'valid_pixels': 100,
                'total_pixels': 100
            },
            {
                'datetime_utc': datetime(2024, 6, 1, 1, 30),
                'precipitation_mm_30min': 2.0,
                'valid_pixels': 100,
                'total_pixels': 100
            }
        ]
        
        hourly_df = imerg_etl.resample_to_hourly(half_hourly_data)
        
        assert len(hourly_df) == 2  # 2 hourly records
        assert 'precipitation_mm_hourly' in hourly_df.columns
        assert hourly_df.iloc[0]['precipitation_mm_hourly'] == 4.0  # 2.5 + 1.5
        assert hourly_df.iloc[1]['precipitation_mm_hourly'] == 5.0  # 3.0 + 2.0
    
    def test_resample_to_hourly_empty(self, imerg_etl):
        """Test resampling with empty data."""
        hourly_df = imerg_etl.resample_to_hourly([])
        assert hourly_df.empty
    
    @patch.object(IMERGETL, 'download_imerg_file')
    @patch.object(IMERGETL, 'process_hdf5_to_precip')
    @patch.object(IMERGETL, 'get_hdf5_path')
    def test_run_etl_success(self, mock_get_path, mock_process, mock_download, imerg_etl, tmp_path):
        """Test successful IMERG ETL run."""
        # Mock file path
        test_file = tmp_path / "test.HDF5"
        test_file.touch()
        mock_get_path.return_value = test_file
        
        # Mock download success
        mock_download.return_value = True
        
        # Mock processing to return precipitation data
        mock_process.side_effect = [
            {
                'datetime_utc': datetime(2024, 6, 1, 0, 0),
                'precipitation_mm_30min': 2.0,
                'valid_pixels': 100,
                'total_pixels': 100
            },
            {
                'datetime_utc': datetime(2024, 6, 1, 0, 30),
                'precipitation_mm_30min': 1.5,
                'valid_pixels': 100,
                'total_pixels': 100
            }
        ]
        
        # Mock lineage tracker and manifest manager
        imerg_etl.lineage_tracker = Mock()
        imerg_etl.manifest_manager = Mock()
        
        # Create output directory
        imerg_etl.interim_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Run ETL for 1 hour (2 half-hourly files)
        start_date = datetime(2024, 6, 1, 0, 0)
        end_date = datetime(2024, 6, 1, 0, 30)
        
        results = imerg_etl.run_etl(start_date, end_date)
        
        assert results["success"] is True
        assert results["timestamps_processed"] == 2
        assert results["total_records"] == 2
        assert "output_file" in results
    
    def test_get_imerg_status_no_file(self, imerg_etl):
        """Test status when no IMERG file exists."""
        status = imerg_etl.get_imerg_status()
        
        assert status["imerg_file_exists"] is False
        assert status["record_count"] == 0
        assert status["latest_data"] is None
        assert status["date_range"] is None
    
    @patch('pandas.read_parquet')
    def test_get_imerg_status_with_file(self, mock_read_parquet, imerg_etl, tmp_path):
        """Test status when IMERG file exists."""
        # Create mock DataFrame
        mock_df = pd.DataFrame({
            'datetime_utc': pd.date_range('2024-06-01', periods=48, freq='h'),
            'precipitation_mm_hourly': np.random.uniform(0, 10, 48),
            'valid_pixels': np.random.randint(80, 100, 48),
            'total_pixels': [100] * 48
        })
        mock_read_parquet.return_value = mock_df
        
        # Create fake file
        imerg_etl.interim_data_dir.mkdir(parents=True, exist_ok=True)
        test_file = imerg_etl.interim_data_dir / "imerg_hourly.parquet"
        test_file.touch()
        
        status = imerg_etl.get_imerg_status()
        
        assert status["imerg_file_exists"] is True
        assert status["record_count"] == 48
        assert status["latest_data"] is not None
        assert status["date_range"] is not None


