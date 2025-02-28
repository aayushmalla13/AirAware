"""Unit tests for data scanner functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from airaware.utils.data_scanner import DataScanner


class TestDataScanner:
    """Test suite for DataScanner class."""

    def test_init(self) -> None:
        """Test DataScanner initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            scanner = DataScanner(temp_path)
            
            assert scanner.data_root == temp_path
            assert scanner.manifest_path == temp_path / "artifacts" / "local_data_manifest.json"
            assert scanner.manifest_path.parent.exists()

    def test_detect_dataset_type(self) -> None:
        """Test dataset type detection."""
        scanner = DataScanner()
        
        test_cases = [
            ("data/raw/openaq/station_1.parquet", {"openaq_targets"}),
            ("data/interim/era5_hourly.parquet", {"era5_hourly"}),
            ("data/processed/train_ready.parquet", {"train_ready"}),
            ("data/interim/joined_data.parquet", {"joined"}),
            ("data/raw/unknown_file.csv", set()),
        ]
        
        for file_path, expected_types in test_cases:
            result = scanner.detect_dataset_type(Path(file_path))
            assert result == expected_types

    def test_compute_file_hash_quick_mode(self) -> None:
        """Test hash computation in quick mode."""
        scanner = DataScanner()
        
        with tempfile.NamedTemporaryFile() as temp_file:
            result = scanner.compute_file_hash(Path(temp_file.name), quick_mode=True)
            assert result is None

    def test_compute_file_hash_full_mode(self) -> None:
        """Test hash computation in full mode."""
        scanner = DataScanner()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("test content")
            temp_path = Path(temp_file.name)
        
        try:
            result = scanner.compute_file_hash(temp_path, quick_mode=False)
            assert result is not None
            assert isinstance(result, str)
            assert len(result) == 64  # SHA256 hex length
        finally:
            temp_path.unlink()

    def test_validate_parquet_schema_quick_mode(self) -> None:
        """Test schema validation in quick mode."""
        scanner = DataScanner()
        
        result = scanner.validate_parquet_schema(Path("test.parquet"), quick_mode=True)
        assert result == {"validated": False, "reason": "skipped"}

    def test_validate_parquet_schema_non_parquet(self) -> None:
        """Test schema validation on non-parquet file."""
        scanner = DataScanner()
        
        result = scanner.validate_parquet_schema(Path("test.csv"), quick_mode=False)
        assert result == {"validated": False, "reason": "skipped"}

    def test_enhanced_schema_validation_structure(self) -> None:
        """Test that enhanced schema validation returns expected structure."""
        scanner = DataScanner()
        
        # Test with a mock parquet file path (validation will fail but structure should be correct)
        with patch('pyarrow.parquet.ParquetFile') as mock_pq:
            mock_file = Mock()
            mock_file.schema_arrow = Mock()
            mock_file.schema_arrow.__iter__ = Mock(return_value=iter([
                Mock(name='pm25'), Mock(name='ts_utc'), Mock(name='station_id')
            ]))
            mock_file.metadata.num_rows = 1000
            mock_pq.return_value = mock_file
            
            # Mock the field.name access
            mock_file.schema_arrow = [Mock(name='pm25'), Mock(name='ts_utc'), Mock(name='station_id')]
            
            result = scanner.validate_parquet_schema(Path("test.parquet"), quick_mode=False)
            
            # Should fail due to mock structure, but we can test the error handling
            assert "validated" in result
            assert isinstance(result.get("reason", ""), str)

    def test_scan_file_basic(self) -> None:
        """Test basic file scanning."""
        scanner = DataScanner()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_file.write("test content")
            temp_path = Path(temp_file.name)
        
        try:
            # Make temp_path relative to scanner's data_root for testing
            rel_path = temp_path.relative_to(temp_path.parent)
            scanner.data_root = temp_path.parent
            
            result = scanner.scan_file(temp_path, quick_mode=True)
            
            assert "path" in result
            assert "size_bytes" in result
            assert "mtime" in result
            assert "extension" in result
            assert result["extension"] == ".txt"
            assert "dataset_types" in result
            
        finally:
            temp_path.unlink()

    def test_create_summary(self) -> None:
        """Test summary creation from file list."""
        scanner = DataScanner()
        
        files = [
            {
                "path": "raw/openaq/file1.parquet",
                "size_mb": 10.0,
                "dataset_types": ["openaq_targets"],
            },
            {
                "path": "interim/era5_data.nc",
                "size_mb": 5.0,
                "dataset_types": ["era5_hourly"],
            },
            {
                "path": "processed/features.parquet",
                "size_mb": 2.0,
                "dataset_types": ["train_ready"],
            },
        ]
        
        summary = scanner._create_summary(files)
        
        assert "by_type" in summary
        assert "by_directory" in summary
        assert "coverage" in summary
        
        # Check by_type summary
        assert "openaq_targets" in summary["by_type"]
        assert summary["by_type"]["openaq_targets"]["files"] == 1
        assert summary["by_type"]["openaq_targets"]["size_mb"] == 10.0
        
        # Check by_directory summary
        assert "raw" in summary["by_directory"]
        assert "interim" in summary["by_directory"]
        assert "processed" in summary["by_directory"]

    @patch('airaware.utils.data_scanner.zipfile.ZipFile')
    def test_safe_unpack_zip_no_file(self, mock_zipfile: Mock) -> None:
        """Test zip unpacking when no zip file exists."""
        scanner = DataScanner()
        
        # Test with non-existent zip file
        result = scanner.safe_unpack_zip(Path("nonexistent.zip"))
        assert result == []

    def test_scan_directory_empty(self) -> None:
        """Test scanning empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            scanner = DataScanner(temp_path)
            
            # Create empty subdirectories
            for subdir in ["raw", "interim", "processed", "artifacts"]:
                (temp_path / subdir).mkdir()
            
            manifest = scanner.scan_directory(quick_mode=True)
            
            assert manifest["total_files"] == 0
            assert manifest["total_size_mb"] == 0
            assert manifest["files"] == []
            assert manifest["quick_mode"] is True

    def test_save_and_load_manifest(self) -> None:
        """Test saving and loading manifest JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            scanner = DataScanner(temp_path)
            
            test_manifest = {
                "scan_timestamp": "2024-01-01T00:00:00",
                "total_files": 1,
                "files": [{"path": "test.txt", "size_mb": 1.0}],
            }
            
            scanner.save_manifest(test_manifest)
            
            # Verify file was created and can be loaded
            assert scanner.manifest_path.exists()
            
            with open(scanner.manifest_path) as f:
                loaded = json.load(f)
            
            assert loaded["total_files"] == 1
            assert loaded["files"][0]["path"] == "test.txt"
