"""Smoke tests for basic AirAware functionality."""

import os
import sys
from pathlib import Path

import pytest


def test_imports() -> None:
    """Test that core modules can be imported."""
    import airaware
    import airaware.utils.data_scanner
    
    # Verify basic attributes
    assert hasattr(airaware, "__version__")
    assert airaware.__version__ == "0.1.0"


def test_environment_setup() -> None:
    """Test that environment is properly configured."""
    # Check that we're in the right directory
    assert Path("pyproject.toml").exists()
    assert Path("airaware").exists()
    assert Path("data").exists()
    
    # Check data structure
    for subdir in ["raw", "interim", "processed", "artifacts"]:
        assert (Path("data") / subdir).exists()


def test_data_scanner_import() -> None:
    """Test that data scanner can be imported and instantiated."""
    from airaware.utils.data_scanner import DataScanner
    
    scanner = DataScanner()
    assert scanner.data_root == Path("data")
    assert scanner.manifest_path == Path("data/artifacts/local_data_manifest.json")


def test_data_scanner_basic_functionality() -> None:
    """Test basic data scanner functionality with sample data."""
    from airaware.utils.data_scanner import DataScanner
    
    scanner = DataScanner()
    
    # Test dataset type detection
    test_paths = [
        Path("data/raw/openaq/station_1.parquet"),
        Path("data/interim/era5_hourly.parquet"),
        Path("data/processed/train_ready.parquet"),
    ]
    
    for path in test_paths:
        dataset_types = scanner.detect_dataset_type(path)
        assert isinstance(dataset_types, set)
        assert len(dataset_types) >= 0  # May be empty for unknown types


def test_python_version() -> None:
    """Test that we're running on supported Python version."""
    assert sys.version_info >= (3, 9), "Python 3.9+ required"


def test_optional_dependencies_importable() -> None:
    """Test that key dependencies can be imported."""
    import pandas
    import numpy
    import requests
    import pydantic
    
    # Just verify they import without errors
    assert pandas.__version__
    assert numpy.__version__

