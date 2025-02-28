"""Tests for OpenAQ data validation components."""

import pandas as pd
import pytest
from datetime import datetime, timedelta

from airaware.etl.data_validator import OpenAQDataValidator


class TestOpenAQDataValidator:
    """Test OpenAQ data validation functionality."""
    
    def create_test_dataframe(self, num_records=100, include_issues=False):
        """Create test DataFrame with OpenAQ-like data."""
        base_time = datetime.now()
        
        data = []
        for i in range(num_records):
            record = {
                'station_id': 3459 + (i % 3),  # 3 different stations
                'sensor_id': 7710 + i,
                'parameter': 'pm25',
                'value': 25.0 + (i % 50),  # Normal PM2.5 values
                'unit': 'µg/m³',
                'datetime_utc': base_time + timedelta(hours=i),
                'latitude': 27.717 + (i % 3) * 0.01,
                'longitude': 85.324 + (i % 3) * 0.01,
                'quality_flag': 'valid',
                'data_source': 'openaq_v3'
            }
            
            if include_issues and i < 10:
                # Introduce various data quality issues
                if i % 4 == 0:
                    record['value'] = None  # Missing value
                elif i % 4 == 1:
                    record['value'] = -5.0  # Negative outlier
                elif i % 4 == 2:
                    record['value'] = 1500.0  # High outlier
                elif i % 4 == 3:
                    record['datetime_utc'] = None  # Missing timestamp
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        if include_issues:
            # Add some duplicate records
            df = pd.concat([df, df.head(5)], ignore_index=True)
        
        return df
    
    def test_schema_validation_valid_data(self):
        """Test schema validation with valid data."""
        validator = OpenAQDataValidator()
        df = self.create_test_dataframe()
        
        is_valid = validator.validate_schema(df)
        assert is_valid is True
    
    def test_schema_validation_missing_columns(self):
        """Test schema validation with missing required columns."""
        validator = OpenAQDataValidator()
        df = self.create_test_dataframe()
        
        # Remove required column
        df = df.drop(columns=['station_id'])
        
        is_valid = validator.validate_schema(df)
        assert is_valid is False
    
    def test_data_quality_validation_clean_data(self):
        """Test data quality validation with clean data."""
        validator = OpenAQDataValidator()
        df = self.create_test_dataframe(num_records=100, include_issues=False)
        
        station_ids = [3459, 3460, 3461]
        metrics = validator.validate_data_quality(df, station_ids)
        
        assert metrics.total_records == 100
        assert metrics.valid_records == 100
        assert metrics.missing_values == 0
        assert metrics.duplicate_records == 0
        assert metrics.outlier_records == 0
        assert metrics.data_completeness == 1.0
        assert metrics.quality_score > 0.9
        assert len(metrics.stations_covered) == 3
    
    def test_data_quality_validation_with_issues(self):
        """Test data quality validation with data issues."""
        validator = OpenAQDataValidator()
        df = self.create_test_dataframe(num_records=100, include_issues=True)
        
        station_ids = [3459, 3460, 3461]
        metrics = validator.validate_data_quality(df, station_ids)
        
        # Should detect issues
        assert metrics.total_records == 105  # 100 + 5 duplicates
        assert metrics.missing_values > 0
        assert metrics.duplicate_records == 5
        assert metrics.outlier_records > 0
        assert metrics.data_completeness < 1.0
        assert metrics.quality_score < 0.95  # Adjusted threshold
    
    def test_data_quality_validation_empty_dataframe(self):
        """Test data quality validation with empty DataFrame."""
        validator = OpenAQDataValidator()
        df = pd.DataFrame()
        
        metrics = validator.validate_data_quality(df, [])
        
        assert metrics.total_records == 0
        assert metrics.valid_records == 0
        assert metrics.data_completeness == 0.0
        assert metrics.quality_score == 0.0
        assert metrics.date_range == ("", "")
        assert metrics.stations_covered == []
    
    def test_quality_score_calculation(self):
        """Test quality score calculation logic."""
        validator = OpenAQDataValidator()
        
        # Perfect data
        score = validator._calculate_quality_score(1.0, 0.0, 0.0, 0.0)
        assert abs(score - 1.0) < 0.001  # Allow floating point precision
        
        # Poor data
        score = validator._calculate_quality_score(0.5, 0.3, 0.2, 0.1)
        assert 0.0 <= score <= 1.0
        assert score < 0.7
        
        # Worst case
        score = validator._calculate_quality_score(0.0, 1.0, 1.0, 1.0)
        assert score == 0.0
    
    def test_generate_quality_report(self):
        """Test quality report generation."""
        validator = OpenAQDataValidator()
        df = self.create_test_dataframe(num_records=50)
        
        station_ids = [3459, 3460, 3461]
        metrics = validator.validate_data_quality(df, station_ids)
        
        report = validator.generate_quality_report(metrics)
        
        assert "OpenAQ Data Quality Report" in report
        assert str(metrics.total_records) in report
        assert f"{metrics.quality_score:.2f}" in report
        assert "stations" in report.lower()
    
    def test_types_compatibility(self):
        """Test data type compatibility checking."""
        validator = OpenAQDataValidator()
        
        import pyarrow as pa
        
        # Compatible types
        assert validator._types_compatible(pa.int32(), pa.int64()) is True
        assert validator._types_compatible(pa.float32(), pa.float64()) is True
        assert validator._types_compatible(pa.string(), pa.string()) is True
        
        # Incompatible types
        assert validator._types_compatible(pa.string(), pa.int64()) is False
        
        # Float can accept int
        assert validator._types_compatible(pa.int32(), pa.float64()) is True
