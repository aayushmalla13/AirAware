"""Data validation and quality checks for OpenAQ ETL pipeline."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DataQualityMetrics(BaseModel):
    """Data quality metrics for OpenAQ measurements."""
    total_records: int
    valid_records: int
    missing_values: int
    duplicate_records: int
    outlier_records: int
    date_range: Tuple[str, str]
    stations_covered: List[int]
    data_completeness: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)


class OpenAQDataValidator:
    """Validates OpenAQ data quality and schema compliance."""
    
    # Expected OpenAQ data schema
    EXPECTED_SCHEMA = pa.schema([
        pa.field("station_id", pa.int64()),
        pa.field("sensor_id", pa.int64()),
        pa.field("parameter", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("unit", pa.string()),
        pa.field("datetime_utc", pa.timestamp('us', tz='UTC')),
        pa.field("latitude", pa.float64()),
        pa.field("longitude", pa.float64()),
        pa.field("quality_flag", pa.string()),
        pa.field("data_source", pa.string()),
    ])
    
    def __init__(self):
        self.validation_results = []
        
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame schema against expected OpenAQ schema."""
        try:
            # Convert DataFrame to PyArrow table for schema validation
            table = pa.Table.from_pandas(df)
            
            # Check required columns
            expected_columns = set(self.EXPECTED_SCHEMA.names)
            actual_columns = set(table.column_names)
            
            missing_columns = expected_columns - actual_columns
            extra_columns = actual_columns - expected_columns
            
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")
                return False
                
            if extra_columns:
                logger.info(f"Extra columns found: {extra_columns}")
            
            # Validate data types for core columns
            for field in self.EXPECTED_SCHEMA:
                if field.name in table.column_names:
                    actual_type = table.column(field.name).type
                    expected_type = field.type
                    
                    # Allow some type flexibility (e.g., int32 vs int64)
                    if not self._types_compatible(actual_type, expected_type):
                        logger.warning(f"Column '{field.name}' type mismatch: expected {expected_type}, got {actual_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
    
    def _types_compatible(self, actual: pa.DataType, expected: pa.DataType) -> bool:
        """Check if data types are compatible."""
        # String types
        if pa.types.is_string(expected) and pa.types.is_string(actual):
            return True
        
        # Numeric types - allow flexibility
        if pa.types.is_integer(expected) and pa.types.is_integer(actual):
            return True
            
        if pa.types.is_floating(expected) and (pa.types.is_floating(actual) or pa.types.is_integer(actual)):
            return True
            
        # Timestamp types
        if pa.types.is_timestamp(expected) and pa.types.is_timestamp(actual):
            return True
            
        # Exact match
        return actual == expected
    
    def validate_data_quality(self, df: pd.DataFrame, station_ids: List[int]) -> DataQualityMetrics:
        """Perform comprehensive data quality validation."""
        logger.info(f"Validating data quality for {len(df)} records")
        
        total_records = len(df)
        
        if total_records == 0:
            return DataQualityMetrics(
                total_records=0,
                valid_records=0,
                missing_values=0,
                duplicate_records=0,
                outlier_records=0,
                date_range=("", ""),
                stations_covered=[],
                data_completeness=0.0,
                quality_score=0.0
            )
        
        # Check for missing values in critical columns
        critical_columns = ['station_id', 'value', 'datetime_utc', 'parameter']
        missing_values = df[critical_columns].isnull().sum().sum()
        
        # Check for duplicates
        duplicate_mask = df.duplicated(subset=['station_id', 'datetime_utc', 'parameter'])
        duplicate_records = duplicate_mask.sum()
        
        # Check for outliers in PM2.5 values
        pm25_data = df[df['parameter'] == 'pm25']
        outlier_records = 0
        
        if not pm25_data.empty:
            # PM2.5 outliers: negative values or extremely high values (>1000 µg/m³)
            outlier_mask = (pm25_data['value'] < 0) | (pm25_data['value'] > 1000)
            outlier_records = outlier_mask.sum()
        
        # Valid records (excluding those with critical missing values or outliers)
        valid_mask = ~df[critical_columns].isnull().any(axis=1) & ~duplicate_mask
        if not pm25_data.empty:
            pm25_outlier_mask = df.index.isin(pm25_data[outlier_mask].index)
            valid_mask = valid_mask & ~pm25_outlier_mask
        
        valid_records = valid_mask.sum()
        
        # Date range
        if 'datetime_utc' in df.columns and not df['datetime_utc'].empty:
            min_date = df['datetime_utc'].min().isoformat()
            max_date = df['datetime_utc'].max().isoformat()
            date_range = (min_date, max_date)
        else:
            date_range = ("", "")
        
        # Stations covered
        stations_covered = df['station_id'].unique().tolist() if 'station_id' in df.columns else []
        
        # Data completeness
        data_completeness = valid_records / total_records if total_records > 0 else 0.0
        
        # Quality score calculation
        quality_score = self._calculate_quality_score(
            data_completeness,
            missing_values / total_records if total_records > 0 else 1.0,
            duplicate_records / total_records if total_records > 0 else 0.0,
            outlier_records / total_records if total_records > 0 else 0.0
        )
        
        metrics = DataQualityMetrics(
            total_records=total_records,
            valid_records=valid_records,
            missing_values=missing_values,
            duplicate_records=duplicate_records,
            outlier_records=outlier_records,
            date_range=date_range,
            stations_covered=stations_covered,
            data_completeness=data_completeness,
            quality_score=quality_score
        )
        
        logger.info(f"Data quality validation complete: {quality_score:.2f} quality score")
        return metrics
    
    def _calculate_quality_score(self, completeness: float, missing_rate: float, 
                                duplicate_rate: float, outlier_rate: float) -> float:
        """Calculate overall data quality score (0-1)."""
        # Weighted scoring
        completeness_weight = 0.4
        missing_penalty_weight = 0.3
        duplicate_penalty_weight = 0.2
        outlier_penalty_weight = 0.1
        
        score = (
            completeness * completeness_weight +
            (1 - missing_rate) * missing_penalty_weight +
            (1 - duplicate_rate) * duplicate_penalty_weight +
            (1 - outlier_rate) * outlier_penalty_weight
        )
        
        return min(max(score, 0.0), 1.0)
    
    def validate_parquet_file(self, file_path: str) -> bool:
        """Validate a Parquet file for schema and basic quality."""
        try:
            # Read file metadata without loading all data
            parquet_file = pq.ParquetFile(file_path)
            
            # Check schema
            schema_valid = True
            try:
                table = parquet_file.read()
                df = table.to_pandas()
                schema_valid = self.validate_schema(df)
            except Exception as e:
                logger.error(f"Failed to read Parquet file {file_path}: {e}")
                return False
            
            # Basic checks
            if parquet_file.metadata.num_rows == 0:
                logger.warning(f"Parquet file {file_path} is empty")
                return False
            
            logger.info(f"Parquet file {file_path} validation: {'passed' if schema_valid else 'failed'}")
            return schema_valid
            
        except Exception as e:
            logger.error(f"Parquet file validation failed for {file_path}: {e}")
            return False
    
    def generate_quality_report(self, metrics: DataQualityMetrics) -> str:
        """Generate a human-readable quality report."""
        report = f"""
OpenAQ Data Quality Report
==========================

Dataset Overview:
- Total Records: {metrics.total_records:,}
- Valid Records: {metrics.valid_records:,}
- Data Completeness: {metrics.data_completeness:.1%}
- Overall Quality Score: {metrics.quality_score:.2f}/1.00

Data Issues:
- Missing Values: {metrics.missing_values:,}
- Duplicate Records: {metrics.duplicate_records:,}  
- Outlier Records: {metrics.outlier_records:,}

Coverage:
- Date Range: {metrics.date_range[0]} to {metrics.date_range[1]}
- Stations Covered: {len(metrics.stations_covered)} stations
- Station IDs: {', '.join(map(str, metrics.stations_covered[:10]))}{'...' if len(metrics.stations_covered) > 10 else ''}

Quality Assessment:
{'✅ EXCELLENT' if metrics.quality_score >= 0.9 else '✅ GOOD' if metrics.quality_score >= 0.7 else '⚠️ FAIR' if metrics.quality_score >= 0.5 else '❌ POOR'} - Quality score: {metrics.quality_score:.2f}
"""
        return report


