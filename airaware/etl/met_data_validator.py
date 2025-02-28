"""Advanced data quality validation for meteorological data (ERA5/IMERG)."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MeteorologicalQualityMetrics(BaseModel):
    """Quality metrics for meteorological data."""
    total_records: int
    temporal_gaps: int
    missing_values: int
    outlier_records: int
    date_range: Tuple[str, str]
    variables_validated: List[str]
    temporal_consistency: float = Field(ge=0, le=1)
    spatial_consistency: float = Field(ge=0, le=1)
    physical_realism: float = Field(ge=0, le=1)
    overall_quality_score: float = Field(ge=0, le=1)


class MeteorologicalDataValidator:
    """Advanced validator for ERA5 and IMERG meteorological data."""
    
    # Physical bounds for meteorological variables
    VARIABLE_BOUNDS = {
        # Wind components (m/s) - reasonable bounds for surface winds
        "u10": (-50, 50),
        "v10": (-50, 50),
        "wind_speed": (0, 50),
        "wind_direction": (0, 360),
        
        # Temperature (Celsius) - reasonable bounds for Earth's surface
        "t2m_celsius": (-60, 60),
        "t2m": (200, 330),  # Kelvin
        
        # Boundary layer height (meters)
        "blh": (10, 5000),
        
        # Precipitation (mm/hour) - reasonable hourly precipitation
        "precipitation_mm_hourly": (0, 100),
        "precipitation_mm_30min": (0, 50),
    }
    
    def __init__(self):
        self.validation_results = []
    
    def validate_era5_data(self, df: pd.DataFrame) -> MeteorologicalQualityMetrics:
        """Validate ERA5 meteorological data quality."""
        logger.info(f"Validating ERA5 data quality for {len(df)} records")
        
        if df.empty:
            return self._create_empty_metrics("ERA5")
        
        # Required ERA5 variables
        era5_variables = ["u10", "v10", "t2m_celsius", "blh", "wind_speed", "wind_direction"]
        missing_vars = [var for var in era5_variables if var not in df.columns]
        
        if missing_vars:
            logger.warning(f"Missing ERA5 variables: {missing_vars}")
        
        available_vars = [var for var in era5_variables if var in df.columns]
        
        # Validate temporal consistency
        temporal_score = self._validate_temporal_consistency(df)
        
        # Validate physical realism
        physical_score = self._validate_physical_bounds(df, available_vars)
        
        # Validate spatial consistency (for derived variables)
        spatial_score = self._validate_spatial_consistency(df)
        
        # Count issues
        total_records = len(df)
        missing_values = df[available_vars].isnull().sum().sum()
        outlier_records = self._count_outliers(df, available_vars)
        temporal_gaps = self._count_temporal_gaps(df)
        
        # Calculate overall quality score
        overall_score = (temporal_score + physical_score + spatial_score) / 3
        
        # Create date range
        date_range = (
            df['datetime_utc'].min().strftime('%Y-%m-%d %H:%M'),
            df['datetime_utc'].max().strftime('%Y-%m-%d %H:%M')
        )
        
        return MeteorologicalQualityMetrics(
            total_records=total_records,
            temporal_gaps=temporal_gaps,
            missing_values=int(missing_values),
            outlier_records=outlier_records,
            date_range=date_range,
            variables_validated=available_vars,
            temporal_consistency=temporal_score,
            spatial_consistency=spatial_score,
            physical_realism=physical_score,
            overall_quality_score=overall_score
        )
    
    def validate_imerg_data(self, df: pd.DataFrame) -> MeteorologicalQualityMetrics:
        """Validate IMERG precipitation data quality."""
        logger.info(f"Validating IMERG data quality for {len(df)} records")
        
        if df.empty:
            return self._create_empty_metrics("IMERG")
        
        # Required IMERG variables
        imerg_variables = ["precipitation_mm_hourly"]
        missing_vars = [var for var in imerg_variables if var not in df.columns]
        
        if missing_vars:
            logger.warning(f"Missing IMERG variables: {missing_vars}")
        
        available_vars = [var for var in imerg_variables if var in df.columns]
        
        # Validate temporal consistency
        temporal_score = self._validate_temporal_consistency(df)
        
        # Validate physical realism for precipitation
        physical_score = self._validate_precipitation_physics(df)
        
        # Spatial consistency is less relevant for aggregated precipitation
        spatial_score = 1.0
        
        # Count issues
        total_records = len(df)
        missing_values = df[available_vars].isnull().sum().sum()
        outlier_records = self._count_outliers(df, available_vars)
        temporal_gaps = self._count_temporal_gaps(df)
        
        # Calculate overall quality score
        overall_score = (temporal_score + physical_score + spatial_score) / 3
        
        # Create date range
        date_range = (
            df['datetime_utc'].min().strftime('%Y-%m-%d %H:%M'),
            df['datetime_utc'].max().strftime('%Y-%m-%d %H:%M')
        )
        
        return MeteorologicalQualityMetrics(
            total_records=total_records,
            temporal_gaps=temporal_gaps,
            missing_values=int(missing_values),
            outlier_records=outlier_records,
            date_range=date_range,
            variables_validated=available_vars,
            temporal_consistency=temporal_score,
            spatial_consistency=spatial_score,
            physical_realism=physical_score,
            overall_quality_score=overall_score
        )
    
    def _create_empty_metrics(self, data_type: str) -> MeteorologicalQualityMetrics:
        """Create empty metrics for no data case."""
        return MeteorologicalQualityMetrics(
            total_records=0,
            temporal_gaps=0,
            missing_values=0,
            outlier_records=0,
            date_range=("", ""),
            variables_validated=[],
            temporal_consistency=0.0,
            spatial_consistency=0.0,
            physical_realism=0.0,
            overall_quality_score=0.0
        )
    
    def _validate_temporal_consistency(self, df: pd.DataFrame) -> float:
        """Validate temporal consistency of the time series."""
        if 'datetime_utc' not in df.columns or len(df) < 2:
            return 0.0
        
        # Sort by time
        df_sorted = df.sort_values('datetime_utc')
        
        # Calculate time differences
        time_diffs = df_sorted['datetime_utc'].diff().dropna()
        
        # Expected hourly interval
        expected_interval = pd.Timedelta(hours=1)
        
        # Count proper intervals
        proper_intervals = (time_diffs == expected_interval).sum()
        total_intervals = len(time_diffs)
        
        if total_intervals == 0:
            return 1.0
        
        consistency_score = proper_intervals / total_intervals
        
        logger.debug(f"Temporal consistency: {consistency_score:.3f}")
        return consistency_score
    
    def _validate_physical_bounds(self, df: pd.DataFrame, variables: List[str]) -> float:
        """Validate that values are within physically realistic bounds."""
        total_checks = 0
        valid_checks = 0
        
        for var in variables:
            if var not in df.columns or var not in self.VARIABLE_BOUNDS:
                continue
            
            min_bound, max_bound = self.VARIABLE_BOUNDS[var]
            values = df[var].dropna()
            
            if len(values) == 0:
                continue
            
            # Check bounds
            within_bounds = ((values >= min_bound) & (values <= max_bound)).sum()
            total_values = len(values)
            
            total_checks += total_values
            valid_checks += within_bounds
        
        if total_checks == 0:
            return 1.0
        
        physical_score = valid_checks / total_checks
        logger.debug(f"Physical realism: {physical_score:.3f}")
        return physical_score
    
    def _validate_spatial_consistency(self, df: pd.DataFrame) -> float:
        """Validate spatial consistency using derived wind relationships."""
        if not all(col in df.columns for col in ['u10', 'v10', 'wind_speed']):
            return 1.0
        
        # Check if derived wind speed matches u10/v10 components
        calculated_wind_speed = np.sqrt(df['u10']**2 + df['v10']**2)
        
        # Allow small numerical differences (1% tolerance)
        wind_speed_diff = np.abs(calculated_wind_speed - df['wind_speed'])
        tolerance = 0.01 * df['wind_speed'].abs()
        
        consistent_records = (wind_speed_diff <= tolerance).sum()
        total_records = len(df)
        
        if total_records == 0:
            return 1.0
        
        spatial_score = consistent_records / total_records
        logger.debug(f"Spatial consistency: {spatial_score:.3f}")
        return spatial_score
    
    def _validate_precipitation_physics(self, df: pd.DataFrame) -> float:
        """Validate precipitation-specific physics."""
        if 'precipitation_mm_hourly' not in df.columns:
            return 1.0
        
        precip_values = df['precipitation_mm_hourly'].dropna()
        
        if len(precip_values) == 0:
            return 1.0
        
        # Check for physical bounds
        min_bound, max_bound = self.VARIABLE_BOUNDS['precipitation_mm_hourly']
        within_bounds = ((precip_values >= min_bound) & (precip_values <= max_bound)).sum()
        
        # Check for reasonable statistics
        # Most hours should have low/no precipitation
        low_precip_hours = (precip_values <= 5.0).sum()  # <= 5mm/hr is reasonable
        reasonable_stats = low_precip_hours / len(precip_values) >= 0.8  # 80% of hours
        
        physical_score = within_bounds / len(precip_values)
        
        # Adjust score based on statistical reasonableness
        if not reasonable_stats:
            physical_score *= 0.8  # Penalize unrealistic precipitation patterns
        
        logger.debug(f"Precipitation physics: {physical_score:.3f}")
        return physical_score
    
    def _count_outliers(self, df: pd.DataFrame, variables: List[str]) -> int:
        """Count outlier records using IQR method."""
        outlier_count = 0
        
        for var in variables:
            if var not in df.columns:
                continue
            
            values = df[var].dropna()
            
            if len(values) < 4:  # Need at least 4 values for IQR
                continue
            
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            # Outliers are beyond 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((values < lower_bound) | (values > upper_bound)).sum()
            outlier_count += outliers
        
        return outlier_count
    
    def _count_temporal_gaps(self, df: pd.DataFrame) -> int:
        """Count temporal gaps larger than expected interval."""
        if 'datetime_utc' not in df.columns or len(df) < 2:
            return 0
        
        # Sort by time
        df_sorted = df.sort_values('datetime_utc')
        
        # Calculate time differences
        time_diffs = df_sorted['datetime_utc'].diff().dropna()
        
        # Expected hourly interval
        expected_interval = pd.Timedelta(hours=1)
        
        # Count gaps larger than expected
        gaps = (time_diffs > expected_interval).sum()
        
        return gaps
    
    def generate_quality_report(self, metrics: MeteorologicalQualityMetrics, data_type: str) -> str:
        """Generate human-readable quality report."""
        report = f"""
{data_type} Data Quality Report
{'=' * (len(data_type) + 19)}

Summary:
- Total Records: {metrics.total_records:,}
- Date Range: {metrics.date_range[0]} to {metrics.date_range[1]}
- Variables: {', '.join(metrics.variables_validated)}

Quality Scores:
- Temporal Consistency: {metrics.temporal_consistency:.1%}
- Spatial Consistency: {metrics.spatial_consistency:.1%}
- Physical Realism: {metrics.physical_realism:.1%}
- Overall Quality: {metrics.overall_quality_score:.1%}

Issues Detected:
- Missing Values: {metrics.missing_values:,}
- Temporal Gaps: {metrics.temporal_gaps}
- Outlier Records: {metrics.outlier_records}

Quality Assessment: {"EXCELLENT" if metrics.overall_quality_score > 0.9 else "GOOD" if metrics.overall_quality_score > 0.7 else "POOR"}
"""
        
        # Add recommendations based on quality score
        if metrics.overall_quality_score < 0.7:
            report += "\nRecommendations:\n"
            if metrics.temporal_consistency < 0.8:
                report += "- Review temporal gaps and data collection frequency\n"
            if metrics.physical_realism < 0.8:
                report += "- Investigate physically unrealistic values\n"
            if metrics.spatial_consistency < 0.8:
                report += "- Check spatial processing and derived variable calculations\n"
        
        return report


