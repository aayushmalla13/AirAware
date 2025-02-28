"""Advanced data completeness analysis and gap detection for meteorological data."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DataGap(BaseModel):
    """Represents a gap in time series data."""
    start_time: datetime
    end_time: datetime
    duration_hours: float
    expected_records: int
    gap_type: str  # "short", "medium", "long", "critical"
    variables_affected: List[str]
    impact_severity: str  # "low", "medium", "high", "critical"


class CompletenessMetrics(BaseModel):
    """Comprehensive data completeness metrics."""
    total_expected_records: int
    total_actual_records: int
    completeness_percentage: float = Field(ge=0, le=100)
    gaps_detected: int
    longest_gap_hours: float
    total_gap_hours: float
    variables_analyzed: List[str]
    data_quality_grade: str  # A, B, C, D, F
    gaps_by_severity: Dict[str, int] = Field(default_factory=dict)


class DataCompletenessAnalyzer:
    """Advanced analyzer for meteorological data completeness."""
    
    # Gap severity thresholds (hours)
    GAP_THRESHOLDS = {
        "short": 6,      # <= 6 hours
        "medium": 24,    # 6-24 hours  
        "long": 72,      # 24-72 hours
        "critical": 168  # > 72 hours (3+ days)
    }
    
    def __init__(self, expected_frequency: str = "1H"):
        """Initialize completeness analyzer.
        
        Args:
            expected_frequency: Expected data frequency (e.g., "1H", "30min")
        """
        self.expected_frequency = expected_frequency
        self.frequency_minutes = self._parse_frequency(expected_frequency)
    
    def analyze_era5_completeness(self, df: pd.DataFrame, 
                                 start_date: datetime, 
                                 end_date: datetime) -> CompletenessMetrics:
        """Analyze completeness of ERA5 meteorological data."""
        logger.info(f"Analyzing ERA5 completeness from {start_date} to {end_date}")
        
        # ERA5 variables to check
        era5_variables = ["u10", "v10", "t2m", "blh", "wind_speed", "wind_direction", "t2m_celsius"]
        available_vars = [var for var in era5_variables if var in df.columns]
        
        return self._analyze_completeness(df, start_date, end_date, available_vars, "ERA5")
    
    def analyze_imerg_completeness(self, df: pd.DataFrame,
                                  start_date: datetime,
                                  end_date: datetime) -> CompletenessMetrics:
        """Analyze completeness of IMERG precipitation data."""
        logger.info(f"Analyzing IMERG completeness from {start_date} to {end_date}")
        
        # IMERG variables to check
        imerg_variables = ["precipitation_mm_hourly"]
        available_vars = [var for var in imerg_variables if var in df.columns]
        
        return self._analyze_completeness(df, start_date, end_date, available_vars, "IMERG")
    
    def _analyze_completeness(self, df: pd.DataFrame, start_date: datetime,
                             end_date: datetime, variables: List[str],
                             data_type: str) -> CompletenessMetrics:
        """Core completeness analysis logic."""
        
        # Create expected time index
        expected_times = pd.date_range(
            start=start_date,
            end=end_date,
            freq=self.expected_frequency
        )
        
        total_expected = len(expected_times)
        
        if df.empty or 'datetime_utc' not in df.columns:
            return self._create_empty_metrics(total_expected, variables)
        
        # Ensure datetime column is properly typed
        df = df.copy()
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        
        # Find actual records in expected time range
        mask = (df['datetime_utc'] >= start_date) & (df['datetime_utc'] <= end_date)
        actual_df = df[mask].copy()
        
        total_actual = len(actual_df)
        completeness_pct = (total_actual / total_expected * 100) if total_expected > 0 else 0
        
        # Detect gaps
        gaps = self._detect_gaps(actual_df['datetime_utc'], expected_times, variables)
        
        # Calculate gap statistics
        total_gap_hours = sum(gap.duration_hours for gap in gaps)
        longest_gap_hours = max((gap.duration_hours for gap in gaps), default=0)
        
        # Categorize gaps by severity
        gaps_by_severity = self._categorize_gaps_by_severity(gaps)
        
        # Assign quality grade
        quality_grade = self._assign_quality_grade(completeness_pct, gaps_by_severity)
        
        logger.info(f"{data_type} completeness: {completeness_pct:.1f}% "
                   f"({total_actual}/{total_expected} records, {len(gaps)} gaps)")
        
        return CompletenessMetrics(
            total_expected_records=total_expected,
            total_actual_records=total_actual,
            completeness_percentage=completeness_pct,
            gaps_detected=len(gaps),
            longest_gap_hours=longest_gap_hours,
            total_gap_hours=total_gap_hours,
            variables_analyzed=variables,
            data_quality_grade=quality_grade,
            gaps_by_severity=gaps_by_severity
        )
    
    def _detect_gaps(self, actual_times: pd.Series, expected_times: pd.DatetimeIndex,
                    variables: List[str]) -> List[DataGap]:
        """Detect gaps in time series data."""
        gaps = []
        
        if len(actual_times) == 0:
            # Complete data gap
            return [DataGap(
                start_time=expected_times[0],
                end_time=expected_times[-1],
                duration_hours=(expected_times[-1] - expected_times[0]).total_seconds() / 3600,
                expected_records=len(expected_times),
                gap_type="critical",
                variables_affected=variables,
                impact_severity="critical"
            )]
        
        # Sort actual times
        actual_times_sorted = actual_times.sort_values()
        
        # Convert to set for faster lookup
        actual_times_set = set(actual_times_sorted)
        
        # Find missing timestamps
        missing_times = []
        for expected_time in expected_times:
            if expected_time not in actual_times_set:
                missing_times.append(expected_time)
        
        if not missing_times:
            return gaps  # No gaps found
        
        # Group consecutive missing times into gaps
        current_gap_start = None
        current_gap_times = []
        
        for i, missing_time in enumerate(missing_times):
            if current_gap_start is None:
                # Start new gap
                current_gap_start = missing_time
                current_gap_times = [missing_time]
            else:
                # Check if this continues the current gap
                expected_next = current_gap_times[-1] + timedelta(minutes=self.frequency_minutes)
                
                if missing_time == expected_next:
                    # Continue current gap
                    current_gap_times.append(missing_time)
                else:
                    # End current gap and start new one
                    gap = self._create_gap_from_times(current_gap_start, current_gap_times[-1],
                                                    len(current_gap_times), variables)
                    gaps.append(gap)
                    
                    # Start new gap
                    current_gap_start = missing_time
                    current_gap_times = [missing_time]
        
        # Don't forget the last gap
        if current_gap_start is not None:
            gap = self._create_gap_from_times(current_gap_start, current_gap_times[-1],
                                            len(current_gap_times), variables)
            gaps.append(gap)
        
        return gaps
    
    def _create_gap_from_times(self, start_time: datetime, end_time: datetime,
                              expected_records: int, variables: List[str]) -> DataGap:
        """Create a DataGap object from start/end times."""
        duration_hours = (end_time - start_time).total_seconds() / 3600 + (self.frequency_minutes / 60)
        gap_type = self._classify_gap_type(duration_hours)
        impact_severity = self._assess_gap_impact(duration_hours, variables)
        
        return DataGap(
            start_time=start_time,
            end_time=end_time,
            duration_hours=duration_hours,
            expected_records=expected_records,
            gap_type=gap_type,
            variables_affected=variables,
            impact_severity=impact_severity
        )
    
    def _classify_gap_type(self, duration_hours: float) -> str:
        """Classify gap type based on duration."""
        if duration_hours <= self.GAP_THRESHOLDS["short"]:
            return "short"
        elif duration_hours <= self.GAP_THRESHOLDS["medium"]:
            return "medium"
        elif duration_hours <= self.GAP_THRESHOLDS["long"]:
            return "long"
        else:
            return "critical"
    
    def _assess_gap_impact(self, duration_hours: float, variables: List[str]) -> str:
        """Assess the impact severity of a gap."""
        # Base severity on duration
        if duration_hours <= 3:
            base_severity = "low"
        elif duration_hours <= 12:
            base_severity = "medium"
        elif duration_hours <= 48:
            base_severity = "high"
        else:
            base_severity = "critical"
        
        # Adjust based on variables affected
        critical_vars = ["t2m", "t2m_celsius", "precipitation_mm_hourly"]
        has_critical_vars = any(var in variables for var in critical_vars)
        
        if has_critical_vars and base_severity in ["low", "medium"]:
            # Upgrade severity for critical variables
            severity_map = {"low": "medium", "medium": "high"}
            return severity_map.get(base_severity, base_severity)
        
        return base_severity
    
    def _categorize_gaps_by_severity(self, gaps: List[DataGap]) -> Dict[str, int]:
        """Categorize gaps by severity level."""
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for gap in gaps:
            severity = gap.impact_severity
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        return severity_counts
    
    def _assign_quality_grade(self, completeness_pct: float, 
                             gaps_by_severity: Dict[str, int]) -> str:
        """Assign overall data quality grade."""
        # Base grade on completeness percentage
        if completeness_pct >= 98:
            base_grade = "A"
        elif completeness_pct >= 95:
            base_grade = "B"
        elif completeness_pct >= 90:
            base_grade = "C"
        elif completeness_pct >= 80:
            base_grade = "D"
        else:
            base_grade = "F"
        
        # Downgrade based on critical gaps
        if gaps_by_severity.get("critical", 0) > 0:
            grade_map = {"A": "C", "B": "C", "C": "D", "D": "F", "F": "F"}
            base_grade = grade_map.get(base_grade, base_grade)
        elif gaps_by_severity.get("high", 0) > 2:
            grade_map = {"A": "B", "B": "C", "C": "D", "D": "F", "F": "F"}
            base_grade = grade_map.get(base_grade, base_grade)
        
        return base_grade
    
    def _create_empty_metrics(self, total_expected: int, 
                             variables: List[str]) -> CompletenessMetrics:
        """Create metrics for empty dataset."""
        return CompletenessMetrics(
            total_expected_records=total_expected,
            total_actual_records=0,
            completeness_percentage=0.0,
            gaps_detected=1 if total_expected > 0 else 0,
            longest_gap_hours=0.0,
            total_gap_hours=0.0,
            variables_analyzed=variables,
            data_quality_grade="F",
            gaps_by_severity={"critical": 1 if total_expected > 0 else 0}
        )
    
    def _parse_frequency(self, frequency: str) -> int:
        """Parse frequency string to minutes."""
        if frequency == "1H":
            return 60
        elif frequency == "30min":
            return 30
        elif frequency == "15min":
            return 15
        else:
            # Default to hourly
            logger.warning(f"Unknown frequency {frequency}, defaulting to 60 minutes")
            return 60
    
    def generate_completeness_report(self, metrics: CompletenessMetrics, 
                                   data_type: str) -> str:
        """Generate human-readable completeness report."""
        report = f"""
{data_type} Data Completeness Report
{'=' * (len(data_type) + 30)}

Overall Summary:
- Data Quality Grade: {metrics.data_quality_grade}
- Completeness: {metrics.completeness_percentage:.1f}%
- Records: {metrics.total_actual_records:,} / {metrics.total_expected_records:,}
- Variables: {', '.join(metrics.variables_analyzed)}

Gap Analysis:
- Total Gaps: {metrics.gaps_detected}
- Total Gap Time: {metrics.total_gap_hours:.1f} hours
- Longest Gap: {metrics.longest_gap_hours:.1f} hours

Gaps by Severity:
- Low Impact: {metrics.gaps_by_severity.get('low', 0)}
- Medium Impact: {metrics.gaps_by_severity.get('medium', 0)}
- High Impact: {metrics.gaps_by_severity.get('high', 0)}
- Critical Impact: {metrics.gaps_by_severity.get('critical', 0)}

Quality Assessment: """
        
        if metrics.data_quality_grade in ["A", "B"]:
            report += "EXCELLENT - Data suitable for high-precision modeling"
        elif metrics.data_quality_grade == "C":
            report += "GOOD - Data suitable for most modeling applications"
        elif metrics.data_quality_grade == "D":
            report += "FAIR - Data may need gap filling for critical applications"
        else:
            report += "POOR - Significant data quality issues detected"
        
        # Add recommendations
        if metrics.completeness_percentage < 90:
            report += "\n\nRecommendations:"
            report += "\n- Investigate data collection issues"
            report += "\n- Consider gap filling techniques"
            
        if metrics.gaps_by_severity.get('critical', 0) > 0:
            report += "\n- Address critical data gaps immediately"
            
        return report
    
    def suggest_gap_filling_strategy(self, gaps: List[DataGap]) -> Dict[str, str]:
        """Suggest gap filling strategies based on gap characteristics."""
        strategies = {}
        
        for gap in gaps:
            gap_id = f"gap_{gap.start_time.strftime('%Y%m%d_%H%M')}"
            
            if gap.duration_hours <= 3:
                # Short gaps - linear interpolation
                strategies[gap_id] = "linear_interpolation"
            elif gap.duration_hours <= 12:
                # Medium gaps - seasonal decomposition + interpolation
                strategies[gap_id] = "seasonal_interpolation"
            elif gap.duration_hours <= 48:
                # Long gaps - historical averaging or model-based
                strategies[gap_id] = "historical_averaging"
            else:
                # Critical gaps - external data source or exclude period
                strategies[gap_id] = "external_data_source"
        
        return strategies


