"""Data quality assessment tools for AirAware feasibility analysis."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DataQualityMetrics(BaseModel):
    """Data quality metrics for a dataset."""
    
    file_path: str
    file_size_mb: float
    row_count: int = 0
    column_count: int = 0
    
    # Completeness metrics
    null_percentage: float = 0.0
    duplicate_percentage: float = 0.0
    
    # Temporal metrics
    date_range_days: int = 0
    temporal_gaps: int = 0
    
    # Value quality metrics
    outlier_percentage: float = 0.0
    negative_values: int = 0
    
    # Overall scores
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    temporal_score: float = 0.0
    overall_quality: float = 0.0
    
    # Issues found
    quality_issues: List[str] = Field(default_factory=list)


class DataQualityAssessor:
    """Assess data quality for OpenAQ and meteorological datasets."""
    
    def __init__(self, data_root: Path = Path("data")) -> None:
        """Initialize data quality assessor.
        
        Args:
            data_root: Root directory containing data files
        """
        self.data_root = data_root
        
    def assess_parquet_file(self, file_path: Path, sample_size: int = 10000) -> DataQualityMetrics:
        """Assess quality of a single parquet file.
        
        Args:
            file_path: Path to parquet file
            sample_size: Maximum rows to sample for analysis
            
        Returns:
            Data quality metrics
        """
        try:
            # Basic file info
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Load data (sample if large)
            df = pd.read_parquet(file_path)
            original_rows = len(df)
            
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"Sampling {sample_size} rows from {original_rows} total")
            
            # Initialize metrics
            metrics = DataQualityMetrics(
                file_path=str(file_path.relative_to(self.data_root)),
                file_size_mb=round(file_size_mb, 2),
                row_count=original_rows,
                column_count=len(df.columns),
            )
            
            # Completeness assessment
            null_count = df.isnull().sum().sum()
            total_cells = len(df) * len(df.columns)
            metrics.null_percentage = round((null_count / total_cells) * 100, 2) if total_cells > 0 else 0
            
            # Duplicate assessment
            if len(df) > 1:
                duplicate_count = df.duplicated().sum()
                metrics.duplicate_percentage = round((duplicate_count / len(df)) * 100, 2)
            
            # Temporal analysis (if date/time columns exist)
            date_columns = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'ts'])]
            if date_columns:
                metrics = self._assess_temporal_quality(df, date_columns[0], metrics)
            
            # Value quality for numeric columns
            numeric_columns = df.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 0:
                metrics = self._assess_value_quality(df, numeric_columns, metrics)
            
            # Calculate overall scores
            metrics = self._calculate_quality_scores(metrics)
            
            logger.info(f"Quality assessment complete for {file_path.name}: {metrics.overall_quality:.2f}/1.0")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to assess quality of {file_path}: {e}")
            return DataQualityMetrics(
                file_path=str(file_path),
                file_size_mb=0.0,
                quality_issues=[f"Assessment failed: {e}"]
            )
    
    def _assess_temporal_quality(
        self, 
        df: pd.DataFrame, 
        date_column: str, 
        metrics: DataQualityMetrics
    ) -> DataQualityMetrics:
        """Assess temporal data quality."""
        try:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            
            # Remove null dates
            valid_dates = df[date_column].dropna()
            
            if len(valid_dates) < 2:
                metrics.quality_issues.append("Insufficient valid dates for temporal analysis")
                return metrics
            
            # Date range
            date_range = valid_dates.max() - valid_dates.min()
            metrics.date_range_days = date_range.days
            
            # Detect gaps (assuming hourly data)
            if len(valid_dates) > 10:
                sorted_dates = valid_dates.sort_values()
                expected_intervals = len(sorted_dates) - 1
                
                # Calculate actual time differences
                time_diffs = sorted_dates.diff().dropna()
                
                # Count significant gaps (> 2 hours for hourly data)
                expected_hour = timedelta(hours=1)
                gap_threshold = timedelta(hours=2)
                
                gaps = time_diffs[time_diffs > gap_threshold]
                metrics.temporal_gaps = len(gaps)
                
                if metrics.temporal_gaps > expected_intervals * 0.1:  # >10% gaps
                    metrics.quality_issues.append(f"High temporal gaps: {metrics.temporal_gaps}")
            
        except Exception as e:
            metrics.quality_issues.append(f"Temporal analysis failed: {e}")
        
        return metrics
    
    def _assess_value_quality(
        self, 
        df: pd.DataFrame, 
        numeric_columns: List[str], 
        metrics: DataQualityMetrics
    ) -> DataQualityMetrics:
        """Assess numeric value quality."""
        try:
            for col in numeric_columns:
                values = df[col].dropna()
                
                if len(values) == 0:
                    continue
                
                # Check for negative values (problematic for measurements like PM2.5)
                if 'pm' in col.lower() or 'concentration' in col.lower():
                    negative_count = (values < 0).sum()
                    metrics.negative_values += negative_count
                    
                    if negative_count > 0:
                        metrics.quality_issues.append(f"Negative values in {col}: {negative_count}")
                
                # Outlier detection using IQR method
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = values[(values < lower_bound) | (values > upper_bound)]
                    outlier_pct = (len(outliers) / len(values)) * 100
                    
                    metrics.outlier_percentage = max(metrics.outlier_percentage, outlier_pct)
                    
                    if outlier_pct > 5:  # >5% outliers
                        metrics.quality_issues.append(f"High outliers in {col}: {outlier_pct:.1f}%")
        
        except Exception as e:
            metrics.quality_issues.append(f"Value analysis failed: {e}")
        
        return metrics
    
    def _calculate_quality_scores(self, metrics: DataQualityMetrics) -> DataQualityMetrics:
        """Calculate overall quality scores."""
        
        # Completeness score (based on null percentage)
        metrics.completeness_score = max(0, 1.0 - (metrics.null_percentage / 100))
        
        # Consistency score (based on duplicates and outliers)
        duplicate_penalty = metrics.duplicate_percentage / 100
        outlier_penalty = min(1.0, metrics.outlier_percentage / 20)  # Cap at 20%
        metrics.consistency_score = max(0, 1.0 - duplicate_penalty - outlier_penalty)
        
        # Temporal score (based on gaps and range)
        if metrics.date_range_days > 0:
            # Prefer longer date ranges and fewer gaps
            range_score = min(1.0, metrics.date_range_days / 30)  # Normalize by 30 days
            gap_penalty = min(1.0, metrics.temporal_gaps / 100)  # Cap at 100 gaps
            metrics.temporal_score = max(0, range_score - gap_penalty)
        else:
            metrics.temporal_score = 0.0
        
        # Overall quality (weighted average)
        weights = [0.4, 0.3, 0.3]  # completeness, consistency, temporal
        scores = [metrics.completeness_score, metrics.consistency_score, metrics.temporal_score]
        
        metrics.overall_quality = sum(w * s for w, s in zip(weights, scores))
        
        # Add quality classification
        if metrics.overall_quality >= 0.8:
            pass  # Excellent quality
        elif metrics.overall_quality >= 0.6:
            metrics.quality_issues.append("Good quality with minor issues")
        elif metrics.overall_quality >= 0.4:
            metrics.quality_issues.append("Fair quality - review recommended")
        else:
            metrics.quality_issues.append("Poor quality - significant issues detected")
        
        return metrics
    
    def assess_directory(self, directory: Path, file_pattern: str = "*.parquet") -> List[DataQualityMetrics]:
        """Assess quality of all matching files in a directory.
        
        Args:
            directory: Directory to scan
            file_pattern: File pattern to match
            
        Returns:
            List of quality metrics for each file
        """
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []
        
        files = list(directory.glob(file_pattern))
        if not files:
            logger.warning(f"No {file_pattern} files found in {directory}")
            return []
        
        logger.info(f"Assessing quality of {len(files)} files in {directory}")
        
        results = []
        for file_path in files:
            metrics = self.assess_parquet_file(file_path)
            results.append(metrics)
        
        return results
    
    def generate_quality_report(self, metrics_list: List[DataQualityMetrics]) -> Dict[str, any]:
        """Generate summary quality report.
        
        Args:
            metrics_list: List of quality metrics
            
        Returns:
            Summary report dictionary
        """
        if not metrics_list:
            return {"error": "No metrics provided"}
        
        # Overall statistics
        total_files = len(metrics_list)
        total_size_mb = sum(m.file_size_mb for m in metrics_list)
        total_rows = sum(m.row_count for m in metrics_list)
        
        # Quality scores
        quality_scores = [m.overall_quality for m in metrics_list if m.overall_quality > 0]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Issue analysis
        all_issues = []
        for metrics in metrics_list:
            all_issues.extend(metrics.quality_issues)
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # File quality distribution
        excellent_files = sum(1 for m in metrics_list if m.overall_quality >= 0.8)
        good_files = sum(1 for m in metrics_list if 0.6 <= m.overall_quality < 0.8)
        fair_files = sum(1 for m in metrics_list if 0.4 <= m.overall_quality < 0.6)
        poor_files = sum(1 for m in metrics_list if m.overall_quality < 0.4)
        
        return {
            "summary": {
                "total_files": total_files,
                "total_size_mb": round(total_size_mb, 2),
                "total_rows": total_rows,
                "average_quality": round(avg_quality, 3),
            },
            "quality_distribution": {
                "excellent": excellent_files,
                "good": good_files,
                "fair": fair_files,
                "poor": poor_files,
            },
            "common_issues": [
                {"issue": issue, "count": count}
                for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ],
            "recommendations": self._generate_recommendations(metrics_list),
        }
    
    def _generate_recommendations(self, metrics_list: List[DataQualityMetrics]) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []
        
        # Check for common issues
        high_null_files = [m for m in metrics_list if m.null_percentage > 20]
        if high_null_files:
            recommendations.append(f"Review data completeness in {len(high_null_files)} files with >20% missing values")
        
        high_duplicate_files = [m for m in metrics_list if m.duplicate_percentage > 10]
        if high_duplicate_files:
            recommendations.append(f"Remove duplicates from {len(high_duplicate_files)} files")
        
        temporal_gap_files = [m for m in metrics_list if m.temporal_gaps > 50]
        if temporal_gap_files:
            recommendations.append(f"Investigate temporal gaps in {len(temporal_gap_files)} files")
        
        negative_value_files = [m for m in metrics_list if m.negative_values > 0]
        if negative_value_files:
            recommendations.append(f"Clean negative measurement values in {len(negative_value_files)} files")
        
        low_quality_files = [m for m in metrics_list if m.overall_quality < 0.6]
        if low_quality_files:
            recommendations.append(f"Consider re-processing {len(low_quality_files)} low-quality files")
        
        if not recommendations:
            recommendations.append("Data quality is good - no major issues detected")
        
        return recommendations


