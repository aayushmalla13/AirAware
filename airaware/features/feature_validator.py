"""Feature validation and quality assessment for ML pipelines."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

logger = logging.getLogger(__name__)


class FeatureQualityMetrics(BaseModel):
    """Comprehensive feature quality metrics."""
    total_features: int
    numeric_features: int
    categorical_features: int
    missing_values_rate: float = Field(ge=0, le=1)
    infinite_values_count: int
    constant_features: int
    duplicate_features: int
    highly_correlated_pairs: int
    data_quality_score: float = Field(ge=0, le=1)
    feature_statistics: Dict = Field(default_factory=dict)
    quality_issues: List[str] = Field(default_factory=list)


class FeatureValidator:
    """Comprehensive feature validation for ML pipelines."""
    
    def __init__(self, 
                 correlation_threshold: float = 0.95,
                 missing_threshold: float = 0.5,
                 variance_threshold: float = 0.001):
        
        self.correlation_threshold = correlation_threshold
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        
        logger.info("FeatureValidator initialized")
    
    def validate_features(self, df: pd.DataFrame, 
                         target_col: str = 'pm25') -> FeatureQualityMetrics:
        """Perform comprehensive feature validation."""
        
        logger.info(f"Validating features for {len(df):,} records")
        
        # Get feature columns (exclude target and metadata)
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime_utc', 'station_id', target_col]]
        
        if not feature_cols:
            logger.warning("No feature columns found for validation")
            return FeatureQualityMetrics(
                total_features=0, numeric_features=0, categorical_features=0,
                missing_values_rate=0.0, infinite_values_count=0,
                constant_features=0, duplicate_features=0, highly_correlated_pairs=0,
                data_quality_score=0.0, quality_issues=["No features to validate"]
            )
        
        # Analyze feature types
        numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df[feature_cols].select_dtypes(exclude=['number']).columns.tolist()
        
        # Run validation checks
        quality_issues = []
        
        # 1. Missing values analysis
        missing_analysis = self._analyze_missing_values(df[feature_cols])
        overall_missing_rate = missing_analysis["overall_rate"]
        
        if overall_missing_rate > 0.1:  # >10% missing
            quality_issues.append(f"High missing values rate: {overall_missing_rate:.1%}")
        
        # 2. Infinite values check
        infinite_count = self._count_infinite_values(df[numeric_cols])
        
        if infinite_count > 0:
            quality_issues.append(f"Found {infinite_count} infinite values")
        
        # 3. Constant features check
        constant_features = self._find_constant_features(df[feature_cols])
        
        if constant_features:
            quality_issues.append(f"Found {len(constant_features)} constant features")
        
        # 4. Duplicate features check
        duplicate_features = self._find_duplicate_features(df[feature_cols])
        
        if duplicate_features:
            quality_issues.append(f"Found {len(duplicate_features)} duplicate feature pairs")
        
        # 5. High correlation check
        high_corr_pairs = self._find_highly_correlated_features(df[numeric_cols])
        
        if high_corr_pairs:
            quality_issues.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
        
        # 6. Low variance check
        low_variance_features = self._find_low_variance_features(df[numeric_cols])
        
        if low_variance_features:
            quality_issues.append(f"Found {len(low_variance_features)} low variance features")
        
        # Calculate feature statistics
        feature_statistics = self._calculate_feature_statistics(df[feature_cols], target_col, df)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            overall_missing_rate, infinite_count, len(constant_features),
            len(duplicate_features), len(high_corr_pairs), len(feature_cols)
        )
        
        logger.info(f"Feature validation complete - Quality score: {quality_score:.1%}")
        
        return FeatureQualityMetrics(
            total_features=len(feature_cols),
            numeric_features=len(numeric_cols),
            categorical_features=len(categorical_cols),
            missing_values_rate=overall_missing_rate,
            infinite_values_count=infinite_count,
            constant_features=len(constant_features),
            duplicate_features=len(duplicate_features),
            highly_correlated_pairs=len(high_corr_pairs),
            data_quality_score=quality_score,
            feature_statistics=feature_statistics,
            quality_issues=quality_issues
        )
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """Analyze missing values patterns."""
        
        missing_analysis = {
            "overall_rate": df.isnull().sum().sum() / (len(df) * len(df.columns)),
            "by_column": {},
            "problematic_columns": []
        }
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_rate = missing_count / len(df)
            
            missing_analysis["by_column"][col] = {
                "count": int(missing_count),
                "rate": float(missing_rate)
            }
            
            if missing_rate > self.missing_threshold:
                missing_analysis["problematic_columns"].append(col)
        
        return missing_analysis
    
    def _count_infinite_values(self, df: pd.DataFrame) -> int:
        """Count infinite values in numeric columns."""
        
        if df.empty:
            return 0
        
        infinite_count = 0
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                infinite_count += np.isinf(df[col]).sum()
        
        return int(infinite_count)
    
    def _find_constant_features(self, df: pd.DataFrame) -> List[str]:
        """Find features with constant values."""
        
        constant_features = []
        
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_features.append(col)
        
        return constant_features
    
    def _find_duplicate_features(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Find duplicate features (identical values)."""
        
        duplicate_pairs = []
        
        # Only check numeric columns for exact duplicates
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Check if columns are identical
                if df[col1].equals(df[col2]):
                    duplicate_pairs.append((col1, col2))
        
        return duplicate_pairs
    
    def _find_highly_correlated_features(self, df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Find highly correlated feature pairs."""
        
        if df.empty or len(df.columns) < 2:
            return []
        
        # Calculate correlation matrix
        corr_matrix = df.corr().abs()
        
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlation = corr_matrix.iloc[i, j]
                
                if correlation > self.correlation_threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        float(correlation)
                    ))
        
        return high_corr_pairs
    
    def _find_low_variance_features(self, df: pd.DataFrame) -> List[str]:
        """Find features with very low variance."""
        
        low_variance_features = []
        
        for col in df.columns:
            if df[col].dtype in ['number']:
                variance = df[col].var()
                
                if variance < self.variance_threshold:
                    low_variance_features.append(col)
        
        return low_variance_features
    
    def _calculate_feature_statistics(self, feature_df: pd.DataFrame, 
                                    target_col: str, full_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive feature statistics."""
        
        statistics = {
            "numeric_stats": {},
            "categorical_stats": {},
            "target_correlations": {},
            "distribution_tests": {}
        }
        
        # Numeric feature statistics
        numeric_cols = feature_df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            col_data = feature_df[col].dropna()
            
            if len(col_data) > 0:
                statistics["numeric_stats"][col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "q25": float(col_data.quantile(0.25)),
                    "median": float(col_data.median()),
                    "q75": float(col_data.quantile(0.75)),
                    "skewness": float(stats.skew(col_data)),
                    "kurtosis": float(stats.kurtosis(col_data))
                }
                
                # Target correlation (if target exists)
                if target_col in full_df.columns:
                    target_data = full_df[target_col].dropna()
                    
                    # Align data for correlation calculation
                    aligned_data = pd.concat([col_data, target_data], axis=1, join='inner').dropna()
                    
                    if len(aligned_data) > 1:
                        correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                        statistics["target_correlations"][col] = float(correlation) if not np.isnan(correlation) else 0.0
                
                # Distribution normality test (for reasonable sample sizes)
                if 20 <= len(col_data) <= 5000:
                    try:
                        statistic, p_value = stats.shapiro(col_data.sample(min(len(col_data), 1000)))
                        statistics["distribution_tests"][col] = {
                            "test": "shapiro",
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "is_normal": p_value > 0.05
                        }
                    except Exception:
                        pass  # Skip if test fails
        
        # Categorical feature statistics
        categorical_cols = feature_df.select_dtypes(exclude=['number']).columns
        
        for col in categorical_cols:
            col_data = feature_df[col].dropna()
            
            if len(col_data) > 0:
                value_counts = col_data.value_counts()
                
                statistics["categorical_stats"][col] = {
                    "unique_values": int(col_data.nunique()),
                    "mode": str(col_data.mode().iloc[0]) if not col_data.mode().empty else "unknown",
                    "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "mode_percentage": float(value_counts.iloc[0] / len(col_data)) if len(value_counts) > 0 else 0.0,
                    "entropy": float(-sum((p := value_counts / len(col_data)) * np.log2(p + 1e-10)))
                }
        
        return statistics
    
    def _calculate_quality_score(self, missing_rate: float, infinite_count: int,
                               constant_features: int, duplicate_features: int,
                               high_corr_pairs: int, total_features: int) -> float:
        """Calculate overall feature quality score."""
        
        if total_features == 0:
            return 0.0
        
        # Start with perfect score
        score = 1.0
        
        # Penalty for missing values (0-50% penalty)
        score -= min(0.5, missing_rate * 2)
        
        # Penalty for infinite values (5% per infinite value, max 20%)
        if infinite_count > 0:
            score -= min(0.2, infinite_count * 0.05)
        
        # Penalty for constant features (2% per constant feature, max 20%)
        if constant_features > 0:
            score -= min(0.2, constant_features / total_features * 0.5)
        
        # Penalty for duplicate features (1% per duplicate pair, max 10%)
        if duplicate_features > 0:
            score -= min(0.1, duplicate_features / total_features * 0.3)
        
        # Penalty for highly correlated features (1% per pair, max 15%)
        if high_corr_pairs > 0:
            score -= min(0.15, high_corr_pairs / total_features * 0.2)
        
        return max(0.0, min(1.0, score))
    
    def generate_validation_report(self, metrics: FeatureQualityMetrics) -> str:
        """Generate human-readable validation report."""
        
        report = f"""
🔍 Feature Validation Report
=============================

📊 Feature Overview:
- Total Features: {metrics.total_features}
- Numeric Features: {metrics.numeric_features}
- Categorical Features: {metrics.categorical_features}

📈 Data Quality Score: {metrics.data_quality_score:.1%}

⚠️ Quality Issues:
"""
        
        if metrics.quality_issues:
            for issue in metrics.quality_issues:
                report += f"- {issue}\n"
        else:
            report += "- No major quality issues detected ✅\n"
        
        report += f"""
📋 Detailed Analysis:
- Missing Values Rate: {metrics.missing_values_rate:.1%}
- Infinite Values: {metrics.infinite_values_count}
- Constant Features: {metrics.constant_features}
- Duplicate Features: {metrics.duplicate_features}
- Highly Correlated Pairs: {metrics.highly_correlated_pairs}

💡 Recommendations:
"""
        
        # Add recommendations based on findings
        if metrics.missing_values_rate > 0.1:
            report += "- Consider imputation strategies for missing values\n"
        
        if metrics.constant_features > 0:
            report += "- Remove constant features as they provide no predictive value\n"
        
        if metrics.duplicate_features > 0:
            report += "- Remove duplicate features to reduce dimensionality\n"
        
        if metrics.highly_correlated_pairs > 5:
            report += "- Consider feature selection to reduce multicollinearity\n"
        
        if metrics.infinite_values_count > 0:
            report += "- Handle infinite values before model training\n"
        
        if not any([metrics.missing_values_rate > 0.1, metrics.constant_features > 0,
                   metrics.duplicate_features > 0, metrics.highly_correlated_pairs > 5,
                   metrics.infinite_values_count > 0]):
            report += "- Feature set is well-prepared for model training ✅\n"
        
        return report
    
    def get_feature_recommendations(self, metrics: FeatureQualityMetrics) -> Dict[str, List[str]]:
        """Get specific feature engineering recommendations."""
        
        recommendations = {
            "remove": [],
            "transform": [],
            "impute": [],
            "investigate": []
        }
        
        # Analyze feature statistics for recommendations
        for feature, stats in metrics.feature_statistics.get("numeric_stats", {}).items():
            
            # High skewness suggests transformation
            if abs(stats.get("skewness", 0)) > 2:
                recommendations["transform"].append(f"{feature} (high skewness: {stats['skewness']:.2f})")
            
            # High kurtosis suggests outliers
            if abs(stats.get("kurtosis", 0)) > 3:
                recommendations["investigate"].append(f"{feature} (high kurtosis: {stats['kurtosis']:.2f})")
        
        # Missing value recommendations
        missing_stats = metrics.feature_statistics.get("missing_analysis", {}).get("by_column", {})
        
        for feature, missing_info in missing_stats.items():
            missing_rate = missing_info.get("rate", 0)
            
            if missing_rate > 0.5:
                recommendations["remove"].append(f"{feature} (missing: {missing_rate:.1%})")
            elif missing_rate > 0.1:
                recommendations["impute"].append(f"{feature} (missing: {missing_rate:.1%})")
        
        return recommendations


