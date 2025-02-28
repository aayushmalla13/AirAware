"""Advanced feature selection with statistical tests and ML-based importance."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureSelectionConfig(BaseModel):
    """Configuration for advanced feature selection."""
    statistical_tests: bool = Field(True, description="Use statistical tests for selection")
    mutual_information: bool = Field(True, description="Use mutual information")
    random_forest_importance: bool = Field(True, description="Use RF feature importance")
    correlation_analysis: bool = Field(True, description="Advanced correlation analysis")
    
    # Selection thresholds
    p_value_threshold: float = Field(0.05, description="P-value threshold for statistical tests")
    mutual_info_threshold: float = Field(0.01, description="Mutual information threshold")
    rf_importance_threshold: float = Field(0.001, description="RF importance threshold")
    
    # Selection methods
    max_features: Optional[int] = Field(None, description="Maximum number of features to select")
    selection_method: str = Field("combined", description="Selection method: statistical, ml, combined")


class FeatureImportanceResult(BaseModel):
    """Result of feature importance analysis."""
    feature_scores: Dict[str, float] = Field(default_factory=dict)
    selected_features: List[str] = Field(default_factory=list)
    method_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    selection_rationale: Dict[str, str] = Field(default_factory=dict)


class AdvancedFeatureSelector:
    """Advanced feature selection using multiple statistical and ML methods."""
    
    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        self.config = config or FeatureSelectionConfig()
        logger.info("AdvancedFeatureSelector initialized")
    
    def select_features(self, df: pd.DataFrame, 
                       target_col: str = 'pm25') -> FeatureImportanceResult:
        """Perform advanced feature selection using multiple methods."""
        
        logger.info(f"Starting advanced feature selection for {len(df):,} records")
        
        # Get feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime_utc', 'station_id', target_col]]
        
        if not feature_cols:
            logger.warning("No features found for selection")
            return FeatureImportanceResult()
        
        # Prepare data
        X, y, feature_names = self._prepare_data(df, feature_cols, target_col)
        
        if X.shape[0] == 0:
            logger.warning("No valid data for feature selection")
            return FeatureImportanceResult()
        
        # Run different selection methods
        method_scores = {}
        
        if self.config.statistical_tests:
            method_scores["statistical"] = self._statistical_selection(X, y, feature_names)
        
        if self.config.mutual_information:
            method_scores["mutual_info"] = self._mutual_info_selection(X, y, feature_names)
        
        if self.config.random_forest_importance:
            method_scores["random_forest"] = self._random_forest_selection(X, y, feature_names)
        
        if self.config.correlation_analysis:
            method_scores["correlation"] = self._correlation_selection(X, y, feature_names)
        
        # Combine scores and select features
        combined_scores = self._combine_scores(method_scores)
        selected_features = self._select_top_features(combined_scores, feature_names)
        
        # Generate selection rationale
        rationale = self._generate_selection_rationale(method_scores, selected_features)
        
        logger.info(f"Feature selection complete: {len(selected_features)}/{len(feature_names)} features selected")
        
        return FeatureImportanceResult(
            feature_scores=combined_scores,
            selected_features=selected_features,
            method_scores=method_scores,
            selection_rationale=rationale
        )
    
    def _prepare_data(self, df: pd.DataFrame, feature_cols: List[str], 
                     target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for feature selection."""
        
        # Select numeric features only for ML methods
        numeric_features = []
        for col in feature_cols:
            if df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                numeric_features.append(col)
        
        # Create feature matrix
        X = df[numeric_features].values
        y = df[target_col].values
        
        # Handle missing values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Scale features for consistency
        if X.shape[0] > 0:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        return X, y, numeric_features
    
    def _statistical_selection(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: List[str]) -> Dict[str, float]:
        """Feature selection using statistical tests."""
        
        logger.debug("Running statistical feature selection")
        
        scores = {}
        
        # F-regression test
        try:
            f_scores, p_values = f_regression(X, y)
            
            for i, (f_score, p_value) in enumerate(zip(f_scores, p_values)):
                feature_name = feature_names[i]
                # Score is inverse of p-value (lower p-value = higher importance)
                score = 1.0 - min(p_value, 1.0) if not np.isnan(p_value) else 0.0
                scores[feature_name] = score
                
        except Exception as e:
            logger.warning(f"Statistical selection failed: {e}")
        
        return scores
    
    def _mutual_info_selection(self, X: np.ndarray, y: np.ndarray,
                              feature_names: List[str]) -> Dict[str, float]:
        """Feature selection using mutual information."""
        
        logger.debug("Running mutual information feature selection")
        
        scores = {}
        
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
            for i, mi_score in enumerate(mi_scores):
                feature_name = feature_names[i]
                scores[feature_name] = float(mi_score) if not np.isnan(mi_score) else 0.0
                
        except Exception as e:
            logger.warning(f"Mutual information selection failed: {e}")
        
        return scores
    
    def _random_forest_selection(self, X: np.ndarray, y: np.ndarray,
                                feature_names: List[str]) -> Dict[str, float]:
        """Feature selection using Random Forest importance."""
        
        logger.debug("Running Random Forest feature selection")
        
        scores = {}
        
        try:
            # Use a simple Random Forest for feature importance
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(X, y)
            importances = rf.feature_importances_
            
            for i, importance in enumerate(importances):
                feature_name = feature_names[i]
                scores[feature_name] = float(importance) if not np.isnan(importance) else 0.0
                
        except Exception as e:
            logger.warning(f"Random Forest selection failed: {e}")
        
        return scores
    
    def _correlation_selection(self, X: np.ndarray, y: np.ndarray,
                              feature_names: List[str]) -> Dict[str, float]:
        """Feature selection using correlation analysis."""
        
        logger.debug("Running correlation feature selection")
        
        scores = {}
        
        try:
            for i, feature_values in enumerate(X.T):
                feature_name = feature_names[i]
                
                # Calculate Pearson correlation
                correlation, p_value = stats.pearsonr(feature_values, y)
                
                # Use absolute correlation as score
                score = abs(correlation) if not np.isnan(correlation) else 0.0
                scores[feature_name] = score
                
        except Exception as e:
            logger.warning(f"Correlation selection failed: {e}")
        
        return scores
    
    def _combine_scores(self, method_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Combine scores from different methods."""
        
        if not method_scores:
            return {}
        
        # Get all features
        all_features = set()
        for scores in method_scores.values():
            all_features.update(scores.keys())
        
        combined_scores = {}
        
        for feature in all_features:
            # Collect scores from all methods
            feature_scores = []
            
            for method, scores in method_scores.items():
                if feature in scores:
                    # Normalize scores to 0-1 range for each method
                    method_values = list(scores.values())
                    if method_values:
                        max_val = max(method_values)
                        min_val = min(method_values)
                        if max_val > min_val:
                            normalized_score = (scores[feature] - min_val) / (max_val - min_val)
                        else:
                            normalized_score = 1.0 if scores[feature] > 0 else 0.0
                        feature_scores.append(normalized_score)
            
            # Combine scores (average)
            if feature_scores:
                combined_scores[feature] = np.mean(feature_scores)
            else:
                combined_scores[feature] = 0.0
        
        return combined_scores
    
    def _select_top_features(self, combined_scores: Dict[str, float],
                            feature_names: List[str]) -> List[str]:
        """Select top features based on combined scores."""
        
        if not combined_scores:
            return feature_names
        
        # Sort features by score
        sorted_features = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Apply selection criteria
        selected_features = []
        
        for feature, score in sorted_features:
            # Apply score thresholds
            if score < 0.1:  # Minimum threshold
                continue
            
            selected_features.append(feature)
            
            # Apply max features limit
            if self.config.max_features and len(selected_features) >= self.config.max_features:
                break
        
        # Ensure we have at least some features
        if not selected_features and sorted_features:
            # Take top 10 features if none meet threshold
            selected_features = [f[0] for f in sorted_features[:10]]
        
        return selected_features
    
    def _generate_selection_rationale(self, method_scores: Dict[str, Dict[str, float]],
                                    selected_features: List[str]) -> Dict[str, str]:
        """Generate rationale for feature selection decisions."""
        
        rationale = {}
        
        for feature in selected_features:
            reasons = []
            
            # Check which methods ranked this feature highly
            for method, scores in method_scores.items():
                if feature in scores:
                    score = scores[feature]
                    # Normalize score for comparison
                    method_values = list(scores.values())
                    if method_values:
                        max_val = max(method_values)
                        if max_val > 0:
                            normalized_score = score / max_val
                            if normalized_score > 0.5:
                                reasons.append(f"High {method} score ({normalized_score:.2f})")
            
            if reasons:
                rationale[feature] = "; ".join(reasons)
            else:
                rationale[feature] = "Selected by combined ranking"
        
        return rationale
    
    def generate_selection_report(self, result: FeatureImportanceResult) -> str:
        """Generate comprehensive feature selection report."""
        
        report = f"""
🎯 Advanced Feature Selection Report
====================================

📊 Selection Summary:
- Features Selected: {len(result.selected_features)}
- Methods Used: {len(result.method_scores)}
- Top Features by Combined Score:
"""
        
        # Show top 10 features
        sorted_features = sorted(
            result.feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for i, (feature, score) in enumerate(sorted_features, 1):
            status = "✅ SELECTED" if feature in result.selected_features else "❌ NOT SELECTED"
            report += f"\n  {i:2d}. {feature}: {score:.3f} - {status}"
        
        # Method comparison
        report += f"\n\n📋 Method Comparison:\n"
        
        for method, scores in result.method_scores.items():
            if scores:
                avg_score = np.mean(list(scores.values()))
                report += f"- {method.title()}: {len(scores)} features, avg score {avg_score:.3f}\n"
        
        # Selection rationale for top features
        if result.selection_rationale:
            report += f"\n💡 Selection Rationale (Top 5):\n"
            
            top_selected = [f for f in sorted_features[:5] if f[0] in result.selected_features]
            
            for feature, _ in top_selected:
                rationale = result.selection_rationale.get(feature, "No specific rationale")
                report += f"- {feature}: {rationale}\n"
        
        return report


