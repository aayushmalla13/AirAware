"""Feature interaction detection for enhanced model performance."""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class InteractionConfig(BaseModel):
    """Configuration for feature interaction detection."""
    max_interactions: int = Field(20, description="Maximum number of interactions to create")
    min_correlation: float = Field(0.1, description="Minimum correlation for interaction")
    interaction_types: List[str] = Field(
        default=["multiplicative", "ratio", "difference"], 
        description="Types of interactions to detect"
    )
    domain_guided: bool = Field(True, description="Use domain knowledge for interactions")


class InteractionResult(BaseModel):
    """Result of interaction detection."""
    interactions_created: int
    interaction_features: List[str] = Field(default_factory=list)
    interaction_scores: Dict[str, float] = Field(default_factory=dict)
    interaction_types: Dict[str, str] = Field(default_factory=dict)


class FeatureInteractionDetector:
    """Detect and create meaningful feature interactions."""
    
    def __init__(self, config: Optional[InteractionConfig] = None):
        self.config = config or InteractionConfig()
        logger.info("FeatureInteractionDetector initialized")
    
    def detect_and_create_interactions(self, df: pd.DataFrame, 
                                     target_col: str = 'pm25') -> Tuple[pd.DataFrame, InteractionResult]:
        """Detect and create feature interactions."""
        
        logger.info(f"Detecting feature interactions for {len(df):,} records")
        
        df = df.copy()
        
        # Get numeric feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime_utc', 'station_id', target_col] 
                       and df[col].dtype in ['int64', 'int32', 'float64', 'float32']]
        
        if len(feature_cols) < 2:
            logger.warning("Need at least 2 numeric features for interactions")
            return df, InteractionResult(interactions_created=0)
        
        # Find promising interactions
        interactions = self._find_promising_interactions(df, feature_cols, target_col)
        
        # Add domain-guided interactions
        if self.config.domain_guided:
            domain_interactions = self._get_domain_interactions(df, feature_cols)
            interactions.extend(domain_interactions)
        
        # Create interaction features
        interaction_features = []
        interaction_scores = {}
        interaction_types = {}
        
        for feature1, feature2, interaction_type, score in interactions[:self.config.max_interactions]:
            feature_name = self._create_interaction_feature(
                df, feature1, feature2, interaction_type
            )
            
            if feature_name:
                interaction_features.append(feature_name)
                interaction_scores[feature_name] = score
                interaction_types[feature_name] = interaction_type
        
        logger.info(f"Created {len(interaction_features)} interaction features")
        
        return df, InteractionResult(
            interactions_created=len(interaction_features),
            interaction_features=interaction_features,
            interaction_scores=interaction_scores,
            interaction_types=interaction_types
        )
    
    def _find_promising_interactions(self, df: pd.DataFrame, feature_cols: List[str],
                                   target_col: str) -> List[Tuple[str, str, str, float]]:
        """Find promising feature interactions based on correlation with target."""
        
        logger.debug("Finding promising feature interactions")
        
        interactions = []
        
        # Consider all pairs of features
        feature_pairs = list(combinations(feature_cols, 2))
        
        for feature1, feature2 in feature_pairs:
            # Skip if either feature has too many missing values
            if (df[feature1].isnull().sum() / len(df) > 0.5 or 
                df[feature2].isnull().sum() / len(df) > 0.5):
                continue
            
            # Try different interaction types
            for interaction_type in self.config.interaction_types:
                try:
                    # Create temporary interaction
                    temp_interaction = self._calculate_interaction(
                        df[feature1], df[feature2], interaction_type
                    )
                    
                    # Calculate correlation with target
                    if target_col in df.columns:
                        valid_mask = ~(temp_interaction.isnull() | df[target_col].isnull())
                        if valid_mask.sum() > 10:  # Need at least 10 valid points
                            correlation, p_value = pearsonr(
                                temp_interaction[valid_mask], 
                                df[target_col][valid_mask]
                            )
                            
                            if abs(correlation) >= self.config.min_correlation and p_value < 0.05:
                                interactions.append((
                                    feature1, feature2, interaction_type, abs(correlation)
                                ))
                
                except Exception:
                    continue  # Skip problematic interactions
        
        # Sort by correlation strength
        interactions.sort(key=lambda x: x[3], reverse=True)
        
        return interactions
    
    def _get_domain_interactions(self, df: pd.DataFrame, 
                               feature_cols: List[str]) -> List[Tuple[str, str, str, float]]:
        """Get domain-guided interactions for air quality."""
        
        logger.debug("Adding domain-guided interactions")
        
        domain_interactions = []
        
        # Wind speed and boundary layer height (ventilation)
        if 'wind_speed' in feature_cols and 'blh' in feature_cols:
            domain_interactions.append(('wind_speed', 'blh', 'multiplicative', 0.8))
        
        # Temperature and wind speed (heat transport)
        if 't2m_celsius' in feature_cols and 'wind_speed' in feature_cols:
            domain_interactions.append(('t2m_celsius', 'wind_speed', 'multiplicative', 0.7))
        
        # Wind speed and stability (dispersion)
        if 'wind_speed' in feature_cols and any('stability' in col for col in feature_cols):
            stability_cols = [col for col in feature_cols if 'stability' in col]
            if stability_cols:
                domain_interactions.append(('wind_speed', stability_cols[0], 'ratio', 0.6))
        
        # Boundary layer and temperature (mixing height effects)
        if 'blh' in feature_cols and 't2m_celsius' in feature_cols:
            domain_interactions.append(('blh', 't2m_celsius', 'ratio', 0.6))
        
        # Lag interactions (PM2.5 persistence with meteorology)
        lag_features = [col for col in feature_cols if col.startswith('lag_')]
        met_features = [col for col in feature_cols if col.startswith('met_')]
        
        if lag_features and met_features:
            # Pick the most recent lag and a key meteorological feature
            recent_lag = min(lag_features, key=lambda x: int(x.split('_')[1][:-1]))
            if 'met_lag_1h_wind_speed' in met_features:
                domain_interactions.append((recent_lag, 'met_lag_1h_wind_speed', 'multiplicative', 0.5))
        
        # Rolling features interactions
        rolling_features = [col for col in feature_cols if col.startswith('rolling_')]
        if len(rolling_features) >= 2:
            # Mean vs std interactions (measure of variability importance)
            mean_features = [col for col in rolling_features if '_mean_' in col]
            std_features = [col for col in rolling_features if '_std_' in col]
            
            for mean_feat in mean_features[:2]:
                for std_feat in std_features[:2]:
                    if mean_feat.replace('_mean_', '_') in std_feat:  # Same base feature
                        domain_interactions.append((mean_feat, std_feat, 'ratio', 0.4))
        
        return domain_interactions
    
    def _calculate_interaction(self, series1: pd.Series, series2: pd.Series, 
                             interaction_type: str) -> pd.Series:
        """Calculate specific type of interaction between two features."""
        
        if interaction_type == "multiplicative":
            return series1 * series2
        
        elif interaction_type == "ratio":
            # Avoid division by zero
            return series1 / (series2 + 1e-8)
        
        elif interaction_type == "difference":
            return series1 - series2
        
        elif interaction_type == "polynomial":
            return series1 ** 2 + series2 ** 2
        
        elif interaction_type == "log_ratio":
            # Log ratio for positive values
            pos_mask = (series1 > 0) & (series2 > 0)
            result = pd.Series(index=series1.index, dtype=float)
            result[pos_mask] = np.log(series1[pos_mask] + 1) - np.log(series2[pos_mask] + 1)
            return result
        
        else:
            raise ValueError(f"Unknown interaction type: {interaction_type}")
    
    def _create_interaction_feature(self, df: pd.DataFrame, feature1: str, 
                                  feature2: str, interaction_type: str) -> Optional[str]:
        """Create and add interaction feature to dataframe."""
        
        try:
            # Calculate interaction
            interaction_values = self._calculate_interaction(
                df[feature1], df[feature2], interaction_type
            )
            
            # Generate feature name
            if interaction_type == "multiplicative":
                feature_name = f"interact_{feature1}_x_{feature2}"
            elif interaction_type == "ratio":
                feature_name = f"interact_{feature1}_div_{feature2}"
            elif interaction_type == "difference":
                feature_name = f"interact_{feature1}_minus_{feature2}"
            else:
                feature_name = f"interact_{feature1}_{interaction_type}_{feature2}"
            
            # Ensure feature name is unique
            counter = 1
            original_name = feature_name
            while feature_name in df.columns:
                feature_name = f"{original_name}_{counter}"
                counter += 1
            
            # Add to dataframe
            df[feature_name] = interaction_values
            
            # Handle infinite values
            if df[feature_name].isinf().any():
                df[feature_name] = df[feature_name].replace([np.inf, -np.inf], np.nan)
            
            # Fill missing values with median
            if df[feature_name].isnull().any():
                median_val = df[feature_name].median()
                if not np.isnan(median_val):
                    df[feature_name] = df[feature_name].fillna(median_val)
                else:
                    df[feature_name] = df[feature_name].fillna(0)
            
            return feature_name
            
        except Exception as e:
            logger.warning(f"Failed to create interaction {feature1} {interaction_type} {feature2}: {e}")
            return None
    
    def analyze_interaction_importance(self, df: pd.DataFrame, 
                                     interaction_features: List[str],
                                     target_col: str = 'pm25') -> Dict[str, Dict]:
        """Analyze the importance of created interactions."""
        
        logger.debug("Analyzing interaction importance")
        
        analysis = {}
        
        for feature in interaction_features:
            if feature not in df.columns or target_col not in df.columns:
                continue
            
            try:
                # Basic statistics
                feature_stats = {
                    "mean": float(df[feature].mean()),
                    "std": float(df[feature].std()),
                    "min": float(df[feature].min()),
                    "max": float(df[feature].max()),
                    "missing_rate": float(df[feature].isnull().sum() / len(df))
                }
                
                # Correlation with target
                valid_mask = ~(df[feature].isnull() | df[target_col].isnull())
                if valid_mask.sum() > 10:
                    correlation, p_value = pearsonr(
                        df[feature][valid_mask], 
                        df[target_col][valid_mask]
                    )
                    
                    feature_stats["correlation_with_target"] = float(correlation)
                    feature_stats["correlation_p_value"] = float(p_value)
                
                analysis[feature] = feature_stats
                
            except Exception as e:
                logger.warning(f"Failed to analyze interaction {feature}: {e}")
        
        return analysis
    
    def generate_interaction_report(self, result: InteractionResult,
                                  analysis: Optional[Dict] = None) -> str:
        """Generate interaction detection report."""
        
        report = f"""
🔄 Feature Interaction Detection Report
========================================

📊 Interaction Summary:
- Interactions Created: {result.interactions_created}
- Interaction Types Used: {len(set(result.interaction_types.values()))}

🎯 Top Interactions by Score:
"""
        
        # Sort interactions by score
        sorted_interactions = sorted(
            result.interaction_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (feature, score) in enumerate(sorted_interactions[:10], 1):
            interaction_type = result.interaction_types.get(feature, "unknown")
            report += f"\n  {i:2d}. {feature}: {score:.3f} ({interaction_type})"
        
        # Interaction type distribution
        type_counts = {}
        for interaction_type in result.interaction_types.values():
            type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
        
        report += f"\n\n📋 Interaction Types:\n"
        for interaction_type, count in type_counts.items():
            report += f"- {interaction_type.title()}: {count} interactions\n"
        
        # Analysis summary if provided
        if analysis:
            high_corr_interactions = [
                feature for feature, stats in analysis.items()
                if stats.get("correlation_with_target", 0) > 0.2
            ]
            
            if high_corr_interactions:
                report += f"\n🎯 High-Impact Interactions (|correlation| > 0.2):\n"
                for feature in high_corr_interactions[:5]:
                    corr = analysis[feature].get("correlation_with_target", 0)
                    report += f"- {feature}: correlation = {corr:.3f}\n"
        
        return report


