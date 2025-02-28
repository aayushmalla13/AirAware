"""
Feature Importance Analysis

This module provides comprehensive feature importance analysis
for air quality forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import warnings

logger = logging.getLogger(__name__)

@dataclass
class FeatureImportanceConfig:
    """Configuration for feature importance analysis"""
    # Analysis parameters
    n_repeats: int = 10  # Number of repeats for permutation importance
    random_state: Optional[int] = None
    
    # Methods to use
    use_permutation_importance: bool = True
    use_tree_importance: bool = True
    use_correlation_analysis: bool = True
    use_mutual_information: bool = True
    
    # Correlation analysis
    correlation_threshold: float = 0.7
    use_absolute_correlation: bool = True
    
    # Mutual information
    mi_discrete_features: bool = False
    mi_n_neighbors: int = 3
    
    # Ranking and selection
    top_k_features: int = 20
    importance_threshold: float = 0.01
    
    # Time series specific
    use_temporal_importance: bool = True
    temporal_windows: List[int] = None  # Will be set to [1, 3, 6, 12, 24] if None
    
    # Stability analysis
    use_stability_analysis: bool = True
    stability_splits: int = 5
    
    def __post_init__(self):
        if self.temporal_windows is None:
            self.temporal_windows = [1, 3, 6, 12, 24]

class FeatureImportanceAnalyzer:
    """Comprehensive feature importance analyzer"""
    
    def __init__(self, config: FeatureImportanceConfig):
        self.config = config
        self.is_fitted = False
        self.importance_results = {}
        self.feature_names = []
        self.stability_results = {}
        
    def fit(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> 'FeatureImportanceAnalyzer':
        """Fit feature importance analyzer"""
        logger.info("🔧 Fitting feature importance analyzer...")
        
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])]
        
        # Permutation importance
        if self.config.use_permutation_importance:
            self._calculate_permutation_importance(model, X, y)
        
        # Tree-based importance
        if self.config.use_tree_importance and hasattr(model, 'feature_importances_'):
            self._calculate_tree_importance(model)
        
        # Correlation analysis
        if self.config.use_correlation_analysis:
            self._calculate_correlation_importance(X, y)
        
        # Mutual information
        if self.config.use_mutual_information:
            self._calculate_mutual_information(X, y)
        
        # Temporal importance
        if self.config.use_temporal_importance:
            self._calculate_temporal_importance(model, X, y)
        
        # Stability analysis
        if self.config.use_stability_analysis:
            self._calculate_stability_analysis(model, X, y)
        
        self.is_fitted = True
        logger.info("✅ Feature importance analyzer fitted")
        
        return self
    
    def _calculate_permutation_importance(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray):
        """Calculate permutation importance"""
        logger.info("Calculating permutation importance...")
        
        try:
            perm_importance = permutation_importance(
                model, X, y, 
                n_repeats=self.config.n_repeats,
                random_state=self.config.random_state
            )
            
            self.importance_results['permutation'] = {
                'importances': perm_importance.importances_mean,
                'std': perm_importance.importances_std,
                'feature_names': self.feature_names
            }
        except Exception as e:
            logger.warning(f"Failed to calculate permutation importance: {e}")
    
    def _calculate_tree_importance(self, model: BaseEstimator):
        """Calculate tree-based feature importance"""
        logger.info("Calculating tree-based importance...")
        
        try:
            if hasattr(model, 'feature_importances_'):
                self.importance_results['tree'] = {
                    'importances': model.feature_importances_,
                    'feature_names': self.feature_names
                }
        except Exception as e:
            logger.warning(f"Failed to calculate tree importance: {e}")
    
    def _calculate_correlation_importance(self, X: np.ndarray, y: np.ndarray):
        """Calculate correlation-based importance"""
        logger.info("Calculating correlation importance...")
        
        try:
            correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                if self.config.use_absolute_correlation:
                    corr = abs(corr)
                correlations.append(corr)
            
            self.importance_results['correlation'] = {
                'importances': np.array(correlations),
                'feature_names': self.feature_names
            }
        except Exception as e:
            logger.warning(f"Failed to calculate correlation importance: {e}")
    
    def _calculate_mutual_information(self, X: np.ndarray, y: np.ndarray):
        """Calculate mutual information importance"""
        logger.info("Calculating mutual information importance...")
        
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            mi_scores = mutual_info_regression(
                X, y,
                discrete_features=self.config.mi_discrete_features,
                n_neighbors=self.config.mi_n_neighbors,
                random_state=self.config.random_state
            )
            
            self.importance_results['mutual_information'] = {
                'importances': mi_scores,
                'feature_names': self.feature_names
            }
        except Exception as e:
            logger.warning(f"Failed to calculate mutual information: {e}")
    
    def _calculate_temporal_importance(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray):
        """Calculate temporal feature importance"""
        logger.info("Calculating temporal importance...")
        
        try:
            temporal_importance = {}
            
            for window in self.config.temporal_windows:
                if window <= X.shape[0]:
                    # Use last 'window' samples for temporal analysis
                    X_window = X[-window:]
                    y_window = y[-window:]
                    
                    # Calculate importance for this window
                    if self.config.use_permutation_importance:
                        perm_importance = permutation_importance(
                            model, X_window, y_window,
                            n_repeats=5,  # Fewer repeats for temporal analysis
                            random_state=self.config.random_state
                        )
                        temporal_importance[f'window_{window}'] = {
                            'importances': perm_importance.importances_mean,
                            'std': perm_importance.importances_std
                        }
            
            self.importance_results['temporal'] = temporal_importance
        except Exception as e:
            logger.warning(f"Failed to calculate temporal importance: {e}")
    
    def _calculate_stability_analysis(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray):
        """Calculate feature importance stability across different splits"""
        logger.info("Calculating stability analysis...")
        
        try:
            from sklearn.model_selection import KFold
            
            kf = KFold(n_splits=self.config.stability_splits, shuffle=True, random_state=self.config.random_state)
            stability_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model on this split
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train, y_train)
                
                # Calculate importance
                if hasattr(model_copy, 'feature_importances_'):
                    stability_scores.append(model_copy.feature_importances_)
                else:
                    # Use permutation importance as fallback
                    perm_importance = permutation_importance(
                        model_copy, X_val, y_val,
                        n_repeats=3,
                        random_state=self.config.random_state
                    )
                    stability_scores.append(perm_importance.importances_mean)
            
            # Calculate stability metrics
            stability_scores = np.array(stability_scores)
            mean_importance = np.mean(stability_scores, axis=0)
            std_importance = np.std(stability_scores, axis=0)
            cv_importance = std_importance / (mean_importance + 1e-8)  # Coefficient of variation
            
            self.stability_results = {
                'mean_importance': mean_importance,
                'std_importance': std_importance,
                'cv_importance': cv_importance,
                'stability_scores': stability_scores,
                'feature_names': self.feature_names
            }
        except Exception as e:
            logger.warning(f"Failed to calculate stability analysis: {e}")
    
    def get_top_features(self, method: str = 'permutation', top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Get top K most important features"""
        if not self.is_fitted:
            raise ValueError("Feature importance analyzer must be fitted first")
        
        if method not in self.importance_results:
            raise ValueError(f"Method '{method}' not available")
        
        top_k = top_k or self.config.top_k_features
        importances = self.importance_results[method]['importances']
        feature_names = self.importance_results[method]['feature_names']
        
        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1]
        top_features = [(feature_names[i], importances[i]) for i in sorted_indices[:top_k]]
        
        return top_features
    
    def get_feature_ranking(self, method: str = 'permutation') -> pd.DataFrame:
        """Get complete feature ranking"""
        if not self.is_fitted:
            raise ValueError("Feature importance analyzer must be fitted first")
        
        if method not in self.importance_results:
            raise ValueError(f"Method '{method}' not available")
        
        importances = self.importance_results[method]['importances']
        feature_names = self.importance_results[method]['feature_names']
        
        # Create ranking DataFrame
        ranking_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'rank': range(1, len(feature_names) + 1)
        })
        
        # Sort by importance
        ranking_df = ranking_df.sort_values('importance', ascending=False).reset_index(drop=True)
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        # Add standard deviation if available
        if 'std' in self.importance_results[method]:
            ranking_df['std'] = self.importance_results[method]['std']
        
        return ranking_df
    
    def get_importance_summary(self) -> Dict[str, Any]:
        """Get comprehensive importance summary"""
        if not self.is_fitted:
            return {}
        
        summary = {
            'methods_used': list(self.importance_results.keys()),
            'n_features': len(self.feature_names),
            'top_features': {},
            'stability_analysis': self.stability_results is not None
        }
        
        # Get top features for each method
        for method in self.importance_results.keys():
            try:
                top_features = self.get_top_features(method, top_k=5)
                summary['top_features'][method] = top_features
            except Exception as e:
                logger.warning(f"Failed to get top features for {method}: {e}")
        
        return summary
    
    def save_results(self, path: str):
        """Save importance results"""
        if not self.is_fitted:
            raise ValueError("Feature importance analyzer must be fitted first")
        
        results = {
            'config': self.config.__dict__,
            'importance_results': self.importance_results,
            'stability_results': self.stability_results,
            'feature_names': self.feature_names,
            'summary': self.get_importance_summary()
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Feature importance results saved to {path}")
    
    def load_results(self, path: str):
        """Load importance results"""
        with open(path, 'r') as f:
            results = json.load(f)
        
        self.config = results['config']
        self.importance_results = results['importance_results']
        self.stability_results = results['stability_results']
        self.feature_names = results['feature_names']
        self.is_fitted = True
        
        logger.info(f"Feature importance results loaded from {path}")
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get analyzer information"""
        return {
            'method': 'feature_importance',
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names),
            'methods_used': list(self.importance_results.keys()) if self.is_fitted else [],
            'config': self.config.__dict__
        }
