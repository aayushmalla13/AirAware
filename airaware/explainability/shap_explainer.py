"""
SHAP Explainer

This module provides SHAP (SHapley Additive exPlanations) based
explainability for air quality forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from sklearn.base import BaseEstimator, RegressorMixin
import warnings

logger = logging.getLogger(__name__)

@dataclass
class SHAPConfig:
    """Configuration for SHAP explainer"""
    # Explainer type
    explainer_type: str = "tree"  # "tree", "linear", "kernel", "deep", "gradient"
    
    # Sampling parameters
    max_samples: int = 100  # Maximum samples for explanation
    sample_size: int = 50  # Sample size for background data
    
    # Tree explainer specific
    use_tree_path_dependent: bool = True
    feature_perturbation: str = "tree_path_dependent"  # "tree_path_dependent", "interventional"
    
    # Kernel explainer specific
    kernel_width: float = 0.25
    l1_reg: str = "aic"  # "aic", "bic", "num_features", float
    
    # Deep explainer specific
    use_deep_explainer: bool = False
    deep_explainer_background: int = 50
    
    # Gradient explainer specific
    use_gradient_explainer: bool = False
    
    # Explanation parameters
    explanation_type: str = "local"  # "local", "global", "both"
    use_waterfall_plot: bool = True
    use_summary_plot: bool = True
    use_force_plot: bool = False
    
    # Aggregation parameters
    aggregate_explanations: bool = True
    aggregation_method: str = "mean"  # "mean", "median", "max"
    
    # Time series specific
    use_temporal_shap: bool = True
    temporal_window: int = 24  # Hours for temporal analysis
    
    # Feature selection
    top_k_features: int = 10
    importance_threshold: float = 0.01

class SHAPExplainer:
    """SHAP-based explainer for model interpretability"""
    
    def __init__(self, config: SHAPConfig):
        self.config = config
        self.is_fitted = False
        self.explainer = None
        self.shap_values = None
        self.feature_names = []
        self.background_data = None
        self.explanation_results = {}
        
    def fit(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'SHAPExplainer':
        """Fit SHAP explainer"""
        logger.info("🔧 Fitting SHAP explainer...")
        
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])]
        
        # Prepare background data
        self._prepare_background_data(X)
        
        # Create explainer
        self._create_explainer(model, X)
        
        # Calculate SHAP values
        self._calculate_shap_values(model, X)
        
        self.is_fitted = True
        logger.info("✅ SHAP explainer fitted")
        
        return self
    
    def _prepare_background_data(self, X: np.ndarray):
        """Prepare background data for SHAP"""
        # Sample background data
        n_samples = min(self.config.sample_size, X.shape[0])
        background_indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
        self.background_data = X[background_indices]
        
        logger.info(f"Prepared background data with {n_samples} samples")
    
    def _create_explainer(self, model: BaseEstimator, X: np.ndarray):
        """Create appropriate SHAP explainer"""
        try:
            import shap
            
            if self.config.explainer_type == "tree":
                # Tree explainer for tree-based models
                if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                    self.explainer = shap.TreeExplainer(
                        model,
                        feature_perturbation=self.config.feature_perturbation
                    )
                else:
                    logger.warning("Model is not tree-based, falling back to kernel explainer")
                    self.config.explainer_type = "kernel"
            
            if self.config.explainer_type == "kernel":
                # Kernel explainer
                self.explainer = shap.KernelExplainer(
                    model.predict,
                    self.background_data,
                    kernel_width=self.config.kernel_width,
                    l1_reg=self.config.l1_reg
                )
            
            elif self.config.explainer_type == "linear":
                # Linear explainer
                self.explainer = shap.LinearExplainer(model, self.background_data)
            
            elif self.config.explainer_type == "deep":
                # Deep explainer (for neural networks)
                if self.config.use_deep_explainer:
                    self.explainer = shap.DeepExplainer(
                        model,
                        self.background_data[:self.config.deep_explainer_background]
                    )
                else:
                    logger.warning("Deep explainer not enabled, falling back to kernel explainer")
                    self.config.explainer_type = "kernel"
                    self.explainer = shap.KernelExplainer(model.predict, self.background_data)
            
            elif self.config.explainer_type == "gradient":
                # Gradient explainer (for differentiable models)
                if self.config.use_gradient_explainer:
                    self.explainer = shap.GradientExplainer(model, self.background_data)
                else:
                    logger.warning("Gradient explainer not enabled, falling back to kernel explainer")
                    self.config.explainer_type = "kernel"
                    self.explainer = shap.KernelExplainer(model.predict, self.background_data)
            
            logger.info(f"Created {self.config.explainer_type} explainer")
            
        except ImportError:
            logger.error("SHAP library not installed. Please install with: pip install shap")
            raise ImportError("SHAP library is required for SHAP explainer")
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            raise
    
    def _calculate_shap_values(self, model: BaseEstimator, X: np.ndarray):
        """Calculate SHAP values"""
        try:
            import shap
            
            # Limit samples for efficiency
            n_samples = min(self.config.max_samples, X.shape[0])
            X_sample = X[:n_samples]
            
            # Calculate SHAP values
            if self.config.explainer_type == "kernel":
                # Kernel explainer needs different approach
                self.shap_values = self.explainer.shap_values(X_sample)
            else:
                # Other explainers
                self.shap_values = self.explainer.shap_values(X_sample)
            
            # Ensure SHAP values are numpy array
            if isinstance(self.shap_values, list):
                self.shap_values = np.array(self.shap_values)
            
            logger.info(f"Calculated SHAP values for {n_samples} samples")
            
        except Exception as e:
            logger.error(f"Failed to calculate SHAP values: {e}")
            raise
    
    def explain_instance(self, instance: np.ndarray, instance_idx: int = 0) -> Dict[str, Any]:
        """Explain a single instance"""
        if not self.is_fitted:
            raise ValueError("SHAP explainer must be fitted first")
        
        try:
            import shap
            
            # Get SHAP values for this instance
            if self.shap_values is not None and instance_idx < len(self.shap_values):
                instance_shap = self.shap_values[instance_idx]
            else:
                # Calculate SHAP values for this specific instance
                instance_shap = self.explainer.shap_values(instance.reshape(1, -1))
                if isinstance(instance_shap, list):
                    instance_shap = np.array(instance_shap)
                instance_shap = instance_shap.flatten()
            
            # Create explanation
            explanation = {
                'instance': instance.tolist(),
                'shap_values': instance_shap.tolist(),
                'feature_names': self.feature_names,
                'feature_contributions': dict(zip(self.feature_names, instance_shap)),
                'prediction': self.explainer.expected_value + np.sum(instance_shap),
                'base_value': self.explainer.expected_value
            }
            
            # Get top contributing features
            feature_contributions = list(zip(self.feature_names, instance_shap))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            explanation['top_features'] = feature_contributions[:self.config.top_k_features]
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain instance: {e}")
            return {}
    
    def explain_global(self) -> Dict[str, Any]:
        """Explain global model behavior"""
        if not self.is_fitted:
            raise ValueError("SHAP explainer must be fitted first")
        
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        try:
            # Calculate global importance
            if self.config.aggregation_method == "mean":
                global_importance = np.mean(np.abs(self.shap_values), axis=0)
            elif self.config.aggregation_method == "median":
                global_importance = np.median(np.abs(self.shap_values), axis=0)
            elif self.config.aggregation_method == "max":
                global_importance = np.max(np.abs(self.shap_values), axis=0)
            else:
                global_importance = np.mean(np.abs(self.shap_values), axis=0)
            
            # Create global explanation
            explanation = {
                'global_importance': global_importance.tolist(),
                'feature_names': self.feature_names,
                'feature_importance': dict(zip(self.feature_names, global_importance)),
                'base_value': self.explainer.expected_value,
                'n_samples': len(self.shap_values)
            }
            
            # Get top important features
            feature_importance = list(zip(self.feature_names, global_importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            explanation['top_features'] = feature_importance[:self.config.top_k_features]
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to create global explanation: {e}")
            return {}
    
    def explain_temporal(self, X: np.ndarray, timestamps: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Explain temporal patterns in SHAP values"""
        if not self.is_fitted:
            raise ValueError("SHAP explainer must be fitted first")
        
        if not self.config.use_temporal_shap:
            return {}
        
        try:
            # Calculate SHAP values for temporal analysis
            if self.shap_values is None:
                self._calculate_shap_values(None, X)
            
            # Group by time windows
            if timestamps is not None:
                # Convert timestamps to hourly bins
                hourly_bins = timestamps.dt.floor('H')
                unique_hours = hourly_bins.unique()
                
                temporal_explanations = {}
                for hour in unique_hours[:self.config.temporal_window]:
                    hour_mask = hourly_bins == hour
                    if np.any(hour_mask):
                        hour_shap = self.shap_values[hour_mask]
                        hour_importance = np.mean(np.abs(hour_shap), axis=0)
                        
                        temporal_explanations[str(hour)] = {
                            'hour': str(hour),
                            'importance': hour_importance.tolist(),
                            'feature_names': self.feature_names,
                            'n_samples': len(hour_shap)
                        }
                
                return {
                    'temporal_explanations': temporal_explanations,
                    'temporal_window': self.config.temporal_window
                }
            else:
                # Simple temporal analysis without timestamps
                window_size = min(self.config.temporal_window, len(self.shap_values))
                recent_shap = self.shap_values[-window_size:]
                recent_importance = np.mean(np.abs(recent_shap), axis=0)
                
                return {
                    'recent_importance': recent_importance.tolist(),
                    'feature_names': self.feature_names,
                    'window_size': window_size
                }
                
        except Exception as e:
            logger.error(f"Failed to create temporal explanation: {e}")
            return {}
    
    def get_explanation_summary(self) -> Dict[str, Any]:
        """Get explanation summary"""
        if not self.is_fitted:
            return {}
        
        summary = {
            'explainer_type': self.config.explainer_type,
            'n_features': len(self.feature_names),
            'n_samples': len(self.shap_values) if self.shap_values is not None else 0,
            'base_value': self.explainer.expected_value if self.explainer else None,
            'explanation_types': []
        }
        
        if self.config.explanation_type in ["local", "both"]:
            summary['explanation_types'].append('local')
        
        if self.config.explanation_type in ["global", "both"]:
            summary['explanation_types'].append('global')
        
        if self.config.use_temporal_shap:
            summary['explanation_types'].append('temporal')
        
        return summary
    
    def save_explainer(self, path: str):
        """Save SHAP explainer and results"""
        if not self.is_fitted:
            raise ValueError("SHAP explainer must be fitted first")
        
        # Note: SHAP explainers are not directly serializable
        # We save the configuration and calculated values instead
        results = {
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'shap_values': self.shap_values.tolist() if self.shap_values is not None else None,
            'base_value': self.explainer.expected_value if self.explainer else None,
            'background_data': self.background_data.tolist() if self.background_data is not None else None,
            'summary': self.get_explanation_summary()
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"SHAP explainer results saved to {path}")
    
    def load_explainer(self, path: str):
        """Load SHAP explainer results"""
        with open(path, 'r') as f:
            results = json.load(f)
        
        self.config = results['config']
        self.feature_names = results['feature_names']
        self.shap_values = np.array(results['shap_values']) if results['shap_values'] is not None else None
        self.background_data = np.array(results['background_data']) if results['background_data'] is not None else None
        self.is_fitted = True
        
        logger.info(f"SHAP explainer results loaded from {path}")
    
    def get_explainer_info(self) -> Dict[str, Any]:
        """Get explainer information"""
        return {
            'method': 'shap',
            'explainer_type': self.config.explainer_type,
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names),
            'n_samples': len(self.shap_values) if self.shap_values is not None else 0,
            'config': self.config.__dict__
        }
