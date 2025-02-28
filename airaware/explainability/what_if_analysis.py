"""
What-If Analysis

This module provides what-if analysis capabilities for exploring
different scenarios and their impact on air quality predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from sklearn.base import BaseEstimator, RegressorMixin
import warnings

logger = logging.getLogger(__name__)

@dataclass
class WhatIfConfig:
    """Configuration for what-if analysis"""
    # Scenario parameters
    n_scenarios: int = 10  # Number of scenarios to generate
    scenario_type: str = "systematic"  # "systematic", "random", "custom"
    
    # Feature perturbation
    perturbation_method: str = "percentage"  # "percentage", "absolute", "standard_deviation"
    perturbation_range: Tuple[float, float] = (-0.2, 0.2)  # Percentage range for perturbation
    perturbation_std: float = 1.0  # Standard deviations for perturbation
    
    # Feature selection
    target_features: List[str] = None  # Features to perturb (None = all features)
    feature_groups: Dict[str, List[str]] = None  # Group features for coordinated changes
    
    # Scenario generation
    use_feature_correlations: bool = True
    respect_feature_bounds: bool = True
    feature_bounds: Dict[str, Tuple[float, float]] = None
    
    # Analysis parameters
    use_sensitivity_analysis: bool = True
    sensitivity_method: str = "sobol"  # "sobol", "morris", "fast"
    sensitivity_n_samples: int = 1000
    
    # Counterfactual analysis
    use_counterfactual: bool = True
    counterfactual_method: str = "genetic"  # "genetic", "gradient", "random"
    counterfactual_n_trials: int = 100
    
    # Visualization
    use_visualization: bool = True
    plot_scenarios: bool = True
    plot_sensitivity: bool = True
    
    # Time series specific
    use_temporal_scenarios: bool = True
    temporal_horizon: int = 24  # Hours ahead to analyze
    temporal_resolution: str = "hourly"  # "hourly", "daily", "weekly"

class WhatIfAnalyzer:
    """What-if analyzer for scenario exploration"""
    
    def __init__(self, config: WhatIfConfig):
        self.config = config
        self.is_fitted = False
        self.model = None
        self.feature_names = []
        self.baseline_data = None
        self.scenarios = {}
        self.sensitivity_results = {}
        self.counterfactual_results = {}
        
    def fit(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'WhatIfAnalyzer':
        """Fit what-if analyzer"""
        logger.info("🔧 Fitting what-if analyzer...")
        
        self.model = model
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])]
        self.baseline_data = X.copy()
        
        # Generate scenarios
        self._generate_scenarios(X)
        
        # Perform sensitivity analysis
        if self.config.use_sensitivity_analysis:
            self._perform_sensitivity_analysis(X, y)
        
        # Perform counterfactual analysis
        if self.config.use_counterfactual:
            self._perform_counterfactual_analysis(X, y)
        
        self.is_fitted = True
        logger.info("✅ What-if analyzer fitted")
        
        return self
    
    def _generate_scenarios(self, X: np.ndarray):
        """Generate what-if scenarios"""
        logger.info("Generating what-if scenarios...")
        
        if self.config.scenario_type == "systematic":
            self._generate_systematic_scenarios(X)
        elif self.config.scenario_type == "random":
            self._generate_random_scenarios(X)
        elif self.config.scenario_type == "custom":
            self._generate_custom_scenarios(X)
        else:
            raise ValueError(f"Unknown scenario type: {self.config.scenario_type}")
    
    def _generate_systematic_scenarios(self, X: np.ndarray):
        """Generate systematic scenarios"""
        # Use the last sample as baseline
        baseline = X[-1].copy()
        
        # Determine features to perturb
        if self.config.target_features is None:
            target_features = list(range(len(self.feature_names)))
        else:
            target_features = [self.feature_names.index(f) for f in self.config.target_features if f in self.feature_names]
        
        scenarios = []
        
        for i, feature_idx in enumerate(target_features):
            if i >= self.config.n_scenarios:
                break
            
            # Create scenario by perturbing this feature
            scenario = baseline.copy()
            
            if self.config.perturbation_method == "percentage":
                perturbation = np.random.uniform(*self.config.perturbation_range)
                scenario[feature_idx] = baseline[feature_idx] * (1 + perturbation)
            elif self.config.perturbation_method == "absolute":
                perturbation = np.random.uniform(*self.config.perturbation_range)
                scenario[feature_idx] = baseline[feature_idx] + perturbation
            elif self.config.perturbation_method == "standard_deviation":
                std = np.std(X[:, feature_idx])
                perturbation = np.random.normal(0, self.config.perturbation_std * std)
                scenario[feature_idx] = baseline[feature_idx] + perturbation
            
            # Respect feature bounds if specified
            if self.config.respect_feature_bounds and self.config.feature_bounds:
                feature_name = self.feature_names[feature_idx]
                if feature_name in self.config.feature_bounds:
                    min_val, max_val = self.config.feature_bounds[feature_name]
                    scenario[feature_idx] = np.clip(scenario[feature_idx], min_val, max_val)
            
            scenarios.append({
                'name': f'scenario_{i+1}',
                'description': f'Perturb {self.feature_names[feature_idx]} by {perturbation:.3f}',
                'baseline': baseline.tolist(),
                'scenario': scenario.tolist(),
                'perturbed_features': [feature_idx],
                'perturbations': [perturbation]
            })
        
        self.scenarios['systematic'] = scenarios
    
    def _generate_random_scenarios(self, X: np.ndarray):
        """Generate random scenarios"""
        baseline = X[-1].copy()
        scenarios = []
        
        for i in range(self.config.n_scenarios):
            scenario = baseline.copy()
            perturbed_features = []
            perturbations = []
            
            # Randomly select features to perturb
            n_features_to_perturb = np.random.randint(1, min(5, len(self.feature_names)) + 1)
            feature_indices = np.random.choice(len(self.feature_names), size=n_features_to_perturb, replace=False)
            
            for feature_idx in feature_indices:
                if self.config.perturbation_method == "percentage":
                    perturbation = np.random.uniform(*self.config.perturbation_range)
                    scenario[feature_idx] = baseline[feature_idx] * (1 + perturbation)
                elif self.config.perturbation_method == "absolute":
                    perturbation = np.random.uniform(*self.config.perturbation_range)
                    scenario[feature_idx] = baseline[feature_idx] + perturbation
                elif self.config.perturbation_method == "standard_deviation":
                    std = np.std(X[:, feature_idx])
                    perturbation = np.random.normal(0, self.config.perturbation_std * std)
                    scenario[feature_idx] = baseline[feature_idx] + perturbation
                
                perturbed_features.append(feature_idx)
                perturbations.append(perturbation)
            
            scenarios.append({
                'name': f'random_scenario_{i+1}',
                'description': f'Random perturbation of {len(perturbed_features)} features',
                'baseline': baseline.tolist(),
                'scenario': scenario.tolist(),
                'perturbed_features': perturbed_features,
                'perturbations': perturbations
            })
        
        self.scenarios['random'] = scenarios
    
    def _generate_custom_scenarios(self, X: np.ndarray):
        """Generate custom scenarios based on feature groups"""
        baseline = X[-1].copy()
        scenarios = []
        
        if self.config.feature_groups is None:
            # Default groups based on feature types
            meteorological = [f for f in self.feature_names if any(term in f.lower() for term in ['temp', 'humidity', 'wind', 'pressure', 'blh', 'u10', 'v10', 't2m'])]
            temporal = [f for f in self.feature_names if any(term in f.lower() for term in ['hour', 'day', 'month', 'season'])]
            pollution = [f for f in self.feature_names if any(term in f.lower() for term in ['pm', 'no2', 'o3', 'co'])]
            other = [f for f in self.feature_names if f not in meteorological + temporal + pollution]
            
            self.config.feature_groups = {
                'meteorological': meteorological,
                'temporal': temporal,
                'pollution': pollution,
                'other': other
            }
        
        for group_name, group_features in self.config.feature_groups.items():
            if not group_features:
                continue
            
            # Create scenario for this group
            scenario = baseline.copy()
            perturbed_features = []
            perturbations = []
            
            for feature_name in group_features:
                if feature_name in self.feature_names:
                    feature_idx = self.feature_names.index(feature_name)
                    
                    if self.config.perturbation_method == "percentage":
                        perturbation = np.random.uniform(*self.config.perturbation_range)
                        scenario[feature_idx] = baseline[feature_idx] * (1 + perturbation)
                    elif self.config.perturbation_method == "absolute":
                        perturbation = np.random.uniform(*self.config.perturbation_range)
                        scenario[feature_idx] = baseline[feature_idx] + perturbation
                    elif self.config.perturbation_method == "standard_deviation":
                        std = np.std(X[:, feature_idx])
                        perturbation = np.random.normal(0, self.config.perturbation_std * std)
                        scenario[feature_idx] = baseline[feature_idx] + perturbation
                    
                    perturbed_features.append(feature_idx)
                    perturbations.append(perturbation)
            
            scenarios.append({
                'name': f'group_{group_name}',
                'description': f'Perturb {group_name} features',
                'baseline': baseline.tolist(),
                'scenario': scenario.tolist(),
                'perturbed_features': perturbed_features,
                'perturbations': perturbations
            })
        
        self.scenarios['custom'] = scenarios
    
    def _perform_sensitivity_analysis(self, X: np.ndarray, y: np.ndarray):
        """Perform sensitivity analysis"""
        logger.info("Performing sensitivity analysis...")
        
        try:
            # Simple sensitivity analysis using variance-based method
            baseline_pred = self.model.predict(X[-1:].reshape(1, -1))[0]
            
            sensitivity_scores = []
            for i in range(len(self.feature_names)):
                # Perturb feature i
                X_perturbed = X[-1:].copy()
                
                if self.config.perturbation_method == "percentage":
                    perturbation = 0.1  # 10% perturbation
                    X_perturbed[0, i] = X_perturbed[0, i] * (1 + perturbation)
                elif self.config.perturbation_method == "absolute":
                    perturbation = np.std(X[:, i]) * 0.1
                    X_perturbed[0, i] = X_perturbed[0, i] + perturbation
                else:
                    perturbation = np.std(X[:, i])
                    X_perturbed[0, i] = X_perturbed[0, i] + perturbation
                
                # Calculate prediction change
                perturbed_pred = self.model.predict(X_perturbed)[0]
                sensitivity = abs(perturbed_pred - baseline_pred)
                sensitivity_scores.append(sensitivity)
            
            self.sensitivity_results = {
                'sensitivity_scores': sensitivity_scores,
                'feature_names': self.feature_names,
                'baseline_prediction': baseline_pred,
                'method': 'variance_based'
            }
            
        except Exception as e:
            logger.warning(f"Failed to perform sensitivity analysis: {e}")
            self.sensitivity_results = {}
    
    def _perform_counterfactual_analysis(self, X: np.ndarray, y: np.ndarray):
        """Perform counterfactual analysis"""
        logger.info("Performing counterfactual analysis...")
        
        try:
            baseline = X[-1].copy()
            baseline_pred = self.model.predict(baseline.reshape(1, -1))[0]
            
            # Simple counterfactual: find minimal changes to achieve target prediction
            target_prediction = baseline_pred * 1.2  # 20% increase
            
            counterfactuals = []
            for i in range(self.config.counterfactual_n_trials):
                # Randomly select features to change
                n_features = np.random.randint(1, min(5, len(self.feature_names)) + 1)
                feature_indices = np.random.choice(len(self.feature_names), size=n_features, replace=False)
                
                # Try different perturbation magnitudes
                for magnitude in [0.1, 0.2, 0.5, 1.0]:
                    counterfactual = baseline.copy()
                    
                    for feature_idx in feature_indices:
                        if self.config.perturbation_method == "percentage":
                            perturbation = magnitude * np.random.choice([-1, 1])
                            counterfactual[feature_idx] = baseline[feature_idx] * (1 + perturbation)
                        else:
                            std = np.std(X[:, feature_idx])
                            perturbation = magnitude * std * np.random.choice([-1, 1])
                            counterfactual[feature_idx] = baseline[feature_idx] + perturbation
                    
                    # Check if this counterfactual achieves the target
                    counterfactual_pred = self.model.predict(counterfactual.reshape(1, -1))[0]
                    
                    if abs(counterfactual_pred - target_prediction) < abs(baseline_pred - target_prediction):
                        counterfactuals.append({
                            'baseline': baseline.tolist(),
                            'counterfactual': counterfactual.tolist(),
                            'baseline_prediction': baseline_pred,
                            'counterfactual_prediction': counterfactual_pred,
                            'target_prediction': target_prediction,
                            'perturbed_features': feature_indices.tolist(),
                            'magnitude': magnitude
                        })
                        
                        if len(counterfactuals) >= 5:  # Limit number of counterfactuals
                            break
                
                if len(counterfactuals) >= 5:
                    break
            
            self.counterfactual_results = {
                'counterfactuals': counterfactuals,
                'target_prediction': target_prediction,
                'baseline_prediction': baseline_pred,
                'method': 'random_search'
            }
            
        except Exception as e:
            logger.warning(f"Failed to perform counterfactual analysis: {e}")
            self.counterfactual_results = {}
    
    def analyze_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Analyze a specific scenario"""
        if not self.is_fitted:
            raise ValueError("What-if analyzer must be fitted first")
        
        # Find scenario
        scenario = None
        for scenario_type, scenarios in self.scenarios.items():
            for s in scenarios:
                if s['name'] == scenario_name:
                    scenario = s
                    break
            if scenario:
                break
        
        if scenario is None:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        # Get predictions
        baseline_pred = self.model.predict(np.array(scenario['baseline']).reshape(1, -1))[0]
        scenario_pred = self.model.predict(np.array(scenario['scenario']).reshape(1, -1))[0]
        
        # Calculate impact
        impact = scenario_pred - baseline_pred
        impact_percentage = (impact / baseline_pred) * 100 if baseline_pred != 0 else 0
        
        return {
            'scenario_name': scenario_name,
            'baseline_prediction': baseline_pred,
            'scenario_prediction': scenario_pred,
            'impact': impact,
            'impact_percentage': impact_percentage,
            'perturbed_features': [self.feature_names[i] for i in scenario['perturbed_features']],
            'perturbations': scenario['perturbations']
        }
    
    def get_scenario_summary(self) -> Dict[str, Any]:
        """Get summary of all scenarios"""
        if not self.is_fitted:
            return {}
        
        summary = {
            'n_scenarios': sum(len(scenarios) for scenarios in self.scenarios.values()),
            'scenario_types': list(self.scenarios.keys()),
            'scenarios': {}
        }
        
        # Analyze each scenario
        for scenario_type, scenarios in self.scenarios.items():
            summary['scenarios'][scenario_type] = []
            for scenario in scenarios:
                try:
                    analysis = self.analyze_scenario(scenario['name'])
                    summary['scenarios'][scenario_type].append(analysis)
                except Exception as e:
                    logger.warning(f"Failed to analyze scenario {scenario['name']}: {e}")
        
        return summary
    
    def get_sensitivity_summary(self) -> Dict[str, Any]:
        """Get sensitivity analysis summary"""
        if not self.is_fitted or not self.sensitivity_results:
            return {}
        
        sensitivity_scores = self.sensitivity_results['sensitivity_scores']
        feature_names = self.sensitivity_results['feature_names']
        
        # Sort by sensitivity
        sorted_indices = np.argsort(sensitivity_scores)[::-1]
        top_sensitive_features = [(feature_names[i], sensitivity_scores[i]) for i in sorted_indices[:10]]
        
        return {
            'top_sensitive_features': top_sensitive_features,
            'sensitivity_scores': sensitivity_scores,
            'feature_names': feature_names,
            'baseline_prediction': self.sensitivity_results['baseline_prediction'],
            'method': self.sensitivity_results['method']
        }
    
    def get_counterfactual_summary(self) -> Dict[str, Any]:
        """Get counterfactual analysis summary"""
        if not self.is_fitted or not self.counterfactual_results:
            return {}
        
        counterfactuals = self.counterfactual_results['counterfactuals']
        
        if not counterfactuals:
            return {'message': 'No counterfactuals found'}
        
        # Sort by how close they are to target
        counterfactuals.sort(key=lambda x: abs(x['counterfactual_prediction'] - x['target_prediction']))
        
        return {
            'best_counterfactuals': counterfactuals[:3],
            'target_prediction': self.counterfactual_results['target_prediction'],
            'baseline_prediction': self.counterfactual_results['baseline_prediction'],
            'n_counterfactuals': len(counterfactuals),
            'method': self.counterfactual_results['method']
        }
    
    def save_results(self, path: str):
        """Save what-if analysis results"""
        if not self.is_fitted:
            raise ValueError("What-if analyzer must be fitted first")
        
        results = {
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'scenarios': self.scenarios,
            'sensitivity_results': self.sensitivity_results,
            'counterfactual_results': self.counterfactual_results,
            'scenario_summary': self.get_scenario_summary(),
            'sensitivity_summary': self.get_sensitivity_summary(),
            'counterfactual_summary': self.get_counterfactual_summary()
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"What-if analysis results saved to {path}")
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get analyzer information"""
        return {
            'method': 'what_if_analysis',
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names),
            'scenario_types': list(self.scenarios.keys()) if self.is_fitted else [],
            'config': self.config.__dict__
        }
