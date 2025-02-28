"""
Explainability Utilities

This module provides utility functions for explainability analysis,
including visualization, reporting, and evaluation metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
from pathlib import Path
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

logger = logging.getLogger(__name__)

def plot_feature_importance(importance_results: Dict[str, Any], 
                          method: str = 'permutation',
                          top_k: int = 20,
                          save_path: Optional[str] = None) -> None:
    """Plot feature importance"""
    if method not in importance_results:
        logger.error(f"Method '{method}' not found in importance results")
        return
    
    # Handle different result structures
    if 'importances' in importance_results[method]:
        importances = importance_results[method]['importances']
        feature_names = importance_results[method]['feature_names']
    elif isinstance(importance_results[method], dict) and 'window_1' in importance_results[method]:
        # Handle temporal importance structure
        logger.warning(f"Temporal importance structure detected for {method}, skipping plot")
        return
    else:
        logger.error(f"Unexpected structure for method '{method}': {importance_results[method].keys()}")
        return
    
    # Sort by importance
    sorted_indices = np.argsort(importances)[::-1][:top_k]
    top_importances = importances[sorted_indices]
    top_features = [feature_names[i] for i in sorted_indices]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_features)), top_importances)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_k} Feature Importance ({method.title()})')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, top_importances)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', va='center', ha='left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.show()

def plot_shap_summary(shap_values: np.ndarray, 
                     feature_names: List[str],
                     X: np.ndarray,
                     save_path: Optional[str] = None) -> None:
    """Plot SHAP summary"""
    try:
        import shap
        
        # Create SHAP explainer for plotting
        explainer = shap.Explainer(lambda x: np.random.random(len(x)), X[:100])
        shap_values_obj = shap.Values(shap_values, X[:100], feature_names)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_obj, X[:100], feature_names=feature_names, show=False)
        plt.title('SHAP Summary Plot')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("SHAP library not available for plotting")
        # Fallback to simple bar plot
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        sorted_indices = np.argsort(mean_shap)[::-1][:20]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_indices)), mean_shap[sorted_indices])
        plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('SHAP Feature Importance (Fallback)')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP fallback plot saved to {save_path}")
        
        plt.show()

def plot_lime_explanation(explanation: Dict[str, Any],
                         save_path: Optional[str] = None) -> None:
    """Plot LIME explanation"""
    if 'feature_contributions' not in explanation:
        logger.error("LIME explanation does not contain feature contributions")
        return
    
    contributions = explanation['feature_contributions']
    feature_names = list(contributions.keys())
    values = list(contributions.values())
    
    # Sort by absolute value
    sorted_indices = np.argsort(np.abs(values))[::-1][:15]
    top_features = [feature_names[i] for i in sorted_indices]
    top_values = [values[i] for i in sorted_indices]
    
    # Create plot
    colors = ['red' if v < 0 else 'green' for v in top_values]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_features)), top_values, color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Contribution to Prediction')
    plt.title('LIME Feature Contributions')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_values)):
        plt.text(bar.get_width() + (0.01 if value >= 0 else -0.01), 
                bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', 
                ha='left' if value >= 0 else 'right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"LIME explanation plot saved to {save_path}")
    
    plt.show()

def plot_what_if_scenarios(scenario_summary: Dict[str, Any],
                          save_path: Optional[str] = None) -> None:
    """Plot what-if scenario analysis"""
    if not scenario_summary or 'scenarios' not in scenario_summary:
        logger.error("No scenario data available for plotting")
        return
    
    # Collect all scenarios
    all_scenarios = []
    for scenario_type, scenarios in scenario_summary['scenarios'].items():
        for scenario in scenarios:
            scenario['type'] = scenario_type
            all_scenarios.append(scenario)
    
    if not all_scenarios:
        logger.warning("No scenarios to plot")
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Impact distribution
    impacts = [s['impact'] for s in all_scenarios]
    axes[0, 0].hist(impacts, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', label='No Impact')
    axes[0, 0].set_xlabel('Prediction Impact')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Scenario Impacts')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Impact by scenario type
    scenario_types = list(set(s['type'] for s in all_scenarios))
    type_impacts = {t: [s['impact'] for s in all_scenarios if s['type'] == t] for t in scenario_types}
    
    axes[0, 1].boxplot([type_impacts[t] for t in scenario_types], labels=scenario_types)
    axes[0, 1].set_ylabel('Prediction Impact')
    axes[0, 1].set_title('Impact by Scenario Type')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Top impactful scenarios
    top_scenarios = sorted(all_scenarios, key=lambda x: abs(x['impact']), reverse=True)[:10]
    scenario_names = [s['scenario_name'] for s in top_scenarios]
    scenario_impacts = [s['impact'] for s in top_scenarios]
    
    colors = ['red' if impact < 0 else 'green' for impact in scenario_impacts]
    axes[1, 0].barh(range(len(scenario_names)), scenario_impacts, color=colors, alpha=0.7)
    axes[1, 0].set_yticks(range(len(scenario_names)))
    axes[1, 0].set_yticklabels(scenario_names)
    axes[1, 0].set_xlabel('Prediction Impact')
    axes[1, 0].set_title('Top 10 Most Impactful Scenarios')
    axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].invert_yaxis()
    
    # 4. Feature perturbation frequency
    all_perturbed_features = []
    for scenario in all_scenarios:
        all_perturbed_features.extend(scenario['perturbed_features'])
    
    if all_perturbed_features:
        feature_counts = pd.Series(all_perturbed_features).value_counts().head(10)
        axes[1, 1].bar(range(len(feature_counts)), feature_counts.values)
        axes[1, 1].set_xticks(range(len(feature_counts)))
        axes[1, 1].set_xticklabels(feature_counts.index, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Most Frequently Perturbed Features')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"What-if scenarios plot saved to {save_path}")
    
    plt.show()

def plot_counterfactual_examples(counterfactual_summary: Dict[str, Any],
                               save_path: Optional[str] = None) -> None:
    """Plot counterfactual examples"""
    if not counterfactual_summary or 'best_counterfactuals' not in counterfactual_summary:
        logger.error("No counterfactual data available for plotting")
        return
    
    counterfactuals = counterfactual_summary['best_counterfactuals']
    
    if not counterfactuals:
        logger.warning("No counterfactuals to plot")
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Prediction changes
    baseline_preds = [cf['baseline_prediction'] for cf in counterfactuals]
    counterfactual_preds = [cf['counterfactual_prediction'] for cf in counterfactuals]
    target_pred = counterfactuals[0]['target_prediction']
    
    x_pos = range(len(counterfactuals))
    axes[0, 0].bar([x - 0.2 for x in x_pos], baseline_preds, width=0.4, label='Baseline', alpha=0.7)
    axes[0, 0].bar([x + 0.2 for x in x_pos], counterfactual_preds, width=0.4, label='Counterfactual', alpha=0.7)
    axes[0, 0].axhline(y=target_pred, color='red', linestyle='--', label='Target')
    axes[0, 0].set_xlabel('Counterfactual Index')
    axes[0, 0].set_ylabel('Prediction')
    axes[0, 0].set_title('Baseline vs Counterfactual Predictions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distance to target
    distances = [abs(cf['counterfactual_prediction'] - cf['target_prediction']) for cf in counterfactuals]
    axes[0, 1].bar(x_pos, distances, alpha=0.7)
    axes[0, 1].set_xlabel('Counterfactual Index')
    axes[0, 1].set_ylabel('Distance to Target')
    axes[0, 1].set_title('Distance to Target Prediction')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Number of perturbed features
    n_perturbed = [len(cf['perturbed_features']) for cf in counterfactuals]
    axes[1, 0].bar(x_pos, n_perturbed, alpha=0.7)
    axes[1, 0].set_xlabel('Counterfactual Index')
    axes[1, 0].set_ylabel('Number of Perturbed Features')
    axes[1, 0].set_title('Complexity of Counterfactuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Perturbation magnitudes
    magnitudes = [cf['magnitude'] for cf in counterfactuals]
    axes[1, 1].bar(x_pos, magnitudes, alpha=0.7)
    axes[1, 1].set_xlabel('Counterfactual Index')
    axes[1, 1].set_ylabel('Perturbation Magnitude')
    axes[1, 1].set_title('Perturbation Magnitudes')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Counterfactual examples plot saved to {save_path}")
    
    plt.show()

def plot_sensitivity_analysis(sensitivity_summary: Dict[str, Any],
                            save_path: Optional[str] = None) -> None:
    """Plot sensitivity analysis results"""
    if not sensitivity_summary or 'top_sensitive_features' not in sensitivity_summary:
        logger.error("No sensitivity data available for plotting")
        return
    
    top_features = sensitivity_summary['top_sensitive_features']
    
    if not top_features:
        logger.warning("No sensitive features to plot")
        return
    
    # Create plot
    plt.figure(figsize=(12, 8))
    feature_names = [f[0] for f in top_features]
    sensitivity_scores = [f[1] for f in top_features]
    
    bars = plt.barh(range(len(feature_names)), sensitivity_scores, alpha=0.7)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Sensitivity Score')
    plt.title('Feature Sensitivity Analysis')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, sensitivity_scores)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', ha='left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sensitivity analysis plot saved to {save_path}")
    
    plt.show()

def create_explainability_report(explainability_results: Dict[str, Any],
                               save_path: Optional[str] = None) -> Dict[str, Any]:
    """Create comprehensive explainability report"""
    report = {
        'summary': {},
        'feature_importance': {},
        'shap_analysis': {},
        'what_if_analysis': {},
        'sensitivity_analysis': {},
        'counterfactual_analysis': {},
        'recommendations': []
    }
    
    # Feature importance summary
    if 'feature_importance' in explainability_results:
        fi_results = explainability_results['feature_importance']
        if 'importance_results' in fi_results:
            for method, results in fi_results['importance_results'].items():
                if 'importances' in results:
                    importances = results['importances']
                    feature_names = results['feature_names']
                    
                    # Get top features
                    sorted_indices = np.argsort(importances)[::-1][:10]
                    top_features = [(feature_names[i], importances[i]) for i in sorted_indices]
                    
                    report['feature_importance'][method] = {
                        'top_features': top_features,
                        'total_features': len(feature_names),
                        'importance_range': [float(np.min(importances)), float(np.max(importances))]
                    }
    
    # SHAP analysis summary
    if 'shap_analysis' in explainability_results:
        shap_results = explainability_results['shap_analysis']
        if 'explanation_summary' in shap_results:
            report['shap_analysis'] = shap_results['explanation_summary']
    
    # What-if analysis summary
    if 'what_if_analysis' in explainability_results:
        whatif_results = explainability_results['what_if_analysis']
        if 'scenario_summary' in whatif_results:
            scenario_summary = whatif_results['scenario_summary']
            
            # Calculate scenario statistics
            all_scenarios = []
            for scenario_type, scenarios in scenario_summary.get('scenarios', {}).items():
                all_scenarios.extend(scenarios)
            
            if all_scenarios:
                impacts = [s['impact'] for s in all_scenarios]
                report['what_if_analysis'] = {
                    'n_scenarios': len(all_scenarios),
                    'mean_impact': float(np.mean(impacts)),
                    'std_impact': float(np.std(impacts)),
                    'max_impact': float(np.max(impacts)),
                    'min_impact': float(np.min(impacts))
                }
    
    # Sensitivity analysis summary
    if 'sensitivity_analysis' in explainability_results:
        sens_results = explainability_results['sensitivity_analysis']
        if 'sensitivity_summary' in sens_results:
            report['sensitivity_analysis'] = sens_results['sensitivity_summary']
    
    # Counterfactual analysis summary
    if 'counterfactual_analysis' in explainability_results:
        cf_results = explainability_results['counterfactual_analysis']
        if 'counterfactual_summary' in cf_results:
            report['counterfactual_analysis'] = cf_results['counterfactual_summary']
    
    # Generate recommendations
    recommendations = []
    
    # Feature importance recommendations
    if report['feature_importance']:
        for method, results in report['feature_importance'].items():
            top_feature = results['top_features'][0]
            recommendations.append(f"Most important feature ({method}): {top_feature[0]} (importance: {top_feature[1]:.3f})")
    
    # What-if analysis recommendations
    if report['what_if_analysis']:
        mean_impact = report['what_if_analysis']['mean_impact']
        if abs(mean_impact) > 0.1:
            recommendations.append(f"Model is sensitive to feature changes (mean impact: {mean_impact:.3f})")
    
    # Sensitivity analysis recommendations
    if report['sensitivity_analysis'] and 'top_sensitive_features' in report['sensitivity_analysis']:
        top_sensitive = report['sensitivity_analysis']['top_sensitive_features'][0]
        recommendations.append(f"Most sensitive feature: {top_sensitive[0]} (sensitivity: {top_sensitive[1]:.3f})")
    
    report['recommendations'] = recommendations
    
    # Overall summary
    report['summary'] = {
        'n_analysis_methods': len([k for k, v in report.items() if isinstance(v, dict) and v]),
        'n_recommendations': len(recommendations),
        'analysis_completeness': len([k for k, v in report.items() if isinstance(v, dict) and v]) / 5.0
    }
    
    # Save report
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Explainability report saved to {save_path}")
    
    return report

def evaluate_explainability_quality(explainability_results: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate the quality of explainability results"""
    quality_metrics = {}
    
    # Completeness score
    expected_methods = ['feature_importance', 'shap_analysis', 'what_if_analysis', 'sensitivity_analysis', 'counterfactual_analysis']
    available_methods = [method for method in expected_methods if method in explainability_results]
    quality_metrics['completeness'] = len(available_methods) / len(expected_methods)
    
    # Consistency score (if multiple methods available)
    if len(available_methods) > 1:
        # Compare top features across methods
        top_features_by_method = {}
        
        for method in available_methods:
            if method == 'feature_importance' and 'importance_results' in explainability_results[method]:
                for submethod, results in explainability_results[method]['importance_results'].items():
                    if 'importances' in results:
                        importances = results['importances']
                        feature_names = results['feature_names']
                        sorted_indices = np.argsort(importances)[::-1][:5]
                        top_features_by_method[f"{method}_{submethod}"] = [feature_names[i] for i in sorted_indices]
        
        if len(top_features_by_method) > 1:
            # Calculate Jaccard similarity between top feature sets
            similarities = []
            methods = list(top_features_by_method.keys())
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    set1 = set(top_features_by_method[methods[i]])
                    set2 = set(top_features_by_method[methods[j]])
                    similarity = len(set1.intersection(set2)) / len(set1.union(set2))
                    similarities.append(similarity)
            
            quality_metrics['consistency'] = np.mean(similarities) if similarities else 0.0
        else:
            quality_metrics['consistency'] = 0.0
    else:
        quality_metrics['consistency'] = 0.0
    
    # Stability score (if available)
    if 'feature_importance' in explainability_results and 'stability_results' in explainability_results['feature_importance']:
        stability_results = explainability_results['feature_importance']['stability_results']
        if 'cv_importance' in stability_results:
            cv_scores = stability_results['cv_importance']
            # Lower CV indicates higher stability
            quality_metrics['stability'] = 1.0 / (1.0 + np.mean(cv_scores))
        else:
            quality_metrics['stability'] = 0.0
    else:
        quality_metrics['stability'] = 0.0
    
    # Overall quality score
    quality_metrics['overall_quality'] = np.mean(list(quality_metrics.values()))
    
    return quality_metrics
