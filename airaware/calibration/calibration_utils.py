"""
Calibration Utilities

This module provides utility functions for calibration methods,
including evaluation metrics, visualization, and helper functions.
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
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

def calculate_coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Calculate empirical coverage of prediction intervals"""
    return np.mean((y_true >= y_lower) & (y_true <= y_upper))

def calculate_interval_width(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Calculate average width of prediction intervals"""
    return np.mean(y_upper - y_lower)

def calculate_calibration_error(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, 
                              target_coverage: float = 0.9) -> float:
    """Calculate calibration error (deviation from target coverage)"""
    empirical_coverage = calculate_coverage(y_true, y_lower, y_upper)
    return abs(empirical_coverage - target_coverage)

def calculate_sharpness(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Calculate sharpness (inverse of average interval width)"""
    return 1.0 / calculate_interval_width(y_lower, y_upper)

def calculate_quantile_score(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Calculate quantile score (pinball loss)"""
    errors = y_true - y_pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))

def calculate_winkler_score(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, 
                           alpha: float = 0.1) -> float:
    """Calculate Winkler score for prediction intervals"""
    width = y_upper - y_lower
    penalty = 2.0 / alpha * np.maximum(0, y_lower - y_true) + 2.0 / alpha * np.maximum(0, y_true - y_upper)
    return np.mean(width + penalty)

def calculate_interval_score(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, 
                           alpha: float = 0.1) -> float:
    """Calculate interval score (combination of coverage and width)"""
    coverage = calculate_coverage(y_true, y_lower, y_upper)
    width = calculate_interval_width(y_lower, y_upper)
    coverage_error = calculate_calibration_error(y_true, y_lower, y_upper, 1 - alpha)
    
    # Combine coverage error and width (lower is better)
    return coverage_error + width / np.std(y_true)  # Normalize width by data std

def evaluate_calibration_quality(y_true: np.ndarray, y_pred: np.ndarray, 
                                y_lower: np.ndarray, y_upper: np.ndarray,
                                target_coverage: float = 0.9) -> Dict[str, float]:
    """Comprehensive evaluation of calibration quality"""
    results = {}
    
    # Basic metrics
    results['mae'] = mean_absolute_error(y_true, y_pred)
    results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Coverage metrics
    results['coverage'] = calculate_coverage(y_true, y_lower, y_upper)
    results['coverage_error'] = calculate_calibration_error(y_true, y_lower, y_upper, target_coverage)
    
    # Width metrics
    results['interval_width'] = calculate_interval_width(y_lower, y_upper)
    results['sharpness'] = calculate_sharpness(y_lower, y_upper)
    
    # Combined scores
    results['winkler_score'] = calculate_winkler_score(y_true, y_lower, y_upper, 1 - target_coverage)
    results['interval_score'] = calculate_interval_score(y_true, y_lower, y_upper, 1 - target_coverage)
    
    return results

def plot_calibration_diagnostics(y_true: np.ndarray, y_pred: np.ndarray,
                                y_lower: np.ndarray, y_upper: np.ndarray,
                                target_coverage: float = 0.9,
                                save_path: Optional[str] = None) -> None:
    """Plot calibration diagnostic plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Prediction vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Prediction vs Actual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Prediction intervals
    n_samples = min(100, len(y_true))  # Show first 100 samples
    indices = np.arange(n_samples)
    axes[1, 0].fill_between(indices, y_lower[:n_samples], y_upper[:n_samples], 
                           alpha=0.3, label='Prediction Interval')
    axes[1, 0].plot(indices, y_true[:n_samples], 'o-', label='Actual', markersize=4)
    axes[1, 0].plot(indices, y_pred[:n_samples], 's-', label='Predicted', markersize=4)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Prediction Intervals (First 100 Samples)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Coverage by quantiles
    n_quantiles = 10
    quantiles = np.linspace(0, 1, n_quantiles + 1)
    coverage_by_quantile = []
    
    for i in range(n_quantiles):
        q_lower = quantiles[i]
        q_upper = quantiles[i + 1]
        mask = (y_pred >= np.quantile(y_pred, q_lower)) & (y_pred <= np.quantile(y_pred, q_upper))
        
        if np.sum(mask) > 0:
            coverage = calculate_coverage(y_true[mask], y_lower[mask], y_upper[mask])
            coverage_by_quantile.append(coverage)
        else:
            coverage_by_quantile.append(0)
    
    axes[1, 1].bar(range(n_quantiles), coverage_by_quantile, alpha=0.7)
    axes[1, 1].axhline(y=target_coverage, color='r', linestyle='--', 
                      label=f'Target Coverage ({target_coverage:.1%})')
    axes[1, 1].set_xlabel('Prediction Quantile')
    axes[1, 1].set_ylabel('Empirical Coverage')
    axes[1, 1].set_title('Coverage by Prediction Quantile')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration diagnostics saved to {save_path}")
    
    plt.show()

def plot_coverage_evolution(coverage_history: List[Dict[str, Any]], 
                           target_coverage: float = 0.9,
                           save_path: Optional[str] = None) -> None:
    """Plot evolution of coverage over time"""
    if not coverage_history:
        logger.warning("No coverage history provided")
        return
    
    # Extract data
    n_samples = [h['n_samples'] for h in coverage_history]
    coverage = [h['coverage'] for h in coverage_history]
    
    plt.figure(figsize=(12, 6))
    plt.plot(n_samples, coverage, 'b-', linewidth=2, label='Empirical Coverage')
    plt.axhline(y=target_coverage, color='r', linestyle='--', 
               label=f'Target Coverage ({target_coverage:.1%})')
    plt.fill_between(n_samples, target_coverage - 0.05, target_coverage + 0.05, 
                    alpha=0.2, color='red', label='Tolerance Band')
    
    plt.xlabel('Number of Samples')
    plt.ylabel('Empirical Coverage')
    plt.title('Coverage Evolution Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Coverage evolution plot saved to {save_path}")
    
    plt.show()

def plot_quantile_calibration(y_true: np.ndarray, y_pred: np.ndarray,
                             quantile_levels: List[float],
                             save_path: Optional[str] = None) -> None:
    """Plot quantile calibration (reliability diagram)"""
    n_quantiles = len(quantile_levels)
    empirical_quantiles = []
    theoretical_quantiles = []
    
    for level in quantile_levels:
        # Calculate empirical quantile
        empirical_quantile = np.quantile(y_true, level)
        empirical_quantiles.append(empirical_quantile)
        
        # Calculate theoretical quantile (from predictions)
        theoretical_quantile = np.quantile(y_pred, level)
        theoretical_quantiles.append(theoretical_quantile)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(theoretical_quantiles, empirical_quantiles, s=100, alpha=0.7)
    plt.plot([min(theoretical_quantiles), max(theoretical_quantiles)], 
             [min(theoretical_quantiles), max(theoretical_quantiles)], 
             'r--', lw=2, label='Perfect Calibration')
    
    plt.xlabel('Theoretical Quantiles (Predictions)')
    plt.ylabel('Empirical Quantiles (Actual)')
    plt.title('Quantile Calibration (Reliability Diagram)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Quantile calibration plot saved to {save_path}")
    
    plt.show()

def create_calibration_report(y_true: np.ndarray, y_pred: np.ndarray,
                            y_lower: np.ndarray, y_upper: np.ndarray,
                            target_coverage: float = 0.9,
                            save_path: Optional[str] = None) -> Dict[str, Any]:
    """Create comprehensive calibration report"""
    report = {}
    
    # Basic statistics
    report['n_samples'] = len(y_true)
    report['target_coverage'] = target_coverage
    
    # Evaluation metrics
    report['evaluation'] = evaluate_calibration_quality(y_true, y_pred, y_lower, y_upper, target_coverage)
    
    # Coverage analysis
    coverage = calculate_coverage(y_true, y_lower, y_upper)
    report['coverage_analysis'] = {
        'empirical_coverage': coverage,
        'coverage_error': abs(coverage - target_coverage),
        'coverage_ratio': coverage / target_coverage,
        'is_well_calibrated': abs(coverage - target_coverage) < 0.05
    }
    
    # Interval analysis
    report['interval_analysis'] = {
        'mean_width': calculate_interval_width(y_lower, y_upper),
        'width_std': np.std(y_upper - y_lower),
        'sharpness': calculate_sharpness(y_lower, y_upper),
        'winkler_score': calculate_winkler_score(y_true, y_lower, y_upper, 1 - target_coverage)
    }
    
    # Residual analysis
    residuals = y_true - y_pred
    report['residual_analysis'] = {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals),
        'normality_test_pvalue': stats.normaltest(residuals)[1]
    }
    
    # Save report
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Calibration report saved to {save_path}")
    
    return report

def compare_calibration_methods(results: Dict[str, Dict[str, Any]], 
                              save_path: Optional[str] = None) -> None:
    """Compare different calibration methods"""
    methods = list(results.keys())
    metrics = ['coverage', 'coverage_error', 'interval_width', 'winkler_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[method]['evaluation'][metric] for method in methods]
        
        bars = axes[i].bar(methods, values, alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
        
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration comparison saved to {save_path}")
    
    plt.show()

def validate_calibration_inputs(y_true: np.ndarray, y_pred: np.ndarray,
                              y_lower: np.ndarray, y_upper: np.ndarray) -> bool:
    """Validate inputs for calibration evaluation"""
    # Check shapes
    if not (y_true.shape == y_pred.shape == y_lower.shape == y_upper.shape):
        logger.error("All arrays must have the same shape")
        return False
    
    # Check for NaN values
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)) or \
       np.any(np.isnan(y_lower)) or np.any(np.isnan(y_upper)):
        logger.error("Arrays contain NaN values")
        return False
    
    # Check for infinite values
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)) or \
       np.any(np.isinf(y_lower)) or np.any(np.isinf(y_upper)):
        logger.error("Arrays contain infinite values")
        return False
    
    # Check interval validity
    if np.any(y_upper < y_lower):
        logger.error("Upper bounds must be greater than or equal to lower bounds")
        return False
    
    return True

def bootstrap_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                y_lower: np.ndarray, y_upper: np.ndarray,
                                target_coverage: float = 0.9,
                                n_bootstrap: int = 1000,
                                confidence_level: float = 0.95) -> Dict[str, Tuple[float, float, float]]:
    """Bootstrap confidence intervals for calibration metrics"""
    n_samples = len(y_true)
    metrics = ['coverage', 'coverage_error', 'interval_width', 'winkler_score']
    bootstrap_results = {metric: [] for metric in metrics}
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_lower_boot = y_lower[indices]
        y_upper_boot = y_upper[indices]
        
        # Calculate metrics
        eval_results = evaluate_calibration_quality(y_true_boot, y_pred_boot, 
                                                  y_lower_boot, y_upper_boot, target_coverage)
        
        for metric in metrics:
            bootstrap_results[metric].append(eval_results[metric])
    
    # Calculate confidence intervals
    confidence_intervals = {}
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    for metric in metrics:
        values = np.array(bootstrap_results[metric])
        mean_val = np.mean(values)
        lower_ci = np.percentile(values, lower_percentile)
        upper_ci = np.percentile(values, upper_percentile)
        confidence_intervals[metric] = (mean_val, lower_ci, upper_ci)
    
    return confidence_intervals
