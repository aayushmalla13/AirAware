"""
Residual Analysis and Diagnostic Plots for AirAware Baseline Models

This module implements comprehensive residual analysis and diagnostic plots
for validating baseline forecasting models and identifying potential issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import warnings
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

@dataclass
class ResidualAnalysisConfig:
    """Configuration for residual analysis."""
    
    # Analysis settings
    include_normality_tests: bool = True
    include_autocorrelation_tests: bool = True
    include_heteroscedasticity_tests: bool = True
    include_stationarity_tests: bool = True
    
    # Plot settings
    plot_residuals: bool = True
    plot_acf_pacf: bool = True
    plot_qq_plot: bool = True
    plot_residual_vs_fitted: bool = True
    plot_residual_vs_time: bool = True
    plot_histogram: bool = True
    
    # Statistical test settings
    significance_level: float = 0.05
    lags_for_acf: int = 40
    lags_for_ljung_box: int = 10
    
    # Output settings
    save_plots: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300

class ResidualAnalyzer:
    """Comprehensive residual analysis for forecasting models."""
    
    def __init__(self, config: Optional[ResidualAnalysisConfig] = None):
        self.config = config or ResidualAnalysisConfig()
        self.results = {}
        
    def analyze_residuals(self, 
                        actuals: List[float],
                        predictions: List[float],
                        timestamps: Optional[List[datetime]] = None,
                        model_name: str = "model") -> Dict[str, Any]:
        """Perform comprehensive residual analysis."""
        
        logger.info(f"Analyzing residuals for {model_name}")
        
        # Convert to numpy arrays
        actuals = np.array(actuals)
        predictions = np.array(predictions)
        
        # Calculate residuals
        residuals = actuals - predictions
        
        # Basic statistics
        basic_stats = self._calculate_basic_statistics(residuals, actuals, predictions)
        
        # Statistical tests
        statistical_tests = self._perform_statistical_tests(residuals)
        
        # Diagnostic plots
        plots = self._create_diagnostic_plots(
            residuals, actuals, predictions, timestamps, model_name
        )
        
        # Model diagnostics
        model_diagnostics = self._calculate_model_diagnostics(residuals, actuals, predictions)
        
        results = {
            'model_name': model_name,
            'basic_statistics': basic_stats,
            'statistical_tests': statistical_tests,
            'model_diagnostics': model_diagnostics,
            'plots': plots,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.results[model_name] = results
        
        logger.info(f"Residual analysis completed for {model_name}")
        return results
    
    def _calculate_basic_statistics(self, 
                                  residuals: np.ndarray,
                                  actuals: np.ndarray,
                                  predictions: np.ndarray) -> Dict[str, float]:
        """Calculate basic residual statistics."""
        
        stats_dict = {
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals)),
            'residual_median': float(np.median(residuals)),
            'residual_min': float(np.min(residuals)),
            'residual_max': float(np.max(residuals)),
            'residual_range': float(np.max(residuals) - np.min(residuals)),
            'residual_skewness': float(stats.skew(residuals)),
            'residual_kurtosis': float(stats.kurtosis(residuals)),
            'mae': float(mean_absolute_error(actuals, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(actuals, predictions))),
            'mape': float(np.mean(np.abs((actuals - predictions) / actuals)) * 100),
            'smape': float(np.mean(2 * np.abs(actuals - predictions) / (np.abs(actuals) + np.abs(predictions))) * 100)
        }
        
        return stats_dict
    
    def _perform_statistical_tests(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Perform statistical tests on residuals."""
        
        tests = {}
        
        # Normality tests
        if self.config.include_normality_tests:
            try:
                # Shapiro-Wilk test
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                tests['shapiro_wilk'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > self.config.significance_level
                }
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
                tests['kolmogorov_smirnov'] = {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_p),
                    'is_normal': ks_p > self.config.significance_level
                }
                
                # Anderson-Darling test
                ad_stat, ad_critical, ad_significance = stats.anderson(residuals, dist='norm')
                tests['anderson_darling'] = {
                    'statistic': float(ad_stat),
                    'critical_values': [float(x) for x in ad_critical],
                    'significance_levels': [float(x) for x in ad_significance],
                    'is_normal': ad_stat < ad_critical[2]  # 5% significance level
                }
                
            except Exception as e:
                logger.warning(f"Normality tests failed: {e}")
                tests['normality_tests_error'] = str(e)
        
        # Autocorrelation tests
        if self.config.include_autocorrelation_tests:
            try:
                # Ljung-Box test
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_result = acorr_ljungbox(residuals, lags=self.config.lags_for_ljung_box, return_df=True)
                tests['ljung_box'] = {
                    'statistics': lb_result['lb_stat'].tolist(),
                    'p_values': lb_result['lb_pvalue'].tolist(),
                    'is_white_noise': all(lb_result['lb_pvalue'] > self.config.significance_level)
                }
                
                # Durbin-Watson test
                from statsmodels.stats.diagnostic import durbin_watson
                dw_stat = durbin_watson(residuals)
                tests['durbin_watson'] = {
                    'statistic': float(dw_stat),
                    'is_uncorrelated': 1.5 < dw_stat < 2.5
                }
                
            except Exception as e:
                logger.warning(f"Autocorrelation tests failed: {e}")
                tests['autocorrelation_tests_error'] = str(e)
        
        # Heteroscedasticity tests
        if self.config.include_heteroscedasticity_tests:
            try:
                # Breusch-Pagan test
                from statsmodels.stats.diagnostic import het_breuschpagan
                # Create a simple regression for the test
                X = np.column_stack([np.ones(len(residuals)), np.arange(len(residuals))])
                bp_stat, bp_p, _, _ = het_breuschpagan(residuals**2, X)
                tests['breusch_pagan'] = {
                    'statistic': float(bp_stat),
                    'p_value': float(bp_p),
                    'is_homoscedastic': bp_p > self.config.significance_level
                }
                
            except Exception as e:
                logger.warning(f"Heteroscedasticity tests failed: {e}")
                tests['heteroscedasticity_tests_error'] = str(e)
        
        # Stationarity tests
        if self.config.include_stationarity_tests:
            try:
                # Augmented Dickey-Fuller test
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(residuals)
                tests['adf'] = {
                    'statistic': float(adf_result[0]),
                    'p_value': float(adf_result[1]),
                    'critical_values': {k: float(v) for k, v in adf_result[4].items()},
                    'is_stationary': adf_result[1] < self.config.significance_level
                }
                
            except Exception as e:
                logger.warning(f"Stationarity tests failed: {e}")
                tests['stationarity_tests_error'] = str(e)
        
        return tests
    
    def _create_diagnostic_plots(self, 
                               residuals: np.ndarray,
                               actuals: np.ndarray,
                               predictions: np.ndarray,
                               timestamps: Optional[List[datetime]],
                               model_name: str) -> Dict[str, str]:
        """Create diagnostic plots."""
        
        plots = {}
        
        try:
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Residual Analysis for {model_name}', fontsize=16)
            
            # 1. Residuals vs Time
            if self.config.plot_residual_vs_time:
                ax1 = axes[0, 0]
                if timestamps:
                    ax1.plot(timestamps, residuals, 'o-', alpha=0.7)
                    ax1.set_xlabel('Time')
                else:
                    ax1.plot(residuals, 'o-', alpha=0.7)
                    ax1.set_xlabel('Observation Index')
                ax1.set_ylabel('Residuals')
                ax1.set_title('Residuals vs Time')
                ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax1.grid(True, alpha=0.3)
            
            # 2. Residuals vs Fitted Values
            if self.config.plot_residual_vs_fitted:
                ax2 = axes[0, 1]
                ax2.scatter(predictions, residuals, alpha=0.7)
                ax2.set_xlabel('Fitted Values')
                ax2.set_ylabel('Residuals')
                ax2.set_title('Residuals vs Fitted Values')
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax2.grid(True, alpha=0.3)
            
            # 3. Q-Q Plot
            if self.config.plot_qq_plot:
                ax3 = axes[0, 2]
                stats.probplot(residuals, dist="norm", plot=ax3)
                ax3.set_title('Q-Q Plot')
                ax3.grid(True, alpha=0.3)
            
            # 4. Histogram of Residuals
            if self.config.plot_histogram:
                ax4 = axes[1, 0]
                ax4.hist(residuals, bins=30, alpha=0.7, density=True, edgecolor='black')
                ax4.set_xlabel('Residuals')
                ax4.set_ylabel('Density')
                ax4.set_title('Histogram of Residuals')
                
                # Overlay normal distribution
                mu, sigma = np.mean(residuals), np.std(residuals)
                x = np.linspace(residuals.min(), residuals.max(), 100)
                ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            # 5. ACF Plot
            if self.config.plot_acf_pacf:
                ax5 = axes[1, 1]
                try:
                    from statsmodels.graphics.tsaplots import plot_acf
                    plot_acf(residuals, lags=self.config.lags_for_acf, ax=ax5, alpha=0.05)
                    ax5.set_title('Autocorrelation Function')
                except Exception as e:
                    ax5.text(0.5, 0.5, f'ACF plot failed: {e}', 
                            transform=ax5.transAxes, ha='center', va='center')
                    ax5.set_title('Autocorrelation Function (Failed)')
            
            # 6. PACF Plot
            if self.config.plot_acf_pacf:
                ax6 = axes[1, 2]
                try:
                    from statsmodels.graphics.tsaplots import plot_pacf
                    plot_pacf(residuals, lags=self.config.lags_for_acf, ax=ax6, alpha=0.05)
                    ax6.set_title('Partial Autocorrelation Function')
                except Exception as e:
                    ax6.text(0.5, 0.5, f'PACF plot failed: {e}', 
                            transform=ax6.transAxes, ha='center', va='center')
                    ax6.set_title('Partial Autocorrelation Function (Failed)')
            
            plt.tight_layout()
            
            # Save plot if requested
            if self.config.save_plots:
                plot_path = f'data/artifacts/residual_analysis_{model_name}.{self.config.plot_format}'
                plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
                plots['diagnostic_plot'] = plot_path
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create diagnostic plots: {e}")
            plots['plot_error'] = str(e)
        
        return plots
    
    def _calculate_model_diagnostics(self, 
                                   residuals: np.ndarray,
                                   actuals: np.ndarray,
                                   predictions: np.ndarray) -> Dict[str, float]:
        """Calculate additional model diagnostics."""
        
        diagnostics = {}
        
        # Residual analysis
        diagnostics['residual_mean_abs'] = float(np.mean(np.abs(residuals)))
        diagnostics['residual_std_abs'] = float(np.std(np.abs(residuals)))
        
        # Percentage of residuals within different thresholds
        thresholds = [1, 2, 3]  # Standard deviations
        for threshold in thresholds:
            within_threshold = np.abs(residuals) <= threshold * np.std(residuals)
            diagnostics[f'residuals_within_{threshold}std'] = float(np.mean(within_threshold))
        
        # Bias metrics
        diagnostics['mean_bias'] = float(np.mean(residuals))
        diagnostics['mean_absolute_bias'] = float(np.mean(np.abs(residuals)))
        
        # Efficiency metrics
        diagnostics['nash_sutcliffe'] = float(1 - np.sum(residuals**2) / np.sum((actuals - np.mean(actuals))**2))
        diagnostics['coefficient_of_determination'] = float(np.corrcoef(actuals, predictions)[0, 1]**2)
        
        # Outlier detection
        q75, q25 = np.percentile(residuals, [75, 25])
        iqr = q75 - q25
        outlier_threshold = 1.5 * iqr
        outliers = np.abs(residuals) > outlier_threshold
        diagnostics['outlier_percentage'] = float(np.mean(outliers))
        
        return diagnostics
    
    def generate_summary_report(self, model_name: str) -> str:
        """Generate a summary report for residual analysis."""
        
        if model_name not in self.results:
            return f"No analysis results found for {model_name}"
        
        result = self.results[model_name]
        
        report = f"""
RESIDUAL ANALYSIS REPORT: {model_name}
{'='*50}

BASIC STATISTICS:
- Mean Residual: {result['basic_statistics']['residual_mean']:.4f}
- Residual Std Dev: {result['basic_statistics']['residual_std']:.4f}
- Residual Range: {result['basic_statistics']['residual_range']:.4f}
- Skewness: {result['basic_statistics']['residual_skewness']:.4f}
- Kurtosis: {result['basic_statistics']['residual_kurtosis']:.4f}

PERFORMANCE METRICS:
- MAE: {result['basic_statistics']['mae']:.4f}
- RMSE: {result['basic_statistics']['rmse']:.4f}
- MAPE: {result['basic_statistics']['mape']:.2f}%
- sMAPE: {result['basic_statistics']['smape']:.2f}%

STATISTICAL TESTS:
"""
        
        # Add statistical test results
        tests = result.get('statistical_tests', {})
        
        if 'shapiro_wilk' in tests:
            sw = tests['shapiro_wilk']
            report += f"- Shapiro-Wilk (Normality): p={sw['p_value']:.4f}, Normal={sw['is_normal']}\n"
        
        if 'ljung_box' in tests:
            lb = tests['ljung_box']
            report += f"- Ljung-Box (Autocorrelation): White Noise={lb['is_white_noise']}\n"
        
        if 'durbin_watson' in tests:
            dw = tests['durbin_watson']
            report += f"- Durbin-Watson: {dw['statistic']:.4f}, Uncorrelated={dw['is_uncorrelated']}\n"
        
        if 'breusch_pagan' in tests:
            bp = tests['breusch_pagan']
            report += f"- Breusch-Pagan (Heteroscedasticity): p={bp['p_value']:.4f}, Homoscedastic={bp['is_homoscedastic']}\n"
        
        if 'adf' in tests:
            adf = tests['adf']
            report += f"- ADF (Stationarity): p={adf['p_value']:.4f}, Stationary={adf['is_stationary']}\n"
        
        # Add model diagnostics
        diagnostics = result.get('model_diagnostics', {})
        report += f"""
MODEL DIAGNOSTICS:
- Nash-Sutcliffe Efficiency: {diagnostics.get('nash_sutcliffe', 'N/A'):.4f}
- R²: {diagnostics.get('coefficient_of_determination', 'N/A'):.4f}
- Outlier Percentage: {diagnostics.get('outlier_percentage', 'N/A'):.2f}%
- Residuals within 2σ: {diagnostics.get('residuals_within_2std', 'N/A'):.2f}%

ANALYSIS TIMESTAMP: {result.get('analysis_timestamp', 'N/A')}
"""
        
        return report
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare residual analysis results across multiple models."""
        
        if not all(name in self.results for name in model_names):
            missing = [name for name in model_names if name not in self.results]
            raise ValueError(f"Missing analysis results for: {missing}")
        
        comparison = {
            'model_names': model_names,
            'metrics_comparison': {},
            'best_model_by_metric': {},
            'statistical_tests_comparison': {}
        }
        
        # Compare basic metrics
        metrics_to_compare = ['mae', 'rmse', 'smape', 'residual_std', 'residual_mean']
        
        for metric in metrics_to_compare:
            comparison['metrics_comparison'][metric] = {}
            for model_name in model_names:
                value = self.results[model_name]['basic_statistics'].get(metric, None)
                comparison['metrics_comparison'][metric][model_name] = value
            
            # Find best model for this metric (lower is better for most metrics)
            valid_values = {k: v for k, v in comparison['metrics_comparison'][metric].items() if v is not None}
            if valid_values:
                best_model = min(valid_values.items(), key=lambda x: x[1])[0]
                comparison['best_model_by_metric'][metric] = best_model
        
        # Compare statistical tests
        for model_name in model_names:
            tests = self.results[model_name].get('statistical_tests', {})
            comparison['statistical_tests_comparison'][model_name] = tests
        
        return comparison
