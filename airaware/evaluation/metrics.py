"""Comprehensive forecasting metrics for time series evaluation."""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MetricConfig(BaseModel):
    """Configuration for forecasting metrics."""
    include_mae: bool = Field(True, description="Include Mean Absolute Error")
    include_rmse: bool = Field(True, description="Include Root Mean Square Error")
    include_smape: bool = Field(True, description="Include Symmetric Mean Absolute Percentage Error")
    include_mase: bool = Field(True, description="Include Mean Absolute Scaled Error")
    include_pinball: bool = Field(True, description="Include Pinball Loss (quantile loss)")
    include_coverage: bool = Field(True, description="Include interval coverage metrics")
    
    # Pinball loss quantiles
    quantiles: List[float] = Field(default=[0.1, 0.5, 0.9], description="Quantiles for pinball loss")
    
    # MASE baseline
    mase_baseline_period: int = Field(24, description="Baseline period for MASE (hours)")
    
    # Coverage intervals
    coverage_levels: List[float] = Field(default=[0.8, 0.9], description="Coverage levels to evaluate")


class MetricResult(BaseModel):
    """Result of metric calculation."""
    metric_name: str
    value: float
    details: Dict = Field(default_factory=dict)


class ForecastingMetrics:
    """
    Comprehensive forecasting metrics for time series evaluation.
    
    Implements standard forecasting metrics including:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Square Error) 
    - sMAPE (Symmetric Mean Absolute Percentage Error)
    - MASE (Mean Absolute Scaled Error)
    - Pinball Loss (Quantile Loss)
    - Coverage metrics for uncertainty quantification
    """
    
    def __init__(self, config: Optional[MetricConfig] = None):
        self.config = config or MetricConfig()
        logger.info("ForecastingMetrics initialized")
    
    def evaluate_point_forecast(self, y_true: Union[List, np.ndarray, pd.Series],
                               y_pred: Union[List, np.ndarray, pd.Series],
                               baseline_data: Optional[Union[List, np.ndarray, pd.Series]] = None) -> Dict[str, MetricResult]:
        """
        Evaluate point forecasts using multiple metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            baseline_data: Historical data for MASE calculation
            
        Returns:
            Dictionary of metric results
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
        
        # Remove missing values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            logger.warning("No valid predictions to evaluate")
            return {}
        
        results = {}
        
        # Mean Absolute Error
        if self.config.include_mae:
            mae = self._calculate_mae(y_true_clean, y_pred_clean)
            results['MAE'] = MetricResult(
                metric_name='MAE',
                value=mae,
                details={'n_observations': len(y_true_clean)}
            )
        
        # Root Mean Square Error
        if self.config.include_rmse:
            rmse = self._calculate_rmse(y_true_clean, y_pred_clean)
            results['RMSE'] = MetricResult(
                metric_name='RMSE',
                value=rmse,
                details={'n_observations': len(y_true_clean)}
            )
        
        # Symmetric Mean Absolute Percentage Error
        if self.config.include_smape:
            smape = self._calculate_smape(y_true_clean, y_pred_clean)
            results['sMAPE'] = MetricResult(
                metric_name='sMAPE',
                value=smape,
                details={'n_observations': len(y_true_clean)}
            )
        
        # Mean Absolute Scaled Error
        if self.config.include_mase and baseline_data is not None:
            mase = self._calculate_mase(y_true_clean, y_pred_clean, baseline_data)
            results['MASE'] = MetricResult(
                metric_name='MASE',
                value=mase,
                details={
                    'n_observations': len(y_true_clean),
                    'baseline_period': self.config.mase_baseline_period
                }
            )
        
        return results
    
    def evaluate_probabilistic_forecast(self, y_true: Union[List, np.ndarray, pd.Series],
                                      quantile_predictions: Dict[str, Union[List, np.ndarray, pd.Series]]) -> Dict[str, MetricResult]:
        """
        Evaluate probabilistic forecasts using quantile-based metrics.
        
        Args:
            y_true: True values
            quantile_predictions: Dictionary with quantile levels as keys (e.g., 'q0.1', 'q0.5', 'q0.9')
            
        Returns:
            Dictionary of metric results
        """
        y_true = np.array(y_true)
        results = {}
        
        # Pinball Loss for each quantile
        if self.config.include_pinball:
            pinball_results = {}
            total_pinball = 0.0
            
            for quantile in self.config.quantiles:
                q_key = f'q{quantile:.1f}'
                
                if q_key in quantile_predictions:
                    y_pred_q = np.array(quantile_predictions[q_key])
                    
                    # Remove missing values
                    mask = ~(np.isnan(y_true) | np.isnan(y_pred_q))
                    y_true_clean = y_true[mask]
                    y_pred_q_clean = y_pred_q[mask]
                    
                    if len(y_true_clean) > 0:
                        pinball = self._calculate_pinball_loss(y_true_clean, y_pred_q_clean, quantile)
                        pinball_results[q_key] = pinball
                        total_pinball += pinball
            
            results['Pinball_Loss'] = MetricResult(
                metric_name='Pinball_Loss',
                value=total_pinball / len(pinball_results) if pinball_results else 0.0,
                details={
                    'by_quantile': pinball_results,
                    'quantiles_evaluated': list(pinball_results.keys())
                }
            )
        
        # Coverage metrics
        if self.config.include_coverage:
            coverage_results = self._calculate_coverage_metrics(y_true, quantile_predictions)
            results.update(coverage_results)
        
        return results
    
    def evaluate_horizon_performance(self, y_true: Union[List, np.ndarray, pd.Series],
                                   y_pred: Union[List, np.ndarray, pd.Series],
                                   horizons: Union[List, np.ndarray],
                                   baseline_data: Optional[Union[List, np.ndarray, pd.Series]] = None) -> Dict[int, Dict[str, MetricResult]]:
        """
        Evaluate performance at different forecast horizons.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            horizons: Forecast horizon for each prediction (in hours)
            baseline_data: Historical data for MASE calculation
            
        Returns:
            Dictionary with horizon as key and metrics as values
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        horizons = np.array(horizons)
        
        horizon_results = {}
        
        # Group by horizon
        unique_horizons = np.unique(horizons)
        
        for horizon in unique_horizons:
            mask = horizons == horizon
            y_true_h = y_true[mask]
            y_pred_h = y_pred[mask]
            
            if len(y_true_h) > 0:
                horizon_results[int(horizon)] = self.evaluate_point_forecast(
                    y_true_h, y_pred_h, baseline_data
                )
        
        return horizon_results
    
    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator > 1e-8
        if np.sum(mask) == 0:
            return 0.0
        
        smape = np.mean(numerator[mask] / denominator[mask]) * 100
        return float(smape)
    
    def _calculate_mase(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       baseline_data: Union[List, np.ndarray, pd.Series]) -> float:
        """Calculate Mean Absolute Scaled Error."""
        baseline_data = np.array(baseline_data)
        
        # Calculate seasonal naive baseline error
        if len(baseline_data) <= self.config.mase_baseline_period:
            # Not enough data for seasonal baseline, use simple naive
            baseline_mae = np.mean(np.abs(np.diff(baseline_data))) if len(baseline_data) > 1 else 1.0
        else:
            # Seasonal naive error
            seasonal_predictions = baseline_data[:-self.config.mase_baseline_period]
            seasonal_actual = baseline_data[self.config.mase_baseline_period:]
            baseline_mae = np.mean(np.abs(seasonal_actual - seasonal_predictions))
        
        if baseline_mae == 0:
            baseline_mae = 1e-8  # Avoid division by zero
        
        mae = self._calculate_mae(y_true, y_pred)
        mase = mae / baseline_mae
        
        return float(mase)
    
    def _calculate_pinball_loss(self, y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        """Calculate Pinball Loss (Quantile Loss) for a specific quantile."""
        error = y_true - y_pred
        loss = np.where(error >= 0, quantile * error, (quantile - 1) * error)
        return float(np.mean(loss))
    
    def _calculate_coverage_metrics(self, y_true: np.ndarray, 
                                  quantile_predictions: Dict[str, np.ndarray]) -> Dict[str, MetricResult]:
        """Calculate interval coverage metrics."""
        results = {}
        
        for coverage_level in self.config.coverage_levels:
            # Determine lower and upper quantiles for this coverage level
            alpha = 1 - coverage_level
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2
            
            lower_key = f'q{lower_q:.1f}'
            upper_key = f'q{upper_q:.1f}'
            
            if lower_key in quantile_predictions and upper_key in quantile_predictions:
                lower_pred = np.array(quantile_predictions[lower_key])
                upper_pred = np.array(quantile_predictions[upper_key])
                
                # Remove missing values
                mask = ~(np.isnan(y_true) | np.isnan(lower_pred) | np.isnan(upper_pred))
                y_true_clean = y_true[mask]
                lower_clean = lower_pred[mask]
                upper_clean = upper_pred[mask]
                
                if len(y_true_clean) > 0:
                    # Calculate coverage
                    in_interval = (y_true_clean >= lower_clean) & (y_true_clean <= upper_clean)
                    coverage = np.mean(in_interval)
                    
                    # Calculate average interval width
                    avg_width = np.mean(upper_clean - lower_clean)
                    
                    # Calculate coverage error
                    coverage_error = abs(coverage - coverage_level)
                    
                    results[f'Coverage_{int(coverage_level*100)}'] = MetricResult(
                        metric_name=f'Coverage_{int(coverage_level*100)}',
                        value=float(coverage),
                        details={
                            'target_coverage': coverage_level,
                            'actual_coverage': float(coverage),
                            'coverage_error': float(coverage_error),
                            'average_width': float(avg_width),
                            'n_observations': len(y_true_clean)
                        }
                    )
        
        return results
    
    def calculate_relative_performance(self, model_metrics: Dict[str, MetricResult],
                                     baseline_metrics: Dict[str, MetricResult]) -> Dict[str, float]:
        """
        Calculate relative performance compared to baseline.
        
        Args:
            model_metrics: Metrics for the model being evaluated
            baseline_metrics: Metrics for the baseline model
            
        Returns:
            Dictionary of relative improvements (positive = better than baseline)
        """
        relative_performance = {}
        
        for metric_name in model_metrics:
            if metric_name in baseline_metrics:
                model_value = model_metrics[metric_name].value
                baseline_value = baseline_metrics[metric_name].value
                
                if baseline_value != 0:
                    # For error metrics (lower is better), calculate improvement
                    if metric_name in ['MAE', 'RMSE', 'sMAPE', 'MASE', 'Pinball_Loss']:
                        improvement = (baseline_value - model_value) / baseline_value
                    else:
                        # For coverage metrics (closer to target is better)
                        improvement = model_value - baseline_value
                    
                    relative_performance[f'{metric_name}_improvement'] = improvement
        
        return relative_performance
    
    def get_metric_summary(self, metrics: Dict[str, MetricResult]) -> Dict[str, float]:
        """Get a summary of key metrics for reporting."""
        
        summary = {}
        
        # Extract key metric values
        for metric_name, result in metrics.items():
            summary[metric_name] = result.value
        
        return summary


