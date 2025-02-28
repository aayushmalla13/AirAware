"""
Rolling Origin Cross-Validation for Time Series Forecasting

This module implements rolling-origin cross-validation specifically designed for
time series forecasting, ensuring proper temporal ordering and avoiding data leakage.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from ..baselines import (
    SeasonalNaiveForecaster, 
    ProphetBaseline, 
    ARIMABaseline, 
    BaselineEnsemble
)
from .metrics import ForecastingMetrics

logger = logging.getLogger(__name__)

@dataclass
class RollingCVConfig:
    """Configuration for rolling-origin cross-validation."""
    
    # Temporal parameters
    initial_train_size: int = field(default=168, metadata={"description": "Initial training window size (hours)"})
    step_size: int = field(default=24, metadata={"description": "Step size between origins (hours)"})
    max_origins: int = field(default=10, metadata={"description": "Maximum number of origins"})
    min_train_size: int = field(default=72, metadata={"description": "Minimum training window size"})
    
    # Forecast horizons
    horizons: List[int] = field(default_factory=lambda: [6, 12, 24], 
                               metadata={"description": "Forecast horizons in hours"})
    
    # Validation parameters
    gap_size: int = field(default=0, metadata={"description": "Gap between train and test (hours)"})
    min_test_size: int = field(default=6, metadata={"description": "Minimum test set size"})
    
    # Parallel processing
    n_jobs: int = field(default=1, metadata={"description": "Number of parallel jobs"})
    random_state: int = field(default=42, metadata={"description": "Random state for reproducibility"})
    
    # Model-specific parameters
    refit_models: bool = field(default=True, metadata={"description": "Refit models for each origin"})
    save_predictions: bool = field(default=False, metadata={"description": "Save individual predictions"})
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.initial_train_size < self.min_train_size:
            raise ValueError("initial_train_size must be >= min_train_size")
        
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        
        if any(h <= 0 for h in self.horizons):
            raise ValueError("All horizons must be positive")
        
        if self.gap_size < 0:
            raise ValueError("gap_size must be non-negative")

@dataclass
class RollingCVResult:
    """Results from rolling-origin cross-validation."""
    
    # Configuration
    config: RollingCVConfig
    model_name: str
    
    # Temporal information
    origins: List[pd.Timestamp]
    train_sizes: List[int]
    test_sizes: List[int]
    
    # Performance metrics
    metrics_by_horizon: Dict[int, Dict[str, float]]
    metrics_by_origin: Dict[int, Dict[str, float]]
    
    # Detailed results
    predictions: Optional[Dict[int, List[float]]] = None
    actuals: Optional[Dict[int, List[float]]] = None
    
    # Metadata
    execution_time: float = 0.0
    n_origins_completed: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class RollingOriginCV:
    """
    Rolling Origin Cross-Validation for Time Series Forecasting.
    
    This implementation ensures proper temporal ordering by:
    1. Using only past data for training
    2. Maintaining chronological order
    3. Avoiding data leakage
    4. Supporting multiple forecast horizons
    """
    
    def __init__(self, config: RollingCVConfig):
        self.config = config
        self.metrics_calculator = ForecastingMetrics()
        
    def evaluate_model(self, 
                      data: pd.DataFrame,
                      model_class: Any,
                      model_name: str,
                      target_col: str = 'pm25',
                      group_col: str = 'station_id',
                      datetime_col: str = 'datetime_utc') -> RollingCVResult:
        """
        Perform rolling-origin cross-validation for a single model.
        
        Args:
            data: Time series data with datetime index
            model_class: Model class to evaluate
            model_name: Name of the model for identification
            target_col: Name of target variable column
            group_col: Name of grouping column (e.g., station_id)
            datetime_col: Name of datetime column
            
        Returns:
            RollingCVResult with comprehensive evaluation metrics
        """
        logger.info(f"Starting rolling-origin CV for {model_name}")
        start_time = datetime.now()
        
        # Prepare data
        data_sorted = self._prepare_data(data, datetime_col, target_col, group_col)
        
        # Generate rolling origins
        origins = self._generate_origins(data_sorted)
        
        if not origins:
            logger.warning("No valid origins found for rolling CV")
            return self._create_empty_result(model_name)
        
        logger.info(f"Generated {len(origins)} rolling origins")
        
        # Initialize result containers
        result = RollingCVResult(
            config=self.config,
            model_name=model_name,
            origins=origins,
            train_sizes=[],
            test_sizes=[],
            metrics_by_horizon={h: {} for h in self.config.horizons},
            metrics_by_origin={},
            predictions={h: [] for h in self.config.horizons} if self.config.save_predictions else None,
            actuals={h: [] for h in self.config.horizons} if self.config.save_predictions else None,
            errors=[]
        )
        
        # Perform rolling evaluation
        if self.config.n_jobs > 1:
            result = self._evaluate_parallel(data_sorted, model_class, result)
        else:
            result = self._evaluate_sequential(data_sorted, model_class, result)
        
        # Calculate final metrics
        result.execution_time = (datetime.now() - start_time).total_seconds()
        result.n_origins_completed = len(result.origins)
        
        logger.info(f"Rolling CV completed for {model_name} in {result.execution_time:.2f}s")
        return result
    
    def _prepare_data(self, 
                      data: pd.DataFrame, 
                      datetime_col: str,
                      target_col: str, 
                      group_col: str) -> pd.DataFrame:
        """Prepare and validate data for rolling CV."""
        
        # Ensure datetime column is properly formatted
        if not pd.api.types.is_datetime64_any_dtype(data[datetime_col]):
            data[datetime_col] = pd.to_datetime(data[datetime_col])
        
        # Sort by datetime and group
        data_sorted = data.sort_values([group_col, datetime_col]).copy()
        
        # Remove duplicates
        data_sorted = data_sorted.drop_duplicates(subset=[group_col, datetime_col])
        
        # Check for sufficient data
        min_required = (self.config.initial_train_size + 
                       max(self.config.horizons) + 
                       self.config.gap_size)
        
        if len(data_sorted) < min_required:
            raise ValueError(f"Insufficient data: need at least {min_required} records, got {len(data_sorted)}")
        
        return data_sorted
    
    def _generate_origins(self, data: pd.DataFrame) -> List[pd.Timestamp]:
        """Generate rolling origin timestamps."""
        
        origins = []
        start_time = data['datetime_utc'].min()
        end_time = data['datetime_utc'].max()
        
        current_origin = start_time + pd.Timedelta(hours=self.config.initial_train_size)
        
        origin_count = 0
        while (current_origin <= end_time - pd.Timedelta(hours=max(self.config.horizons) + self.config.gap_size) 
               and origin_count < self.config.max_origins):
            
            origins.append(current_origin)
            current_origin += pd.Timedelta(hours=self.config.step_size)
            origin_count += 1
        
        return origins
    
    def _evaluate_sequential(self, 
                           data: pd.DataFrame, 
                           model_class: Any, 
                           result: RollingCVResult) -> RollingCVResult:
        """Perform sequential rolling evaluation."""
        
        for i, origin in enumerate(result.origins):
            try:
                logger.info(f"Evaluating origin {i+1}/{len(result.origins)}: {origin}")
                
                # Split data for this origin
                train_data, test_data = self._split_data_at_origin(data, origin)
                
                if train_data is None or test_data is None:
                    result.errors.append(f"Origin {origin}: Insufficient data for split")
                    continue
                
                # Record split sizes
                result.train_sizes.append(len(train_data))
                result.test_sizes.append(len(test_data))
                
                # Fit model
                model = model_class()
                model.fit(train_data, target_col='pm25', group_col='station_id')
                
                # Evaluate each horizon
                origin_metrics = {}
                for horizon in self.config.horizons:
                    horizon_metrics = self._evaluate_horizon(
                        model, test_data, horizon, origin, result
                    )
                    
                    if horizon_metrics:
                        origin_metrics.update(horizon_metrics)
                        
                        # Update horizon-level metrics
                        for metric_name, metric_value in horizon_metrics.items():
                            if metric_name not in result.metrics_by_horizon[horizon]:
                                result.metrics_by_horizon[horizon][metric_name] = []
                            result.metrics_by_horizon[horizon][metric_name].append(metric_value)
                
                result.metrics_by_origin[i] = origin_metrics
                
            except Exception as e:
                error_msg = f"Origin {origin}: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
        
        return result
    
    def _evaluate_parallel(self, 
                          data: pd.DataFrame, 
                          model_class: Any, 
                          result: RollingCVResult) -> RollingCVResult:
        """Perform parallel rolling evaluation."""
        
        def evaluate_origin(origin_data):
            origin, train_data, test_data = origin_data
            
            try:
                # Fit model
                model = model_class()
                model.fit(train_data, target_col='pm25', group_col='station_id')
                
                # Evaluate each horizon
                origin_metrics = {}
                for horizon in self.config.horizons:
                    horizon_metrics = self._evaluate_horizon(
                        model, test_data, horizon, origin, result
                    )
                    
                    if horizon_metrics:
                        origin_metrics.update(horizon_metrics)
                
                return origin, origin_metrics, None
                
            except Exception as e:
                return origin, {}, str(e)
        
        # Prepare data for parallel processing
        origin_data = []
        for i, origin in enumerate(result.origins):
            train_data, test_data = self._split_data_at_origin(data, origin)
            if train_data is not None and test_data is not None:
                origin_data.append((origin, train_data, test_data))
                result.train_sizes.append(len(train_data))
                result.test_sizes.append(len(test_data))
        
        # Execute parallel evaluation
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            future_to_origin = {
                executor.submit(evaluate_origin, data): data[0] 
                for data in origin_data
            }
            
            for future in as_completed(future_to_origin):
                origin, origin_metrics, error = future.result()
                
                if error:
                    result.errors.append(f"Origin {origin}: {error}")
                else:
                    result.metrics_by_origin[origin] = origin_metrics
                    
                    # Update horizon-level metrics
                    for horizon in self.config.horizons:
                        for metric_name, metric_value in origin_metrics.items():
                            if metric_name not in result.metrics_by_horizon[horizon]:
                                result.metrics_by_horizon[horizon][metric_name] = []
                            result.metrics_by_horizon[horizon][metric_name].append(metric_value)
        
        return result
    
    def _split_data_at_origin(self, 
                            data: pd.DataFrame, 
                            origin: pd.Timestamp) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Split data at a specific origin timestamp."""
        
        # Training data: all data before origin
        train_data = data[data['datetime_utc'] < origin].copy()
        
        # Test data: data after origin + gap
        test_start = origin + pd.Timedelta(hours=self.config.gap_size)
        test_end = test_start + pd.Timedelta(hours=max(self.config.horizons))
        test_data = data[
            (data['datetime_utc'] >= test_start) & 
            (data['datetime_utc'] <= test_end)
        ].copy()
        
        # Validate split
        if len(train_data) < self.config.min_train_size:
            return None, None
        
        if len(test_data) < self.config.min_test_size:
            return None, None
        
        return train_data, test_data
    
    def _evaluate_horizon(self, 
                         model: Any, 
                         test_data: pd.DataFrame, 
                         horizon: int,
                         origin: pd.Timestamp,
                         result: RollingCVResult) -> Dict[str, float]:
        """Evaluate model performance for a specific horizon."""
        
        try:
            # Generate forecast
            forecast_start = origin + pd.Timedelta(hours=self.config.gap_size)
            forecast = model.forecast(forecast_start, horizon)
            
            # Get actual values for the same period
            actual_start = forecast_start
            actual_end = actual_start + pd.Timedelta(hours=horizon)
            
            actual_data = test_data[
                (test_data['datetime_utc'] >= actual_start) & 
                (test_data['datetime_utc'] < actual_end)
            ].copy()
            
            if len(actual_data) == 0:
                return {}
            
            # Align predictions with actuals
            # Handle multi-station predictions by taking first station's predictions
            if len(forecast.predictions) > len(actual_data):
                # Multi-station predictions - take first station
                predictions = forecast.predictions[:len(actual_data)]
            else:
                predictions = forecast.predictions
            
            actuals = actual_data['pm25'].tolist()
            
            # Ensure same length
            min_length = min(len(predictions), len(actuals))
            predictions = predictions[:min_length]
            actuals = actuals[:min_length]
            
            # Calculate metrics
            metric_results = self.metrics_calculator.evaluate_point_forecast(
                y_true=actuals,
                y_pred=predictions
            )
            
            # Convert MetricResult objects to simple float values
            metrics = {}
            for metric_name, metric_result in metric_results.items():
                metrics[metric_name] = metric_result.value
            
            # Save predictions if requested
            if self.config.save_predictions:
                result.predictions[horizon].extend(predictions)
                result.actuals[horizon].extend(actuals)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Horizon {horizon} evaluation failed: {e}")
            return {}
    
    def _create_empty_result(self, model_name: str) -> RollingCVResult:
        """Create empty result for failed evaluation."""
        return RollingCVResult(
            config=self.config,
            model_name=model_name,
            origins=[],
            train_sizes=[],
            test_sizes=[],
            metrics_by_horizon={h: {} for h in self.config.horizons},
            metrics_by_origin={},
            errors=[f"No valid origins found for {model_name}"]
        )
    
    def compare_models(self, 
                      data: pd.DataFrame,
                      models: Dict[str, Any],
                      target_col: str = 'pm25',
                      group_col: str = 'station_id',
                      datetime_col: str = 'datetime_utc') -> Dict[str, RollingCVResult]:
        """
        Compare multiple models using rolling-origin CV.
        
        Args:
            data: Time series data
            models: Dictionary mapping model names to model classes
            target_col: Name of target variable column
            group_col: Name of grouping column
            datetime_col: Name of datetime column
            
        Returns:
            Dictionary mapping model names to RollingCVResult objects
        """
        results = {}
        
        for model_name, model_class in models.items():
            logger.info(f"Evaluating {model_name} with rolling-origin CV")
            results[model_name] = self.evaluate_model(
                data=data,
                model_class=model_class,
                model_name=model_name,
                target_col=target_col,
                group_col=group_col,
                datetime_col=datetime_col
            )
        
        return results
    
    def generate_summary_report(self, results: Dict[str, RollingCVResult]) -> str:
        """Generate a comprehensive summary report of rolling CV results."""
        
        report = []
        report.append("=" * 80)
        report.append("ROLLING-ORIGIN CROSS-VALIDATION SUMMARY REPORT")
        report.append("=" * 80)
        
        # Configuration summary
        report.append(f"\nConfiguration:")
        report.append(f"  • Initial train size: {self.config.initial_train_size} hours")
        report.append(f"  • Step size: {self.config.step_size} hours")
        report.append(f"  • Max origins: {self.config.max_origins}")
        report.append(f"  • Horizons: {self.config.horizons} hours")
        report.append(f"  • Gap size: {self.config.gap_size} hours")
        
        # Model comparison
        report.append(f"\nModel Performance Summary:")
        report.append("-" * 50)
        
        for model_name, result in results.items():
            report.append(f"\n{model_name}:")
            report.append(f"  • Origins completed: {result.n_origins_completed}")
            report.append(f"  • Execution time: {result.execution_time:.2f}s")
            report.append(f"  • Errors: {len(result.errors)}")
            
            if result.errors:
                report.append(f"  • Error details: {result.errors[:3]}...")
            
            # Average metrics by horizon
            for horizon in self.config.horizons:
                if horizon in result.metrics_by_horizon:
                    metrics = result.metrics_by_horizon[horizon]
                    if metrics:
                        report.append(f"  • Horizon {horizon}h:")
                        for metric_name, values in metrics.items():
                            if values:
                                avg_value = np.mean(values)
                                std_value = np.std(values)
                                report.append(f"    - {metric_name}: {avg_value:.3f} ± {std_value:.3f}")
        
        # Best model by horizon
        report.append(f"\nBest Model by Horizon:")
        report.append("-" * 30)
        
        for horizon in self.config.horizons:
            best_model = None
            best_mae = float('inf')
            
            for model_name, result in results.items():
                if (horizon in result.metrics_by_horizon and 
                    'mae' in result.metrics_by_horizon[horizon] and
                    result.metrics_by_horizon[horizon]['mae']):
                    
                    avg_mae = np.mean(result.metrics_by_horizon[horizon]['mae'])
                    if avg_mae < best_mae:
                        best_mae = avg_mae
                        best_model = model_name
            
            if best_model:
                report.append(f"  • {horizon}h: {best_model} (MAE: {best_mae:.3f})")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
