"""Comprehensive evaluation framework for baseline forecasting models."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .metrics import ForecastingMetrics, MetricConfig
from ..baselines import (
    SeasonalNaiveForecaster, ProphetBaseline, ARIMABaseline, BaselineEnsemble
)

logger = logging.getLogger(__name__)


class BaselineEvaluationConfig(BaseModel):
    """Configuration for baseline evaluation."""
    # Data splits
    train_end_date: str = Field("2024-12-31", description="End date for training data")
    validation_end_date: str = Field("2025-06-30", description="End date for validation data")
    
    # Forecast horizons to evaluate (success criteria: 6h, 12h, 24h)
    forecast_horizons: List[int] = Field([6, 12, 24], description="Forecast horizons in hours")
    
    # Rolling evaluation
    rolling_window_days: int = Field(7, description="Rolling window size in days")
    rolling_step_days: int = Field(1, description="Rolling step size in days")
    
    # Models to evaluate
    evaluate_seasonal_naive: bool = Field(True)
    evaluate_prophet: bool = Field(True)
    evaluate_arima: bool = Field(True)
    evaluate_ensemble: bool = Field(True)
    
    # Evaluation settings
    min_test_samples: int = Field(100, description="Minimum test samples per horizon")
    calculate_relative_improvement: bool = Field(True, description="Calculate improvement vs seasonal-naive")
    
    # Output settings
    save_predictions: bool = Field(True, description="Save detailed predictions")
    save_visualizations: bool = Field(True, description="Save evaluation plots")


class ModelPerformance(BaseModel):
    """Performance metrics for a single model."""
    model_name: str
    horizon_hours: int
    mae: float
    rmse: float
    smape: float
    mase: Optional[float] = None
    pinball_loss: Optional[float] = None
    coverage_80: Optional[float] = None
    coverage_90: Optional[float] = None
    n_predictions: int
    relative_improvement_mae: Optional[float] = None


class BaselineEvaluationResult(BaseModel):
    """Complete evaluation result for all baseline models."""
    evaluation_config: BaselineEvaluationConfig
    evaluation_timestamp: str
    data_summary: Dict
    model_performances: List[ModelPerformance]
    champion_model: Dict
    summary_metrics: Dict
    horizon_analysis: Dict
    rolling_analysis: Optional[Dict] = None


class BaselineEvaluator:
    """
    Comprehensive evaluator for baseline forecasting models.
    
    Evaluates multiple baselines on multiple horizons using time series cross-validation.
    Provides detailed performance comparison and identifies champion model.
    """
    
    def __init__(self, config: Optional[BaselineEvaluationConfig] = None):
        self.config = config or BaselineEvaluationConfig()
        self.console = Console()
        self.metrics_calculator = ForecastingMetrics()
        
        logger.info("BaselineEvaluator initialized")
    
    def evaluate_all_baselines(self, df: pd.DataFrame, 
                              target_col: str = 'pm25',
                              group_col: Optional[str] = 'station_id') -> BaselineEvaluationResult:
        """
        Evaluate all baseline models on the provided dataset.
        
        Args:
            df: Complete dataset with datetime_utc, target, and features
            target_col: Name of target variable column
            group_col: Name of grouping column (e.g., station_id)
            
        Returns:
            Comprehensive evaluation results
        """
        self.console.print("🚀 Starting Comprehensive Baseline Evaluation", style="bold blue")
        
        start_time = datetime.now()
        
        # Split data
        train_df, val_df, test_df = self._split_data(df)
        
        data_summary = {
            'total_records': len(df),
            'train_records': len(train_df),
            'validation_records': len(val_df),
            'test_records': len(test_df),
            'stations': df[group_col].nunique() if group_col else 1,
            'date_range': {
                'start': df['datetime_utc'].min().isoformat(),
                'end': df['datetime_utc'].max().isoformat()
            },
            'target_stats': {
                'mean': float(df[target_col].mean()),
                'std': float(df[target_col].std()),
                'min': float(df[target_col].min()),
                'max': float(df[target_col].max())
            }
        }
        
        self.console.print(f"📊 Data Summary:")
        self.console.print(f"  • Total Records: {data_summary['total_records']:,}")
        self.console.print(f"  • Train/Val/Test: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
        self.console.print(f"  • Stations: {data_summary['stations']}")
        self.console.print(f"  • PM2.5 Range: {data_summary['target_stats']['min']:.1f} - {data_summary['target_stats']['max']:.1f} μg/m³")
        
        # Initialize models
        models = self._initialize_models()
        
        # Fit models
        fitted_models = self._fit_models(models, train_df, target_col, group_col)
        
        # Evaluate models
        all_performances = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            total_evaluations = len(fitted_models) * len(self.config.forecast_horizons)
            task = progress.add_task("Evaluating models...", total=total_evaluations)
            
            for model_name, model in fitted_models.items():
                for horizon in self.config.forecast_horizons:
                    progress.update(task, description=f"Evaluating {model_name} @ {horizon}h...")
                    
                    performance = self._evaluate_single_model(
                        model, model_name, horizon, test_df, target_col, group_col
                    )
                    
                    if performance:
                        all_performances.append(performance)
                    
                    progress.advance(task)
        
        # Calculate relative improvements
        if self.config.calculate_relative_improvement:
            all_performances = self._calculate_relative_improvements(all_performances)
        
        # Analyze results
        champion_model = self._identify_champion_model(all_performances)
        summary_metrics = self._calculate_summary_metrics(all_performances)
        horizon_analysis = self._analyze_horizon_performance(all_performances)
        
        # Optional rolling analysis
        rolling_analysis = None
        if self.config.rolling_window_days > 0:
            self.console.print("📈 Performing rolling evaluation...")
            rolling_analysis = self._perform_rolling_evaluation(fitted_models, test_df, target_col, group_col)
        
        evaluation_result = BaselineEvaluationResult(
            evaluation_config=self.config,
            evaluation_timestamp=start_time.isoformat(),
            data_summary=data_summary,
            model_performances=all_performances,
            champion_model=champion_model,
            summary_metrics=summary_metrics,
            horizon_analysis=horizon_analysis,
            rolling_analysis=rolling_analysis
        )
        
        # Display results
        self._display_results(evaluation_result)
        
        return evaluation_result
    
    def _split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into train/validation/test sets."""
        
        train_end = pd.to_datetime(self.config.train_end_date, utc=True)
        val_end = pd.to_datetime(self.config.validation_end_date, utc=True)
        
        train_df = df[df['datetime_utc'] <= train_end].copy()
        val_df = df[(df['datetime_utc'] > train_end) & (df['datetime_utc'] <= val_end)].copy()
        test_df = df[df['datetime_utc'] > val_end].copy()
        
        return train_df, val_df, test_df
    
    def _initialize_models(self) -> Dict:
        """Initialize baseline models based on configuration."""
        
        models = {}
        
        if self.config.evaluate_seasonal_naive:
            models['seasonal_naive'] = SeasonalNaiveForecaster()
        
        if self.config.evaluate_prophet:
            models['prophet'] = ProphetBaseline()
        
        if self.config.evaluate_arima:
            models['arima'] = ARIMABaseline()
        
        if self.config.evaluate_ensemble:
            models['ensemble'] = BaselineEnsemble()
        
        return models
    
    def _fit_models(self, models: Dict, train_df: pd.DataFrame, 
                   target_col: str, group_col: Optional[str]) -> Dict:
        """Fit all models on training data."""
        
        fitted_models = {}
        
        for model_name, model in models.items():
            try:
                self.console.print(f"🔧 Fitting {model_name}...")
                model.fit(train_df, target_col, group_col)
                fitted_models[model_name] = model
                self.console.print(f"  ✅ {model_name} fitted successfully")
                
            except Exception as e:
                self.console.print(f"  ❌ {model_name} fitting failed: {e}")
        
        return fitted_models
    
    def _evaluate_single_model(self, model, model_name: str, horizon: int,
                              test_df: pd.DataFrame, target_col: str, 
                              group_col: Optional[str]) -> Optional[ModelPerformance]:
        """Evaluate a single model at a specific horizon."""
        
        try:
            # Generate predictions for the test set
            predictions = []
            actuals = []
            
            # Group by station if applicable
            if group_col:
                stations = test_df[group_col].unique()
            else:
                stations = [None]
            
            for station in stations:
                station_data = test_df[test_df[group_col] == station] if station is not None else test_df
                
                # Create prediction windows
                for i in range(0, len(station_data) - horizon, horizon):
                    start_idx = i + horizon  # Start prediction after horizon
                    
                    if start_idx >= len(station_data):
                        break
                    
                    # Get forecast timestamp
                    forecast_time = station_data.iloc[start_idx]['datetime_utc']
                    
                    try:
                        # Generate prediction
                        forecast = model.forecast(forecast_time, horizon, station)
                        
                        # Get actual values for the forecast period
                        end_time = forecast_time + pd.Timedelta(hours=horizon-1)
                        actual_window = station_data[
                            (station_data['datetime_utc'] >= forecast_time) & 
                            (station_data['datetime_utc'] <= end_time)
                        ]
                        
                        if len(actual_window) == horizon and len(forecast.predictions) == horizon:
                            predictions.extend(forecast.predictions)
                            actuals.extend(actual_window[target_col].tolist())
                    
                    except Exception as e:
                        continue  # Skip failed predictions
            
            if len(predictions) < self.config.min_test_samples:
                logger.warning(f"Insufficient test samples for {model_name} @ {horizon}h: {len(predictions)}")
                return None
            
            # Calculate metrics
            predictions_array = np.array(predictions)
            actuals_array = np.array(actuals)
            
            # Remove any NaN values
            mask = ~(np.isnan(predictions_array) | np.isnan(actuals_array))
            predictions_clean = predictions_array[mask]
            actuals_clean = actuals_array[mask]
            
            if len(predictions_clean) == 0:
                return None
            
            # Calculate metrics
            mae = float(np.mean(np.abs(actuals_clean - predictions_clean)))
            rmse = float(np.sqrt(np.mean((actuals_clean - predictions_clean) ** 2)))
            
            # sMAPE
            denominator = (np.abs(actuals_clean) + np.abs(predictions_clean)) / 2
            mask_nonzero = denominator > 1e-8
            smape = float(np.mean(np.abs(actuals_clean[mask_nonzero] - predictions_clean[mask_nonzero]) / denominator[mask_nonzero]) * 100)
            
            return ModelPerformance(
                model_name=model_name,
                horizon_hours=horizon,
                mae=mae,
                rmse=rmse,
                smape=smape,
                n_predictions=len(predictions_clean)
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name} @ {horizon}h: {e}")
            return None
    
    def _calculate_relative_improvements(self, performances: List[ModelPerformance]) -> List[ModelPerformance]:
        """Calculate relative improvements compared to seasonal-naive baseline."""
        
        # Find seasonal-naive performance for each horizon
        baseline_mae = {}
        
        for perf in performances:
            if perf.model_name == 'seasonal_naive':
                baseline_mae[perf.horizon_hours] = perf.mae
        
        # Calculate improvements
        for perf in performances:
            if perf.horizon_hours in baseline_mae:
                baseline = baseline_mae[perf.horizon_hours]
                if baseline > 0:
                    improvement = (baseline - perf.mae) / baseline
                    perf.relative_improvement_mae = improvement
        
        return performances
    
    def _identify_champion_model(self, performances: List[ModelPerformance]) -> Dict:
        """Identify the best performing model overall."""
        
        if not performances:
            return {}
        
        # Group by model and calculate average MAE
        model_avg_mae = {}
        model_counts = {}
        
        for perf in performances:
            if perf.model_name not in model_avg_mae:
                model_avg_mae[perf.model_name] = 0
                model_counts[perf.model_name] = 0
            
            model_avg_mae[perf.model_name] += perf.mae
            model_counts[perf.model_name] += 1
        
        # Calculate averages
        for model_name in model_avg_mae:
            model_avg_mae[model_name] /= model_counts[model_name]
        
        # Find champion
        champion_name = min(model_avg_mae.keys(), key=lambda k: model_avg_mae[k])
        
        return {
            'model_name': champion_name,
            'average_mae': model_avg_mae[champion_name],
            'all_model_maes': model_avg_mae
        }
    
    def _calculate_summary_metrics(self, performances: List[ModelPerformance]) -> Dict:
        """Calculate summary metrics across all models and horizons."""
        
        if not performances:
            return {}
        
        summary = {
            'total_evaluations': len(performances),
            'models_evaluated': len(set(p.model_name for p in performances)),
            'horizons_evaluated': len(set(p.horizon_hours for p in performances)),
            'average_mae': np.mean([p.mae for p in performances]),
            'average_rmse': np.mean([p.rmse for p in performances]),
            'average_smape': np.mean([p.smape for p in performances])
        }
        
        # Success criteria check: beat seasonal-naive by ≥10-15% at horizons {6, 12, 24}
        success_criteria = {}
        for horizon in [6, 12, 24]:
            horizon_perfs = [p for p in performances if p.horizon_hours == horizon]
            
            if horizon_perfs:
                seasonal_naive_mae = next((p.mae for p in horizon_perfs if p.model_name == 'seasonal_naive'), None)
                
                if seasonal_naive_mae:
                    improvements = []
                    for p in horizon_perfs:
                        if p.model_name != 'seasonal_naive' and p.relative_improvement_mae is not None:
                            improvements.append(p.relative_improvement_mae)
                    
                    if improvements:
                        best_improvement = max(improvements)
                        success_criteria[f'{horizon}h'] = {
                            'best_improvement_pct': best_improvement * 100,
                            'meets_10pct_target': best_improvement >= 0.10,
                            'meets_15pct_target': best_improvement >= 0.15
                        }
        
        summary['success_criteria'] = success_criteria
        
        return summary
    
    def _analyze_horizon_performance(self, performances: List[ModelPerformance]) -> Dict:
        """Analyze performance by forecast horizon."""
        
        horizon_analysis = {}
        
        for horizon in self.config.forecast_horizons:
            horizon_perfs = [p for p in performances if p.horizon_hours == horizon]
            
            if horizon_perfs:
                horizon_analysis[f'{horizon}h'] = {
                    'models': [p.model_name for p in horizon_perfs],
                    'mae_values': [p.mae for p in horizon_perfs],
                    'best_model': min(horizon_perfs, key=lambda p: p.mae).model_name,
                    'best_mae': min(p.mae for p in horizon_perfs),
                    'mae_range': {
                        'min': min(p.mae for p in horizon_perfs),
                        'max': max(p.mae for p in horizon_perfs),
                        'std': np.std([p.mae for p in horizon_perfs])
                    }
                }
        
        return horizon_analysis
    
    def _perform_rolling_evaluation(self, fitted_models: Dict, test_df: pd.DataFrame,
                                   target_col: str, group_col: Optional[str]) -> Dict:
        """Perform rolling window evaluation (simplified version)."""
        
        # This would implement rolling cross-validation
        # For now, return a placeholder
        return {
            'rolling_window_days': self.config.rolling_window_days,
            'rolling_step_days': self.config.rolling_step_days,
            'note': 'Rolling evaluation implementation pending'
        }
    
    def _display_results(self, result: BaselineEvaluationResult):
        """Display evaluation results in a formatted table."""
        
        from rich.table import Table
        
        # Performance table
        table = Table(title="🏆 Baseline Model Performance Comparison")
        
        table.add_column("Model", style="cyan")
        table.add_column("Horizon", justify="center")
        table.add_column("MAE", justify="right", style="green")
        table.add_column("RMSE", justify="right")
        table.add_column("sMAPE (%)", justify="right")
        table.add_column("Improvement (%)", justify="right", style="magenta")
        table.add_column("N Samples", justify="right", style="dim")
        
        for perf in result.model_performances:
            improvement_str = f"{perf.relative_improvement_mae*100:+.1f}" if perf.relative_improvement_mae else "—"
            
            table.add_row(
                perf.model_name,
                f"{perf.horizon_hours}h",
                f"{perf.mae:.2f}",
                f"{perf.rmse:.2f}",
                f"{perf.smape:.1f}",
                improvement_str,
                str(perf.n_predictions)
            )
        
        self.console.print(table)
        
        # Champion model
        if result.champion_model:
            self.console.print(f"\n🥇 Champion Model: {result.champion_model['model_name']} (MAE: {result.champion_model['average_mae']:.2f})")
        
        # Success criteria
        if 'success_criteria' in result.summary_metrics:
            self.console.print(f"\n🎯 Success Criteria Analysis:")
            for horizon, criteria in result.summary_metrics['success_criteria'].items():
                target_10 = "✅" if criteria['meets_10pct_target'] else "❌"
                target_15 = "✅" if criteria['meets_15pct_target'] else "❌"
                self.console.print(f"  • {horizon}: {criteria['best_improvement_pct']:+.1f}% improvement {target_10} 10% {target_15} 15%")
    
    def save_results(self, result: BaselineEvaluationResult, output_path: str = "data/artifacts/baseline_evaluation.json"):
        """Save evaluation results to file."""
        
        import json
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        result_dict = result.model_dump()
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.console.print(f"💾 Results saved to {output_path}")
        
        return output_path


