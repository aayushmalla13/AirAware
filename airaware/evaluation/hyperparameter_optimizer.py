"""Hyperparameter optimization for baseline forecasting models using Optuna."""

import logging
from typing import Dict, List, Optional, Any, Tuple
import json
import os

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from pydantic import BaseModel, Field

from ..baselines import (
    SeasonalNaiveForecaster, ProphetBaseline, ARIMABaseline, BaselineEnsemble
)
from ..baselines.seasonal_naive import SeasonalNaiveConfig
from ..baselines.prophet_baseline import ProphetConfig
from ..baselines.arima_baseline import ARIMAConfig
from ..baselines.baseline_ensemble import EnsembleConfig
from .metrics import ForecastingMetrics

logger = logging.getLogger(__name__)


class HyperparameterOptimizationConfig(BaseModel):
    """Configuration for hyperparameter optimization."""
    
    # Optimization settings
    n_trials: int = Field(50, description="Number of optimization trials")
    timeout: Optional[int] = Field(None, description="Timeout in seconds")
    n_jobs: int = Field(1, description="Number of parallel jobs")
    
    # Model-specific settings
    optimize_seasonal_naive: bool = Field(True)
    optimize_prophet: bool = Field(True)
    optimize_arima: bool = Field(True)
    optimize_ensemble: bool = Field(True)
    
    # Validation settings
    validation_split: float = Field(0.2, description="Validation split ratio")
    cv_folds: int = Field(3, description="Cross-validation folds")
    
    # Optimization objective
    objective_metric: str = Field("mae", description="Metric to optimize")
    direction: str = Field("minimize", description="Optimization direction")
    
    # Storage settings
    study_name: str = Field("airquality_hyperopt", description="Optuna study name")
    storage_url: Optional[str] = Field(None, description="Optuna storage URL")
    
    # Pruning settings
    enable_pruning: bool = Field(True, description="Enable pruning")
    pruning_threshold: float = Field(0.1, description="Pruning threshold")


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer for baseline forecasting models.
    
    Uses Optuna for Bayesian optimization with TPE sampler and median pruner.
    Supports optimization of all baseline models with cross-validation.
    """
    
    def __init__(self, config: Optional[HyperparameterOptimizationConfig] = None):
        self.config = config or HyperparameterOptimizationConfig()
        self.metrics_calculator = ForecastingMetrics()
        
        # Setup Optuna study
        self.study = self._create_study()
        
        logger.info(f"HyperparameterOptimizer initialized with {self.config.n_trials} trials")
    
    def optimize_models(self, 
                      train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      target_col: str = 'pm25',
                      group_col: Optional[str] = 'station_id') -> Dict[str, Any]:
        """
        Optimize hyperparameters for all enabled models.
        
        Args:
            train_df: Training data
            test_df: Test data for final evaluation
            target_col: Target variable column name
            group_col: Grouping column name
            
        Returns:
            Dictionary with optimization results for each model
        """
        logger.info("Starting hyperparameter optimization")
        
        results = {}
        
        # Optimize each model type
        if self.config.optimize_seasonal_naive:
            results['seasonal_naive'] = self._optimize_seasonal_naive(train_df, test_df, target_col, group_col)
        
        if self.config.optimize_prophet:
            results['prophet'] = self._optimize_prophet(train_df, test_df, target_col, group_col)
        
        if self.config.optimize_arima:
            results['arima'] = self._optimize_arima(train_df, test_df, target_col, group_col)
        
        if self.config.optimize_ensemble:
            results['ensemble'] = self._optimize_ensemble(train_df, test_df, target_col, group_col)
        
        # Save results
        self._save_optimization_results(results)
        
        logger.info("Hyperparameter optimization completed")
        return results
    
    def _optimize_seasonal_naive(self, 
                                train_df: pd.DataFrame,
                                test_df: pd.DataFrame,
                                target_col: str,
                                group_col: Optional[str]) -> Dict[str, Any]:
        """Optimize Seasonal Naive hyperparameters."""
        
        def objective(trial):
            # Define hyperparameter search space
            seasonal_period = trial.suggest_categorical('seasonal_period', [24, 168, 720])  # 24h, weekly, monthly
            min_samples = trial.suggest_int('min_samples', 2, 10)
            
            # Create config
            config = SeasonalNaiveConfig(
                seasonal_period=seasonal_period,
                min_samples=min_samples
            )
            
            # Evaluate model
            score = self._evaluate_model(
                SeasonalNaiveForecaster(config),
                train_df, test_df, target_col, group_col
            )
            
            return score
        
        # Run optimization
        study = optuna.create_study(
            direction=self.config.direction,
            sampler=TPESampler(),
            pruner=MedianPruner() if self.config.enable_pruning else None
        )
        
        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }
    
    def _optimize_prophet(self, 
                         train_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         target_col: str,
                         group_col: Optional[str]) -> Dict[str, Any]:
        """Optimize Prophet hyperparameters."""
        
        def objective(trial):
            # Define hyperparameter search space
            yearly_seasonality = trial.suggest_categorical('yearly_seasonality', [True, False])
            weekly_seasonality = trial.suggest_categorical('weekly_seasonality', [True, False])
            daily_seasonality = trial.suggest_categorical('daily_seasonality', [True, False])
            
            seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
            changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True)
            seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True)
            
            # External regressor settings
            include_meteorological = trial.suggest_categorical('include_meteorological', [True, False])
            include_calendar_events = trial.suggest_categorical('include_calendar_events', [True, False])
            include_pollution_features = trial.suggest_categorical('include_pollution_features', [True, False])
            
            # Create config
            config = ProphetConfig(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                include_meteorological=include_meteorological,
                include_calendar_events=include_calendar_events,
                include_pollution_features=include_pollution_features
            )
            
            # Evaluate model
            score = self._evaluate_model(
                ProphetBaseline(config),
                train_df, test_df, target_col, group_col
            )
            
            return score
        
        # Run optimization
        study = optuna.create_study(
            direction=self.config.direction,
            sampler=TPESampler(),
            pruner=MedianPruner() if self.config.enable_pruning else None
        )
        
        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }
    
    def _optimize_arima(self, 
                       train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       target_col: str,
                       group_col: Optional[str]) -> Dict[str, Any]:
        """Optimize ARIMA hyperparameters."""
        
        def objective(trial):
            # Define hyperparameter search space
            max_p = trial.suggest_int('max_p', 1, 5)
            max_d = trial.suggest_int('max_d', 0, 2)
            max_q = trial.suggest_int('max_q', 1, 5)
            
            max_P = trial.suggest_int('max_P', 0, 2)
            max_D = trial.suggest_int('max_D', 0, 1)
            max_Q = trial.suggest_int('max_Q', 0, 2)
            
            seasonal_period = trial.suggest_categorical('seasonal_period', [24, 168])
            trend = trial.suggest_categorical('trend', ['n', 'c', 't', 'ct'])
            
            # Create config
            config = ARIMAConfig(
                max_p=max_p,
                max_d=max_d,
                max_q=max_q,
                max_P=max_P,
                max_D=max_D,
                max_Q=max_Q,
                seasonal_period=seasonal_period,
                trend=trend
            )
            
            # Evaluate model
            score = self._evaluate_model(
                ARIMABaseline(config),
                train_df, test_df, target_col, group_col
            )
            
            return score
        
        # Run optimization
        study = optuna.create_study(
            direction=self.config.direction,
            sampler=TPESampler(),
            pruner=MedianPruner() if self.config.enable_pruning else None
        )
        
        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }
    
    def _optimize_ensemble(self, 
                         train_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         target_col: str,
                         group_col: Optional[str]) -> Dict[str, Any]:
        """Optimize Ensemble hyperparameters."""
        
        def objective(trial):
            # Define hyperparameter search space
            include_seasonal_naive = trial.suggest_categorical('include_seasonal_naive', [True, False])
            include_prophet = trial.suggest_categorical('include_prophet', [True, False])
            include_arima = trial.suggest_categorical('include_arima', [True, False])
            
            # Weight optimization
            if include_seasonal_naive and include_prophet and include_arima:
                weight_sn = trial.suggest_float('weight_seasonal_naive', 0.1, 0.8)
                weight_prophet = trial.suggest_float('weight_prophet', 0.1, 0.8)
                weight_arima = 1.0 - weight_sn - weight_prophet
                
                if weight_arima < 0.1:
                    weight_arima = 0.1
                    weight_sn = (weight_sn / (weight_sn + weight_prophet)) * 0.9
                    weight_prophet = (weight_prophet / (weight_sn + weight_prophet)) * 0.9
            
            # Create config
            config = EnsembleConfig(
                include_seasonal_naive=include_seasonal_naive,
                include_prophet=include_prophet,
                include_arima=include_arima,
                weights={
                    'seasonal_naive': weight_sn if include_seasonal_naive else 0,
                    'prophet': weight_prophet if include_prophet else 0,
                    'arima': weight_arima if include_arima else 0
                }
            )
            
            # Evaluate model
            score = self._evaluate_model(
                BaselineEnsemble(config),
                train_df, test_df, target_col, group_col
            )
            
            return score
        
        # Run optimization
        study = optuna.create_study(
            direction=self.config.direction,
            sampler=TPESampler(),
            pruner=MedianPruner() if self.config.enable_pruning else None
        )
        
        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }
    
    def _evaluate_model(self, 
                       model: Any,
                       train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       target_col: str,
                       group_col: Optional[str]) -> float:
        """Evaluate model using cross-validation."""
        
        # Split data for validation
        n_train = int(len(train_df) * (1 - self.config.validation_split))
        val_train_df = train_df.iloc[:n_train]
        val_test_df = train_df.iloc[n_train:]
        
        try:
            # Fit model
            model.fit(val_train_df, target_col=target_col, group_col=group_col)
            
            # Generate predictions
            test_timestamps = val_test_df['datetime_utc'].head(24).tolist()
            forecast = model.predict(test_timestamps, station_id=val_test_df['station_id'].iloc[0] if group_col else None)
            
            # Calculate metrics
            actual_values = val_test_df['pm25'].head(24).tolist()
            predictions = forecast.predictions
            
            # Ensure same length
            min_len = min(len(actual_values), len(predictions))
            actual_values = actual_values[:min_len]
            predictions = predictions[:min_len]
            
            # Calculate target metric
            if self.config.objective_metric == 'mae':
                return np.mean(np.abs(np.array(actual_values) - np.array(predictions)))
            elif self.config.objective_metric == 'rmse':
                return np.sqrt(np.mean((np.array(actual_values) - np.array(predictions))**2))
            elif self.config.objective_metric == 'smape':
                smape_values = []
                for actual, pred in zip(actual_values, predictions):
                    if actual + pred != 0:
                        smape_values.append(200 * abs(actual - pred) / (abs(actual) + abs(pred)))
                return np.mean(smape_values) if smape_values else 0.0
            else:
                return np.mean(np.abs(np.array(actual_values) - np.array(predictions)))
                
        except Exception as e:
            logger.warning(f"Model evaluation failed: {e}")
            return float('inf')
    
    def _create_study(self) -> optuna.Study:
        """Create Optuna study."""
        
        sampler = TPESampler()
        pruner = MedianPruner() if self.config.enable_pruning else None
        
        if self.config.storage_url:
            study = optuna.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage_url,
                direction=self.config.direction,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )
        else:
            study = optuna.create_study(
                direction=self.config.direction,
                sampler=sampler,
                pruner=pruner
            )
        
        return study
    
    def _save_optimization_results(self, results: Dict[str, Any]):
        """Save optimization results to file."""
        
        output_dir = "data/artifacts"
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare results for JSON serialization
        serializable_results = {}
        for model_name, result in results.items():
            serializable_results[model_name] = {
                'best_params': result['best_params'],
                'best_value': result['best_value'],
                'n_trials': result['n_trials']
            }
        
        # Save to file
        output_path = os.path.join(output_dir, "hyperparameter_optimization_results.json")
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Optimization results saved to {output_path}")
    
    def get_best_model_configs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get best model configurations from optimization results."""
        
        best_configs = {}
        
        for model_name, result in results.items():
            if model_name == 'seasonal_naive':
                best_configs[model_name] = SeasonalNaiveConfig(**result['best_params'])
            elif model_name == 'prophet':
                best_configs[model_name] = ProphetConfig(**result['best_params'])
            elif model_name == 'arima':
                best_configs[model_name] = ARIMAConfig(**result['best_params'])
            elif model_name == 'ensemble':
                best_configs[model_name] = EnsembleConfig(**result['best_params'])
        
        return best_configs
