"""Parallel evaluation framework for baseline forecasting models."""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pickle
import tempfile
import os

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn

from .metrics import ForecastingMetrics, MetricConfig
from ..baselines import (
    SeasonalNaiveForecaster, ProphetBaseline, ARIMABaseline, BaselineEnsemble
)

logger = logging.getLogger(__name__)


class ParallelEvaluationConfig(BaseModel):
    """Configuration for parallel evaluation."""
    # Parallelization settings
    max_workers: int = Field(mp.cpu_count(), description="Maximum number of parallel workers")
    use_threading: bool = Field(False, description="Use threading instead of multiprocessing")
    chunk_size: int = Field(1000, description="Chunk size for data processing")
    
    # Model evaluation settings
    forecast_horizons: List[int] = Field([6, 12, 24], description="Forecast horizons in hours")
    models_to_evaluate: List[str] = Field(
        ["seasonal_naive", "prophet", "arima", "ensemble"], 
        description="Models to evaluate in parallel"
    )
    
    # Performance settings
    enable_model_caching: bool = Field(True, description="Cache fitted models")
    cache_dir: str = Field("data/artifacts/model_cache", description="Directory for model cache")
    
    # Progress tracking
    show_progress: bool = Field(True, description="Show progress bars")
    progress_update_interval: float = Field(0.1, description="Progress update interval in seconds")


class ModelEvaluationTask(BaseModel):
    """Task for parallel model evaluation."""
    model_name: str
    model_class: str
    config: Dict[str, Any]
    train_data_path: str
    test_data_path: str
    station_id: Optional[str] = None
    horizon_hours: int = 24


class ParallelEvaluationResult(BaseModel):
    """Result from parallel evaluation."""
    model_name: str
    success: bool
    mae: Optional[float] = None
    rmse: Optional[float] = None
    smape: Optional[float] = None
    execution_time: float
    error_message: Optional[str] = None
    predictions: Optional[List[float]] = None
    actuals: Optional[List[float]] = None


class ParallelEvaluator:
    """
    Parallel evaluator for baseline forecasting models.
    
    Provides significant speedup by training and evaluating models in parallel,
    with optional model caching to avoid redundant training.
    """
    
    def __init__(self, config: Optional[ParallelEvaluationConfig] = None):
        self.config = config or ParallelEvaluationConfig()
        self.console = Console()
        self.metrics_calculator = ForecastingMetrics()
        
        # Setup model cache directory
        if self.config.enable_model_caching:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        logger.info(f"ParallelEvaluator initialized with {self.config.max_workers} workers")
    
    def evaluate_models_parallel(self, 
                               train_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               target_col: str = 'pm25',
                               group_col: Optional[str] = 'station_id') -> Dict[str, ParallelEvaluationResult]:
        """
        Evaluate multiple models in parallel.
        
        Args:
            train_df: Training data
            test_df: Test data
            target_col: Target variable column name
            group_col: Grouping column name
            
        Returns:
            Dictionary of evaluation results by model name
        """
        self.console.print("🚀 Starting Parallel Model Evaluation", style="bold blue")
        
        start_time = datetime.now()
        
        # Prepare evaluation tasks
        tasks = self._prepare_evaluation_tasks(train_df, test_df, target_col, group_col)
        
        # Execute tasks in parallel
        results = self._execute_tasks_parallel(tasks)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        self.console.print(f"✅ Parallel evaluation completed in {execution_time:.2f}s", style="bold green")
        
        return results
    
    def _prepare_evaluation_tasks(self, 
                                 train_df: pd.DataFrame,
                                 test_df: pd.DataFrame,
                                 target_col: str,
                                 group_col: Optional[str]) -> List[ModelEvaluationTask]:
        """Prepare tasks for parallel execution."""
        
        tasks = []
        
        # Save data to temporary files for multiprocessing
        train_path = self._save_dataframe_to_temp(train_df)
        test_path = self._save_dataframe_to_temp(test_df)
        
        # Get unique stations for evaluation
        stations = train_df[group_col].unique() if group_col else [None]
        
        # Limit to first 3 stations for speed and avoid empty data issues
        stations_to_evaluate = stations[:3] if len(stations) > 3 else stations
        
        # Ensure we have overlapping stations between train and test
        test_stations = test_df[group_col].unique() if group_col else [None]
        overlapping_stations = [s for s in stations_to_evaluate if s in test_stations]
        
        if not overlapping_stations:
            logger.warning("No overlapping stations between train and test data")
            return []
        
        logger.info(f"Evaluating stations: {overlapping_stations}")
        
        for model_name in self.config.models_to_evaluate:
            for station_id in overlapping_stations:
                for horizon in self.config.forecast_horizons:
                    
                    # Check cache first
                    cache_key = self._get_cache_key(model_name, station_id, horizon, train_df.shape[0])
                    if self.config.enable_model_caching and self._is_model_cached(cache_key):
                        logger.info(f"Using cached model for {model_name} (station: {station_id}, horizon: {horizon})")
                        continue
                    
                    task = ModelEvaluationTask(
                        model_name=model_name,
                        model_class=self._get_model_class_name(model_name),
                        config=self._get_model_config(model_name),
                        train_data_path=train_path,
                        test_data_path=test_path,
                        station_id=str(station_id) if station_id else None,
                        horizon_hours=horizon
                    )
                    tasks.append(task)
        
        logger.info(f"Prepared {len(tasks)} evaluation tasks")
        return tasks
    
    def _execute_tasks_parallel(self, tasks: List[ModelEvaluationTask]) -> Dict[str, ParallelEvaluationResult]:
        """Execute tasks in parallel using ProcessPoolExecutor or ThreadPoolExecutor."""
        
        results = {}
        
        if self.config.use_threading:
            executor_class = ThreadPoolExecutor
        else:
            executor_class = ProcessPoolExecutor
        
        with executor_class(max_workers=self.config.max_workers) as executor:
            
            # Submit all tasks
            future_to_task = {
                executor.submit(self._evaluate_single_model, task): task 
                for task in tasks
            }
            
            # Process completed tasks with progress bar
            if self.config.show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=self.console
                ) as progress:
                    
                    task_progress = progress.add_task(
                        f"Evaluating {len(tasks)} models...", 
                        total=len(tasks)
                    )
                    
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            result = future.result()
                            results[f"{task.model_name}_{task.station_id}_{task.horizon_hours}h"] = result
                            
                            if result.success:
                                progress.update(task_progress, 
                                              description=f"✅ {task.model_name} ({task.horizon_hours}h)")
                            else:
                                progress.update(task_progress, 
                                              description=f"❌ {task.model_name} ({task.horizon_hours}h)")
                            
                        except Exception as e:
                            logger.error(f"Task {task.model_name} failed: {e}")
                            results[f"{task.model_name}_{task.station_id}_{task.horizon_hours}h"] = ParallelEvaluationResult(
                                model_name=task.model_name,
                                success=False,
                                execution_time=0.0,
                                error_message=str(e)
                            )
                        
                        progress.advance(task_progress)
            else:
                # Process without progress bar
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results[f"{task.model_name}_{task.station_id}_{task.horizon_hours}h"] = result
                    except Exception as e:
                        logger.error(f"Task {task.model_name} failed: {e}")
                        results[f"{task.model_name}_{task.station_id}_{task.horizon_hours}h"] = ParallelEvaluationResult(
                            model_name=task.model_name,
                            success=False,
                            execution_time=0.0,
                            error_message=str(e)
                        )
        
        return results
    
    def _evaluate_single_model(self, task: ModelEvaluationTask) -> ParallelEvaluationResult:
        """Evaluate a single model (runs in separate process)."""
        
        start_time = datetime.now()
        
        try:
            # Load data
            train_df = pd.read_parquet(task.train_data_path)
            test_df = pd.read_parquet(task.test_data_path)
            
            # Initialize model
            model = self._create_model_instance(task.model_class, task.config)
            
            # Filter data for specific station if needed
            if task.station_id:
                train_df_filtered = train_df[train_df['station_id'] == int(task.station_id)]
                test_df_filtered = test_df[test_df['station_id'] == int(task.station_id)]
                
                # Check if we have enough data
                if len(train_df_filtered) < 10:
                    logger.warning(f"Insufficient training data for station {task.station_id}: {len(train_df_filtered)} records")
                    return ParallelEvaluationResult(
                        model_name=task.model_name,
                        success=False,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        error_message=f"Insufficient training data for station {task.station_id}"
                    )
                
                if len(test_df_filtered) < task.horizon_hours:
                    logger.warning(f"Insufficient test data for station {task.station_id}: {len(test_df_filtered)} records")
                    return ParallelEvaluationResult(
                        model_name=task.model_name,
                        success=False,
                        execution_time=(datetime.now() - start_time).total_seconds(),
                        error_message=f"Insufficient test data for station {task.station_id}"
                    )
                
                train_df = train_df_filtered
                test_df = test_df_filtered
            
            # Fit model
            model.fit(train_df, target_col='pm25', group_col='station_id')
            
            # Generate predictions
            test_timestamps = test_df['datetime_utc'].head(task.horizon_hours).tolist()
            forecast = model.predict(test_timestamps, station_id=int(task.station_id) if task.station_id else None)
            
            # Calculate metrics
            actual_values = test_df['pm25'].head(task.horizon_hours).tolist()
            predictions = forecast.predictions
            
            # Ensure same length
            min_len = min(len(actual_values), len(predictions))
            actual_values = actual_values[:min_len]
            predictions = predictions[:min_len]
            
            # Calculate metrics
            mae = np.mean(np.abs(np.array(actual_values) - np.array(predictions)))
            rmse = np.sqrt(np.mean((np.array(actual_values) - np.array(predictions))**2))
            
            # Calculate sMAPE
            smape_values = []
            for actual, pred in zip(actual_values, predictions):
                if actual + pred != 0:
                    smape_values.append(200 * abs(actual - pred) / (abs(actual) + abs(pred)))
            smape = np.mean(smape_values) if smape_values else 0.0
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ParallelEvaluationResult(
                model_name=task.model_name,
                success=True,
                mae=mae,
                rmse=rmse,
                smape=smape,
                execution_time=execution_time,
                predictions=predictions,
                actuals=actual_values
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Model {task.model_name} evaluation failed: {e}")
            
            return ParallelEvaluationResult(
                model_name=task.model_name,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _create_model_instance(self, model_class: str, config: Dict[str, Any]):
        """Create model instance from class name and config."""
        
        if model_class == "SeasonalNaiveForecaster":
            from ..baselines.seasonal_naive import SeasonalNaiveConfig
            return SeasonalNaiveForecaster(SeasonalNaiveConfig(**config))
        
        elif model_class == "ProphetBaseline":
            from ..baselines.prophet_baseline import ProphetConfig
            return ProphetBaseline(ProphetConfig(**config))
        
        elif model_class == "ARIMABaseline":
            from ..baselines.arima_baseline import ARIMAConfig
            return ARIMABaseline(ARIMAConfig(**config))
        
        elif model_class == "BaselineEnsemble":
            from ..baselines.baseline_ensemble import EnsembleConfig
            return BaselineEnsemble(EnsembleConfig(**config))
        
        else:
            raise ValueError(f"Unknown model class: {model_class}")
    
    def _get_model_class_name(self, model_name: str) -> str:
        """Get model class name from model name."""
        mapping = {
            "seasonal_naive": "SeasonalNaiveForecaster",
            "prophet": "ProphetBaseline", 
            "arima": "ARIMABaseline",
            "ensemble": "BaselineEnsemble"
        }
        return mapping.get(model_name, model_name)
    
    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get default config for model."""
        return {}  # Use default configs
    
    def _save_dataframe_to_temp(self, df: pd.DataFrame) -> str:
        """Save dataframe to temporary parquet file."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        df.to_parquet(temp_file.name)
        return temp_file.name
    
    def _get_cache_key(self, model_name: str, station_id: Optional[str], horizon: int, data_size: int) -> str:
        """Generate cache key for model."""
        return f"{model_name}_{station_id}_{horizon}h_{data_size}"
    
    def _is_model_cached(self, cache_key: str) -> bool:
        """Check if model is cached."""
        cache_path = os.path.join(self.config.cache_dir, f"{cache_key}.pkl")
        return os.path.exists(cache_path)
    
    def _save_model_to_cache(self, model, cache_key: str):
        """Save fitted model to cache."""
        cache_path = os.path.join(self.config.cache_dir, f"{cache_key}.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump(model, f)
    
    def _load_model_from_cache(self, cache_key: str):
        """Load fitted model from cache."""
        cache_path = os.path.join(self.config.cache_dir, f"{cache_key}.pkl")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        import glob
        temp_files = glob.glob("/tmp/tmp*.parquet")
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass
