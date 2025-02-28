"""Evaluation framework for time series forecasting models."""

from .metrics import ForecastingMetrics, MetricConfig, MetricResult
from .baseline_evaluator import BaselineEvaluator, BaselineEvaluationConfig, BaselineEvaluationResult
from .rolling_cv import RollingOriginCV, RollingCVConfig, RollingCVResult
from .residual_analysis import (
    ResidualAnalyzer,
    ResidualAnalysisConfig
)
from .parallel_evaluator import ParallelEvaluator, ParallelEvaluationConfig, ParallelEvaluationResult

# Optional hyperparameter optimizer (requires optuna)
try:
    from .hyperparameter_optimizer import HyperparameterOptimizer, HyperparameterOptimizationConfig
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

__all__ = [
    "ForecastingMetrics",
    "MetricConfig",
    "MetricResult",
    "BaselineEvaluator",
    "BaselineEvaluationConfig",
    "BaselineEvaluationResult",
    "RollingOriginCV",
    "RollingCVConfig",
    "RollingCVResult",
    "ResidualAnalyzer",
    "ResidualAnalysisConfig",
    "ParallelEvaluator",
    "ParallelEvaluationConfig",
    "ParallelEvaluationResult",
]

if HYPEROPT_AVAILABLE:
    __all__.extend([
        "HyperparameterOptimizer",
        "HyperparameterOptimizationConfig"
    ])
