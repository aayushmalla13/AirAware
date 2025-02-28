"""
Explainability Module

This module provides explainability and what-if analysis capabilities
for air quality forecasting models.
"""

from .feature_importance import FeatureImportanceAnalyzer, FeatureImportanceConfig
from .shap_explainer import SHAPExplainer, SHAPConfig
from .what_if_analysis import WhatIfAnalyzer, WhatIfConfig
from .explainability_utils import (
    plot_feature_importance,
    plot_shap_summary,
    plot_lime_explanation,
    plot_what_if_scenarios,
    plot_counterfactual_examples,
    plot_sensitivity_analysis,
    create_explainability_report,
    evaluate_explainability_quality
)

__all__ = [
    "FeatureImportanceAnalyzer",
    "FeatureImportanceConfig",
    "SHAPExplainer",
    "SHAPConfig",
    "WhatIfAnalyzer",
    "WhatIfConfig",
    "plot_feature_importance",
    "plot_shap_summary",
    "plot_lime_explanation",
    "plot_what_if_scenarios",
    "plot_counterfactual_examples",
    "plot_sensitivity_analysis",
    "create_explainability_report",
    "evaluate_explainability_quality"
]
