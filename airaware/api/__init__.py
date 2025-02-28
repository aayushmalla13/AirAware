"""
AirAware API Module

This module provides the FastAPI backend for the AirAware air quality forecasting system.
It includes endpoints for forecasting, explainability, what-if analysis, and model management.
"""

from .app import app
from .models import (
    ForecastRequest,
    ForecastResponse,
    ExplainabilityRequest,
    ExplainabilityResponse,
    WhatIfRequest,
    WhatIfResponse,
    ModelInfo,
    HealthResponse
)
from .services import (
    ModelService,
    ForecastService,
    ExplainabilityService,
    WhatIfService
)

__all__ = [
    "app",
    "ForecastRequest",
    "ForecastResponse", 
    "ExplainabilityRequest",
    "ExplainabilityResponse",
    "WhatIfRequest",
    "WhatIfResponse",
    "ModelInfo",
    "HealthResponse",
    "ModelService",
    "ForecastService",
    "ExplainabilityService",
    "WhatIfService"
]
