"""
Pydantic models for AirAware API

This module defines the request and response models for the FastAPI endpoints.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ModelType(str, Enum):
    """Available model types"""
    PROPHET = "prophet"
    PATCHTST = "patchtst"


class Language(str, Enum):
    """Supported languages"""
    EN = "en"
    NE = "ne"


class AgentType(str, Enum):
    """Available agent types"""
    HEALTH = "health"
    FORECAST = "forecast"
    DATA = "data"
    NOTIFICATION = "notification"
    ORCHESTRATOR = "orchestrator"


class WorkflowType(str, Enum):
    """Available workflow types"""
    FORECAST_GENERATION = "forecast_generation"
    HEALTH_ASSESSMENT = "health_assessment"
    DATA_QUALITY_CHECK = "data_quality_check"
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"


class ForecastHorizon(int, Enum):
    """Available forecast horizons in hours"""
    H6 = 6
    H12 = 12
    H24 = 24


class UncertaintyLevel(float, Enum):
    """Available uncertainty levels"""
    P80 = 0.8
    P90 = 0.9
    P95 = 0.95


class StationInfo(BaseModel):
    """Station information"""
    station_id: str = Field(..., description="Station identifier")
    name: str = Field(..., description="Station name")
    latitude: float = Field(..., description="Station latitude")
    longitude: float = Field(..., description="Station longitude")
    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country code")


class ForecastPoint(BaseModel):
    """Single forecast point"""
    timestamp: datetime = Field(..., description="Forecast timestamp")
    pm25_mean: float = Field(..., description="Mean PM2.5 prediction")
    pm25_lower: Optional[float] = Field(None, description="Lower bound (uncertainty)")
    pm25_upper: Optional[float] = Field(None, description="Upper bound (uncertainty)")
    confidence_level: Optional[float] = Field(None, description="Confidence level")


class ForecastRequest(BaseModel):
    """Request model for forecasting"""
    station_ids: List[str] = Field(..., description="List of station IDs to forecast")
    horizon_hours: ForecastHorizon = Field(24, description="Forecast horizon in hours")
    model_type: ModelType = Field(ModelType.PROPHET, description="Model to use for forecasting")
    uncertainty_level: UncertaintyLevel = Field(0.9, description="Uncertainty level")
    language: Language = Field(Language.EN, description="Response language")
    include_explanations: bool = Field(False, description="Include feature importance explanations")


class ForecastResponse(BaseModel):
    """Response model for forecasting"""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    station_forecasts: Dict[str, List[ForecastPoint]] = Field(..., description="Forecasts by station")
    model_info: Dict[str, Any] = Field(..., description="Model information and metadata")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    language: Language = Field(..., description="Response language")


class ExplainabilityRequest(BaseModel):
    """Request model for explainability analysis"""
    station_id: str = Field(..., description="Station ID to analyze")
    horizon_hours: ForecastHorizon = Field(24, description="Forecast horizon")
    model_type: ModelType = Field(ModelType.PROPHET, description="Model to explain")
    methods: List[str] = Field(["permutation", "tree"], description="Explainability methods")
    language: Language = Field(Language.EN, description="Response language")


class FeatureImportance(BaseModel):
    """Feature importance result"""
    feature_name: str = Field(..., description="Feature name")
    importance_score: float = Field(..., description="Importance score")
    method: str = Field(..., description="Method used for calculation")


class ExplainabilityResponse(BaseModel):
    """Response model for explainability analysis"""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    station_id: str = Field(..., description="Station ID analyzed")
    feature_importance: List[FeatureImportance] = Field(..., description="Feature importance results")
    shap_values: Optional[Dict[str, Any]] = Field(None, description="SHAP values if requested")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    language: Language = Field(..., description="Response language")


class WhatIfRequest(BaseModel):
    """Request model for what-if analysis"""
    station_id: str = Field(..., description="Station ID to analyze")
    horizon_hours: ForecastHorizon = Field(24, description="Forecast horizon")
    model_type: ModelType = Field(ModelType.PROPHET, description="Model to use")
    scenarios: List[Dict[str, Any]] = Field(..., description="What-if scenarios")
    language: Language = Field(Language.EN, description="Response language")


class WhatIfScenario(BaseModel):
    """What-if scenario result"""
    scenario_name: str = Field(..., description="Scenario name")
    scenario_description: str = Field(..., description="Scenario description")
    baseline_forecast: List[ForecastPoint] = Field(..., description="Baseline forecast")
    scenario_forecast: List[ForecastPoint] = Field(..., description="Scenario forecast")
    impact_analysis: Dict[str, Any] = Field(..., description="Impact analysis results")


class WhatIfResponse(BaseModel):
    """Response model for what-if analysis"""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    station_id: str = Field(..., description="Station ID analyzed")
    scenarios: List[WhatIfScenario] = Field(..., description="What-if scenario results")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    language: Language = Field(..., description="Response language")


class ModelInfo(BaseModel):
    """Model information"""
    model_type: ModelType = Field(..., description="Model type")
    version: str = Field(..., description="Model version")
    training_date: datetime = Field(..., description="Training date")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    is_available: bool = Field(..., description="Whether model is available")
    last_updated: datetime = Field(..., description="Last update timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    models: Dict[str, ModelInfo] = Field(..., description="Available models")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model"""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID if available")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# Agent-related models
class AgentStatus(str, Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"
    MAINTENANCE = "maintenance"


class AgentInfo(BaseModel):
    """Agent information"""
    agent_id: str = Field(..., description="Agent identifier")
    agent_name: str = Field(..., description="Agent name")
    agent_type: AgentType = Field(..., description="Agent type")
    status: AgentStatus = Field(..., description="Current agent status")
    enabled: bool = Field(..., description="Whether agent is enabled")
    uptime_percentage: float = Field(..., description="Agent uptime percentage")
    last_execution: Optional[datetime] = Field(None, description="Last execution time")
    metrics: Dict[str, Any] = Field(..., description="Agent performance metrics")


class AgentRequest(BaseModel):
    """Request model for agent operations"""
    agent_type: AgentType = Field(..., description="Type of agent to execute")
    station_id: str = Field(..., description="Station ID for analysis")
    user_id: Optional[str] = Field(None, description="User ID for personalized services")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    language: Language = Field(Language.EN, description="Response language")


class AgentResponse(BaseModel):
    """Response model for agent operations"""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    agent_type: AgentType = Field(..., description="Agent type that processed the request")
    status: str = Field(..., description="Operation status")
    result: Dict[str, Any] = Field(..., description="Agent execution results")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    language: Language = Field(..., description="Response language")


class WorkflowRequest(BaseModel):
    """Request model for workflow execution"""
    workflow_type: WorkflowType = Field(..., description="Type of workflow to execute")
    station_id: str = Field(..., description="Station ID for analysis")
    user_id: Optional[str] = Field(None, description="User ID for personalized services")
    pm25_data: Dict[str, Any] = Field(default_factory=dict, description="PM2.5 data")
    forecast_data: List[Dict[str, Any]] = Field(default_factory=list, description="Forecast data")
    language: Language = Field(Language.EN, description="Response language")


class WorkflowResponse(BaseModel):
    """Response model for workflow execution"""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    workflow_type: WorkflowType = Field(..., description="Workflow type executed")
    status: str = Field(..., description="Workflow status")
    workflow_result: Dict[str, Any] = Field(..., description="Workflow execution results")
    system_insights: List[Dict[str, Any]] = Field(..., description="System insights generated")
    system_health: Dict[str, Any] = Field(..., description="System health status")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    language: Language = Field(..., description="Response language")


class SystemStatusResponse(BaseModel):
    """System status response"""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Status timestamp")
    agents: Dict[str, AgentInfo] = Field(..., description="Agent status information")
    workflows: Dict[str, Any] = Field(..., description="Workflow status information")
    insights: Dict[str, Any] = Field(..., description="System insights")
    system_health: str = Field(..., description="System health status")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
