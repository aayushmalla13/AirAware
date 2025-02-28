"""
Intelligent Agents Module for AirAware

This module provides intelligent agents for automated air quality management,
health advisory, forecast optimization, and system monitoring.
"""

from .base_agent import BaseAgent, AgentConfig, AgentStatus
from .health_agent import HealthAdvisoryAgent, HealthAgentConfig
from .forecast_agent import ForecastOptimizationAgent, ForecastAgentConfig
from .data_agent import DataQualityAgent, DataAgentConfig
from .notification_agent import NotificationAgent, NotificationAgentConfig
from .orchestrator import AgentOrchestrator, OrchestratorConfig

__all__ = [
    "BaseAgent",
    "AgentConfig", 
    "AgentStatus",
    "HealthAdvisoryAgent",
    "HealthAgentConfig",
    "ForecastOptimizationAgent",
    "ForecastAgentConfig",
    "DataQualityAgent",
    "DataAgentConfig",
    "NotificationAgent",
    "NotificationAgentConfig",
    "AgentOrchestrator",
    "OrchestratorConfig"
]
