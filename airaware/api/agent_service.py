"""
Agent Service for AirAware API

This module provides agent management and execution services.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from ..agents import (
    AgentOrchestrator, OrchestratorConfig,
    HealthAdvisoryAgent, HealthAgentConfig,
    ForecastOptimizationAgent, ForecastAgentConfig,
    DataQualityAgent, DataAgentConfig,
    NotificationAgent, NotificationAgentConfig
)
from .models import (
    AgentRequest, AgentResponse, WorkflowRequest, WorkflowResponse,
    SystemStatusResponse, AgentInfo, AgentType, WorkflowType, AgentStatus
)

logger = logging.getLogger(__name__)


class AgentService:
    """Service for managing and executing intelligent agents"""
    
    def __init__(self):
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.agents: Dict[str, Any] = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize the agent service"""
        if self.initialized:
            return
        
        try:
            logger.info("Initializing Agent Service...")
            
            # Create orchestrator configuration
            orchestrator_config = OrchestratorConfig(
                agent_id="api_orchestrator",
                agent_name="API Orchestrator",
                agent_type="orchestrator",
                enabled=True,
                log_level="INFO",
                # Agent configurations
                health_agent_config=HealthAgentConfig(
                    agent_id="health_agent",
                    agent_name="Health Advisory Agent",
                    agent_type="health"
                ),
                forecast_agent_config=ForecastAgentConfig(
                    agent_id="forecast_agent",
                    agent_name="Forecast Optimization Agent",
                    agent_type="forecast"
                ),
                data_agent_config=DataAgentConfig(
                    agent_id="data_agent",
                    agent_name="Data Quality Monitoring Agent",
                    agent_type="data"
                ),
                notification_agent_config=NotificationAgentConfig(
                    agent_id="notification_agent",
                    agent_name="Notification Agent",
                    agent_type="notification"
                )
            )
            
            # Create orchestrator
            self.orchestrator = AgentOrchestrator(orchestrator_config)
            
            # Start orchestrator
            success = await self.orchestrator.start()
            if not success:
                raise Exception("Failed to start orchestrator")
            
            # Start all agents
            start_results = await self.orchestrator.start_all_agents()
            failed_agents = [agent_id for agent_id, result in start_results.items() if not result]
            
            if failed_agents:
                logger.warning(f"Some agents failed to start: {failed_agents}")
            
            # Store agent references
            self.agents = self.orchestrator.agents
            
            self.initialized = True
            logger.info("Agent Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Agent Service: {e}")
            raise
    
    async def execute_agent(self, request: AgentRequest) -> AgentResponse:
        """Execute a specific agent"""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Executing {request.agent_type} agent for station {request.station_id}")
            
            # Get the appropriate agent
            agent = self.agents.get(request.agent_type.value)
            if not agent:
                raise ValueError(f"Agent type {request.agent_type} not found")
            
            # Prepare context
            context = {
                "station_id": request.station_id,
                "user_id": request.user_id,
                "language": request.language.value,
                **request.context
            }
            
            # Add available models for forecast agent
            if request.agent_type.value == "forecast":
                context["available_models"] = ["prophet", "patchtst"]
                context["horizon_hours"] = request.context.get("horizon_hours", 24)
            
            # Execute agent
            result = await agent.run(context)
            
            processing_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                request_id=request_id,
                timestamp=datetime.now(),
                agent_type=request.agent_type,
                status="success",
                result=result,
                processing_time_ms=processing_time,
                language=request.language
            )
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                request_id=request_id,
                timestamp=datetime.now(),
                agent_type=request.agent_type,
                status="error",
                result={"error": str(e)},
                processing_time_ms=processing_time,
                language=request.language
            )
    
    async def execute_workflow(self, request: WorkflowRequest) -> WorkflowResponse:
        """Execute a workflow using the orchestrator"""
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Executing {request.workflow_type} workflow for station {request.station_id}")
            
            # Prepare context
            context = {
                "workflow_type": request.workflow_type.value,
                "station_id": request.station_id,
                "user_id": request.user_id,
                "pm25_data": request.pm25_data,
                "forecast_data": request.forecast_data,
                "language": request.language.value
            }
            
            # Execute workflow
            result = await self.orchestrator.run(context)
            
            processing_time = (time.time() - start_time) * 1000
            
            return WorkflowResponse(
                request_id=request_id,
                timestamp=datetime.now(),
                workflow_type=request.workflow_type,
                status=result.get("status", "success"),
                workflow_result=result.get("workflow_result", {}),
                system_insights=result.get("system_insights", []),
                system_health=result.get("system_health", {}),
                processing_time_ms=processing_time,
                language=request.language
            )
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return WorkflowResponse(
                request_id=request_id,
                timestamp=datetime.now(),
                workflow_type=request.workflow_type,
                status="error",
                workflow_result={"error": str(e)},
                system_insights=[],
                system_health={"status": "error"},
                processing_time_ms=processing_time,
                language=request.language
            )
    
    async def get_system_status(self) -> SystemStatusResponse:
        """Get comprehensive system status"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get orchestrator status
            orchestrator_status = self.orchestrator.get_system_status()
            
            # Convert agent status to AgentInfo objects
            agents_info = {}
            for agent_id, agent in self.agents.items():
                agent_status = agent.get_status()
                agents_info[agent_id] = AgentInfo(
                    agent_id=agent_status["agent_id"],
                    agent_name=agent_status["agent_name"],
                    agent_type=AgentType(agent_status["agent_type"]),
                    status=AgentStatus(agent_status["status"]),
                    enabled=agent_status["enabled"],
                    uptime_percentage=agent_status["metrics"]["uptime_percentage"],
                    last_execution=datetime.fromisoformat(agent_status["metrics"]["last_execution_time"]) if agent_status["metrics"]["last_execution_time"] else None,
                    metrics=agent_status["metrics"]
                )
            
            return SystemStatusResponse(
                status="healthy" if orchestrator_status["system_health"] == "healthy" else "degraded",
                timestamp=datetime.now(),
                agents=agents_info,
                workflows=orchestrator_status["workflows"],
                insights=orchestrator_status["insights"],
                system_health=orchestrator_status["system_health"],
                uptime_seconds=orchestrator_status.get("uptime_seconds", 0)
            )
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return SystemStatusResponse(
                status="error",
                timestamp=datetime.now(),
                agents={},
                workflows={},
                insights={},
                system_health="error",
                uptime_seconds=0
            )
    
    async def health_check(self) -> bool:
        """Perform health check on all agents"""
        if not self.initialized:
            return False
        
        try:
            return await self.orchestrator.health_check()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the agent service"""
        if self.orchestrator:
            try:
                await self.orchestrator.stop_all_agents()
                await self.orchestrator.stop()
                logger.info("Agent Service shutdown completed")
            except Exception as e:
                logger.error(f"Error during agent service shutdown: {e}")
        
        self.initialized = False
