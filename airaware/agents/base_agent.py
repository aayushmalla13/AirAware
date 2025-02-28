"""
Base Agent Class for AirAware Intelligent Agents

This module provides the base class and configuration for all intelligent agents.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import json
from pathlib import Path


class AgentStatus(str, Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"
    MAINTENANCE = "maintenance"


@dataclass
class AgentConfig:
    """Base configuration for all agents"""
    # Agent identification
    agent_id: str
    agent_name: str
    agent_type: str
    
    # Execution settings
    enabled: bool = True
    max_execution_time: int = 300  # 5 minutes
    retry_attempts: int = 3
    retry_delay: int = 30  # seconds
    
    # Monitoring settings
    log_level: str = "INFO"
    metrics_enabled: bool = True
    health_check_interval: int = 60  # seconds
    
    # Data persistence
    state_file: Optional[str] = None
    metrics_file: Optional[str] = None
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    last_execution_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "average_execution_time": self.average_execution_time,
            "last_execution_time": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "consecutive_failures": self.consecutive_failures,
            "uptime_percentage": self.uptime_percentage
        }


class BaseAgent(ABC):
    """Base class for all intelligent agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(f"agent.{config.agent_id}")
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # Agent state
        self.status = AgentStatus.IDLE
        self.start_time: Optional[datetime] = None
        self.last_health_check: Optional[datetime] = None
        self.metrics = AgentMetrics()
        
        # Execution tracking
        self._current_task: Optional[asyncio.Task] = None
        self._execution_times: List[float] = []
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {
            "start": [],
            "stop": [],
            "error": [],
            "success": [],
            "status_change": []
        }
        
        # Load previous state if available
        self._load_state()
        
        # Ensure status is IDLE after initialization
        self.status = AgentStatus.IDLE
        self.logger.info(f"Agent {self.config.agent_id} initialized with status: {self.status}")
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main logic
        
        Args:
            context: Execution context with relevant data
            
        Returns:
            Execution results
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Perform health check for the agent
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    async def start(self) -> bool:
        """Start the agent"""
        try:
            self.logger.info(f"Starting agent {self.config.agent_id}")
            
            # Emit start event
            await self._emit_event("start", {"agent_id": self.config.agent_id})
            
            # Update status to running during initialization
            self.status = AgentStatus.RUNNING
            self.start_time = datetime.now()
            
            # Save state
            self._save_state()
            
            # Set status to idle after successful initialization
            self.status = AgentStatus.IDLE
            self.logger.info(f"Agent {self.config.agent_id} status set to: {self.status}")
            
            self.logger.info(f"Agent {self.config.agent_id} started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start agent {self.config.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            await self._emit_event("error", {"agent_id": self.config.agent_id, "error": str(e)})
            return False
    
    async def stop(self) -> bool:
        """Stop the agent"""
        try:
            self.logger.info(f"Stopping agent {self.config.agent_id}")
            
            # Cancel current task if running
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
                try:
                    await self._current_task
                except asyncio.CancelledError:
                    pass
            
            # Update status
            self.status = AgentStatus.STOPPED
            
            # Emit stop event
            await self._emit_event("stop", {"agent_id": self.config.agent_id})
            
            # Save state
            self._save_state()
            
            self.logger.info(f"Agent {self.config.agent_id} stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop agent {self.config.agent_id}: {e}")
            return False
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent with retry logic"""
        if not self.config.enabled:
            self.logger.warning(f"Agent {self.config.agent_id} is disabled")
            return {"status": "disabled", "message": "Agent is disabled"}
        
        if self.status == AgentStatus.RUNNING:
            self.logger.warning(f"Agent {self.config.agent_id} is already running (status: {self.status})")
            return {"status": "busy", "message": "Agent is already running"}
        
        execution_start = time.time()
        self.status = AgentStatus.RUNNING
        self.metrics.total_executions += 1
        
        try:
            # Create execution task with timeout
            self._current_task = asyncio.create_task(
                asyncio.wait_for(
                    self.execute(context),
                    timeout=self.config.max_execution_time
                )
            )
            
            # Execute with retry logic
            result = await self._execute_with_retry(context)
            
            # Update metrics
            execution_time = time.time() - execution_start
            self._update_metrics(execution_time, success=True)
            
            # Emit success event
            await self._emit_event("success", {
                "agent_id": self.config.agent_id,
                "execution_time": execution_time,
                "result": result
            })
            
            self.logger.info(f"Agent {self.config.agent_id} executed successfully in {execution_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Agent {self.config.agent_id} execution timed out after {self.config.max_execution_time}s"
            self.logger.error(error_msg)
            self._update_metrics(time.time() - execution_start, success=False)
            await self._emit_event("error", {"agent_id": self.config.agent_id, "error": error_msg})
            return {"status": "timeout", "error": error_msg}
            
        except Exception as e:
            error_msg = f"Agent {self.config.agent_id} execution failed: {e}"
            self.logger.error(error_msg)
            self._update_metrics(time.time() - execution_start, success=False)
            await self._emit_event("error", {"agent_id": self.config.agent_id, "error": error_msg})
            return {"status": "error", "error": error_msg}
            
        finally:
            self.status = AgentStatus.IDLE
            self._current_task = None
            self._save_state()
    
    async def _execute_with_retry(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with retry logic"""
        last_error = None
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt} for agent {self.config.agent_id}")
                    await asyncio.sleep(self.config.retry_delay)
                
                return await self._current_task
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed for agent {self.config.agent_id}: {e}")
                
                if attempt == self.config.retry_attempts:
                    raise last_error
    
    def _update_metrics(self, execution_time: float, success: bool):
        """Update agent metrics"""
        self.metrics.last_execution_time = datetime.now()
        self._execution_times.append(execution_time)
        
        # Keep only last 100 execution times for average calculation
        if len(self._execution_times) > 100:
            self._execution_times = self._execution_times[-100:]
        
        self.metrics.average_execution_time = sum(self._execution_times) / len(self._execution_times)
        
        if success:
            self.metrics.successful_executions += 1
            self.metrics.last_success_time = datetime.now()
            self.metrics.consecutive_failures = 0
        else:
            self.metrics.failed_executions += 1
            self.metrics.last_error_time = datetime.now()
            self.metrics.consecutive_failures += 1
        
        # Calculate uptime percentage
        if self.metrics.total_executions > 0:
            self.metrics.uptime_percentage = (
                self.metrics.successful_executions / self.metrics.total_executions
            ) * 100
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to registered handlers"""
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_type}: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type in self._event_handlers:
            self._event_handlers[event_type].append(handler)
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
    
    def remove_event_handler(self, event_type: str, handler: Callable):
        """Remove event handler"""
        if event_type in self._event_handlers and handler in self._event_handlers[event_type]:
            self._event_handlers[event_type].remove(handler)
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "agent_id": self.config.agent_id,
            "agent_name": self.config.agent_name,
            "agent_type": self.config.agent_type,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "metrics": self.metrics.to_dict()
        }
    
    def _save_state(self):
        """Save agent state to file"""
        if self.config.state_file:
            try:
                state = {
                    "status": self.status.value,
                    "start_time": self.start_time.isoformat() if self.start_time else None,
                    "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
                    "metrics": self.metrics.to_dict(),
                    "execution_times": self._execution_times
                }
                
                state_path = Path(self.config.state_file)
                state_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(state_path, 'w') as f:
                    json.dump(state, f, indent=2)
                    
            except Exception as e:
                self.logger.error(f"Failed to save state for agent {self.config.agent_id}: {e}")
    
    def _load_state(self):
        """Load agent state from file"""
        if self.config.state_file:
            try:
                state_path = Path(self.config.state_file)
                if state_path.exists():
                    with open(state_path, 'r') as f:
                        state = json.load(f)
                    
                    # Restore state
                    self.status = AgentStatus(state.get("status", "idle"))
                    
                    if state.get("start_time"):
                        self.start_time = datetime.fromisoformat(state["start_time"])
                    
                    if state.get("last_health_check"):
                        self.last_health_check = datetime.fromisoformat(state["last_health_check"])
                    
                    # Restore metrics
                    metrics_data = state.get("metrics", {})
                    self.metrics = AgentMetrics(
                        total_executions=metrics_data.get("total_executions", 0),
                        successful_executions=metrics_data.get("successful_executions", 0),
                        failed_executions=metrics_data.get("failed_executions", 0),
                        average_execution_time=metrics_data.get("average_execution_time", 0.0),
                        last_execution_time=datetime.fromisoformat(metrics_data["last_execution_time"]) if metrics_data.get("last_execution_time") else None,
                        last_success_time=datetime.fromisoformat(metrics_data["last_success_time"]) if metrics_data.get("last_success_time") else None,
                        last_error_time=datetime.fromisoformat(metrics_data["last_error_time"]) if metrics_data.get("last_error_time") else None,
                        consecutive_failures=metrics_data.get("consecutive_failures", 0),
                        uptime_percentage=metrics_data.get("uptime_percentage", 100.0)
                    )
                    
                    # Restore execution times
                    self._execution_times = state.get("execution_times", [])
                    
            except Exception as e:
                self.logger.error(f"Failed to load state for agent {self.config.agent_id}: {e}")
    
    def __str__(self) -> str:
        return f"{self.config.agent_name} ({self.config.agent_id}) - {self.status.value}"
    
    def __repr__(self) -> str:
        return f"BaseAgent(id={self.config.agent_id}, status={self.status.value})"
