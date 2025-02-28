"""
FastAPI application for AirAware

This module provides the main FastAPI application with all endpoints.
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .models import (
    ForecastRequest, ForecastResponse,
    ExplainabilityRequest, ExplainabilityResponse,
    WhatIfRequest, WhatIfResponse,
    ModelInfo, HealthResponse, ErrorResponse,
    ModelType, Language,
    AgentRequest, AgentResponse, WorkflowRequest, WorkflowResponse,
    SystemStatusResponse
)
from .services import ModelService, ForecastService, ExplainabilityService, WhatIfService
from .agent_service import AgentService
from .cross_station_service import CrossStationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
model_service = None
forecast_service = None
explainability_service = None
whatif_service = None
agent_service = None
cross_station_service = None
start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global model_service, forecast_service, explainability_service, whatif_service, agent_service, cross_station_service, start_time
    
    # Startup
    logger.info("🚀 Starting AirAware API...")
    start_time = time.time()
    
    try:
        # Initialize services
        model_service = ModelService()
        forecast_service = ForecastService(model_service)
        explainability_service = ExplainabilityService(model_service)
        whatif_service = WhatIfService(model_service)
        agent_service = AgentService()
        cross_station_service = CrossStationService()
        
        # Initialize agent service
        await agent_service.initialize()
        
        logger.info("✅ AirAware API started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start AirAware API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AirAware API...")
    if agent_service:
        await agent_service.shutdown()


# Create FastAPI app
app = FastAPI(
    title="AirAware API",
    description="Production-grade PM₂.₅ nowcasting for Kathmandu Valley",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get services
def get_model_service() -> ModelService:
    if model_service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    return model_service


def get_forecast_service() -> ForecastService:
    if forecast_service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    return forecast_service


def get_explainability_service() -> ExplainabilityService:
    if explainability_service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    return explainability_service


def get_whatif_service() -> WhatIfService:
    if whatif_service is None:
        raise HTTPException(status_code=503, detail="Service not available")
    return whatif_service


def get_agent_service() -> AgentService:
    """Get Agent service dependency"""
    if agent_service is None:
        raise HTTPException(status_code=503, detail="Agent service not available")
    return agent_service


def get_cross_station_service() -> CrossStationService:
    """Get cross-station service dependency"""
    if cross_station_service is None:
        raise HTTPException(status_code=503, detail="Cross-station service not available")
    return cross_station_service


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            error_message="An internal error occurred",
            timestamp=datetime.now()
        ).dict()
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global start_time
    
    try:
        # Get model information
        models = {}
        for model_type in ModelType:
            try:
                model_info = model_service.get_model_info(model_type)
                models[model_type.value] = model_info
            except Exception as e:
                logger.warning(f"Failed to get info for model {model_type}: {e}")
                models[model_type.value] = ModelInfo(
                    model_type=model_type,
                    version="unknown",
                    training_date=datetime.now(),
                    performance_metrics={},
                    is_available=False,
                    last_updated=datetime.now()
                )
        
        uptime = time.time() - start_time if start_time else 0
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            models=models,
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Model information endpoint
@app.get("/models", response_model=Dict[str, ModelInfo])
async def get_models(model_service: ModelService = Depends(get_model_service)):
    """Get information about available models"""
    try:
        models = {}
        for model_type in ModelType:
            models[model_type.value] = model_service.get_model_info(model_type)
        
        return models
        
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")


# Station information endpoint
@app.get("/stations")
async def get_stations(forecast_service: ForecastService = Depends(get_forecast_service)):
    """Get list of available stations"""
    try:
        stations = forecast_service.get_available_stations()
        return {"stations": stations}
        
    except Exception as e:
        logger.error(f"Failed to get stations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get station information")


# Forecast endpoint
@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(
    request: ForecastRequest,
    forecast_service: ForecastService = Depends(get_forecast_service)
):
    """Generate PM₂.₅ forecasts"""
    try:
        logger.info(f"Generating forecast for stations: {request.station_ids}")
        
        # Validate request
        if not request.station_ids:
            raise HTTPException(status_code=400, detail="No stations specified")
        
        if request.horizon_hours not in [6, 12, 24]:
            raise HTTPException(status_code=400, detail="Invalid horizon. Must be 6, 12, or 24 hours")
        
        # Generate forecast
        response = forecast_service.generate_forecast(request)
        
        logger.info(f"Forecast generated successfully for {len(request.station_ids)} stations")
        return response
        
    except ValueError as e:
        logger.warning(f"Invalid forecast request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail="Forecast generation failed")


# Explainability endpoint
@app.post("/explainability", response_model=ExplainabilityResponse)
async def analyze_explainability(
    request: ExplainabilityRequest,
    explainability_service: ExplainabilityService = Depends(get_explainability_service)
):
    """Analyze feature importance and model explainability"""
    try:
        logger.info(f"Analyzing explainability for station: {request.station_id}")
        
        # Validate request
        if not request.station_id:
            raise HTTPException(status_code=400, detail="No station specified")
        
        if not request.methods:
            raise HTTPException(status_code=400, detail="No methods specified")
        
        # Perform analysis
        response = explainability_service.analyze_explainability(request)
        
        logger.info(f"Explainability analysis completed for station {request.station_id}")
        return response
        
    except ValueError as e:
        logger.warning(f"Invalid explainability request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Explainability analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Explainability analysis failed")


# What-if analysis endpoint
@app.post("/what-if", response_model=WhatIfResponse)
async def analyze_what_if(
    request: WhatIfRequest,
    whatif_service: WhatIfService = Depends(get_whatif_service)
):
    """Perform what-if scenario analysis"""
    try:
        logger.info(f"Analyzing what-if scenarios for station: {request.station_id}")
        
        # Validate request
        if not request.station_id:
            raise HTTPException(status_code=400, detail="No station specified")
        
        if not request.scenarios:
            raise HTTPException(status_code=400, detail="No scenarios specified")
        
        # Perform analysis
        response = whatif_service.analyze_what_if(request)
        
        logger.info(f"What-if analysis completed for station {request.station_id}")
        return response
        
    except ValueError as e:
        logger.warning(f"Invalid what-if request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"What-if analysis failed: {e}")
        raise HTTPException(status_code=500, detail="What-if analysis failed")


# Cross-station endpoints
@app.post("/cross-station/forecast")
async def generate_cross_station_forecast(
    station_id: str,
    horizon_hours: int = 24,
    include_external_stations: List[str] = ["india", "china"],
    cross_station_service: CrossStationService = Depends(get_cross_station_service)
):
    """Generate cross-station forecast using external data"""
    try:
        return cross_station_service.generate_cross_station_forecast(
            station_id, horizon_hours, include_external_stations
        )
    except Exception as e:
        logger.error(f"Cross-station forecast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cross-station/external-stations")
async def get_external_stations_info(
    cross_station_service: CrossStationService = Depends(get_cross_station_service)
):
    """Get information about available external stations"""
    try:
        return cross_station_service.get_external_stations_info()
    except Exception as e:
        logger.error(f"Failed to get external stations info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cross-station/spatial-features")
async def get_spatial_correlation_features(
    cross_station_service: CrossStationService = Depends(get_cross_station_service)
):
    """Get spatial correlation feature information"""
    try:
        return cross_station_service.get_spatial_correlation_features()
    except Exception as e:
        logger.error(f"Failed to get spatial correlation features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cross-station/model-performance")
async def get_cross_station_model_performance(
    cross_station_service: CrossStationService = Depends(get_cross_station_service)
):
    """Get cross-station model performance metrics"""
    try:
        return cross_station_service.get_model_performance()
    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cross-station/retrain")
async def retrain_cross_station_model(
    station_id: str,
    include_external_stations: List[str] = ["india", "china"],
    cross_station_service: CrossStationService = Depends(get_cross_station_service)
):
    """Retrain the cross-station model with fresh data"""
    try:
        return cross_station_service.retrain_model(station_id, include_external_stations)
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Agent endpoints
@app.post("/agents/execute", response_model=AgentResponse)
async def execute_agent(
    request: AgentRequest,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Execute a specific intelligent agent"""
    try:
        logger.info(f"Executing {request.agent_type} agent for station {request.station_id}")
        
        response = await agent_service.execute_agent(request)
        
        logger.info(f"Agent execution completed successfully")
        return response
        
    except ValueError as e:
        logger.warning(f"Invalid agent request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(status_code=500, detail="Agent execution failed")


@app.post("/workflows/execute", response_model=WorkflowResponse)
async def execute_workflow(
    request: WorkflowRequest,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Execute an intelligent workflow"""
    try:
        logger.info(f"Executing {request.workflow_type} workflow for station {request.station_id}")
        
        response = await agent_service.execute_workflow(request)
        
        logger.info(f"Workflow execution completed successfully")
        return response
        
    except ValueError as e:
        logger.warning(f"Invalid workflow request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail="Workflow execution failed")


@app.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status(
    agent_service: AgentService = Depends(get_agent_service)
):
    """Get comprehensive system status including all agents"""
    try:
        logger.info("Getting system status")
        
        response = await agent_service.get_system_status()
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@app.post("/system/clear-cache")
async def clear_cache(
    model_service: ModelService = Depends(get_model_service),
    forecast_service: ForecastService = Depends(get_forecast_service)
):
    """Clear model cache for testing"""
    try:
        model_service.model_cache.clear()
        logger.info("Model cache cleared")
        return {"status": "success", "message": "Model cache cleared"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AirAware API",
        "description": "Production-grade PM₂.₅ nowcasting with Intelligent Agents",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "stations": "/stations",
            "forecast": "/forecast",
            "explainability": "/explainability",
            "what-if": "/what-if",
            "agents": "/agents/execute",
            "workflows": "/workflows/execute",
            "system_status": "/system/status",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "airaware.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
