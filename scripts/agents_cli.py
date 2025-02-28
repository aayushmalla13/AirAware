#!/usr/bin/env python3
"""
Command-line interface for AirAware Intelligent Agents

This script provides a CLI for managing and testing the intelligent agents system.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from airaware.agents import (
    AgentOrchestrator, OrchestratorConfig,
    HealthAdvisoryAgent, HealthAgentConfig,
    ForecastOptimizationAgent, ForecastAgentConfig,
    DataQualityAgent, DataAgentConfig,
    NotificationAgent, NotificationAgentConfig
)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/agents_cli.log')
        ]
    )


async def start_orchestrator(args):
    """Start the agent orchestrator"""
    print("🚀 Starting AirAware Agent Orchestrator...")
    
    # Create orchestrator configuration
    config = OrchestratorConfig(
        agent_id="orchestrator",
        agent_name="AirAware Orchestrator",
        agent_type="orchestrator",
        enabled=True,
        log_level=args.log_level
    )
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(config)
    
    # Start orchestrator
    success = await orchestrator.start()
    if not success:
        print("❌ Failed to start orchestrator")
        return
    
    print("✅ Orchestrator started successfully")
    
    # Start all agents
    print("🔄 Starting all agents...")
    start_results = orchestrator.start_all_agents()
    
    for agent_id, result in start_results.items():
        if result:
            print(f"✅ {agent_id} agent started")
        else:
            print(f"❌ {agent_id} agent failed to start")
    
    # Run health check
    print("🏥 Running system health check...")
    health_status = await orchestrator.health_check()
    if health_status:
        print("✅ System health check passed")
    else:
        print("❌ System health check failed")
    
    # Display system status
    status = orchestrator.get_system_status()
    print("\n📊 System Status:")
    print(json.dumps(status, indent=2, default=str))
    
    print("\n🎯 Orchestrator is running. Press Ctrl+C to stop.")


async def test_health_agent(args):
    """Test the health advisory agent"""
    print("🏥 Testing Health Advisory Agent...")
    
    # Create health agent configuration
    config = HealthAgentConfig(
        agent_id="health_test",
        agent_name="Health Advisory Test",
        agent_type="health",
        enabled=True,
        log_level=args.log_level
    )
    
    # Create health agent
    agent = HealthAdvisoryAgent(config)
    
    # Start agent
    success = await agent.start()
    if not success:
        print("❌ Failed to start health agent")
        return
    
    # Test context
    context = {
        "pm25_data": {
            "current": args.pm25_level,
            "forecast": [args.pm25_level + i for i in range(24)]
        },
        "user_id": args.user_id,
        "location": args.location,
        "forecast_horizon": 24
    }
    
    # Execute agent
    print(f"🔍 Testing with PM2.5 level: {args.pm25_level} μg/m³")
    result = await agent.run(context)
    
    print("📋 Health Advisory Result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Stop agent
    await agent.stop()
    print("✅ Health agent test completed")


async def test_forecast_agent(args):
    """Test the forecast optimization agent"""
    print("🔮 Testing Forecast Optimization Agent...")
    
    # Create forecast agent configuration
    config = ForecastAgentConfig(
        agent_id="forecast_test",
        agent_name="Forecast Optimization Test",
        agent_type="forecast",
        enabled=True,
        log_level=args.log_level
    )
    
    # Create forecast agent
    agent = ForecastOptimizationAgent(config)
    
    # Start agent
    success = await agent.start()
    if not success:
        print("❌ Failed to start forecast agent")
        return
    
    # Test context
    context = {
        "station_id": args.station_id,
        "horizon_hours": args.horizon_hours,
        "available_models": ["prophet", "patchtst"],
        "current_conditions": {
            "pm25_level": args.pm25_level,
            "weather_conditions": "calm"
        }
    }
    
    # Execute agent
    print(f"🔍 Testing with station: {args.station_id}, horizon: {args.horizon_hours}h")
    result = await agent.run(context)
    
    print("📋 Forecast Optimization Result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Stop agent
    await agent.stop()
    print("✅ Forecast agent test completed")


async def test_data_agent(args):
    """Test the data quality monitoring agent"""
    print("📊 Testing Data Quality Monitoring Agent...")
    
    # Create data agent configuration
    config = DataAgentConfig(
        agent_id="data_test",
        agent_name="Data Quality Test",
        agent_type="data",
        enabled=True,
        log_level=args.log_level
    )
    
    # Create data agent
    agent = DataQualityAgent(config)
    
    # Start agent
    success = await agent.start()
    if not success:
        print("❌ Failed to start data agent")
        return
    
    # Test data
    import pandas as pd
    import numpy as np
    
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'pm25': np.random.normal(30, 10, 100),
        'temperature': np.random.normal(20, 5, 100),
        'humidity': np.random.uniform(30, 80, 100)
    })
    
    # Add some anomalies
    data.loc[10:15, 'pm25'] = 200  # High PM2.5
    data.loc[20:25, 'temperature'] = -100  # Invalid temperature
    
    # Test context
    context = {
        "data": data,
        "station_id": args.station_id,
        "data_source": "test"
    }
    
    # Execute agent
    print(f"🔍 Testing with station: {args.station_id}")
    result = await agent.run(context)
    
    print("📋 Data Quality Result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Stop agent
    await agent.stop()
    print("✅ Data agent test completed")


async def test_notification_agent(args):
    """Test the notification agent"""
    print("📢 Testing Notification Agent...")
    
    # Create notification agent configuration
    config = NotificationAgentConfig(
        agent_id="notification_test",
        agent_name="Notification Test",
        agent_type="notification",
        enabled=True,
        log_level=args.log_level,
        email_enabled=args.email_enabled,
        smtp_username=args.smtp_username,
        smtp_password=args.smtp_password
    )
    
    # Create notification agent
    agent = NotificationAgent(config)
    
    # Start agent
    success = await agent.start()
    if not success:
        print("❌ Failed to start notification agent")
        return
    
    # Add test user preferences
    from airaware.agents.notification_agent import UserPreferences
    user_prefs = UserPreferences(
        user_id=args.user_id,
        email=args.email,
        pm25_warning_enabled=True,
        pm25_critical_enabled=True,
        pm25_emergency_enabled=True,
        email_notifications=True
    )
    agent.add_user_preferences(user_prefs)
    
    # Test context
    context = {
        "pm25_data": {
            "current": args.pm25_level,
            "forecast": [args.pm25_level + i for i in range(24)]
        },
        "station_id": args.station_id,
        "user_id": args.user_id,
        "forecast_data": [
            {"pm25_mean": args.pm25_level + i} for i in range(24)
        ]
    }
    
    # Execute agent
    print(f"🔍 Testing with PM2.5 level: {args.pm25_level} μg/m³")
    result = await agent.run(context)
    
    print("📋 Notification Result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Stop agent
    await agent.stop()
    print("✅ Notification agent test completed")


async def run_workflow(args):
    """Run a specific workflow"""
    print(f"🔄 Running workflow: {args.workflow_type}")
    
    # Create orchestrator configuration
    config = OrchestratorConfig(
        agent_id="workflow_runner",
        agent_name="Workflow Runner",
        agent_type="orchestrator",
        enabled=True,
        log_level=args.log_level
    )
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(config)
    
    # Start orchestrator
    success = await orchestrator.start()
    if not success:
        print("❌ Failed to start orchestrator")
        return
    
    # Start all agents
    start_results = orchestrator.start_all_agents()
    failed_agents = [agent_id for agent_id, result in start_results.items() if not result]
    
    if failed_agents:
        print(f"⚠️ Some agents failed to start: {failed_agents}")
    
    # Prepare workflow context
    context = {
        "workflow_type": args.workflow_type,
        "station_id": args.station_id,
        "user_id": args.user_id,
        "pm25_data": {
            "current": args.pm25_level,
            "forecast": [args.pm25_level + i for i in range(24)]
        },
        "forecast_data": [
            {"pm25_mean": args.pm25_level + i} for i in range(24)
        ]
    }
    
    # Execute workflow
    print(f"🚀 Executing {args.workflow_type} workflow...")
    result = await orchestrator.run(context)
    
    print("📋 Workflow Result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Stop orchestrator
    await orchestrator.stop()
    print("✅ Workflow execution completed")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="AirAware Intelligent Agents CLI")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start orchestrator command
    start_parser = subparsers.add_parser("start", help="Start the agent orchestrator")
    
    # Test health agent command
    health_parser = subparsers.add_parser("test-health", help="Test the health advisory agent")
    health_parser.add_argument("--pm25-level", type=float, default=50.0, help="PM2.5 level for testing")
    health_parser.add_argument("--user-id", default="test_user", help="User ID for testing")
    health_parser.add_argument("--location", default="kathmandu", help="Location for testing")
    
    # Test forecast agent command
    forecast_parser = subparsers.add_parser("test-forecast", help="Test the forecast optimization agent")
    forecast_parser.add_argument("--station-id", default="test_station", help="Station ID for testing")
    forecast_parser.add_argument("--horizon-hours", type=int, default=24, help="Forecast horizon in hours")
    forecast_parser.add_argument("--pm25-level", type=float, default=30.0, help="Current PM2.5 level")
    
    # Test data agent command
    data_parser = subparsers.add_parser("test-data", help="Test the data quality monitoring agent")
    data_parser.add_argument("--station-id", default="test_station", help="Station ID for testing")
    
    # Test notification agent command
    notification_parser = subparsers.add_parser("test-notification", help="Test the notification agent")
    notification_parser.add_argument("--pm25-level", type=float, default=60.0, help="PM2.5 level for testing")
    notification_parser.add_argument("--station-id", default="test_station", help="Station ID for testing")
    notification_parser.add_argument("--user-id", default="test_user", help="User ID for testing")
    notification_parser.add_argument("--email", help="Email address for testing")
    notification_parser.add_argument("--email-enabled", action="store_true", help="Enable email notifications")
    notification_parser.add_argument("--smtp-username", help="SMTP username")
    notification_parser.add_argument("--smtp-password", help="SMTP password")
    
    # Run workflow command
    workflow_parser = subparsers.add_parser("run-workflow", help="Run a specific workflow")
    workflow_parser.add_argument("--workflow-type", required=True, 
                                choices=["forecast_generation", "health_assessment", "data_quality_check"],
                                help="Type of workflow to run")
    workflow_parser.add_argument("--station-id", default="test_station", help="Station ID for testing")
    workflow_parser.add_argument("--user-id", default="test_user", help="User ID for testing")
    workflow_parser.add_argument("--pm25-level", type=float, default=40.0, help="PM2.5 level for testing")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Run the appropriate command
    try:
        if args.command == "start":
            asyncio.run(start_orchestrator(args))
        elif args.command == "test-health":
            asyncio.run(test_health_agent(args))
        elif args.command == "test-forecast":
            asyncio.run(test_forecast_agent(args))
        elif args.command == "test-data":
            asyncio.run(test_data_agent(args))
        elif args.command == "test-notification":
            asyncio.run(test_notification_agent(args))
        elif args.command == "run-workflow":
            asyncio.run(run_workflow(args))
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.exception("CLI execution failed")


if __name__ == "__main__":
    main()
