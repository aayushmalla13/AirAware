#!/usr/bin/env python3
"""
Start the complete AirAware system

This script starts both the FastAPI backend and Streamlit UI for the AirAware system.
"""

import subprocess
import argparse
import logging
import time
import signal
import sys
from pathlib import Path
import threading
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AirAwareLauncher:
    """Launcher for the complete AirAware system"""
    
    def __init__(self, api_port=8000, ui_port=8501, api_host="0.0.0.0", ui_host="localhost"):
        self.api_port = api_port
        self.ui_port = ui_port
        self.api_host = api_host
        self.ui_host = ui_host
        self.api_process = None
        self.ui_process = None
        self.running = False
        
    def check_api_health(self, timeout=30):
        """Check if API is healthy"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://{self.api_host}:{self.api_port}/health", timeout=5)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        return False
    
    def start_api(self):
        """Start the FastAPI server"""
        logger.info("🚀 Starting FastAPI server...")
        
        try:
            self.api_process = subprocess.Popen([
                sys.executable, "scripts/start_api.py",
                "--host", self.api_host,
                "--port", str(self.api_port),
                "--log-level", "info"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for API to be healthy
            if self.check_api_health():
                logger.info("✅ FastAPI server started successfully")
                return True
            else:
                logger.error("❌ FastAPI server failed to start or become healthy")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start FastAPI server: {e}")
            return False
    
    def start_ui(self):
        """Start the Streamlit UI"""
        logger.info("🚀 Starting Streamlit UI...")
        
        try:
            self.ui_process = subprocess.Popen([
                sys.executable, "scripts/start_ui.py",
                "--host", self.ui_host,
                "--port", str(self.ui_port)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Give UI time to start
            time.sleep(5)
            logger.info("✅ Streamlit UI started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Streamlit UI: {e}")
            return False
    
    def start(self):
        """Start both API and UI"""
        logger.info("🌍 Starting AirAware system...")
        
        # Start API first
        if not self.start_api():
            return False
        
        # Start UI
        if not self.start_ui():
            self.stop()
            return False
        
        self.running = True
        
        # Display URLs
        logger.info("=" * 60)
        logger.info("🌍 AirAware System Started Successfully!")
        logger.info("=" * 60)
        logger.info(f"📊 API Documentation: http://{self.api_host}:{self.api_port}/docs")
        logger.info(f"🔍 API Health Check: http://{self.api_host}:{self.api_port}/health")
        logger.info(f"🖥️  Streamlit UI: http://{self.ui_host}:{self.ui_port}")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop the system")
        logger.info("=" * 60)
        
        return True
    
    def stop(self):
        """Stop both API and UI"""
        logger.info("🛑 Stopping AirAware system...")
        
        self.running = False
        
        if self.ui_process:
            logger.info("Stopping Streamlit UI...")
            self.ui_process.terminate()
            try:
                self.ui_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.ui_process.kill()
            self.ui_process = None
        
        if self.api_process:
            logger.info("Stopping FastAPI server...")
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.api_process.kill()
            self.api_process = None
        
        logger.info("✅ AirAware system stopped")
    
    def run(self):
        """Run the system until interrupted"""
        if not self.start():
            sys.exit(1)
        
        try:
            # Keep running until interrupted
            while self.running:
                time.sleep(1)
                
                # Check if processes are still running
                if self.api_process and self.api_process.poll() is not None:
                    logger.error("❌ API process died unexpectedly")
                    break
                
                if self.ui_process and self.ui_process.poll() is not None:
                    logger.error("❌ UI process died unexpectedly")
                    break
                    
        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt signal")
        finally:
            self.stop()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Start the complete AirAware system")
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for the FastAPI server (default: 8000)"
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=8501,
        help="Port for the Streamlit UI (default: 8501)"
    )
    parser.add_argument(
        "--api-host",
        default="0.0.0.0",
        help="Host for the FastAPI server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--ui-host",
        default="localhost",
        help="Host for the Streamlit UI (default: localhost)"
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    # Create and run launcher
    launcher = AirAwareLauncher(
        api_port=args.api_port,
        ui_port=args.ui_port,
        api_host=args.api_host,
        ui_host=args.ui_host
    )
    
    launcher.run()


if __name__ == "__main__":
    main()
