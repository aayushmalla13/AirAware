#!/usr/bin/env python3
"""
Start the AirAware Streamlit UI

This script starts the Streamlit user interface for the AirAware system.
"""

import streamlit.web.cli as stcli
import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to start the Streamlit UI"""
    parser = argparse.ArgumentParser(description="Start AirAware Streamlit UI")
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the Streamlit app on (default: 8501)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to run the Streamlit app on (default: localhost)"
    )
    parser.add_argument(
        "--theme",
        default="light",
        choices=["light", "dark"],
        help="Streamlit theme (default: light)"
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting AirAware Streamlit UI...")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Theme: {args.theme}")
    
    # Streamlit app path
    app_path = project_root / "airaware" / "ui" / "app.py"
    
    if not app_path.exists():
        logger.error(f"❌ Streamlit app not found at {app_path}")
        sys.exit(1)
    
    try:
        # Start Streamlit
        sys.argv = [
            "streamlit",
            "run",
            str(app_path),
            "--server.port", str(args.port),
            "--server.address", args.host,
            "--theme.base", args.theme,
            "--browser.gatherUsageStats", "false"
        ]
        
        stcli.main()
        
    except KeyboardInterrupt:
        logger.info("🛑 Streamlit UI stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start Streamlit UI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
