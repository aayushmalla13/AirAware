#!/usr/bin/env python3
"""
OpenAQ Data Ingestion - Main Launcher

This is the main entry point for the organized OpenAQ data ingestion system.
It properly handles the new folder structure and provides easy access to all features.

Usage:
    python main.py [options]

Options:
    --test      Run test ingestion
    --full      Run full-scale ingestion  
    --analyze   Analyze existing data
    --config    Show configuration

Directory Structure:
    ├── main.py                 # This file - main launcher
    ├── scripts/                # All Python scripts
    │   ├── launch_ingestion.py # Smart launcher with checks
    │   ├── run_full_ingestion.py # Full-scale ingestion
    │   ├── test_ingestion.py   # Test ingestion
    │   ├── analyze_results.py  # Data analysis
    │   └── ingest_openaq_data.py # Original script
    ├── config/                 # Configuration files
    │   └── config.py          # Main configuration
    ├── data/                   # Data storage
    │   ├── full_data/         # Full ingestion results
    │   └── test_data/         # Test ingestion results
    ├── docs/                   # Documentation
    │   └── README.md          # Project documentation
    └── logs/                   # Log files
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add scripts directory to Python path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Change to root directory for relative paths to work
os.chdir(Path(__file__).parent)

def print_banner():
    """Print the OpenAQ banner"""
    print("🌍" + "="*70 + "🌍")
    print("   OpenAQ Air Quality Data Ingestion System")
    print("   Nepal, India & China - Full Coverage")
    print("🌍" + "="*70 + "🌍")
    print()

def show_structure():
    """Show the organized directory structure"""
    print("📁 Project Structure:")
    print("├── main.py                 # Main launcher (this file)")
    print("├── scripts/                # All Python scripts")
    print("│   ├── launch_ingestion.py # Smart launcher with checks")
    print("│   ├── run_full_ingestion.py # Full-scale ingestion")
    print("│   ├── test_ingestion.py   # Test ingestion")
    print("│   ├── analyze_results.py  # Data analysis")
    print("│   └── ingest_openaq_data.py # Original development script")
    print("├── config/                 # Configuration files")
    print("│   └── config.py          # Main configuration")
    print("├── data/                   # Data storage")
    print("│   ├── full_data/         # Full ingestion results")
    print("│   └── test_data/         # Test ingestion results")
    print("├── docs/                   # Documentation")
    print("│   └── README.md          # Project documentation")
    print("├── logs/                   # Log files")
    print("└── venv/                   # Python virtual environment")
    print()

def check_environment():
    """Check if we're in the right environment"""
    if not (Path("scripts").exists() and Path("config").exists()):
        print("❌ Error: Not in the correct directory or missing folders")
        print("   Make sure you're in the openaq_ingestion root directory")
        return False
        
    if not Path("venv").exists():
        print("⚠️  Warning: Virtual environment not found")
        print("   Consider creating one: python -m venv venv")
        
    return True

def run_test():
    """Run test ingestion"""
    print("🧪 Running test ingestion...")
    try:
        result = subprocess.run([
            sys.executable, 
            "scripts/test_ingestion.py"
        ], check=True)
        print("✅ Test ingestion completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed: {e}")
        return False

def run_full():
    """Run full-scale ingestion"""
    print("🚀 Running full-scale ingestion...")
    print("⚠️  This will take 4-8 hours and collect millions of records")
    
    confirm = input("Are you sure you want to proceed? (yes/no): ").lower().strip()
    if confirm not in ['yes', 'y']:
        print("❌ Full ingestion cancelled")
        return False
        
    try:
        result = subprocess.run([
            sys.executable, 
            "scripts/run_full_ingestion.py"
        ], check=True)
        print("✅ Full ingestion completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Full ingestion failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Ingestion interrupted by user")
        return False

def run_analysis():
    """Run data analysis"""
    print("📊 Running data analysis...")
    
    # Check for data directories
    data_dirs = []
    if Path("data/full_data").exists() and list(Path("data/full_data").glob("*.parquet")):
        data_dirs.append("data/full_data")
    if Path("data/test_data").exists() and list(Path("data/test_data").glob("*.parquet")):
        data_dirs.append("data/test_data")
        
    if not data_dirs:
        print("❌ No data found to analyze!")
        print("   Run test or full ingestion first")
        return False
        
    # Use the most recent data
    data_dir = data_dirs[0] if "full_data" in data_dirs[0] else data_dirs[0]
    print(f"🔍 Analyzing data in: {data_dir}")
    
    try:
        result = subprocess.run([
            sys.executable, 
            "scripts/analyze_results.py",
            data_dir
        ], check=True)
        print("✅ Analysis completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e}")
        return False

def show_config():
    """Show current configuration"""
    print("⚙️  Current Configuration:")
    config_file = Path("config/config.py")
    
    if not config_file.exists():
        print("❌ Config file not found!")
        return False
        
    print(f"📄 Config file: {config_file}")
    print()
    
    # Read and display key config values
    with open(config_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            if any(key in line for key in ['TARGET_COUNTRIES', 'REQUEST_DELAY', 'BATCH_SIZE', 'OUTPUT_DIR']):
                print(f"  {line}")
                
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="OpenAQ Data Ingestion System - Organized Structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--test", action="store_true", 
                       help="Run test ingestion (quick validation)")
    parser.add_argument("--full", action="store_true",
                       help="Run full-scale ingestion (4-8 hours)")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze existing data")
    parser.add_argument("--config", action="store_true",
                       help="Show current configuration")
    parser.add_argument("--structure", action="store_true",
                       help="Show project structure")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
        
    # Handle different modes
    success = True
    
    if args.structure:
        show_structure()
    elif args.config:
        success = show_config()
    elif args.test:
        success = run_test()
    elif args.full:
        success = run_full()
    elif args.analyze:
        success = run_analysis()
    else:
        # Interactive mode
        print("🎯 Available Options:")
        print("  1. Test ingestion (30 minutes)")
        print("  2. Full ingestion (4-8 hours)")
        print("  3. Analyze data")
        print("  4. Show configuration")
        print("  5. Show project structure")
        print("  6. Exit")
        print()
        
        while True:
            try:
                choice = input("Select option (1-6): ").strip()
                
                if choice == '1':
                    success = run_test()
                    break
                elif choice == '2':
                    success = run_full()
                    break
                elif choice == '3':
                    success = run_analysis()
                    break
                elif choice == '4':
                    success = show_config()
                    break
                elif choice == '5':
                    show_structure()
                    continue
                elif choice == '6':
                    print("👋 Goodbye!")
                    sys.exit(0)
                else:
                    print("❌ Invalid choice. Please select 1-6.")
                    continue
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)
    
    # Final status
    if success:
        print("\n🎉 Operation completed successfully!")
        print("\n📚 Next steps:")
        print("  • Check data files in data/ directory")
        print("  • Review logs for detailed information")
        print("  • Run analysis if you haven't already")
        print("  • Use data for air quality prediction models")
    else:
        print("\n❌ Operation failed!")
        print("  • Check logs for error details")
        print("  • Verify environment setup")
        print("  • Try test mode first if running full ingestion")
        sys.exit(1)

if __name__ == "__main__":
    main()