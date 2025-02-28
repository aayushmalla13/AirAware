#!/usr/bin/env python3
"""
Quick Launch Script for OpenAQ Full-Scale Data Ingestion

This script provides an easy way to start the full-scale data collection
with proper environment setup and configuration.

Usage:
    python launch_ingestion.py [--test] [--config CONFIG_FILE]

Options:
    --test      Run a test version (limited data for validation)
    --config    Use custom configuration file (default: config.py)
    --analyze   Analyze existing data instead of collecting new data
    --resume    Resume interrupted ingestion (if checkpointing enabled)

Examples:
    python launch_ingestion.py                    # Full ingestion
    python launch_ingestion.py --test             # Test run only
    python launch_ingestion.py --analyze          # Analyze results
"""

import sys
import argparse
import subprocess
from pathlib import Path


def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking environment setup...")
    
    # Check if we're in a virtual environment
    venv_check = subprocess.run([sys.executable, "-c", "import sys; print(hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))"], 
                               capture_output=True, text=True)
    
    if venv_check.stdout.strip() != "True":
        print("⚠️  Warning: Not running in a virtual environment")
        print("   Recommended: source venv/bin/activate")
    else:
        print("✅ Virtual environment detected")
    
    # Check required packages
    required_packages = ['pandas', 'requests', 'pyarrow']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies satisfied")
    return True


def estimate_resources():
    """Provide resource estimates for the full ingestion"""
    print("\n📊 RESOURCE REQUIREMENTS ESTIMATE")
    print("=" * 50)
    print("🕐 Time estimate: 4-8 hours")
    print("💾 Storage estimate: 5-15 GB")
    print("🌐 Network: ~10,000 API requests")
    print("💻 Memory: 2-4 GB RAM recommended")
    print("🔄 CPU: Light usage (mostly I/O bound)")
    print()
    print("📋 What will be collected:")
    print("  • Nepal: ~15 locations, ~70 sensors")
    print("  • India: ~1,000+ locations, ~5,000+ sensors")
    print("  • China: ~2,000+ locations, ~10,000+ sensors")
    print("  • Estimated total: 3,000+ locations, 15,000+ sensors")
    print("  • Estimated measurements: 10-50 million records")


def run_test_ingestion():
    """Run the test ingestion script"""
    print("\n🧪 Running test ingestion...")
    
    test_script = Path("test_ingestion.py")
    if not test_script.exists():
        print("❌ Test script not found!")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(test_script)], check=True)
        print("✅ Test ingestion completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Test ingestion failed: {e}")
        return False


def run_full_ingestion():
    """Run the full-scale ingestion script"""
    print("\n🚀 Starting full-scale ingestion...")
    
    full_script = Path("run_full_ingestion.py")
    if not full_script.exists():
        print("❌ Full ingestion script not found!")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(full_script)], check=True)
        print("✅ Full ingestion completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Full ingestion failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Ingestion interrupted by user")
        return False


def run_analysis():
    """Run the data analysis script"""
    print("\n📊 Running data analysis...")
    
    analysis_script = Path("analyze_results.py")
    if not analysis_script.exists():
        print("❌ Analysis script not found!")
        return False
    
    # Check if data exists
    data_dirs = [Path("../data/full_data"), Path("../data/test_data")]
    existing_data = [d for d in data_dirs if d.exists() and list(d.glob("*.parquet"))]
    
    if not existing_data:
        print("❌ No data found to analyze!")
        print("   Run ingestion first with: python launch_ingestion.py")
        return False
    
    # Use the most recent data directory
    data_dir = max(existing_data, key=lambda d: max(f.stat().st_mtime for f in d.glob("*.parquet")))
    
    try:
        result = subprocess.run([sys.executable, str(analysis_script), str(data_dir)], check=True)
        print("✅ Analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e}")
        return False


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Launch OpenAQ data ingestion or analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--test", action="store_true", 
                       help="Run test ingestion instead of full ingestion")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze existing data instead of collecting")
    parser.add_argument("--config", type=str, default="../config/config.py",
                       help="Configuration file to use")
    parser.add_argument("--skip-check", action="store_true",
                       help="Skip environment checks")
    
    args = parser.parse_args()
    
    print("🌍 OpenAQ Data Ingestion Launcher")
    print("=" * 40)
    
    # Environment checks
    if not args.skip_check:
        if not check_environment():
            print("\n❌ Environment check failed!")
            print("   Fix the issues above and try again")
            print("   Or use --skip-check to proceed anyway")
            sys.exit(1)
    
    # Handle different modes
    if args.analyze:
        success = run_analysis()
    elif args.test:
        estimate_resources()
        print("\n🧪 TEST MODE: Limited data collection for validation")
        confirm = input("\nProceed with test ingestion? (y/N): ").lower().strip()
        if confirm in ['y', 'yes']:
            success = run_test_ingestion()
        else:
            print("❌ Test cancelled by user")
            sys.exit(0)
    else:
        estimate_resources()
        print("\n⚠️  FULL MODE: Complete data collection (may take hours)")
        print("   This will collect ALL available data from Nepal, India, and China")
        confirm = input("\nProceed with full ingestion? (y/N): ").lower().strip()
        if confirm in ['y', 'yes']:
            success = run_full_ingestion()
        else:
            print("❌ Full ingestion cancelled by user")
            sys.exit(0)
    
    # Final status
    if success:
        print("\n🎉 Operation completed successfully!")
        
        if not args.analyze:
            print("\n🔍 Next steps:")
            print("  1. Analyze results: python launch_ingestion.py --analyze")
            print("  2. Explore data files in the output directory")
            print("  3. Use the data for air quality prediction models")
    else:
        print("\n❌ Operation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()