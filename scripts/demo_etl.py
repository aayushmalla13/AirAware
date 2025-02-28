#!/usr/bin/env python3
"""Demo script for OpenAQ ETL pipeline functionality.

Demonstrates the ETL pipeline with mock data to show all features working.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from rich.console import Console
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from airaware.etl.openaq_etl import OpenAQETL
from airaware.etl.data_validator import OpenAQDataValidator
from airaware.etl.manifest_manager import ManifestManager


def create_mock_openaq_data(station_id: int, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Create mock OpenAQ data for testing."""
    date_range = pd.date_range(start_date, end_date, freq='h')
    
    data = []
    for dt in date_range:
        # Generate realistic PM2.5 values with some variation
        base_value = 30.0 + station_id % 20  # Different base for each station
        variation = 10.0 * (0.5 - np.random.random())  # ±10 µg/m³ variation
        pm25_value = max(0, base_value + variation)
        
        data.append({
            'station_id': station_id,
            'sensor_id': 7710 + station_id,
            'parameter': 'pm25',
            'value': pm25_value,
            'unit': 'µg/m³',
            'datetime_utc': dt,
            'latitude': 27.717 + (station_id % 3) * 0.01,
            'longitude': 85.324 + (station_id % 3) * 0.01,
            'quality_flag': 'valid',
            'data_source': 'openaq_mock'
        })
    
    return pd.DataFrame(data)


def demo_data_validation():
    """Demonstrate data validation functionality."""
    console = Console()
    console.print("\n[bold blue]🔍 Demo: Data Validation[/bold blue]")
    
    # Create mock data with some issues
    validator = OpenAQDataValidator()
    
    # Clean data
    clean_df = create_mock_openaq_data(3459, datetime.now() - timedelta(days=7), datetime.now())
    clean_metrics = validator.validate_data_quality(clean_df, [3459])
    
    console.print(f"✅ Clean data validation:")
    console.print(f"   Records: {clean_metrics.total_records}")
    console.print(f"   Quality score: {clean_metrics.quality_score:.3f}")
    console.print(f"   Completeness: {clean_metrics.data_completeness:.1%}")
    
    # Add some data issues
    problematic_df = clean_df.copy()
    
    # Add missing values
    problematic_df.loc[0:5, 'value'] = None
    
    # Add outliers
    problematic_df.loc[10:12, 'value'] = 2000.0  # Unrealistic high values
    
    # Add duplicates
    problematic_df = pd.concat([problematic_df, problematic_df.head(3)], ignore_index=True)
    
    problem_metrics = validator.validate_data_quality(problematic_df, [3459])
    
    console.print(f"\n⚠️ Problematic data validation:")
    console.print(f"   Records: {problem_metrics.total_records}")
    console.print(f"   Quality score: {problem_metrics.quality_score:.3f}")
    console.print(f"   Missing values: {problem_metrics.missing_values}")
    console.print(f"   Duplicates: {problem_metrics.duplicate_records}")
    console.print(f"   Outliers: {problem_metrics.outlier_records}")


def demo_manifest_management():
    """Demonstrate manifest management functionality."""
    console = Console()
    console.print("\n[bold blue]📋 Demo: Manifest Management[/bold blue]")
    
    from airaware.etl.manifest_manager import ETLArtifact
    
    manifest_manager = ManifestManager("data/artifacts/demo_manifest.json")
    
    # Create some mock artifacts
    artifacts = [
        ETLArtifact(
            file_path="data/raw/openaq/station_3459/year=2025/month=09/data.parquet",
            file_type="parquet",
            etl_stage="raw",
            data_source="openaq",
            station_id=3459,
            date_partition="2025-09",
            created_at=datetime.now(),
            file_size_bytes=1024 * 1024,  # 1MB
            record_count=720,  # 30 days * 24 hours
            data_quality_score=0.95
        ),
        ETLArtifact(
            file_path="data/interim/targets.parquet",
            file_type="parquet",
            etl_stage="interim", 
            data_source="openaq",
            station_id=None,
            date_partition=None,
            created_at=datetime.now(),
            file_size_bytes=5 * 1024 * 1024,  # 5MB
            record_count=2160,  # 3 stations * 720 records
            data_quality_score=0.92
        )
    ]
    
    # Add artifacts to manifest
    for artifact in artifacts:
        manifest_manager.add_etl_artifact(artifact)
    
    # Display stats
    stats = manifest_manager.get_etl_stats()
    
    console.print(f"✅ ETL Artifacts: {stats.get('total_etl_artifacts', 0)}")
    console.print(f"✅ Total Size: {stats.get('total_etl_size_mb', 0):.1f} MB")
    console.print(f"✅ Total Records: {stats.get('total_records', 0):,}")
    console.print(f"✅ By Source: {stats.get('by_source', {})}")
    console.print(f"✅ By Stage: {stats.get('by_stage', {})}")


def demo_etl_pipeline_structure():
    """Demonstrate ETL pipeline structure and configuration."""
    console = Console()
    console.print("\n[bold blue]🚀 Demo: ETL Pipeline Structure[/bold blue]")
    
    # Initialize ETL (will load CP-2 configuration)
    etl = OpenAQETL(
        raw_data_dir="data/raw/openaq_demo",
        interim_data_dir="data/interim_demo",
        use_local=True,
        max_workers=2
    )
    
    console.print(f"✅ Stations configured: {len(etl.stations)}")
    
    for station in etl.stations:
        console.print(f"   • Station {station['station_id']}: {station['name']} (Q={station['quality_score']:.2f})")
    
    # Demonstrate partition generation
    start_date = datetime(2025, 8, 1)
    end_date = datetime(2025, 9, 30)
    
    partitions = etl.generate_date_partitions(start_date, end_date)
    console.print(f"\n✅ Generated {len(partitions)} partitions: {partitions}")
    
    # Show file path structure
    for station in etl.stations[:2]:  # Show first 2 stations
        station_id = station['station_id']
        for partition in partitions[:3]:  # Show first 3 partitions
            file_path = etl.get_partition_file_path(station_id, partition)
            console.print(f"   📁 {file_path}")


def main():
    """Run ETL pipeline demonstration."""
    console = Console()
    
    console.print("[bold green]🎭 OpenAQ ETL Pipeline Demo[/bold green]")
    console.print("=" * 60)
    
    try:
        # Demo 1: Data Validation
        demo_data_validation()
        
        # Demo 2: Manifest Management
        demo_manifest_management()
        
        # Demo 3: ETL Pipeline Structure
        demo_etl_pipeline_structure()
        
        console.print("\n[bold green]✅ All ETL Components Demonstrated Successfully![/bold green]")
        console.print("\n[blue]💡 ETL Pipeline Features:[/blue]")
        console.print("   🔄 Reuse-or-fetch logic for efficient data management")
        console.print("   📦 Partitioned Parquet storage by station/year/month")
        console.print("   🔍 Comprehensive data validation and quality scoring")
        console.print("   📋 Manifest integration with CP-0 data scanner")
        console.print("   ⚡ Parallel processing for multiple stations")
        console.print("   🛡️ Schema validation and error handling")
        console.print("   📊 Rich CLI with progress tracking")
        
        console.print(f"\n[bold]🎯 Ready for production ETL with live OpenAQ API![/bold]")
        
    except Exception as e:
        console.print(f"[red]❌ Demo failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
