#!/usr/bin/env python3
"""Test script for CP-3 enhancements: Quality monitoring, lineage tracking, and circuit breaker."""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from airaware.etl.quality_monitor import QualityMonitor, QualityAlert
from airaware.etl.lineage_tracker import DataLineageTracker
from airaware.etl.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerManager
from airaware.etl.data_validator import OpenAQDataValidator
import pandas as pd
import numpy as np
from rich.console import Console

console = Console()

def test_quality_monitoring():
    """Test quality monitoring system."""
    console.print("\n🔍 Testing Quality Monitoring System...")
    
    # Create test data with quality issues
    np.random.seed(42)
    test_data = {
        'station_id': [3460] * 100,
        'value': np.concatenate([
            np.random.normal(25, 5, 80),  # Normal PM2.5 values
            [np.nan] * 10,                # Missing values
            [-5, 600, 999, -10, 850] * 2  # Outliers
        ]),
        'datetime_utc': pd.date_range('2025-09-26 00:00:00', periods=100, freq='h'),
        'unit': ['µg/m³'] * 100,
        'quality_flag': ['valid'] * 100
    }
    
    df = pd.DataFrame(test_data)
    
    # Test quality monitor
    monitor = QualityMonitor()
    alerts = monitor.assess_data_quality(3460, df)
    
    console.print(f"✅ Generated {len(alerts)} quality alerts")
    for alert in alerts:
        console.print(f"  • [{alert.severity.upper()}] {alert.message}")
    
    # Save and retrieve alerts
    monitor.save_alerts(alerts)
    active_alerts = monitor.get_active_alerts()
    console.print(f"✅ {len(active_alerts)} active alerts saved and retrieved")
    
    # Generate quality report
    report = monitor.generate_quality_report()
    console.print("✅ Quality report generated successfully")
    
    return len(alerts) > 0

def test_lineage_tracking():
    """Test data lineage tracking system."""
    console.print("\n📋 Testing Data Lineage Tracking...")
    
    tracker = DataLineageTracker()
    
    # Simulate ETL pipeline events
    extract_id = tracker.track_extraction(
        source="openaq_v3_api", 
        station_id=3460, 
        record_count=100,
        metadata={"api_version": "v3", "date_range": "2025-09-26"}
    )
    console.print(f"✅ Tracked extraction event: {extract_id[:8]}...")
    
    transform_id = tracker.track_transformation(
        source="raw/openaq/station_3460",
        target="interim/cleaned_data",
        operation="data_cleaning",
        record_count=85,
        parent_event_id=extract_id,
        metadata={"removed_nulls": 10, "removed_outliers": 5}
    )
    console.print(f"✅ Tracked transformation event: {transform_id[:8]}...")
    
    validation_id = tracker.track_validation(
        source="interim/cleaned_data",
        quality_score=0.85,
        station_id=3460,
        parent_event_id=transform_id,
        metadata={"completeness": 0.9, "outliers": 0.05}
    )
    console.print(f"✅ Tracked validation event: {validation_id[:8]}...")
    
    load_id = tracker.track_load(
        source="interim/cleaned_data",
        target="data/raw/openaq/station_3460/year=2025/month=09/data.parquet",
        record_count=85,
        parent_event_id=validation_id,
        metadata={"compression": "snappy", "file_size_mb": 2.1}
    )
    console.print(f"✅ Tracked load event: {load_id[:8]}...")
    
    # Generate lineage report
    summary = tracker.get_lineage_summary()
    console.print(f"✅ Lineage summary: {summary['total_events']} events, {summary['total_records']} records")
    
    # Test station-specific lineage
    station_events = tracker.get_lineage_for_station(3460)
    console.print(f"✅ Found {len(station_events)} events for station 3460")
    
    return len(station_events) >= 2  # At least 2 events for station 3460

def test_circuit_breaker():
    """Test circuit breaker pattern."""
    console.print("\n⚡ Testing Circuit Breaker Pattern...")
    
    # Create circuit breaker with low threshold for testing
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=5,
        expected_exception=Exception
    )
    
    manager = CircuitBreakerManager()
    breaker = manager.get_breaker("test_api", config)
    
    # Test normal operation
    def successful_operation():
        return "success"
    
    result = breaker.call(successful_operation)
    console.print(f"✅ Normal operation: {result}")
    
    # Test failure handling
    def failing_operation():
        raise Exception("API failure")
    
    failure_count = 0
    for i in range(5):
        try:
            breaker.call(failing_operation)
        except Exception:
            failure_count += 1
    
    status = breaker.get_status()
    console.print(f"✅ Circuit breaker state: {status['state']} after {failure_count} failures")
    
    # Test manager
    all_status = manager.get_all_status()
    console.print(f"✅ Circuit breaker manager tracking {len(all_status)} breakers")
    
    return status['state'] == 'open'

def test_enhanced_data_validation():
    """Test enhanced data validation with new metrics."""
    console.print("\n🔬 Testing Enhanced Data Validation...")
    
    validator = OpenAQDataValidator()
    
    # Create test data with various quality issues
    np.random.seed(42)
    test_data = {
        'station_id': [3460] * 200,
        'value': np.concatenate([
            np.random.normal(20, 3, 150),  # Normal values
            [np.nan] * 30,                 # Missing values
            [-10, 500, 999, -5] * 5       # Outliers
        ]),
        'datetime_utc': pd.date_range('2025-09-25 00:00:00', periods=200, freq='30min'),
        'parameter': ['pm25'] * 200,
        'unit': ['µg/m³'] * 200,
        'quality_flag': ['valid'] * 200
    }
    
    df = pd.DataFrame(test_data)
    
    # Test validation
    metrics = validator.validate_data_quality(df, [3460])
    
    console.print(f"✅ Data quality metrics calculated:")
    console.print(f"  • Quality score: {metrics.quality_score:.3f}")
    console.print(f"  • Completeness: {metrics.data_completeness:.1%}")
    console.print(f"  • Missing values: {metrics.missing_values}")
    console.print(f"  • Duplicates: {metrics.duplicate_records}")
    console.print(f"  • Outliers: {metrics.outlier_records}")
    
    return 0.5 <= metrics.quality_score <= 0.9  # Expect moderate quality due to issues

def main():
    """Run all enhancement tests."""
    console.print("🚀 CP-3 Enhancement Testing Suite")
    console.print("=" * 50)
    
    tests = [
        ("Quality Monitoring", test_quality_monitoring),
        ("Lineage Tracking", test_lineage_tracking),
        ("Circuit Breaker", test_circuit_breaker),
        ("Enhanced Validation", test_enhanced_data_validation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "✅ PASS" if success else "⚠️ PARTIAL"
        except Exception as e:
            results[test_name] = f"❌ FAIL: {e}"
    
    # Summary
    console.print("\n📊 Test Results Summary:")
    console.print("=" * 30)
    
    passed = 0
    for test_name, result in results.items():
        console.print(f"{result} {test_name}")
        if "PASS" in result:
            passed += 1
    
    console.print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        console.print("\n🎉 All CP-3 enhancements working perfectly!")
    else:
        console.print("\n⚠️ Some enhancements need attention.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
