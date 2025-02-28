#!/usr/bin/env python3
"""
Calibration Enhancements CLI

Command-line interface for testing advanced calibration enhancements
including temporal calibration, multi-target calibration, and monitoring.
"""

import argparse
import logging
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from airaware.calibration import (
    TemporalCalibrator, TemporalCalibrationConfig,
    MultiTargetCalibrator, MultiTargetCalibrationConfig,
    CalibrationMonitor, CalibrationMonitorConfig,
    ConformalPredictor, ConformalConfig,
    calculate_coverage, calculate_interval_width, calculate_calibration_error,
    evaluate_calibration_quality, plot_calibration_diagnostics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_temporal_calibration(args):
    """Test temporal calibration"""
    logger.info("🔧 Testing temporal calibration...")
    
    # Load data
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return
    
    data = pd.read_parquet(args.data_path)
    logger.info(f"Loaded data: {len(data)} records")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split data
    split_idx = int(len(data) * 0.6)
    val_split_idx = int(len(data) * 0.8)
    
    train_data = data.iloc[:split_idx]
    cal_data = data.iloc[split_idx:val_split_idx]
    test_data = data.iloc[val_split_idx:]
    
    logger.info(f"Train: {len(train_data)}, Calibration: {len(cal_data)}, Test: {len(test_data)}")
    
    # Prepare features
    feature_cols = [col for col in train_data.columns if col not in ['datetime_utc', args.target_col]]
    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_data[col]):
            numeric_cols.append(col)
    
    if not numeric_cols:
        numeric_cols = ['hour', 'day_of_week', 'month']
        for col in numeric_cols:
            if col not in train_data.columns:
                if col == 'hour':
                    train_data[col] = train_data['datetime_utc'].dt.hour
                    cal_data[col] = cal_data['datetime_utc'].dt.hour
                    test_data[col] = test_data['datetime_utc'].dt.hour
                elif col == 'day_of_week':
                    train_data[col] = train_data['datetime_utc'].dt.dayofweek
                    cal_data[col] = cal_data['datetime_utc'].dt.dayofweek
                    test_data[col] = test_data['datetime_utc'].dt.dayofweek
                elif col == 'month':
                    train_data[col] = train_data['datetime_utc'].dt.month
                    cal_data[col] = cal_data['datetime_utc'].dt.month
                    test_data[col] = test_data['datetime_utc'].dt.month
    
    X_train = train_data[numeric_cols].values
    y_train = train_data[args.target_col].values
    X_cal = cal_data[numeric_cols].values
    y_cal = cal_data[args.target_col].values
    X_test = test_data[numeric_cols].values
    y_test = test_data[args.target_col].values
    
    # Train base model
    from sklearn.ensemble import RandomForestRegressor
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    
    # Get calibration predictions
    y_pred_cal = base_model.predict(X_cal)
    calibration_scores = np.abs(y_cal - y_pred_cal)
    
    # Configure temporal calibrator
    config = TemporalCalibrationConfig(
        use_temporal_weighting=args.use_temporal_weighting,
        temporal_decay=args.temporal_decay,
        use_seasonal_calibration=args.use_seasonal,
        use_hourly_calibration=args.use_hourly,
        use_weekly_calibration=args.use_weekly
    )
    
    # Create and fit temporal calibrator
    temporal_calibrator = TemporalCalibrator(config)
    start_time = time.time()
    temporal_calibrator.fit(cal_data['datetime_utc'], calibration_scores)
    training_time = time.time() - start_time
    
    # Apply temporal calibration to test data
    y_pred_test = base_model.predict(X_test)
    base_scores = np.abs(y_test - y_pred_test)
    calibrated_scores = temporal_calibrator.predict(test_data['datetime_utc'], base_scores)
    
    # Calculate quantile threshold
    threshold = np.quantile(calibrated_scores, 1 - args.alpha)
    
    # Calculate prediction intervals
    lower_bound = y_pred_test - threshold
    upper_bound = y_pred_test + threshold
    
    # Calculate metrics
    mae = np.mean(np.abs(y_test - y_pred_test))
    rmse = np.sqrt(np.mean((y_test - y_pred_test)**2))
    coverage = calculate_coverage(y_test, lower_bound, upper_bound)
    interval_width = calculate_interval_width(lower_bound, upper_bound)
    calibration_error = calculate_calibration_error(y_test, lower_bound, upper_bound, 1-args.alpha)
    
    # Save results
    results = {
        'model_type': 'Temporal Calibration',
        'config': config.__dict__,
        'training_time': training_time,
        'test_metrics': {
            'mae': mae,
            'rmse': rmse,
            'coverage': coverage,
            'interval_width': interval_width,
            'calibration_error': calibration_error,
            'threshold': threshold
        },
        'temporal_summary': temporal_calibrator.get_temporal_summary()
    }
    
    results_path = os.path.join(args.output_dir, 'temporal_calibration_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Temporal calibration completed!")
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Coverage: {coverage:.3f}")
    logger.info(f"Calibration Error: {calibration_error:.4f}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Results saved to: {results_path}")

def test_multi_target_calibration(args):
    """Test multi-target calibration"""
    logger.info("🔧 Testing multi-target calibration...")
    
    # Load data
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return
    
    data = pd.read_parquet(args.data_path)
    logger.info(f"Loaded data: {len(data)} records")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split data
    split_idx = int(len(data) * 0.6)
    val_split_idx = int(len(data) * 0.8)
    
    train_data = data.iloc[:split_idx]
    cal_data = data.iloc[split_idx:val_split_idx]
    test_data = data.iloc[val_split_idx:]
    
    logger.info(f"Train: {len(train_data)}, Calibration: {len(cal_data)}, Test: {len(test_data)}")
    
    # Prepare features
    feature_cols = [col for col in train_data.columns if col not in ['datetime_utc', args.target_col]]
    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_data[col]):
            numeric_cols.append(col)
    
    if not numeric_cols:
        numeric_cols = ['hour', 'day_of_week', 'month']
        for col in numeric_cols:
            if col not in train_data.columns:
                if col == 'hour':
                    train_data[col] = train_data['datetime_utc'].dt.hour
                    cal_data[col] = cal_data['datetime_utc'].dt.hour
                    test_data[col] = test_data['datetime_utc'].dt.hour
                elif col == 'day_of_week':
                    train_data[col] = train_data['datetime_utc'].dt.dayofweek
                    cal_data[col] = cal_data['datetime_utc'].dt.dayofweek
                    test_data[col] = test_data['datetime_utc'].dt.dayofweek
                elif col == 'month':
                    train_data[col] = train_data['datetime_utc'].dt.month
                    cal_data[col] = cal_data['datetime_utc'].dt.month
                    test_data[col] = test_data['datetime_utc'].dt.month
    
    X_train = train_data[numeric_cols].values
    y_train = train_data[args.target_col].values
    X_cal = cal_data[numeric_cols].values
    y_cal = cal_data[args.target_col].values
    X_test = test_data[numeric_cols].values
    y_test = test_data[args.target_col].values
    
    # Create synthetic multi-target data
    y_cal_multi = np.column_stack([y_cal, y_cal * 0.8, y_cal * 1.2])  # Synthetic pm25, pm10, no2
    y_test_multi = np.column_stack([y_test, y_test * 0.8, y_test * 1.2])
    
    # Train base models for each target
    from sklearn.ensemble import RandomForestRegressor
    base_models = {}
    target_names = ['pm25', 'pm10', 'no2']
    
    for i, target in enumerate(target_names):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        base_models[target] = model
    
    # Configure multi-target calibrator
    config = MultiTargetCalibrationConfig(
        target_columns=target_names,
        use_joint_calibration=args.use_joint,
        use_individual_calibration=args.use_individual,
        use_correlation_calibration=args.use_correlation,
        use_ensemble_calibration=args.use_ensemble
    )
    
    # Create and fit multi-target calibrator
    multi_target_calibrator = MultiTargetCalibrator(config)
    start_time = time.time()
    multi_target_calibrator.fit(X_cal, y_cal_multi, base_models)
    training_time = time.time() - start_time
    
    # Make predictions
    predictions = multi_target_calibrator.predict(X_test, base_models)
    
    # Calculate metrics for each target
    target_metrics = {}
    for target in target_names:
        if target in predictions:
            pred = predictions[target]
            y_true = y_test_multi[:, target_names.index(target)]
            
            mae = np.mean(np.abs(y_true - pred['predictions']))
            rmse = np.sqrt(np.mean((y_true - pred['predictions'])**2))
            coverage = calculate_coverage(y_true, pred['lower_bound'], pred['upper_bound'])
            interval_width = calculate_interval_width(pred['lower_bound'], pred['upper_bound'])
            calibration_error = calculate_calibration_error(y_true, pred['lower_bound'], pred['upper_bound'], 1-args.alpha)
            
            target_metrics[target] = {
                'mae': mae,
                'rmse': rmse,
                'coverage': coverage,
                'interval_width': interval_width,
                'calibration_error': calibration_error
            }
    
    # Save results
    results = {
        'model_type': 'Multi-Target Calibration',
        'config': config.__dict__,
        'training_time': training_time,
        'target_metrics': target_metrics,
        'multi_target_summary': multi_target_calibrator.get_multi_target_summary()
    }
    
    results_path = os.path.join(args.output_dir, 'multi_target_calibration_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Multi-target calibration completed!")
    for target, metrics in target_metrics.items():
        logger.info(f"{target.upper()} - MAE: {metrics['mae']:.4f}, Coverage: {metrics['coverage']:.3f}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Results saved to: {results_path}")

def test_calibration_monitoring(args):
    """Test calibration monitoring"""
    logger.info("🔧 Testing calibration monitoring...")
    
    # Load data
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return
    
    data = pd.read_parquet(args.data_path)
    logger.info(f"Loaded data: {len(data)} records")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split data
    split_idx = int(len(data) * 0.6)
    val_split_idx = int(len(data) * 0.8)
    
    train_data = data.iloc[:split_idx]
    cal_data = data.iloc[split_idx:val_split_idx]
    test_data = data.iloc[val_split_idx:]
    
    logger.info(f"Train: {len(train_data)}, Calibration: {len(cal_data)}, Test: {len(test_data)}")
    
    # Prepare features
    feature_cols = [col for col in train_data.columns if col not in ['datetime_utc', args.target_col]]
    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_data[col]):
            numeric_cols.append(col)
    
    if not numeric_cols:
        numeric_cols = ['hour', 'day_of_week', 'month']
        for col in numeric_cols:
            if col not in train_data.columns:
                if col == 'hour':
                    train_data[col] = train_data['datetime_utc'].dt.hour
                    cal_data[col] = cal_data['datetime_utc'].dt.hour
                    test_data[col] = test_data['datetime_utc'].dt.hour
                elif col == 'day_of_week':
                    train_data[col] = train_data['datetime_utc'].dt.dayofweek
                    cal_data[col] = cal_data['datetime_utc'].dt.dayofweek
                    test_data[col] = test_data['datetime_utc'].dt.dayofweek
                elif col == 'month':
                    train_data[col] = train_data['datetime_utc'].dt.month
                    cal_data[col] = cal_data['datetime_utc'].dt.month
                    test_data[col] = test_data['datetime_utc'].dt.month
    
    X_train = train_data[numeric_cols].values
    y_train = train_data[args.target_col].values
    X_cal = cal_data[numeric_cols].values
    y_cal = cal_data[args.target_col].values
    X_test = test_data[numeric_cols].values
    y_test = test_data[args.target_col].values
    
    # Train base model
    from sklearn.ensemble import RandomForestRegressor
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    
    # Configure conformal predictor
    config = ConformalConfig(alpha=args.alpha, method="quantile")
    conformal_predictor = ConformalPredictor(config)
    conformal_predictor.fit(base_model, X_cal, y_cal, X_train, y_train)
    
    # Configure calibration monitor
    monitor_config = CalibrationMonitorConfig(
        monitor_frequency=args.monitor_frequency,
        alert_threshold=args.alert_threshold,
        use_drift_detection=args.use_drift_detection,
        drift_threshold=args.drift_threshold
    )
    
    # Create and start monitor
    monitor = CalibrationMonitor(monitor_config)
    monitor.start_monitoring()
    
    # Simulate streaming predictions
    start_time = time.time()
    batch_size = 100
    n_batches = len(X_test) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_test))
        
        X_batch = X_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        
        # Get predictions
        predictions = conformal_predictor.predict(X_batch, base_model)
        
        # Update monitor
        monitor.update(
            y_batch,
            predictions['predictions'],
            predictions['lower_bound'],
            predictions['upper_bound']
        )
        
        # Log progress
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{n_batches} batches")
    
    # Stop monitoring
    monitor.stop_monitoring()
    monitoring_time = time.time() - start_time
    
    # Get monitoring summary
    summary = monitor.get_monitoring_summary()
    alerts = monitor.get_alerts()
    
    # Save results
    results = {
        'model_type': 'Calibration Monitoring',
        'config': monitor_config.__dict__,
        'monitoring_time': monitoring_time,
        'monitoring_summary': summary,
        'alerts': alerts,
        'n_alerts': len(alerts)
    }
    
    results_path = os.path.join(args.output_dir, 'calibration_monitoring_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Calibration monitoring completed!")
    logger.info(f"Monitoring time: {monitoring_time:.2f} seconds")
    logger.info(f"Total predictions: {summary.get('n_predictions', 0)}")
    logger.info(f"Recent coverage: {summary.get('recent_coverage', 0):.3f}")
    logger.info(f"Total alerts: {len(alerts)}")
    logger.info(f"Results saved to: {results_path}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Calibration Enhancements CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--data-path', type=str, required=True,
                              help='Path to the data file (parquet format)')
    common_parser.add_argument('--output-dir', type=str, required=True,
                              help='Output directory for results')
    common_parser.add_argument('--target-col', type=str, default='pm25',
                              help='Target column name')
    common_parser.add_argument('--alpha', type=float, default=0.1,
                              help='Significance level (1 - coverage probability)')
    
    # Temporal calibration
    temporal_parser = subparsers.add_parser('temporal', parents=[common_parser],
                                          help='Test temporal calibration')
    temporal_parser.add_argument('--use-temporal-weighting', action='store_true',
                                help='Use temporal weighting')
    temporal_parser.add_argument('--temporal-decay', type=float, default=0.99,
                                help='Temporal decay factor')
    temporal_parser.add_argument('--use-seasonal', action='store_true',
                                help='Use seasonal calibration')
    temporal_parser.add_argument('--use-hourly', action='store_true',
                                help='Use hourly calibration')
    temporal_parser.add_argument('--use-weekly', action='store_true',
                                help='Use weekly calibration')
    
    # Multi-target calibration
    multi_target_parser = subparsers.add_parser('multi-target', parents=[common_parser],
                                              help='Test multi-target calibration')
    multi_target_parser.add_argument('--use-joint', action='store_true',
                                    help='Use joint calibration')
    multi_target_parser.add_argument('--use-individual', action='store_true',
                                    help='Use individual calibration')
    multi_target_parser.add_argument('--use-correlation', action='store_true',
                                    help='Use correlation calibration')
    multi_target_parser.add_argument('--use-ensemble', action='store_true',
                                    help='Use ensemble calibration')
    
    # Calibration monitoring
    monitoring_parser = subparsers.add_parser('monitoring', parents=[common_parser],
                                            help='Test calibration monitoring')
    monitoring_parser.add_argument('--monitor-frequency', type=int, default=100,
                                  help='Monitor every N predictions')
    monitoring_parser.add_argument('--alert-threshold', type=float, default=0.05,
                                  help='Alert threshold for coverage deviation')
    monitoring_parser.add_argument('--use-drift-detection', action='store_true',
                                  help='Use drift detection')
    monitoring_parser.add_argument('--drift-threshold', type=float, default=0.1,
                                  help='Drift detection threshold')
    
    args = parser.parse_args()
    
    if args.command == 'temporal':
        test_temporal_calibration(args)
    elif args.command == 'multi-target':
        test_multi_target_calibration(args)
    elif args.command == 'monitoring':
        test_calibration_monitoring(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
