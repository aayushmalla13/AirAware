#!/usr/bin/env python3
"""
Calibration CLI

Command-line interface for training and evaluating calibration methods
for uncertainty quantification in air quality forecasting.
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
    ConformalPredictor, ConformalConfig,
    QuantileCalibrator, QuantileCalibrationConfig,
    AdaptiveConformalPredictor, AdaptiveConformalConfig,
    EnsembleCalibrator, EnsembleCalibrationConfig,
    calculate_coverage, calculate_interval_width, calculate_calibration_error,
    evaluate_calibration_quality, plot_calibration_diagnostics,
    create_calibration_report, compare_calibration_methods
)
from airaware.baselines import (
    BaselineEnsemble, EnsembleConfig
)
from airaware.deep_models import (
    PatchTSTForecaster, PatchTSTConfig,
    SimpleTFTForecaster, SimpleTFTConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_conformal_calibration(args):
    """Train conformal calibration"""
    logger.info("🔧 Training conformal calibration...")
    
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
    
    # Train base model
    if args.base_model == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif args.base_model == "xgboost":
        from xgboost import XGBRegressor
        base_model = XGBRegressor(n_estimators=100, random_state=42)
    elif args.base_model == "lightgbm":
        from lightgbm import LGBMRegressor
        base_model = LGBMRegressor(n_estimators=100, random_state=42)
    else:
        logger.error(f"Unknown base model: {args.base_model}")
        return
    
    # Prepare features for training - only numeric columns
    feature_cols = [col for col in train_data.columns if col not in ['datetime_utc', args.target_col]]
    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_data[col]):
            numeric_cols.append(col)
    
    if not numeric_cols:
        # Create dummy features if no numeric features
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
    logger.info(f"Training {args.base_model} base model...")
    base_model.fit(X_train, y_train)
    
    # Configure conformal predictor
    config = ConformalConfig(
        alpha=args.alpha,
        method="quantile"
    )
    
    # Create conformal predictor
    conformal_predictor = ConformalPredictor(config)
    
    # Fit conformal predictor
    start_time = time.time()
    conformal_predictor.fit(base_model, X_cal, y_cal, X_train, y_train)
    training_time = time.time() - start_time
    
    # Save model
    model_path = os.path.join(args.output_dir, 'conformal_model.pkl')
    conformal_predictor.save_predictor(model_path)
    
    # Evaluate on test set
    test_predictions = conformal_predictor.predict(X_test, base_model)
    test_actuals = y_test
    
    # Calculate metrics
    mae = np.mean(np.abs(test_actuals - test_predictions['predictions']))
    rmse = np.sqrt(np.mean((test_actuals - test_predictions['predictions'])**2))
    
    # Calculate coverage
    coverage = calculate_coverage(
        test_actuals, 
        test_predictions['lower_bound'], 
        test_predictions['upper_bound']
    )
    
    # Calculate interval width
    interval_width = calculate_interval_width(
        test_predictions['lower_bound'], 
        test_predictions['upper_bound']
    )
    
    # Calculate calibration error
    calibration_error = calculate_calibration_error(
        test_actuals,
        test_predictions['lower_bound'],
        test_predictions['upper_bound'],
        target_coverage=1-args.alpha
    )
    
    # Save results
    results = {
        'model_type': 'Conformal Calibration',
        'base_model': args.base_model,
        'config': config.__dict__,
        'training_time': training_time,
        'test_metrics': {
            'mae': mae,
            'rmse': rmse,
            'coverage': coverage,
            'interval_width': interval_width,
            'calibration_error': calibration_error
        },
        'predictor_info': conformal_predictor.get_predictor_info()
    }
    
    results_path = os.path.join(args.output_dir, 'conformal_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Conformal calibration completed!")
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Coverage: {coverage:.3f}")
    logger.info(f"Calibration Error: {calibration_error:.4f}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Results saved to: {results_path}")

def train_quantile_calibration(args):
    """Train quantile calibration"""
    logger.info("🔧 Training quantile calibration...")
    
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
    
    # Train base model
    if args.base_model == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif args.base_model == "xgboost":
        from xgboost import XGBRegressor
        base_model = XGBRegressor(n_estimators=100, random_state=42)
    elif args.base_model == "lightgbm":
        from lightgbm import LGBMRegressor
        base_model = LGBMRegressor(n_estimators=100, random_state=42)
    else:
        logger.error(f"Unknown base model: {args.base_model}")
        return
    
    # Prepare features for training - only numeric columns
    feature_cols = [col for col in train_data.columns if col not in ['datetime_utc', args.target_col]]
    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_data[col]):
            numeric_cols.append(col)
    
    if not numeric_cols:
        # Create dummy features if no numeric features
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
    logger.info(f"Training {args.base_model} base model...")
    base_model.fit(X_train, y_train)
    
    # Configure quantile calibrator
    config = QuantileCalibrationConfig(
        target_quantiles=args.quantile_levels,
        calibration_method="isotonic" if args.use_isotonic else "linear"
    )
    
    # Create quantile calibrator
    quantile_calibrator = QuantileCalibrator(config)
    
    # Fit quantile calibrator
    start_time = time.time()
    quantile_calibrator.fit(base_model, X_cal, y_cal, X_train, y_train)
    training_time = time.time() - start_time
    
    # Save model
    model_path = os.path.join(args.output_dir, 'quantile_model.pkl')
    quantile_calibrator.save_calibrator(model_path)
    
    # Evaluate on test set
    test_predictions = quantile_calibrator.predict(X_test, base_model)
    test_actuals = y_test
    
    # Calculate metrics
    mae = np.mean(np.abs(test_actuals - test_predictions['predictions']))
    rmse = np.sqrt(np.mean((test_actuals - test_predictions['predictions'])**2))
    
    # Calculate coverage for each quantile level
    coverage_results = {}
    for level in args.quantile_levels:
        if level < 0.5:
            lower_key = f'lower_{level:.3f}'
            if lower_key in test_predictions:
                coverage = calculate_coverage(
                    test_actuals,
                    test_predictions[lower_key],
                    test_predictions['predictions'] + np.inf  # Upper bound is infinity for lower quantiles
                )
                coverage_results[f'lower_{level:.3f}_coverage'] = coverage
        else:
            upper_key = f'upper_{level:.3f}'
            if upper_key in test_predictions:
                coverage = calculate_coverage(
                    test_actuals,
                    test_predictions['predictions'] - np.inf,  # Lower bound is negative infinity for upper quantiles
                    test_predictions[upper_key]
                )
                coverage_results[f'upper_{level:.3f}_coverage'] = coverage
    
    # Save results
    results = {
        'model_type': 'Quantile Calibration',
        'base_model': args.base_model,
        'config': config.__dict__,
        'training_time': training_time,
        'test_metrics': {
            'mae': mae,
            'rmse': rmse,
            'coverage_results': coverage_results
        },
        'calibrator_info': quantile_calibrator.get_calibrator_info()
    }
    
    results_path = os.path.join(args.output_dir, 'quantile_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Quantile calibration completed!")
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Coverage results: {coverage_results}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Results saved to: {results_path}")

def train_adaptive_conformal(args):
    """Train adaptive conformal calibration"""
    logger.info("🔧 Training adaptive conformal calibration...")
    
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
    
    # Train base model
    if args.base_model == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif args.base_model == "xgboost":
        from xgboost import XGBRegressor
        base_model = XGBRegressor(n_estimators=100, random_state=42)
    elif args.base_model == "lightgbm":
        from lightgbm import LGBMRegressor
        base_model = LGBMRegressor(n_estimators=100, random_state=42)
    else:
        logger.error(f"Unknown base model: {args.base_model}")
        return
    
    # Prepare features for training - only numeric columns
    feature_cols = [col for col in train_data.columns if col not in ['datetime_utc', args.target_col]]
    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_data[col]):
            numeric_cols.append(col)
    
    if not numeric_cols:
        # Create dummy features if no numeric features
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
    logger.info(f"Training {args.base_model} base model...")
    base_model.fit(X_train, y_train)
    
    # Configure adaptive conformal predictor
    config = AdaptiveConformalConfig(
        alpha=args.alpha,
        adaptation_method=args.adaptation_method,
        adaptation_rate=args.adaptation_rate,
        use_online_adaptation=args.use_online_adaptation,
        use_sliding_window=args.use_sliding_window,
        window_size=args.window_size
    )
    
    # Create adaptive conformal predictor
    adaptive_predictor = AdaptiveConformalPredictor(config)
    
    # Fit adaptive conformal predictor
    start_time = time.time()
    adaptive_predictor.fit(base_model, X_cal, y_cal, X_train, y_train)
    training_time = time.time() - start_time
    
    # Save model
    model_path = os.path.join(args.output_dir, 'adaptive_conformal_model.pkl')
    adaptive_predictor.save_predictor(model_path)
    
    # Evaluate on test set
    test_predictions = adaptive_predictor.predict(X_test, base_model)
    test_actuals = y_test
    
    # Calculate metrics
    mae = np.mean(np.abs(test_actuals - test_predictions['predictions']))
    rmse = np.sqrt(np.mean((test_actuals - test_predictions['predictions'])**2))
    
    # Calculate coverage
    coverage = calculate_coverage(
        test_actuals, 
        test_predictions['lower_bound'], 
        test_predictions['upper_bound']
    )
    
    # Calculate interval width
    interval_width = calculate_interval_width(
        test_predictions['lower_bound'], 
        test_predictions['upper_bound']
    )
    
    # Calculate calibration error
    calibration_error = calculate_calibration_error(
        test_actuals,
        test_predictions['lower_bound'],
        test_predictions['upper_bound'],
        target_coverage=1-args.alpha
    )
    
    # Get adaptation summary
    adaptation_summary = adaptive_predictor.get_adaptation_summary()
    
    # Save results
    results = {
        'model_type': 'Adaptive Conformal Calibration',
        'base_model': args.base_model,
        'config': config.__dict__,
        'training_time': training_time,
        'test_metrics': {
            'mae': mae,
            'rmse': rmse,
            'coverage': coverage,
            'interval_width': interval_width,
            'calibration_error': calibration_error
        },
        'adaptation_summary': adaptation_summary,
        'predictor_info': adaptive_predictor.get_predictor_info()
    }
    
    results_path = os.path.join(args.output_dir, 'adaptive_conformal_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Adaptive conformal calibration completed!")
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Coverage: {coverage:.3f}")
    logger.info(f"Calibration Error: {calibration_error:.4f}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Results saved to: {results_path}")

def compare_calibration_methods_cli(args):
    """Compare different calibration methods"""
    logger.info("🔍 Comparing calibration methods...")
    
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
    
    # Train base model
    if args.base_model == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif args.base_model == "xgboost":
        from xgboost import XGBRegressor
        base_model = XGBRegressor(n_estimators=100, random_state=42)
    elif args.base_model == "lightgbm":
        from lightgbm import LGBMRegressor
        base_model = LGBMRegressor(n_estimators=100, random_state=42)
    else:
        logger.error(f"Unknown base model: {args.base_model}")
        return
    
    # Prepare features for training - only numeric columns
    feature_cols = [col for col in train_data.columns if col not in ['datetime_utc', args.target_col]]
    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_data[col]):
            numeric_cols.append(col)
    
    if not numeric_cols:
        # Create dummy features if no numeric features
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
    logger.info(f"Training {args.base_model} base model...")
    base_model.fit(X_train, y_train)
    
    # Test different calibration methods
    methods = ['conformal', 'quantile', 'adaptive_conformal']
    results = {}
    
    for method in methods:
        logger.info(f"Testing {method} calibration...")
        
        if method == 'conformal':
            config = ConformalConfig(alpha=args.alpha)
            calibrator = ConformalPredictor(config)
            calibrator.fit(base_model, X_cal, y_cal, X_train, y_train)
            predictions = calibrator.predict(X_test, base_model)
            
        elif method == 'quantile':
            config = QuantileCalibrationConfig(target_quantiles=[args.alpha/2, 1-args.alpha/2])
            calibrator = QuantileCalibrator(config)
            calibrator.fit(base_model, X_cal, y_cal, X_train, y_train)
            predictions = calibrator.predict(X_test, base_model)
            
        elif method == 'adaptive_conformal':
            config = AdaptiveConformalConfig(alpha=args.alpha)
            calibrator = AdaptiveConformalPredictor(config)
            calibrator.fit(base_model, X_cal, y_cal, X_train, y_train)
            predictions = calibrator.predict(X_test, base_model)
        
        # Evaluate
        test_actuals = y_test
        evaluation = evaluate_calibration_quality(
            test_actuals,
            predictions['predictions'],
            predictions['lower_bound'],
            predictions['upper_bound'],
            target_coverage=1-args.alpha
        )
        
        results[method] = {'evaluation': evaluation}
    
    # Compare methods
    compare_calibration_methods(results, 
                              save_path=os.path.join(args.output_dir, 'calibration_comparison.png'))
    
    # Save comparison results
    comparison_path = os.path.join(args.output_dir, 'calibration_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Calibration comparison completed!")
    logger.info(f"Results saved to: {comparison_path}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Calibration CLI for Air Quality Forecasting')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--data-path', type=str, required=True,
                              help='Path to the data file (parquet format)')
    common_parser.add_argument('--output-dir', type=str, required=True,
                              help='Output directory for results')
    common_parser.add_argument('--target-col', type=str, default='pm25',
                              help='Target column name')
    common_parser.add_argument('--base-model', type=str, default='random_forest',
                              choices=['random_forest', 'xgboost', 'lightgbm'],
                              help='Base model for calibration')
    common_parser.add_argument('--alpha', type=float, default=0.1,
                              help='Significance level (1 - coverage probability)')
    
    # Conformal calibration
    conformal_parser = subparsers.add_parser('conformal', parents=[common_parser],
                                           help='Train conformal calibration')
    conformal_parser.add_argument('--conformal-method', type=str, default='split',
                                 choices=['split', 'cv', 'jackknife'],
                                 help='Conformal prediction method')
    conformal_parser.add_argument('--use-cv', action='store_true',
                                 help='Use cross-validation for conformal prediction')
    conformal_parser.add_argument('--n-folds', type=int, default=5,
                                 help='Number of CV folds')
    
    # Quantile calibration
    quantile_parser = subparsers.add_parser('quantile', parents=[common_parser],
                                          help='Train quantile calibration')
    quantile_parser.add_argument('--quantile-levels', type=float, nargs='+',
                                default=[0.05, 0.95],
                                help='Quantile levels for calibration')
    quantile_parser.add_argument('--use-isotonic', action='store_true',
                                help='Use isotonic regression for calibration')
    quantile_parser.add_argument('--use-platt', action='store_true',
                                help='Use Platt scaling for calibration')
    
    # Adaptive conformal calibration
    adaptive_parser = subparsers.add_parser('adaptive-conformal', parents=[common_parser],
                                          help='Train adaptive conformal calibration')
    adaptive_parser.add_argument('--adaptation-method', type=str, default='online',
                                choices=['online', 'batch', 'sliding_window'],
                                help='Adaptation method')
    adaptive_parser.add_argument('--adaptation-rate', type=float, default=0.01,
                                help='Learning rate for adaptation')
    adaptive_parser.add_argument('--use-online-adaptation', action='store_true',
                                help='Use online adaptation')
    adaptive_parser.add_argument('--use-sliding-window', action='store_true',
                                help='Use sliding window adaptation')
    adaptive_parser.add_argument('--window-size', type=int, default=1000,
                                help='Sliding window size')
    
    # Compare methods
    compare_parser = subparsers.add_parser('compare', parents=[common_parser],
                                         help='Compare different calibration methods')
    
    args = parser.parse_args()
    
    if args.command == 'conformal':
        train_conformal_calibration(args)
    elif args.command == 'quantile':
        train_quantile_calibration(args)
    elif args.command == 'adaptive-conformal':
        train_adaptive_conformal(args)
    elif args.command == 'compare':
        compare_calibration_methods_cli(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
