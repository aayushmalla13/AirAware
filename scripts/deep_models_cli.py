#!/usr/bin/env python3
"""
CLI for Deep Learning Models (CP-7)

This script provides a command-line interface for training and evaluating
deep learning models including PatchTST and TFT.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from airaware.deep_models import (
    PatchTSTForecaster, PatchTSTConfig,
    TFTForecaster, TFTConfig,
    DeepEnsemble, DeepEnsembleConfig,
    DeepModelTrainer, TrainingConfig,
    DeepDataPreprocessor, PreprocessingConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_parser():
    """Create command line parser"""
    parser = argparse.ArgumentParser(description="Deep Learning Models CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # PatchTST command
    patchtst_parser = subparsers.add_parser('patchtst', help='Train PatchTST model')
    patchtst_parser.add_argument('--data-path', type=str, default='data/processed/real_features.parquet',
                               help='Path to training data')
    patchtst_parser.add_argument('--output-dir', type=str, default='data/artifacts/deep_models',
                               help='Output directory for models and results')
    patchtst_parser.add_argument('--target-col', type=str, default='pm25',
                               help='Target column name')
    patchtst_parser.add_argument('--context-length', type=int, default=96,
                               help='Input sequence length')
    patchtst_parser.add_argument('--prediction-length', type=int, default=24,
                               help='Forecast horizon')
    patchtst_parser.add_argument('--epochs', type=int, default=50,
                               help='Number of training epochs')
    patchtst_parser.add_argument('--batch-size', type=int, default=32,
                               help='Batch size')
    patchtst_parser.add_argument('--learning-rate', type=float, default=1e-4,
                               help='Learning rate')
    patchtst_parser.add_argument('--device', type=str, default='auto',
                               help='Device to use (auto, cpu, cuda)')
    patchtst_parser.add_argument('--days', type=int, default=None,
                               help='If set, filter to last N days by datetime_utc before training')
    
    # TFT command
    tft_parser = subparsers.add_parser('tft', help='Train TFT model')
    
    # Simple TFT command removed
    
    tft_parser.add_argument('--data-path', type=str, default='data/processed/real_features.parquet',
                          help='Path to training data')
    tft_parser.add_argument('--output-dir', type=str, default='data/artifacts/deep_models',
                          help='Output directory for models and results')
    tft_parser.add_argument('--target-col', type=str, default='pm25',
                          help='Target column name')
    tft_parser.add_argument('--context-length', type=int, default=96,
                          help='Input sequence length')
    tft_parser.add_argument('--prediction-length', type=int, default=24,
                          help='Forecast horizon')
    tft_parser.add_argument('--epochs', type=int, default=50,
                          help='Number of training epochs')
    tft_parser.add_argument('--batch-size', type=int, default=32,
                          help='Batch size')
    tft_parser.add_argument('--learning-rate', type=float, default=1e-3,
                          help='Learning rate')
    tft_parser.add_argument('--device', type=str, default='auto',
                          help='Device to use (auto, cpu, cuda)')
    tft_parser.add_argument('--days', type=int, default=None,
                          help='If set, filter to last N days by datetime_utc before training')
    
    # Ensemble command
    ensemble_parser = subparsers.add_parser('ensemble', help='Train deep ensemble')
    ensemble_parser.add_argument('--data-path', type=str, default='data/processed/real_features.parquet',
                               help='Path to training data')
    ensemble_parser.add_argument('--output-dir', type=str, default='data/artifacts/deep_models',
                               help='Output directory for models and results')
    ensemble_parser.add_argument('--target-col', type=str, default='pm25',
                               help='Target column name')
    ensemble_parser.add_argument('--n-models', type=int, default=3,
                               help='Number of models in ensemble')
    ensemble_parser.add_argument('--epochs', type=int, default=30,
                               help='Number of training epochs per model')
    ensemble_parser.add_argument('--device', type=str, default='auto',
                               help='Device to use (auto, cpu, cuda)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    eval_parser.add_argument('--model-path', type=str, required=True,
                           help='Path to trained model')
    eval_parser.add_argument('--test-data', type=str, required=True,
                           help='Path to test data')
    eval_parser.add_argument('--target-col', type=str, default='pm25',
                           help='Target column name')
    eval_parser.add_argument('--output-dir', type=str, default='data/artifacts/deep_models',
                           help='Output directory for results')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare different models')
    compare_parser.add_argument('--models', type=str, nargs='+', required=True,
                              help='Paths to model directories')
    compare_parser.add_argument('--test-data', type=str, required=True,
                              help='Path to test data')
    compare_parser.add_argument('--target-col', type=str, default='pm25',
                              help='Target column name')
    compare_parser.add_argument('--output-dir', type=str, default='data/artifacts/deep_models',
                              help='Output directory for results')
    
    return parser

def train_patchtst(args):
    """Train PatchTST model"""
    logger.info("🚀 Training PatchTST model...")
    
    # Load data
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return
    
    data = pd.read_parquet(args.data_path)
    # Optional: restrict to last N days
    if args.days is not None and 'datetime_utc' in data.columns:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=args.days)
        data = data[data['datetime_utc'] >= cutoff].reset_index(drop=True)
    logger.info(f"Loaded data: {len(data)} records")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure model
    config = PatchTSTConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Create model
    model = PatchTSTForecaster(config)
    
    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    # Train model
    start_time = time.time()
    history = model.fit(train_data, val_data, args.target_col)
    training_time = time.time() - start_time
    
    # Save model
    model_path = os.path.join(args.output_dir, 'patchtst_model.pth')
    model.save_model(model_path)
    
    # Evaluate on validation set
    val_predictions = model.predict(val_data, args.target_col)
    val_actuals = val_data[args.target_col].values
    
    # Handle prediction shape mismatch
    if val_predictions.ndim > 1:
        # Take the first prediction for each sample
        val_predictions = val_predictions[:, 0] if val_predictions.shape[1] > 0 else val_predictions.flatten()
    
    # Ensure same length
    min_len = min(len(val_actuals), len(val_predictions))
    val_actuals = val_actuals[:min_len]
    val_predictions = val_predictions[:min_len]
    
    # Calculate metrics
    mae = np.mean(np.abs(val_actuals - val_predictions))
    rmse = np.sqrt(np.mean((val_actuals - val_predictions)**2))
    
    # Save results
    results = {
        'model_type': 'PatchTST',
        'config': config.__dict__,
        'training_time': training_time,
        'training_history': history,
        'validation_metrics': {
            'mae': mae,
            'rmse': rmse
        },
        'model_info': model.get_model_info()
    }
    
    results_path = os.path.join(args.output_dir, 'patchtst_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ PatchTST training completed!")
    logger.info(f"Validation MAE: {mae:.4f}")
    logger.info(f"Validation RMSE: {rmse:.4f}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Results saved to: {results_path}")

    # Simple TFT training removed

def train_tft(args):
    """Train TFT model"""
    logger.info("🚀 Training TFT model...")
    
    # Load data
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return
    
    data = pd.read_parquet(args.data_path)
    if args.days is not None and 'datetime_utc' in data.columns:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=args.days)
        data = data[data['datetime_utc'] >= cutoff].reset_index(drop=True)
    logger.info(f"Loaded data: {len(data)} records")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure model
    config = TFTConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Create model
    model = TFTForecaster(config)
    
    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    # Train model
    start_time = time.time()
    history = model.fit(train_data, val_data, args.target_col)
    training_time = time.time() - start_time
    
    # Save model
    model_path = os.path.join(args.output_dir, 'tft_model.pth')
    model.save_model(model_path)
    
    # Evaluate on validation set
    val_predictions = model.predict(val_data, args.target_col)
    val_actuals = val_data[args.target_col].values
    
    # Calculate metrics (use median quantile for TFT)
    median_preds = val_predictions['q0.5']
    mae = np.mean(np.abs(val_actuals - median_preds))
    rmse = np.sqrt(np.mean((val_actuals - median_preds)**2))
    
    # Save results
    results = {
        'model_type': 'TFT',
        'config': config.__dict__,
        'training_time': training_time,
        'training_history': history,
        'validation_metrics': {
            'mae': mae,
            'rmse': rmse
        },
        'model_info': model.get_model_info()
    }
    
    results_path = os.path.join(args.output_dir, 'tft_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ TFT training completed!")
    logger.info(f"Validation MAE: {mae:.4f}")
    logger.info(f"Validation RMSE: {rmse:.4f}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Results saved to: {results_path}")

def train_ensemble(args):
    """Train deep ensemble"""
    logger.info("🚀 Training Deep Ensemble...")
    
    # Load data
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return
    
    data = pd.read_parquet(args.data_path)
    logger.info(f"Loaded data: {len(data)} records")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure ensemble
    patchtst_config = PatchTSTConfig(
        num_epochs=args.epochs,
        device=args.device
    )
    tft_config = TFTConfig(
        num_epochs=args.epochs,
        device=args.device
    )
    
    ensemble_config = DeepEnsembleConfig(
        patchtst_config=patchtst_config,
        tft_config=tft_config,
        n_models=args.n_models,
        device=args.device
    )
    
    # Create ensemble
    ensemble = DeepEnsemble(ensemble_config)
    
    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    # Train ensemble
    start_time = time.time()
    history = ensemble.fit(train_data, val_data, args.target_col)
    training_time = time.time() - start_time
    
    # Save ensemble
    ensemble_path = os.path.join(args.output_dir, 'deep_ensemble')
    ensemble.save_ensemble(ensemble_path)
    
    # Evaluate on validation set
    val_predictions = ensemble.predict(val_data, args.target_col)
    val_actuals = val_data[args.target_col].values
    
    # Calculate metrics
    mae = np.mean(np.abs(val_actuals - val_predictions['mean']))
    rmse = np.sqrt(np.mean((val_actuals - val_predictions['mean'])**2))
    
    # Save results
    results = {
        'model_type': 'Deep Ensemble',
        'config': ensemble_config.__dict__,
        'training_time': training_time,
        'training_history': history,
        'validation_metrics': {
            'mae': mae,
            'rmse': rmse
        },
        'ensemble_info': ensemble.get_ensemble_info()
    }
    
    results_path = os.path.join(args.output_dir, 'ensemble_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Deep Ensemble training completed!")
    logger.info(f"Validation MAE: {mae:.4f}")
    logger.info(f"Validation RMSE: {rmse:.4f}")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Results saved to: {results_path}")

def evaluate_model(args):
    """Evaluate trained model"""
    logger.info("🔍 Evaluating model...")
    
    # Load test data
    if not os.path.exists(args.test_data):
        logger.error(f"Test data file not found: {args.test_data}")
        return
    
    test_data = pd.read_parquet(args.test_data)
    logger.info(f"Loaded test data: {len(test_data)} records")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model (determine type from path)
    if 'patchtst' in args.model_path:
        model = PatchTSTForecaster(PatchTSTConfig())
        model.load_model(args.model_path)
        predictions = model.predict(test_data, args.target_col)
        
    elif 'tft' in args.model_path:
        model = TFTForecaster(TFTConfig())
        model.load_model(args.model_path)
        pred_dict = model.predict(test_data, args.target_col)
        predictions = pred_dict['q0.5']  # Use median quantile
        
    elif 'ensemble' in args.model_path:
        ensemble = DeepEnsemble(DeepEnsembleConfig())
        ensemble.load_ensemble(args.model_path)
        pred_dict = ensemble.predict(test_data, args.target_col)
        predictions = pred_dict['mean']
        
    else:
        logger.error("Unknown model type")
        return
    
    # Calculate metrics
    actuals = test_data[args.target_col].values
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions)**2))
    
    # Save results
    results = {
        'model_path': args.model_path,
        'test_metrics': {
            'mae': mae,
            'rmse': rmse
        },
        'n_test_samples': len(test_data)
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Model evaluation completed!")
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Results saved to: {results_path}")

def compare_models(args):
    """Compare different models"""
    logger.info("🔍 Comparing models...")
    
    # Load test data
    if not os.path.exists(args.test_data):
        logger.error(f"Test data file not found: {args.test_data}")
        return
    
    test_data = pd.read_parquet(args.test_data)
    logger.info(f"Loaded test data: {len(test_data)} records")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    comparison_results = {}
    
    for model_path in args.models:
        logger.info(f"Evaluating model: {model_path}")
        
        try:
            # Load and evaluate model
            if 'patchtst' in model_path:
                model = PatchTSTForecaster(PatchTSTConfig())
                model.load_model(model_path)
                predictions = model.predict(test_data, args.target_col)
                
            elif 'tft' in model_path:
                model = TFTForecaster(TFTConfig())
                model.load_model(model_path)
                pred_dict = model.predict(test_data, args.target_col)
                predictions = pred_dict['q0.5']
                
            elif 'ensemble' in model_path:
                ensemble = DeepEnsemble(DeepEnsembleConfig())
                ensemble.load_ensemble(model_path)
                pred_dict = ensemble.predict(test_data, args.target_col)
                predictions = pred_dict['mean']
                
            else:
                logger.warning(f"Unknown model type for {model_path}")
                continue
            
            # Calculate metrics
            actuals = test_data[args.target_col].values
            mae = np.mean(np.abs(actuals - predictions))
            rmse = np.sqrt(np.mean((actuals - predictions)**2))
            
            comparison_results[model_path] = {
                'mae': mae,
                'rmse': rmse
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_path}: {e}")
            continue
    
    # Save comparison results
    results_path = os.path.join(args.output_dir, 'model_comparison.json')
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Print comparison
    logger.info("📊 Model Comparison Results:")
    logger.info("-" * 50)
    for model_path, metrics in comparison_results.items():
        logger.info(f"{model_path}:")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    
    logger.info(f"✅ Model comparison completed!")
    logger.info(f"Results saved to: {results_path}")

def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == 'patchtst':
            train_patchtst(args)
        elif args.command == 'tft':
            train_tft(args)
        # simple-tft command removed
        elif args.command == 'ensemble':
            train_ensemble(args)
        elif args.command == 'evaluate':
            evaluate_model(args)
        elif args.command == 'compare':
            compare_models(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        raise

if __name__ == "__main__":
    main()
