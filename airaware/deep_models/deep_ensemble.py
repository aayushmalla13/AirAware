"""
Deep Ensemble for combining multiple deep learning models

This module implements ensemble methods for combining predictions from
multiple deep learning models (PatchTST, TFT, etc.) with uncertainty quantification.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from collections import defaultdict

from .patchtst import PatchTSTForecaster, PatchTSTConfig
from .tft import TFTForecaster, TFTConfig

logger = logging.getLogger(__name__)

@dataclass
class DeepEnsembleConfig:
    """Configuration for deep ensemble"""
    # Model configurations
    patchtst_config: PatchTSTConfig = None
    tft_config: TFTConfig = None
    
    # Ensemble parameters
    n_models: int = 3  # Number of models in ensemble
    ensemble_method: str = "mean"  # "mean", "median", "weighted", "stacking"
    
    # Uncertainty quantification
    use_uncertainty: bool = True
    uncertainty_method: str = "ensemble"  # "ensemble", "dropout", "both"
    
    # Training parameters
    train_models_independently: bool = True
    use_cross_validation: bool = False
    cv_folds: int = 5
    
    # Device
    device: str = "auto"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set default configs if not provided
        if self.patchtst_config is None:
            self.patchtst_config = PatchTSTConfig()
        if self.tft_config is None:
            self.tft_config = TFTConfig()

class DeepEnsemble:
    """Deep ensemble for combining multiple deep learning models"""
    
    def __init__(self, config: DeepEnsembleConfig):
        self.config = config
        self.models = {}
        self.model_weights = {}
        self.is_fitted = False
        self.training_history = {}
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ensemble models"""
        # PatchTST models
        for i in range(self.config.n_models):
            model_name = f"patchtst_{i}"
            self.models[model_name] = PatchTSTForecaster(self.config.patchtst_config)
        
        # TFT models
        for i in range(self.config.n_models):
            model_name = f"tft_{i}"
            self.models[model_name] = TFTForecaster(self.config.tft_config)
        
        # Initialize equal weights
        for model_name in self.models.keys():
            self.model_weights[model_name] = 1.0 / len(self.models)
    
    def _prepare_data(self, data: pd.DataFrame, target_col: str = 'pm25') -> Dict[str, Any]:
        """Prepare data for training/prediction"""
        # Ensure data is sorted by datetime
        data = data.sort_values('datetime_utc').reset_index(drop=True)
        
        # Extract features and target
        feature_cols = [col for col in data.columns if col not in ['datetime_utc', target_col, 'station_id']]
        features = data[feature_cols].values
        target = data[target_col].values
        
        # Create static features for TFT
        if 'station_id' in data.columns:
            station_ids = data['station_id'].unique()
            station_to_idx = {sid: i for i, sid in enumerate(station_ids)}
            static_features = np.array([station_to_idx[sid] for sid in data['station_id']])
            static_one_hot = np.zeros((len(static_features), len(station_ids)))
            static_one_hot[np.arange(len(static_features)), static_features] = 1
        else:
            static_one_hot = np.zeros((len(data), 1))
        
        return {
            'data': data,
            'features': features,
            'target': target,
            'static_features': static_one_hot,
            'feature_cols': feature_cols
        }
    
    def fit(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None, 
            target_col: str = 'pm25') -> Dict[str, Any]:
        """Train the ensemble models"""
        logger.info("🚀 Training Deep Ensemble...")
        
        # Prepare data
        train_prepared = self._prepare_data(train_data, target_col)
        val_prepared = self._prepare_data(val_data, target_col) if val_data is not None else None
        
        # Train models independently
        if self.config.train_models_independently:
            for model_name, model in self.models.items():
                logger.info(f"Training {model_name}...")
                
                try:
                    if 'patchtst' in model_name:
                        # Train PatchTST
                        history = model.fit(train_data, val_data, target_col)
                    elif 'tft' in model_name:
                        # Train TFT
                        history = model.fit(train_data, val_data, target_col)
                    
                    self.training_history[model_name] = history
                    logger.info(f"✅ {model_name} training completed")
                    
                except Exception as e:
                    logger.error(f"❌ {model_name} training failed: {e}")
                    # Remove failed model
                    del self.models[model_name]
                    del self.model_weights[model_name]
        
        # Update model weights
        self._update_model_weights(val_data, target_col)
        
        self.is_fitted = True
        logger.info("✅ Deep Ensemble training completed!")
        
        return self.training_history
    
    def _update_model_weights(self, val_data: Optional[pd.DataFrame], target_col: str):
        """Update model weights based on validation performance"""
        if val_data is None or self.config.ensemble_method != "weighted":
            return
        
        logger.info("Updating model weights based on validation performance...")
        
        # Get validation predictions
        val_predictions = {}
        val_actuals = val_data[target_col].values
        
        for model_name, model in self.models.items():
            try:
                if 'patchtst' in model_name:
                    preds = model.predict(val_data, target_col)
                    val_predictions[model_name] = preds
                elif 'tft' in model_name:
                    preds = model.predict(val_data, target_col)
                    # Use median quantile for TFT
                    val_predictions[model_name] = preds['q0.5']
            except Exception as e:
                logger.warning(f"Failed to get predictions from {model_name}: {e}")
                continue
        
        # Calculate weights based on MAE
        weights = {}
        for model_name, preds in val_predictions.items():
            mae = np.mean(np.abs(val_actuals - preds))
            weights[model_name] = 1.0 / (mae + 1e-8)  # Inverse MAE
        
        # Normalize weights
        total_weight = sum(weights.values())
        for model_name in weights:
            weights[model_name] /= total_weight
            self.model_weights[model_name] = weights[model_name]
        
        logger.info(f"Updated model weights: {weights}")
    
    def predict(self, data: pd.DataFrame, target_col: str = 'pm25') -> Dict[str, np.ndarray]:
        """Make ensemble predictions with uncertainty quantification"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        logger.info("Making ensemble predictions...")
        
        # Get predictions from all models
        model_predictions = {}
        model_uncertainties = {}
        
        for model_name, model in self.models.items():
            try:
                if 'patchtst' in model_name:
                    preds = model.predict(data, target_col)
                    model_predictions[model_name] = preds
                    # Simple uncertainty estimation for PatchTST
                    model_uncertainties[model_name] = np.std(preds) * np.ones_like(preds)
                    
                elif 'tft' in model_name:
                    preds = model.predict(data, target_col)
                    # Use median for point prediction
                    model_predictions[model_name] = preds['q0.5']
                    # Use quantile range for uncertainty
                    model_uncertainties[model_name] = (preds['q0.75'] - preds['q0.25']) / 2
                    
            except Exception as e:
                logger.warning(f"Failed to get predictions from {model_name}: {e}")
                continue
        
        if not model_predictions:
            raise ValueError("No models available for prediction")
        
        # Combine predictions
        if self.config.ensemble_method == "mean":
            ensemble_pred = np.mean(list(model_predictions.values()), axis=0)
        elif self.config.ensemble_method == "median":
            ensemble_pred = np.median(list(model_predictions.values()), axis=0)
        elif self.config.ensemble_method == "weighted":
            weighted_pred = np.zeros_like(list(model_predictions.values())[0])
            for model_name, preds in model_predictions.items():
                weighted_pred += self.model_weights[model_name] * preds
            ensemble_pred = weighted_pred
        else:
            # Default to mean
            ensemble_pred = np.mean(list(model_predictions.values()), axis=0)
        
        # Calculate ensemble uncertainty
        if self.config.use_uncertainty:
            if self.config.uncertainty_method == "ensemble":
                # Epistemic uncertainty from model disagreement
                pred_array = np.array(list(model_predictions.values()))
                epistemic_uncertainty = np.std(pred_array, axis=0)
                
                # Aleatoric uncertainty from individual models
                aleatoric_uncertainty = np.mean(list(model_uncertainties.values()), axis=0)
                
                # Total uncertainty
                total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
                
            else:
                # Simple average of individual uncertainties
                total_uncertainty = np.mean(list(model_uncertainties.values()), axis=0)
        else:
            total_uncertainty = np.zeros_like(ensemble_pred)
        
        # Create prediction intervals
        prediction_intervals = {
            'mean': ensemble_pred,
            'std': total_uncertainty,
            'lower_95': ensemble_pred - 1.96 * total_uncertainty,
            'upper_95': ensemble_pred + 1.96 * total_uncertainty,
            'lower_68': ensemble_pred - total_uncertainty,
            'upper_68': ensemble_pred + total_uncertainty
        }
        
        # Add individual model predictions for analysis
        prediction_intervals['individual_models'] = model_predictions
        
        logger.info("✅ Ensemble predictions completed")
        return prediction_intervals
    
    def evaluate(self, test_data: pd.DataFrame, target_col: str = 'pm25') -> Dict[str, float]:
        """Evaluate ensemble performance"""
        logger.info("Evaluating ensemble performance...")
        
        # Get predictions
        predictions = self.predict(test_data, target_col)
        actuals = test_data[target_col].values
        
        # Calculate metrics
        mae = np.mean(np.abs(actuals - predictions['mean']))
        rmse = np.sqrt(np.mean((actuals - predictions['mean'])**2))
        
        # Coverage metrics
        lower_95 = predictions['lower_95']
        upper_95 = predictions['upper_95']
        coverage_95 = np.mean((actuals >= lower_95) & (actuals <= upper_95))
        
        lower_68 = predictions['lower_68']
        upper_68 = predictions['upper_68']
        coverage_68 = np.mean((actuals >= lower_68) & (actuals <= upper_68))
        
        # Sharpness (average width of prediction intervals)
        sharpness_95 = np.mean(upper_95 - lower_95)
        sharpness_68 = np.mean(upper_68 - lower_68)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'coverage_95': coverage_95,
            'coverage_68': coverage_68,
            'sharpness_95': sharpness_95,
            'sharpness_68': sharpness_68
        }
        
        logger.info(f"Ensemble metrics: {metrics}")
        return metrics
    
    def save_ensemble(self, path: str):
        """Save the entire ensemble"""
        ensemble_state = {
            'config': self.config,
            'model_weights': self.model_weights,
            'is_fitted': self.is_fitted,
            'training_history': self.training_history
        }
        
        # Save ensemble metadata
        with open(f"{path}_ensemble.json", 'w') as f:
            json.dump(ensemble_state, f, indent=2, default=str)
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = f"{path}_{model_name}.pth"
            model.save_model(model_path)
        
        logger.info(f"Ensemble saved to {path}")
    
    def load_ensemble(self, path: str):
        """Load the entire ensemble"""
        # Load ensemble metadata
        with open(f"{path}_ensemble.json", 'r') as f:
            ensemble_state = json.load(f)
        
        self.config = ensemble_state['config']
        self.model_weights = ensemble_state['model_weights']
        self.is_fitted = ensemble_state['is_fitted']
        self.training_history = ensemble_state['training_history']
        
        # Load individual models
        for model_name in self.models.keys():
            model_path = f"{path}_{model_name}.pth"
            if Path(model_path).exists():
                self.models[model_name].load_model(model_path)
        
        logger.info(f"Ensemble loaded from {path}")
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get ensemble information"""
        model_info = {}
        for model_name, model in self.models.items():
            model_info[model_name] = model.get_model_info()
        
        return {
            'ensemble_type': 'Deep Ensemble',
            'n_models': len(self.models),
            'ensemble_method': self.config.ensemble_method,
            'use_uncertainty': self.config.use_uncertainty,
            'model_weights': self.model_weights,
            'is_fitted': self.is_fitted,
            'models': model_info,
            'training_history': self.training_history
        }
