"""
Model Trainer for Deep Learning Models

This module provides training utilities and configurations for deep learning models,
including hyperparameter optimization, cross-validation, and model selection.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 10
    
    # Optimization
    optimizer: str = "adam"  # "adam", "adamw", "sgd", "rmsprop"
    weight_decay: float = 1e-5
    scheduler: str = "plateau"  # "plateau", "cosine", "step", "none"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Regularization
    dropout: float = 0.1
    early_stopping: bool = True
    gradient_clipping: float = 1.0
    
    # Validation
    validation_split: float = 0.2
    use_cross_validation: bool = False
    cv_folds: int = 5
    
    # Device
    device: str = "auto"
    
    # Logging
    log_interval: int = 10
    save_best_model: bool = True
    model_save_path: str = "models"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class HyperparameterSearchConfig:
    """Configuration for hyperparameter search"""
    # Search parameters
    n_trials: int = 50
    timeout: Optional[int] = None  # seconds
    direction: str = "minimize"  # "minimize" or "maximize"
    
    # Search space
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-2)
    batch_size_options: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    dropout_range: Tuple[float, float] = (0.0, 0.5)
    
    # Pruning
    use_pruning: bool = True
    pruning_patience: int = 5
    
    # Parallelization
    n_jobs: int = 1

class DeepModelTrainer:
    """Trainer for deep learning models with advanced features"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.training_history = {}
        self.best_model_state = None
        self.best_metric = float('inf')
        
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer based on configuration"""
        if self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        elif self.config.optimizer.lower() == "rmsprop":
            return torch.optim.RMSprop(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration"""
        if self.config.scheduler.lower() == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor
            )
        elif self.config.scheduler.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler.lower() == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=self.config.scheduler_factor
            )
        else:
            return None
    
    def train_model(self, model: nn.Module, train_loader: torch.utils.data.DataLoader,
                   val_loader: Optional[torch.utils.data.DataLoader] = None,
                   criterion: nn.Module = nn.MSELoss()) -> Dict[str, List[float]]:
        """Train a model with the given configuration"""
        logger.info("🚀 Starting model training...")
        
        # Move model to device
        model = model.to(self.config.device)
        
        # Create optimizer and scheduler
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.config.device)
                target = target.to(self.config.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clipping)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data = data.to(self.config.device)
                        target = target.to(self.config.device)
                        output = model(data)
                        loss = criterion(output, target)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)
                
                # Learning rate scheduling
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()
                
                # Early stopping
                if self.config.early_stopping:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        
                        # Save best model
                        if self.config.save_best_model:
                            self.best_model_state = model.state_dict().copy()
                            self.best_metric = avg_val_loss
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                history['val_loss'].append(0.0)
            
            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
            
            # Logging
            if (epoch + 1) % self.config.log_interval == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f if val_loader else 'N/A'}, "
                          f"LR: {current_lr:.2e}")
        
        # Load best model if early stopping was used
        if self.config.early_stopping and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            logger.info("Loaded best model state")
        
        logger.info("✅ Model training completed!")
        return history
    
    def cross_validate(self, model_factory: Callable, train_data: pd.DataFrame,
                      target_col: str = 'pm25', n_splits: int = 5) -> Dict[str, List[float]]:
        """Perform time series cross-validation"""
        logger.info(f"🚀 Starting {n_splits}-fold cross-validation...")
        
        # Create time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(train_data)):
            logger.info(f"Fold {fold + 1}/{n_splits}")
            
            # Split data
            train_fold = train_data.iloc[train_idx]
            val_fold = train_data.iloc[val_idx]
            
            # Create model for this fold
            model = model_factory()
            
            # Prepare data (this would need to be implemented based on the specific model)
            # For now, we'll use a placeholder
            train_loader = self._prepare_data_loader(train_fold, target_col)
            val_loader = self._prepare_data_loader(val_fold, target_col)
            
            # Train model
            history = self.train_model(model, train_loader, val_loader)
            
            # Evaluate
            train_mae, train_rmse = self._evaluate_model(model, train_loader)
            val_mae, val_rmse = self._evaluate_model(model, val_loader)
            
            cv_scores['train_mae'].append(train_mae)
            cv_scores['val_mae'].append(val_mae)
            cv_scores['train_rmse'].append(train_rmse)
            cv_scores['val_rmse'].append(val_rmse)
            
            logger.info(f"Fold {fold + 1} - Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")
        
        # Calculate mean and std
        cv_summary = {}
        for metric, scores in cv_scores.items():
            cv_summary[f'{metric}_mean'] = np.mean(scores)
            cv_summary[f'{metric}_std'] = np.std(scores)
        
        logger.info("✅ Cross-validation completed!")
        logger.info(f"CV Results: {cv_summary}")
        
        return cv_summary
    
    def hyperparameter_search(self, model_factory: Callable, train_data: pd.DataFrame,
                            val_data: pd.DataFrame, target_col: str = 'pm25',
                            search_config: HyperparameterSearchConfig = None) -> Dict[str, Any]:
        """Perform hyperparameter optimization using Optuna"""
        if search_config is None:
            search_config = HyperparameterSearchConfig()
        
        logger.info("🚀 Starting hyperparameter search...")
        
        def objective(trial):
            # Sample hyperparameters
            lr = trial.suggest_float('learning_rate', *search_config.learning_rate_range, log=True)
            batch_size = trial.suggest_categorical('batch_size', search_config.batch_size_options)
            dropout = trial.suggest_float('dropout', *search_config.dropout_range)
            
            # Create model with sampled hyperparameters
            model = model_factory(learning_rate=lr, batch_size=batch_size, dropout=dropout)
            
            # Prepare data
            train_loader = self._prepare_data_loader(train_data, target_col, batch_size)
            val_loader = self._prepare_data_loader(val_data, target_col, batch_size)
            
            # Train model
            history = self.train_model(model, train_loader, val_loader)
            
            # Return validation loss
            return min(history['val_loss'])
        
        # Create study
        study = optuna.create_study(
            direction=search_config.direction,
            sampler=TPESampler(),
            pruner=MedianPruner() if search_config.use_pruning else None
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=search_config.n_trials,
            timeout=search_config.timeout,
            n_jobs=search_config.n_jobs
        )
        
        logger.info("✅ Hyperparameter search completed!")
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best value: {study.best_value}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }
    
    def _prepare_data_loader(self, data: pd.DataFrame, target_col: str, 
                           batch_size: Optional[int] = None) -> torch.utils.data.DataLoader:
        """Prepare data loader (placeholder implementation)"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # This is a placeholder - actual implementation would depend on the specific model
        # For now, return a dummy loader
        dummy_data = torch.randn(len(data), 10)  # Placeholder
        dummy_target = torch.randn(len(data), 1)  # Placeholder
        
        dataset = torch.utils.data.TensorDataset(dummy_data, dummy_target)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def _evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Evaluate model performance"""
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.config.device)
                output = model(data)
                predictions.extend(output.cpu().numpy())
                targets.extend(target.numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        
        return mae, rmse
    
    def save_training_history(self, path: str):
        """Save training history"""
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {path}")
    
    def load_training_history(self, path: str):
        """Load training history"""
        with open(path, 'r') as f:
            self.training_history = json.load(f)
        logger.info(f"Training history loaded from {path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        return {
            'config': self.config,
            'training_history': self.training_history,
            'best_metric': self.best_metric,
            'device': self.config.device
        }
