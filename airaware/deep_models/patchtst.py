"""
PatchTST (Patch Time Series Transformer) Implementation

A state-of-the-art transformer-based model for time series forecasting
that uses patching to improve efficiency and performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class PatchTSTConfig:
    """Configuration for PatchTST model"""
    # Model architecture
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 512
    n_layers: int = 3
    dropout: float = 0.1
    
    # Patching parameters
    patch_len: int = 16
    stride: int = 8
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 10
    
    # Data parameters
    context_length: int = 96  # Input sequence length
    prediction_length: int = 24  # Forecast horizon
    
    # Feature parameters
    n_features: int = 1  # Number of input features
    use_external_features: bool = True
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PatchEmbedding(nn.Module):
    """Patch embedding layer for time series"""
    
    def __init__(self, patch_len: int, stride: int, d_model: int, n_features: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_features = n_features
        
        # Linear projection for patches
        self.patch_projection = nn.Linear(patch_len * n_features, d_model)
        
    def forward(self, x):
        # x: (batch_size, seq_len, n_features)
        batch_size, seq_len, n_features = x.shape
        
        # Ensure we have enough data for at least one patch
        if seq_len < self.patch_len:
            # Pad the sequence if it's too short
            padding = self.patch_len - seq_len
            x = F.pad(x, (0, 0, padding, 0), mode='replicate')
            seq_len = self.patch_len
        
        # Create patches
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len, :].contiguous()
            patches.append(patch)
        
        if not patches:
            # Handle case where sequence is too short
            patch = x[:, -self.patch_len:, :].contiguous()
            patches = [patch]
        
        # Stack patches: (batch_size, n_patches, patch_len * n_features)
        patches = torch.stack(patches, dim=1)
        patches = patches.view(batch_size, patches.size(1), -1)
        
        # Project to d_model
        patches = self.patch_projection(patches)  # (batch_size, n_patches, d_model)
        
        return patches

class PatchTSTModel(nn.Module):
    """PatchTST model implementation"""
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding (initialize with default n_features if available)
        if config.n_features is not None:
            self.patch_embedding = PatchEmbedding(
                patch_len=config.patch_len,
                stride=config.stride,
                d_model=config.d_model,
                n_features=config.n_features
            )
        else:
            self.patch_embedding = None
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.prediction_length)
        
        # External features projection (if used)
        if config.use_external_features:
            self.external_projection = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, external_features=None):
        # x: (batch_size, seq_len, n_features)
        batch_size = x.size(0)
        
        # Initialize patch embedding if not done yet
        if self.patch_embedding is None:
            n_features = x.size(-1)
            self.patch_embedding = PatchEmbedding(
                patch_len=self.config.patch_len,
                stride=self.config.stride,
                d_model=self.config.d_model,
                n_features=n_features
            ).to(x.device)
        
        # Patch embedding
        patches = self.patch_embedding(x)  # (batch_size, n_patches, d_model)
        
        # Add positional encoding
        patches = patches.transpose(0, 1)  # (n_patches, batch_size, d_model)
        patches = self.pos_encoding(patches)
        patches = patches.transpose(0, 1)  # (batch_size, n_patches, d_model)
        
        # Transformer encoding
        encoded = self.transformer(patches)  # (batch_size, n_patches, d_model)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)  # (batch_size, d_model)
        
        # Add external features if available
        if external_features is not None and self.config.use_external_features:
            external_proj = self.external_projection(external_features)
            pooled = pooled + external_proj
        
        # Output projection
        output = self.output_projection(pooled)  # (batch_size, prediction_length)
        
        return output

class PatchTSTForecaster:
    """PatchTST forecaster with training and prediction capabilities"""
    
    def __init__(self, config: PatchTSTConfig):
        self.config = config
        self.model = PatchTSTModel(config).to(config.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.criterion = nn.MSELoss()
        self.is_fitted = False
        self.training_history = []
        
        # Data normalization
        self.feature_scalers = {}
        self.target_scaler = None
        
    def _prepare_data(self, data: pd.DataFrame, target_col: str = 'pm25') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training/prediction"""
        # Ensure data is sorted by datetime
        data = data.sort_values('datetime_utc').reset_index(drop=True)
        
        # Extract features and target - only numeric columns
        feature_cols = [col for col in data.columns if col not in ['datetime_utc', target_col, 'station_id', 'data_source', 'parameter', 'unit', 'quality']]
        
        # Filter to only numeric columns
        numeric_cols = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                numeric_cols.append(col)
        
        if not numeric_cols:
            # If no numeric features, create dummy features
            numeric_cols = ['hour_of_day', 'day_of_week', 'month']
            for col in numeric_cols:
                if col not in data.columns:
                    if col == 'hour_of_day':
                        data[col] = data['datetime_utc'].dt.hour
                    elif col == 'day_of_week':
                        data[col] = data['datetime_utc'].dt.dayofweek
                    elif col == 'month':
                        data[col] = data['datetime_utc'].dt.month
        
        features = data[numeric_cols].values
        target = data[target_col].values
        
        # Normalize features
        if not self.is_fitted:
            from sklearn.preprocessing import StandardScaler
            self.feature_scalers = {}
            for i, col in enumerate(numeric_cols):
                scaler = StandardScaler()
                features[:, i] = scaler.fit_transform(features[:, i].reshape(-1, 1)).flatten()
                self.feature_scalers[col] = scaler
        else:
            for i, col in enumerate(numeric_cols):
                if col in self.feature_scalers:
                    features[:, i] = self.feature_scalers[col].transform(features[:, i].reshape(-1, 1)).flatten()
        
        # Normalize target
        if not self.is_fitted:
            from sklearn.preprocessing import StandardScaler
            self.target_scaler = StandardScaler()
            target = self.target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
        else:
            if self.target_scaler is not None:
                target = self.target_scaler.transform(target.reshape(-1, 1)).flatten()
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(data) - self.config.context_length - self.config.prediction_length + 1):
            seq_features = features[i:i+self.config.context_length]
            seq_target = target[i+self.config.context_length:i+self.config.context_length+self.config.prediction_length]
            
            sequences.append(seq_features)
            targets.append(seq_target)
        
        if not sequences:
            # Handle case where data is too short
            seq_features = features[-self.config.context_length:]
            seq_target = target[-self.config.prediction_length:]
            sequences = [seq_features]
            targets = [seq_target]
        
        # Convert to tensors
        X = torch.FloatTensor(np.array(sequences))
        y = torch.FloatTensor(np.array(targets))
        
        # Update config with actual number of features
        self.config.n_features = features.shape[1]
        
        return X, y, torch.FloatTensor(features)
    
    def fit(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None, 
            target_col: str = 'pm25') -> Dict[str, List[float]]:
        """Train the PatchTST model"""
        logger.info("🚀 Training PatchTST model...")
        
        # Prepare training data
        X_train, y_train, _ = self._prepare_data(train_data, target_col)

        # Ensure patch settings are compatible with context length
        if self.config.context_length % self.config.patch_len != 0:
            # Choose the largest patch_len that divides context_length and is <= current patch_len
            for p in range(self.config.patch_len, 0, -1):
                if self.config.context_length % p == 0:
                    logger.info(f"Adjusting patch_len from {self.config.patch_len} to {p} to divide context_length {self.config.context_length}")
                    self.config.patch_len = p
                    break
            # Recreate patch embedding below
        
        # Recreate patch embedding with the ACTUAL number of features
        n_features = X_train.shape[-1]
        if getattr(self.model, 'patch_embedding', None) is None or \
           getattr(self.model.patch_embedding, 'n_features', None) != n_features or \
           getattr(self.model.patch_embedding, 'patch_len', None) != self.config.patch_len:
            self.model.patch_embedding = PatchEmbedding(
                patch_len=self.config.patch_len,
                stride=self.config.stride,
                d_model=self.config.d_model,
                n_features=n_features
            ).to(self.config.device)
            # Update config for persistence
            self.config.n_features = n_features

        # Prepare validation data if provided
        if val_data is not None:
            X_val, y_val, _ = self._prepare_data(val_data, target_col)
            # Keep embedding consistent
            if X_val.shape[-1] != n_features:
                logger.warning("Validation features dimension differs from training; aligning to training dim")
                # Trim or pad to match training n_features
                if X_val.shape[-1] > n_features:
                    X_val = X_val[..., :n_features]
                else:
                    pad = n_features - X_val.shape[-1]
                    X_val = torch.cat([X_val, torch.zeros(X_val.size(0), X_val.size(1), pad)], dim=-1)
        else:
            X_val, y_val = None, None
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        
        if X_val is not None:
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False
            )
        else:
            val_loader = None
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.config.device)
                batch_y = batch_y.to(self.config.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.config.device)
                        batch_y = batch_y.to(self.config.device)
                        
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                history['val_loss'].append(0.0)
            
            if (epoch + 1) % 10 == 0:
                if val_loader is not None:
                    val_str = f"{avg_val_loss:.4f}"
                else:
                    val_str = "N/A"
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_str}"
                )
        
        self.is_fitted = True
        self.training_history = history
        
        logger.info("✅ PatchTST training completed!")
        return history
    
    def predict(self, data: pd.DataFrame, target_col: str = 'pm25') -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        X, _, _ = self._prepare_data(data, target_col)
        
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X), self.config.batch_size):
                batch_X = X[i:i+self.config.batch_size].to(self.config.device)
                batch_pred = self.model(batch_X)
                predictions.append(batch_pred.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        
        # Inverse transform predictions
        if self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions)
        
        return predictions
    
    def save_model(self, path: str):
        """Save the trained model"""
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_scalers': self.feature_scalers,
            'target_scaler': self.target_scaler,
            'is_fitted': self.is_fitted,
            'training_history': self.training_history
        }
        
        torch.save(model_state, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        # Use weights_only=False for trusted model files with custom config classes
        model_state = torch.load(path, map_location=self.config.device, weights_only=False)
        
        # Update config from saved state
        self.config = model_state['config']
        self.feature_scalers = model_state['feature_scalers']
        self.target_scaler = model_state['target_scaler']
        self.is_fitted = model_state['is_fitted']
        self.training_history = model_state['training_history']
        
        # Reinitialize model with saved config to match architecture
        self.model = PatchTSTModel(self.config)
        self.model.to(self.config.device)
        
        # Initialize patch embedding with the correct n_features from saved config
        if self.config.n_features is not None:
            self.model.patch_embedding = PatchEmbedding(
                patch_len=self.config.patch_len,
                stride=self.config.stride,
                d_model=self.config.d_model,
                n_features=self.config.n_features
            ).to(self.config.device)
        
        # Load state dict
        self.model.load_state_dict(model_state['model_state_dict'])
        
        logger.info(f"Model loaded from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': 'PatchTST',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': self.config,
            'is_fitted': self.is_fitted,
            'device': self.config.device,
            'training_history': self.training_history
        }
