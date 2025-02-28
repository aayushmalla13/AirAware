"""
Temporal Fusion Transformer (TFT) Implementation

A state-of-the-art deep learning model for time series forecasting
that handles both static and time-varying features with attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class TFTConfig:
    """Configuration for TFT model"""
    # Model architecture
    hidden_size: int = 64
    lstm_layers: int = 2
    dropout: float = 0.1
    attention_head_size: int = 4
    num_attention_heads: int = 4
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 10
    
    # Data parameters
    context_length: int = 96  # Input sequence length
    prediction_length: int = 24  # Forecast horizon
    
    # Feature parameters
    num_static_features: int = 0  # Static features (e.g., station_id)
    num_time_varying_features: int = 1  # Time-varying features (e.g., PM2.5)
    num_known_features: int = 0  # Known future features (e.g., weather forecasts)
    
    # Quantile regression
    quantiles: List[float] = None
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.quantiles is None:
            self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

class VariableSelectionNetwork(nn.Module):
    """Variable selection network for TFT"""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Variable selection weights
        self.variable_selection = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        # Apply variable selection
        weights = torch.softmax(self.variable_selection(x), dim=-1)
        selected = weights * x
        
        return selected, weights

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for TFT"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 dropout: float = 0.1, context_size: int = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        
        # Main layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
        # Context layer (if provided)
        if context_size is not None:
            self.context_layer = nn.Linear(context_size, hidden_size)
        else:
            self.context_layer = None
        
        # Gating layer
        self.gate = nn.Linear(hidden_size, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size)
        
    def forward(self, x, context=None):
        # x: (batch_size, seq_len, input_size)
        # context: (batch_size, seq_len, context_size) or None
        
        # First linear layer
        h1 = F.elu(self.linear1(x))
        
        # Add context if provided
        if self.context_layer is not None and context is not None:
            context_proj = self.context_layer(context)
            h1 = h1 + context_proj
        
        # Second linear layer
        h2 = F.elu(self.linear2(h1))
        h2 = self.dropout(h2)
        
        # Third linear layer
        h3 = self.linear3(h2)
        
        # Gating
        gate = torch.sigmoid(self.gate(h2))
        output = gate * h3
        
        # Residual connection and layer norm
        if x.size(-1) == self.output_size:
            output = output + x
        else:
            # Project input to output size
            output = output + F.linear(x, torch.eye(self.output_size, x.size(-1)).to(x.device))
        
        output = self.layer_norm(output)
        
        return output

class TemporalSelfAttention(nn.Module):
    """Temporal self-attention mechanism for TFT"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = x.size()
        
        # Store residual
        residual = x
        
        # Linear projections
        Q = self.q_linear(x)  # (batch_size, seq_len, hidden_size)
        K = self.k_linear(x)  # (batch_size, seq_len, hidden_size)
        V = self.v_linear(x)  # (batch_size, seq_len, hidden_size)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_size)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        
        # Output projection
        output = self.out_linear(attn_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output, attn_weights

class TFTModel(nn.Module):
    """Temporal Fusion Transformer model"""
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        
        # Feature embeddings (will be updated after data preparation)
        self.static_embedding = None
        self.time_varying_embedding = None
        self.known_embedding = None
        
        # Variable selection networks
        self.static_variable_selection = VariableSelectionNetwork(
            config.hidden_size, config.hidden_size, config.dropout
        )
        self.time_varying_variable_selection = VariableSelectionNetwork(
            config.hidden_size, config.hidden_size, config.dropout
        )
        self.known_variable_selection = VariableSelectionNetwork(
            config.hidden_size, config.hidden_size, config.dropout
        )
        
        # Gated Residual Networks
        self.static_grn = GatedResidualNetwork(
            config.hidden_size, config.hidden_size, config.hidden_size, config.dropout
        )
        self.time_varying_grn = GatedResidualNetwork(
            config.hidden_size, config.hidden_size, config.hidden_size, config.dropout
        )
        self.known_grn = GatedResidualNetwork(
            config.hidden_size, config.hidden_size, config.hidden_size, config.dropout
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            config.hidden_size, config.hidden_size, 
            config.lstm_layers, batch_first=True, dropout=config.dropout
        )
        
        # Temporal self-attention
        self.temporal_attention = TemporalSelfAttention(
            config.hidden_size, config.num_attention_heads, config.dropout
        )
        
        # Output layers for quantile regression
        self.output_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.prediction_length)
            for _ in config.quantiles
        ])
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, static_features, time_varying_features, known_features=None):
        # static_features: (batch_size, num_static_features)
        # time_varying_features: (batch_size, seq_len, num_time_varying_features)
        # known_features: (batch_size, seq_len, num_known_features) or None
        
        batch_size, seq_len, _ = time_varying_features.size()
        
        # Initialize embeddings if not done yet
        if self.static_embedding is None:
            num_static = static_features.size(-1)
            num_time_varying = time_varying_features.size(-1)
            num_known = known_features.size(-1) if known_features is not None else 0
            
            self.static_embedding = nn.Linear(num_static, self.config.hidden_size).to(static_features.device)
            self.time_varying_embedding = nn.Linear(num_time_varying, self.config.hidden_size).to(time_varying_features.device)
            if num_known > 0:
                self.known_embedding = nn.Linear(num_known, self.config.hidden_size).to(time_varying_features.device)
        
        # Embed features
        static_emb = self.static_embedding(static_features)  # (batch_size, hidden_size)
        time_varying_emb = self.time_varying_embedding(time_varying_features)  # (batch_size, seq_len, hidden_size)
        
        if known_features is not None:
            known_emb = self.known_embedding(known_features)  # (batch_size, seq_len, hidden_size)
        else:
            known_emb = None
        
        # Variable selection
        static_selected, static_weights = self.static_variable_selection(
            static_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
        )
        time_varying_selected, time_varying_weights = self.time_varying_variable_selection(time_varying_emb)
        
        if known_emb is not None:
            known_selected, known_weights = self.known_variable_selection(known_emb)
        else:
            known_selected = None
            known_weights = None
        
        # Gated Residual Networks
        static_processed = self.static_grn(static_selected)
        time_varying_processed = self.time_varying_grn(time_varying_selected)
        
        if known_selected is not None:
            known_processed = self.known_grn(known_selected)
        else:
            known_processed = None
        
        # Combine features
        combined_features = static_processed + time_varying_processed
        if known_processed is not None:
            combined_features = combined_features + known_processed
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined_features)  # (batch_size, seq_len, hidden_size)
        
        # Temporal self-attention
        attn_out, attn_weights = self.temporal_attention(lstm_out)
        
        # Global average pooling
        pooled = attn_out.mean(dim=1)  # (batch_size, hidden_size)
        
        # Output projections for quantile regression
        outputs = []
        for output_layer in self.output_layers:
            output = output_layer(pooled)  # (batch_size, prediction_length)
            outputs.append(output)
        
        return outputs, attn_weights

class TFTForecaster:
    """TFT forecaster with training and prediction capabilities"""
    
    def __init__(self, config: TFTConfig):
        self.config = config
        self.model = TFTModel(config).to(config.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.is_fitted = False
        self.training_history = []
        
        # Data normalization
        self.feature_scalers = {}
        self.target_scaler = None
        
    def _prepare_data(self, data: pd.DataFrame, target_col: str = 'pm25') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training/prediction"""
        # Ensure data is sorted by datetime
        data = data.sort_values('datetime_utc').reset_index(drop=True)
        
        # Extract features - only numeric columns
        feature_cols = [col for col in data.columns if col not in ['datetime_utc', target_col, 'station_id', 'data_source', 'parameter', 'unit', 'quality', 'quality_flag']]
        
        # Filter to only numeric columns
        numeric_cols = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                numeric_cols.append(col)
        
        if not numeric_cols:
            # If no numeric features, create dummy features
            numeric_cols = ['hour', 'day_of_week', 'month']
            for col in numeric_cols:
                if col not in data.columns:
                    if col == 'hour':
                        data[col] = data['datetime_utc'].dt.hour
                    elif col == 'day_of_week':
                        data[col] = data['datetime_utc'].dt.dayofweek
                    elif col == 'month':
                        data[col] = data['datetime_utc'].dt.month
        
        features = data[numeric_cols].values
        target = data[target_col].values
        
        # Create static features (station_id as one-hot)
        if 'station_id' in data.columns:
            station_ids = data['station_id'].unique()
            station_to_idx = {sid: i for i, sid in enumerate(station_ids)}
            static_features = np.array([station_to_idx[sid] for sid in data['station_id']])
            # Convert to one-hot
            static_one_hot = np.zeros((len(static_features), len(station_ids)))
            static_one_hot[np.arange(len(static_features)), static_features] = 1
        else:
            static_one_hot = np.zeros((len(data), 1))  # Dummy static feature
        
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
        static_seqs = []
        
        for i in range(len(data) - self.config.context_length - self.config.prediction_length + 1):
            seq_features = features[i:i+self.config.context_length]
            # Static features should be the same for all time steps in the sequence
            seq_static = static_one_hot[i]  # Take first static feature for the sequence
            seq_target = target[i+self.config.context_length:i+self.config.context_length+self.config.prediction_length]
            
            sequences.append(seq_features)
            static_seqs.append(seq_static)
            targets.append(seq_target)
        
        if not sequences:
            # Handle case where data is too short
            seq_features = features[-self.config.context_length:]
            seq_static = static_one_hot[-1]  # Take last static feature
            seq_target = target[-self.config.prediction_length:]
            sequences = [seq_features]
            static_seqs = [seq_static]
            targets = [seq_target]
        
        # Convert to tensors
        X = torch.FloatTensor(np.array(sequences))
        X_static = torch.FloatTensor(np.array(static_seqs))
        y = torch.FloatTensor(np.array(targets))
        
        # Update config with actual number of features
        self.config.num_time_varying_features = features.shape[1]
        self.config.num_static_features = static_one_hot.shape[1]
        
        return X, X_static, y, torch.FloatTensor(features)
    
    def _quantile_loss(self, predictions, targets, quantiles):
        """Calculate quantile loss"""
        losses = []
        for i, q in enumerate(quantiles):
            error = targets - predictions[i]
            loss = torch.max((q - 1) * error, q * error)
            losses.append(loss.mean())
        return torch.stack(losses).mean()
    
    def fit(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None, 
            target_col: str = 'pm25') -> Dict[str, List[float]]:
        """Train the TFT model"""
        logger.info("🚀 Training TFT model...")
        
        # Prepare training data
        X_train, X_static_train, y_train, _ = self._prepare_data(train_data, target_col)
        
        # Prepare validation data if provided
        if val_data is not None:
            X_val, X_static_val, y_val, _ = self._prepare_data(val_data, target_col)
        else:
            X_val, X_static_val, y_val = None, None, None
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, X_static_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        
        if X_val is not None:
            val_dataset = torch.utils.data.TensorDataset(X_val, X_static_val, y_val)
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
            
            for batch_X, batch_X_static, batch_y in train_loader:
                batch_X = batch_X.to(self.config.device)
                batch_X_static = batch_X_static.to(self.config.device)
                batch_y = batch_y.to(self.config.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions, _ = self.model(batch_X_static, batch_X)
                
                # Calculate quantile loss
                loss = self._quantile_loss(predictions, batch_y, self.config.quantiles)
                
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
                    for batch_X, batch_X_static, batch_y in val_loader:
                        batch_X = batch_X.to(self.config.device)
                        batch_X_static = batch_X_static.to(self.config.device)
                        batch_y = batch_y.to(self.config.device)
                        
                        predictions, _ = self.model(batch_X_static, batch_X)
                        loss = self._quantile_loss(predictions, batch_y, self.config.quantiles)
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
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f if val_loader else 'N/A'}")
        
        self.is_fitted = True
        self.training_history = history
        
        logger.info("✅ TFT training completed!")
        return history
    
    def predict(self, data: pd.DataFrame, target_col: str = 'pm25') -> Dict[str, np.ndarray]:
        """Make predictions with uncertainty quantification"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        X, X_static, _, _ = self._prepare_data(data, target_col)
        
        predictions = {f'q{q}': [] for q in self.config.quantiles}
        
        with torch.no_grad():
            for i in range(0, len(X), self.config.batch_size):
                batch_X = X[i:i+self.config.batch_size].to(self.config.device)
                batch_X_static = X_static[i:i+self.config.batch_size].to(self.config.device)
                
                batch_predictions, _ = self.model(batch_X_static, batch_X)
                
                for j, q in enumerate(self.config.quantiles):
                    pred = batch_predictions[j].cpu().numpy()
                    predictions[f'q{q}'].append(pred)
        
        # Concatenate predictions
        for q in self.config.quantiles:
            predictions[f'q{q}'] = np.concatenate(predictions[f'q{q}'], axis=0)
            
            # Inverse transform predictions
            if self.target_scaler is not None:
                predictions[f'q{q}'] = self.target_scaler.inverse_transform(predictions[f'q{q}'])
        
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
        self.model = TFTModel(self.config)
        self.model.to(self.config.device)
        
        # Load state dict
        self.model.load_state_dict(model_state['model_state_dict'])
        
        logger.info(f"Model loaded from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': 'TFT',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': self.config,
            'is_fitted': self.is_fitted,
            'device': self.config.device,
            'training_history': self.training_history,
            'quantiles': self.config.quantiles
        }
